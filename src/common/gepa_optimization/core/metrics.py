from typing import Callable, Optional, List
from core.template_handler import ParsingOutput
from solver.logic_evaluator import LogicEvaluator
from rouge_score import rouge_scorer
from tqdm import tqdm
import logging
import time
from core.le_evaluator import LEEvaluator

####################################################################
# Logical Equivalence Evaluation
####################################################################

# Global LE evaluator instance (reused across calls for efficiency)
_le_evaluator: Optional[LEEvaluator] = None

# Global ROUGE scorer instance (reused across calls for efficiency)
_rouge_scorer: Optional[rouge_scorer.RougeScorer] = None

def _get_le_evaluator(max_literals: int = 12) -> LEEvaluator:
    """Get or create a global LE evaluator instance."""
    global _le_evaluator
    if _le_evaluator is None or _le_evaluator.max_literals != max_literals:
        _le_evaluator = LEEvaluator(max_literals=max_literals)
    return _le_evaluator

def _get_rouge_scorer() -> rouge_scorer.RougeScorer:
    """Get or create a global ROUGE scorer instance."""
    global _rouge_scorer
    if _rouge_scorer is None:
        logging.info("Initializing ROUGE scorer (first time only)...")
        _rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        logging.info("ROUGE scorer initialized successfully")
    return _rouge_scorer

def _compute_le_score(
    proof_steps: List[str],
    reference_proof: List[str],
    invalid_indices: List[int],
    max_literals: int = 12
) -> float:
    """Compute LE score by comparing proof steps individually.

    Each generated step finds its best matching reference step.
    Final score is the average of all best matches.

    Args:
        proof_steps: List of generated proof step strings
        reference_proof: Reference proof as a list of formula strings
        invalid_indices: List of indices of invalid proof steps (to skip)
        max_literals: Maximum number of literals for LE computation

    Returns:
        LE score (0.0-1.0) as average of best step matches
    """
    # Handle reference_proof as either string or list
    assert isinstance(reference_proof, list) and len(reference_proof) > 0, "Reference proof must be a non-empty list of strings."
    ref_steps = [s.strip() for s in reference_proof if s.strip()]

    if not ref_steps:
        return 0.0

    # Get valid generated steps (skip invalid ones)
    valid_gen_steps = [
        proof_steps[i].strip()
        for i in range(len(proof_steps))
        if i not in invalid_indices and proof_steps[i].strip()
    ]

    if not valid_gen_steps:
        return 0.0

    le_evaluator = _get_le_evaluator(max_literals)

    # Compare each generated step to find best matching reference
    scores = []
    for gen_step in valid_gen_steps:
        best_score = 0.0
        for ref_step in ref_steps:
            result = le_evaluator.compute_le_score(gen_step, ref_step)
            if result.success:
                best_score = max(best_score, result.score)
        scores.append(best_score)

    return sum(scores) / len(scores) if scores else 0.0


###################################################################
# Metric computation for individual completions
###################################################################

def compute_single_completion_metrics(
        completion: str,
        completion_parser: Callable[[str], ParsingOutput],
        ground_truth_answer: Optional[str] = None,
        ground_truth_proof: Optional[List[str]] = None
    ) -> dict:
    """Compute metrics for a single completion, including both logic and answer metrics.

    Args:
        completion: The model completion string
        completion_parser: Function mapping completion string to ParsingOutput
        ground_truth_answer: Optional ground truth answer for answer correctness checking

    Returns:
        Dictionary containing:
            Logic metrics (computed if proof is present in ParsingOutput):
                - 'parse': 1 if parsing succeeded, 0 otherwise
                - 'valid_fol': 1 if minimum valid formulas exist, 0 otherwise
                - 'is_consistent': 1 if no contradictions found, 0 otherwise
                - 'final_step_entail': 1 if final step is entailed, 0 otherwise
                - 'validity_rate': float (0.0-1.0) proportion of valid formulas
                - 'valid_fact_count': int number of valid fact formulas
                - 'valid_proof_count': int number of valid proof step formulas
                - 'total_valid_formulas': int total number of valid formulas

            Answer metrics (computed if answer is present in ParsingOutput and ground_truth_answer is provided):
                - 'exact_match': 1 if answer matches ground truth exactly, 0 otherwise
                - 'substring_match': 1 if answer is substring of ground truth or vice versa, 0 otherwise
                - 'rouge_l': ROUGE-L F1 score between answer and ground truth (0.0 to 1.0)
    """
    logging.debug("Starting metric computation...")

    metrics = {
        'parse': 0,
        'valid_fol': 0,
        'is_consistent': 0,
        'final_step_entail': 0,
        'validity_rate': 0.0,
        'valid_fact_count': 0,
        'valid_proof_count': 0,
        'total_valid_formulas': 0
    }

    # Try to parse the output
    logging.debug("Parsing completion...")
    parsing = completion_parser(completion)
    if not parsing.success:
        logging.info(f"Parsing failed! format_quality={parsing.format_quality if hasattr(parsing, 'format_quality') else 'N/A'}")
        # Add answer metrics as 0 if ground_truth_answer is provided
        if ground_truth_answer is not None:
            metrics['exact_match'] = 0
            metrics['substring_match'] = 0
            metrics['rouge_l'] = 0.0
        return metrics  # Parsing failed

    metrics['parse'] = 1

    # Compute answer metrics if both answer and ground truth are available
    if ground_truth_answer is not None and parsing.answer is not None:
        logging.info("Computing answer metrics...")
        predicted = parsing.answer.strip().lower()
        ground_truth = ground_truth_answer.strip().lower()

        # Exact match
        logging.info(f"Exact match check: predicted='{predicted}' vs ground_truth='{ground_truth}'")
        if predicted == ground_truth:
            metrics['exact_match'] = 1
            metrics['substring_match'] = 1  # Exact match is also a substring match
        # Substring match
        elif predicted in ground_truth or ground_truth in predicted:
            metrics['exact_match'] = 0
            metrics['substring_match'] = 1
        else:
            metrics['exact_match'] = 0
            metrics['substring_match'] = 0

        # ROUGE-L
        logging.debug("Computing ROUGE-L score...")
        scorer = _get_rouge_scorer()
        rouge_scores = scorer.score(ground_truth, predicted)
        metrics['rouge_l'] = rouge_scores['rougeL'].fmeasure
        logging.debug(f"ROUGE-L score computed: {metrics['rouge_l']:.4f}")
    elif ground_truth_answer is not None:
        # Ground truth provided but no answer in parsing output
        logging.info(f"Answer extraction failed: parsing.answer is None, ground_truth='{ground_truth_answer}'")
        metrics['exact_match'] = 0
        metrics['substring_match'] = 0
        metrics['rouge_l'] = 0.0

    # Compute logic metrics if proof is available
    logging.debug("Computing logic metrics...")
    facts = parsing.facts
    proof = parsing.proof

    if not proof:
        logging.debug("No proof found in parsing output")
        return metrics  # No proof to evaluate

    # Split proof into individual steps (one per line, filter empty lines)
    proof_steps = [step.strip() for step in proof.split('\n') if step.strip()]
    logging.debug(f"Found {len(proof_steps)} proof steps")

    if not proof_steps:
        logging.debug("No valid proof steps after filtering")
        return metrics  # No valid proof steps

    # If facts are not present and only one proof step, consider it invalid FOL
    if not facts and len(proof_steps) == 1:
        logging.debug("Only one proof step with no facts - skipping logic evaluation")
        return metrics

    logging.debug(f"Initializing LogicEvaluator with {len(facts) if facts else 0} facts and {len(proof_steps)} proof steps")
    logic_evaluator = LogicEvaluator(facts, proof_steps, timeout=5)

    # Extract validity metrics
    metrics['validity_rate'] = logic_evaluator.get_valid_rate()
    valid_counts = logic_evaluator.get_valid_formula_count()
    metrics['valid_fact_count'] = valid_counts['facts']
    metrics['valid_proof_count'] = valid_counts['proof_steps']
    metrics['total_valid_formulas'] = valid_counts['total']

    # Check if minimum valid formulas exist to proceed
    if not logic_evaluator.has_minimum_valid_formulas():
        return metrics  # Not enough valid formulas
    metrics['valid_fol'] = 1

    if ground_truth_proof:
        metrics['le_score'] = _compute_le_score(
            proof_steps=proof_steps,
            reference_proof=ground_truth_proof,
            invalid_indices=logic_evaluator.invalid_proof_indices,
        )

    # Check consistency (uses only valid formulas)
    logging.debug("Checking consistency with Prover9...")
    is_consistent = logic_evaluator.is_consistent()
    if not is_consistent:
        logging.debug("Inconsistency detected")
        return metrics  # Contradiction found
    logging.debug("Consistency check passed")
    metrics['is_consistent'] = 1

    # Check final step entailment (uses only valid formulas)
    logging.debug("Checking final step entailment...")
    if logic_evaluator.is_entailed():
        metrics['final_step_entail'] = 1

    logging.debug("Metric computation completed successfully")
    return metrics


def _get_default_failed_metrics(include_answer_metrics: bool = False) -> dict:
    """Return default (all zeros) metrics for failed/timed-out computation.

    Args:
        include_answer_metrics: If True, include answer evaluation metrics (exact_match, etc.)

    Returns:
        Dictionary with all metric keys set to 0 or 0.0
    """
    metrics = {
        'parse': 0,
        'valid_fol': 0,
        'is_consistent': 0,
        'final_step_entail': 0,
        'validity_rate': 0.0,
        'valid_fact_count': 0,
        'valid_proof_count': 0,
        'total_valid_formulas': 0
    }

    if include_answer_metrics:
        metrics.update({
            'exact_match': 0,
            'substring_match': 0,
            'rouge_l': 0.0
        })

    return metrics