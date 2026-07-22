"""
GEPA Prompt Optimization for FOL Proof Chain Generation

This script optimizes prompts for First-Order Logic (FOL) proof generation using
the GEPA (Genetic-Pareto) optimizer from DSPy. It combines standard logic evaluation
metrics with LLM-as-a-judge assessment for comprehensive prompt optimization.

The optimization targets two main aspects:
1. Standard metrics: Parsing success, FOL validity, consistency, entailment
2. LLM judge metrics: Format quality and logical reasoning quality
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import dspy
import pandas as pd
import os

# Import project modules
from core.template_handler import TemplateHandler
from core.dataset_handler import DatasetHandler
from core.utils import ConfigArgumentParser


# ============================================================================
# LLM AS JUDGE - EVALUATION PROMPTS
# ============================================================================

# Prompt for separated facts and proof sections
LLM_JUDGE_PROMPT_SEPARATED = """You are an expert evaluator assessing First-Order Logic (FOL) proof chain generation quality for multi-hop reasoning tasks.

The task requires:
1. ALL content must be inside a single <logic>...</logic> block
2. Within the <logic> block, facts and proof steps MUST BE CLEARLY SEPARATED using section markers (like "FACTS:" and "PROOF:", or "---FACTS---" and "---PROOF---", or similar clear separators)
3. ONLY valid FOL formulas allowed - NO numbered lists, NO periods after formulas, NO natural language sentences
4. Every formula must use ONLY these symbols: ∀ ∃ ∧ ∨ ¬ → ↔ ⊕ ( ) , and alphanumeric predicates/constants

========================================
CRITICAL FORMAT VIOLATIONS (Always score FORMAT=1)
========================================

❌ NEVER ACCEPTABLE:
- Numbered lists: "1. P(x)" or "Step 1: P(x)"
- Periods after formulas: "P(x)."
- Invalid symbols: "∴" (therefore), "." (period), "•", etc. (colons are OK only in section markers)
- Natural language in proof: "From premises 1 and 2, it follows that..."
- Separate <facts> and <proof> blocks OUTSIDE <logic>
- Missing <logic> delimiters entirely
- NO CLEAR SEPARATOR between facts and proof sections (this is CRITICAL)

✓ CORRECT FORMAT (with clear separation):
<logic>
FACTS:
Predicate1(constant1)
Predicate2(constant2, constant3)

PROOF:
Predicate1(constant1)
Predicate1(constant1) → Predicate3(constant1)
Predicate3(constant1)
</logic>

✓ ALSO ACCEPTABLE (alternative separators):
<logic>
---FACTS---
Predicate1(constant1)

---PROOF---
Predicate1(constant1) → Predicate3(constant1)
</logic>

✗ WRONG (no clear separator):
<logic>
Predicate1(constant1)
Predicate2(constant2)
Predicate1(constant1) → Predicate3(constant1)
Predicate3(constant1)
</logic>

========================================
EVALUATION CRITERIA (1-5 scale each)
========================================

**1. FORMAT_QUALITY (1-5)**
Score 5: Perfect - single <logic> block, CLEAR FACTS/PROOF separation (with explicit markers), valid FOL only
Score 4: Minor issues - has separation markers but slight formatting problems (e.g., inconsistent spacing)
Score 3: Moderate - weak or unclear separation (e.g., only whitespace/blank line to separate), or some invalid symbols
Score 2: Major - NO CLEAR SEPARATOR between facts and proof, or numbered lists, or multiple periods
Score 1: Critical - natural language, missing delimiters, or separate <facts>/<proof> blocks outside <logic>

IMPORTANT: If there is NO CLEAR TEXTUAL SEPARATOR (like "FACTS:", "PROOF:", "---FACTS---", etc.) between the facts section and proof section, FORMAT_QUALITY MUST be 2 or lower.

**2. LOGIC_QUALITY (1-5)**
Score 5: Excellent - facts from contexts, valid FOL syntax, coherent proof chain
Score 4: Good - mostly valid FOL, minor logical gaps but conclusion follows
Score 3: Moderate - some valid FOL, weak connections between steps
Score 2: Poor - significant FOL syntax errors, weak proof chain
Score 1: Failed - invalid FOL, hallucinated facts, no logical connection

========================================
FOL Syntax (ONLY VALID SYMBOLS):
========================================
Operators: ∀ ∃ ∧ ∨ ¬ → ↔ ⊕ ( ) ,
Predicates: Alphanumeric, e.g., Dog(x), Capital(city, country)
Examples:
✓ Valid: ∀x (Dog(x) ∧ Friendly(x) → Trustworthy(x))
✓ Valid: Capital(Paris, France)
✗ Invalid: 1. Capital(Paris, France).
✗ Invalid: Therefore, Capital(Paris, France)

========================================
EXAMPLES WITH SCORING:
========================================

EXAMPLE 1 (Perfect - FORMAT: 5, LOGIC: 4):
<logic>
FACTS:
BornIn(1924, LouisCha)
PenName(LouisCha, JinYong)
Featured(BeggarsSect, JinYong)

PROOF:
BornIn(1924, LouisCha)
PenName(LouisCha, JinYong)
Featured(BeggarsSect, JinYong)
∃x (PenName(x, JinYong) ∧ Featured(BeggarsSect, x))
</logic>
✓ Clear "FACTS:" and "PROOF:" separators, valid FOL

EXAMPLE 2 (Poor separation - FORMAT: 2, LOGIC: 3):
<logic>
Capital(Paris, France)
Located(France, Europe)

Capital(Paris, France) ∧ Located(France, Europe)
∃x (Capital(Paris, x) ∧ Located(x, Europe))
</logic>
✗ NO clear separator between facts and proof (only blank line)

EXAMPLE 3 (Alternative separator - FORMAT: 5, LOGIC: 4):
<logic>
--- FACTS ---
Capital(Paris, France)
Located(France, Europe)

--- PROOF ---
Capital(Paris, France) ∧ Located(France, Europe)
</logic>
✓ Clear "--- FACTS ---" and "--- PROOF ---" separators

========================================
OUTPUT FORMAT
========================================

You must respond EXACTLY in this format:

REASONING: [2-3 sentences analyzing format and logic quality, specifically mentioning whether clear FACTS/PROOF separators are present]
FORMAT_QUALITY: [1-5]
LOGIC_QUALITY: [1-5]
"""

# Prompt for unified proof only (no separate facts section)
LLM_JUDGE_PROMPT_UNIFIED = """You are an expert evaluator assessing First-Order Logic (FOL) proof chain generation quality for multi-hop reasoning tasks.

The task requires:
1. ALL content must be inside a single <logic>...</logic> block
2. Unified proof chain with NO separation between facts and proof steps
3. ONLY valid FOL formulas allowed - NO numbered lists, NO periods after formulas, NO natural language sentences
4. Every formula must use ONLY these symbols: ∀ ∃ ∧ ∨ ¬ → ↔ ⊕ ( ) , and alphanumeric predicates/constants

========================================
CRITICAL FORMAT VIOLATIONS (Always score FORMAT=1)
========================================

❌ NEVER ACCEPTABLE:
- Numbered lists: "1. P(x)" or "Step 1: P(x)"
- Periods after formulas: "P(x)."
- Invalid symbols: "∴" (therefore), "." (period), "•", ":", etc.
- Natural language in proof: "From premises 1 and 2, it follows that..."
- Section markers like "FACTS:" or "PROOF:" (unified format has NO sections)
- Missing <logic> delimiters entirely

✓ CORRECT FORMAT:
<logic>
Predicate1(constant1)
Predicate2(constant2, constant3)
Predicate1(constant1) → Predicate3(constant1)
Predicate3(constant1)
</logic>

========================================
EVALUATION CRITERIA (1-5 scale each)
========================================

**1. FORMAT_QUALITY (1-5)**
Score 5: Perfect - single <logic> block, valid FOL only, unified format (no FACTS/PROOF separators)
Score 4: Minor issues - slight formatting problems but all FOL valid
Score 3: Moderate - some invalid symbols or has FACTS/PROOF separators
Score 2: Major - numbered lists, periods, or multiple violations
Score 1: Critical - natural language, missing delimiters

**2. LOGIC_QUALITY (1-5)**
Score 5: Excellent - all formulas valid FOL syntax, coherent proof chain
Score 4: Good - mostly valid FOL, minor logical gaps but conclusion follows
Score 3: Moderate - some valid FOL, weak connections between steps
Score 2: Poor - significant FOL syntax errors, weak proof chain
Score 1: Failed - invalid FOL, no logical connection

========================================
FOL Syntax (ONLY VALID SYMBOLS):
========================================
Operators: ∀ ∃ ∧ ∨ ¬ → ↔ ⊕ ( ) ,
Predicates: Alphanumeric, e.g., Dog(x), Capital(city, country)
Examples:
✓ Valid: ∀x (Dog(x) ∧ Friendly(x) → Trustworthy(x))
✓ Valid: Capital(Paris, France)
✗ Invalid: 1. Capital(Paris, France).
✗ Invalid: Therefore, Capital(Paris, France)

========================================
EXAMPLE (What to expect):
========================================

<logic>
BornIn(1924, LouisCha)
PenName(LouisCha, JinYong)
Featured(BeggarsSect, JinYong)
BornIn(1924, LouisCha) ∧ PenName(LouisCha, JinYong)
∃x (PenName(x, JinYong) ∧ Featured(BeggarsSect, x))
</logic>

FORMAT_QUALITY: 5 (perfect structure, valid FOL only, unified format)
LOGIC_QUALITY: 4 (valid but simple reasoning)

========================================
OUTPUT FORMAT
========================================

You must respond EXACTLY in this format:

REASONING: [2-3 sentences analyzing format and logic quality]
FORMAT_QUALITY: [1-5]
LOGIC_QUALITY: [1-5]
"""


# ============================================================================
# LLM JUDGE METRIC FUNCTION
# ============================================================================

def create_llm_judge_metric(judge_lm: dspy.LM, facts_proof_divided: bool = False, weight: float = 1.0):
    """
    Create LLM-as-judge metric function for format and logic quality evaluation.

    Args:
        judge_lm: Language model for judging
        facts_proof_divided: Whether facts and proof are in separate sections
        weight: Weight for this metric in combined score

    Returns:
        Metric function compatible with GEPA
    """
    judge_prompt = LLM_JUDGE_PROMPT_SEPARATED if facts_proof_divided else LLM_JUDGE_PROMPT_UNIFIED

    def llm_judge_metric(example: Dict, pred: Any, trace=None) -> dspy.Prediction:
        """Evaluate format and logic quality using LLM judge."""
        try:
            completion = str(pred.completions[0]) if hasattr(pred, 'completions') and pred.completions else str(pred)

            question = example.get('question', '')
            contexts = example.get('contexts', [])
            contexts_str = '\n\n'.join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])

            eval_input = f"""{judge_prompt}

========================================
INPUT INFORMATION
========================================

QUESTION:
{question}

CONTEXTS:
{contexts_str}

========================================
GENERATED OUTPUT
========================================

{completion}

========================================
EVALUATION TASK
========================================

Evaluate the format quality and logic quality of the generated output above."""

            with dspy.settings.context(lm=judge_lm):
                eval_response = judge_lm(eval_input)
                if isinstance(eval_response, list):
                    eval_response = eval_response[0] if eval_response else ""

            import re
            eval_response_clean = re.sub(r'<think>.*?</think>', '', str(eval_response), flags=re.DOTALL)
            reasoning_match = re.search(r'REASONING:\s*(.*?)(?=FORMAT_QUALITY:|$)', str(eval_response_clean), re.DOTALL)
            format_match = re.search(r'FORMAT_QUALITY:\s*(\d+)', str(eval_response_clean))
            logic_match = re.search(r'LOGIC_QUALITY:\s*(\d+)', str(eval_response_clean))

            format_score = int(format_match.group(1)) if format_match else 0
            logic_score = int(logic_match.group(1)) if logic_match else 0
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "Failed to parse"

            if not all([format_match, logic_match]):
                logging.warning(f"Failed to parse judge response: {eval_response}")

            overall_score = (format_score + logic_score) / 10.0
            feedback = f"JUDGE EVAL: {reasoning}\nFORMAT={format_score}/5, LOGIC={logic_score}/5"

            return dspy.Prediction(
                score=overall_score * weight,
                feedback=feedback,
                format_score=format_score,
                logic_score=logic_score
            )

        except Exception as e:
            logging.error(f"LLM judge evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return dspy.Prediction(score=0.0, feedback=f"Judge evaluation failed: {e}")

    return llm_judge_metric


# ============================================================================
# STANDARD METRICS FUNCTION
# ============================================================================

def create_standard_metric(template_handler: TemplateHandler, weight: float = 1.0):
    """
    Create standard metrics function using parsing and logic evaluation.

    Args:
        template_handler: Template handler for parsing
        weight: Weight for this metric in combined score

    Returns:
        Metric function compatible with GEPA
    """
    from solver.logic_evaluator import LogicEvaluator

    def standard_metric(example: Dict, pred: Any, trace=None) -> dspy.Prediction:
        """Evaluate using manual parsing and logic metrics."""
        try:
            completion = str(pred.completions[0]) if hasattr(pred, 'completions') and pred.completions else str(pred)

            logic_start = template_handler.parsing_config.logic_start
            logic_end = template_handler.parsing_config.logic_end

            if logic_start not in completion or logic_end not in completion:
                return dspy.Prediction(
                    score=0.0,
                    feedback="✗ Parse failed: logic delimiters not found",
                    metrics={'parse': 0, 'valid_rate': 0.0}
                )

            start_count = completion.count(logic_start)
            end_count = completion.count(logic_end)

            if start_count != 1 or end_count != 1:
                return dspy.Prediction(
                    score=0.0,
                    feedback=f"✗ Parse failed: found {start_count} start and {end_count} end delimiters (expected 1 each)",
                    metrics={'parse': 0, 'valid_rate': 0.0}
                )

            start_idx = completion.find(logic_start) + len(logic_start)
            end_idx = completion.find(logic_end)
            logic_block = completion[start_idx:end_idx].strip()

            formulas = []
            for line in logic_block.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if 'fact' in line.lower() and len(line) < 20:
                    continue
                if 'proof' in line.lower() and len(line) < 20:
                    continue
                formulas.append(line)

            if len(formulas) == 0:
                return dspy.Prediction(
                    score=0.0,
                    feedback="✗ Parse failed: no formulas found in logic block",
                    metrics={'parse': 0, 'valid_rate': 0.0}
                )

            evaluator = LogicEvaluator(facts=None, proof_steps=formulas)
            valid_rate = evaluator.get_valid_rate()
            score = valid_rate

            feedback_parts = [
                "✓ Parsed",
                f"Valid formulas: {valid_rate:.1%}"
            ]
            feedback = " | ".join(feedback_parts)

            return dspy.Prediction(
                score=score * weight,
                feedback=feedback,
                metrics={
                    'parse': 1,
                    'valid_rate': valid_rate,
                    'num_formulas': len(formulas)
                }
            )

        except Exception as e:
            logging.error(f"Standard metric evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return dspy.Prediction(score=0.0, feedback=f"Standard eval failed: {e}")

    return standard_metric


# ============================================================================
# COMBINED METRIC FUNCTION
# ============================================================================

def create_combined_metric(
    standard_metric_fn,
    llm_judge_metric_fn,
    standard_weight: float = 0.6,
    judge_weight: float = 0.4
):
    """
    Create combined metric function that uses both standard and LLM judge metrics.

    Args:
        standard_metric_fn: Standard metric function
        llm_judge_metric_fn: LLM judge metric function
        standard_weight: Weight for standard metrics
        judge_weight: Weight for LLM judge metrics

    Returns:
        Combined metric function
    """

    def combined_metric(example: Dict, pred: Any, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
        """Evaluate using both standard and LLM judge metrics."""
        try:
            standard_result = standard_metric_fn(example, pred, trace)
            judge_result = llm_judge_metric_fn(example, pred, trace)

            combined_score = (standard_result.score * standard_weight +
                            judge_result.score * judge_weight)

            combined_feedback = (
                f"STANDARD ({standard_weight:.1f}x): {standard_result.feedback}\n"
                f"JUDGE ({judge_weight:.1f}x): {judge_result.feedback}\n"
                f"COMBINED: {combined_score:.3f}"
            )

            return dspy.Prediction(
                score=combined_score,
                feedback=combined_feedback,
                standard_score=standard_result.score,
                judge_score=judge_result.score
            )

        except Exception as e:
            logging.error(f"Combined metric evaluation failed: {e}")
            return dspy.Prediction(score=0.0, feedback=f"Combined eval failed: {e}")

    return combined_metric


# ============================================================================
# FOL GENERATION MODULE
# ============================================================================

class FOLProofGenerationModule(dspy.Module):
    """DSPy module for FOL proof chain generation."""

    def __init__(self, template_handler: TemplateHandler):
        super().__init__()
        self.template_handler = template_handler

        separate_sections = template_handler.parsing_config.facts_proof_divided

        if separate_sections:
            self.generate = dspy.ChainOfThought(
                "question, contexts -> facts, proof, answer",
                instructions=(
                    "You are solving a multi-hop reasoning question using First-Order Logic (FOL). "
                    "Extract relevant facts from contexts and create a proof chain to derive the answer.\n\n"
                    "FACTS: Extract FOL facts directly from the contexts. "
                    "Use proper FOL syntax with quantifiers (∀, ∃), logical operators (∧, ∨, →, ¬), and predicates. "
                    "Each fact should be grounded in the provided contexts - do not hallucinate.\n\n"
                    "PROOF: Build a logical derivation chain where each step follows from facts and previous steps. "
                    "The proof should bridge from the extracted facts to the conclusion. "
                    "The final step must be entailed by the chain (verifiable by theorem prover). "
                    "Use valid FOL syntax and proper inference rules.\n\n"
                    "ANSWER: Provide the final answer to the question based on the proof chain. "
                    "Keep it concise and directly address what was asked.\n\n"
                    "Critical: All FOL formulas must be syntactically correct and logically sound."
                )
            )
        else:
            self.generate = dspy.ChainOfThought(
                "question, contexts -> proof, answer",
                instructions=(
                    "You are solving a multi-hop reasoning question using First-Order Logic (FOL). "
                    "Create a proof chain combining facts and derivation steps to derive the answer.\n\n"
                    "PROOF: First establish FOL facts from the contexts, then build a logical derivation chain. "
                    "Each step should follow logically from previous steps. "
                    "Use proper FOL syntax: quantifiers (∀, ∃), operators (∧, ∨, →, ¬), and predicates. "
                    "The final step must be entailed by the chain (verifiable by theorem prover). "
                    "Ground all facts in the provided contexts - do not hallucinate.\n\n"
                    "ANSWER: Provide the final answer to the question based on the proof. "
                    "Keep it concise and directly address what was asked.\n\n"
                    "Critical: All FOL formulas must be syntactically correct and logically sound."
                )
            )

        self.separate_sections = separate_sections

    def forward(self, question: str, contexts: List[str]):
        contexts_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        result = self.generate(question=question, contexts=contexts_str)

        if self.separate_sections:
            if not hasattr(result, 'facts'):
                result.facts = ""
            if not hasattr(result, 'proof'):
                result.proof = ""
        else:
            if not hasattr(result, 'facts'):
                result.facts = ""
            if not hasattr(result, 'proof'):
                result.proof = ""

        if not hasattr(result, 'answer'):
            result.answer = ""

        return result


# ============================================================================
# DATASET LOADING AND MERGING
# ============================================================================

def load_and_merge_datasets(
    dataset_names: List[str],
    split: str,
    downsample: float,
    seed: int
) -> List[Dict]:
    """
    Load and merge multiple datasets.

    Args:
        dataset_names: List of dataset names to load
        split: Dataset split (train or test)
        downsample: Ratio to downsample (0-1)
        seed: Random seed

    Returns:
        Merged dataset as list of examples
    """
    random.seed(seed)
    all_examples = []

    for dataset_name in dataset_names:
        logging.info(f"Loading dataset: {dataset_name} (split: {split})")
        handler = DatasetHandler(dataset_name=dataset_name)
        dataset = handler.load_dataset()[split]

        examples = [dict(example) for example in dataset]

        if downsample < 1.0:
            n_samples = max(1, int(len(examples) * downsample))
            examples = random.sample(examples, n_samples)
            logging.info(f"Downsampled {dataset_name} to {len(examples)} examples")

        all_examples.extend(examples)

    logging.info(f"Total merged examples: {len(all_examples)}")
    return all_examples


# ============================================================================
# MAIN FUNCTION
# ============================================================================

OUTPUT_ROOT = Path('output/gepa')

def main():
    parser = ConfigArgumentParser(
        description='GEPA Prompt Optimization for FOL Proof Generation'
    )

    parser.add_argument('--fol_model', type=str,
                       help='HuggingFace model name for FOL generation (to optimize prompt for)')
    parser.add_argument('--initial_template', type=str,
                       help='Initial template name (determines prompt and parsing behavior)')
    parser.add_argument('--dataset', type=str, nargs='+',
                       help='One or more dataset names (will be merged if multiple)')
    parser.add_argument('--downsample', type=float, default=1.0,
                       help='Downsample ratio for quicker tests (0-1)')

    parser.add_argument('--judge_model', type=str,
                       default='RedHatAI/granite-3.1-8b-instruct-quantized.w4a16',
                       help='Judge model for LLM-as-judge evaluation')
    parser.add_argument('--vllm_url', type=str, default='http://localhost',
                       help='vLLM server URL')
    parser.add_argument('--fol_port', type=int, default=8000,
                       help='vLLM port for FOL model')
    parser.add_argument('--judge_port', type=int, default=8001,
                       help='vLLM port for judge model')

    parser.add_argument('--fol_max_len', type=int, default=2048,
                       help='Max sequence length for FOL model server')
    parser.add_argument('--fol_gpu_mem', type=float, default=0.30,
                       help='GPU memory utilization for FOL model server')
    parser.add_argument('--judge_max_len', type=int, default=4096,
                       help='Max sequence length for judge model server')
    parser.add_argument('--judge_gpu_mem', type=float, default=0.65,
                       help='GPU memory utilization for judge model server')

    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'test'],
                       help='Dataset split to use for optimization')
    parser.add_argument('--num_train_samples', type=int, default=30,
                       help='Number of training samples for GEPA')
    parser.add_argument('--num_val_samples', type=int, default=15,
                       help='Number of validation samples for GEPA')

    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Temperature for FOL model generation')
    parser.add_argument('--max_tokens', type=int, default=1024,
                       help='Maximum tokens for FOL model generation')

    parser.add_argument('--judge_temperature', type=float, default=0.3,
                       help='Temperature for judge model')
    parser.add_argument('--judge_max_tokens', type=int, default=2048,
                       help='Maximum tokens for judge model')
    parser.add_argument('--judge_enable_thinking', action='store_true',
                       help='Enable thinking mode for Qwen3 judge models')
    parser.add_argument('--facts_proof_divided', action='store_true',
                       help='Whether facts and proof should be in separate sections (affects judge prompt)')

    parser.add_argument('--standard_weight', type=float, default=0.6,
                       help='Weight for standard metrics (parsing + logic)')
    parser.add_argument('--judge_weight', type=float, default=0.4,
                       help='Weight for LLM judge metrics')

    parser.add_argument('--gepa_auto', type=str, choices=['light', 'medium', 'heavy'],
                       help='GEPA auto mode preset')
    parser.add_argument('--max_full_evals', type=int,
                       help='Maximum number of full evaluations')
    parser.add_argument('--max_metric_calls', type=int,
                       help='Maximum number of metric calls')

    parser.add_argument('--num_threads', type=int, default=8,
                       help='Number of parallel threads for GEPA')

    parser.add_argument('--output_dir', type=str, default='gepa_default_run',
                       help='Directory to save optimization results')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='synfol-gepa',
                       help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='W&B run name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.getLogger('litellm').setLevel(logging.WARNING)
    logging.getLogger('LiteLLM').setLevel(logging.WARNING)
    logging.getLogger('dspy').setLevel(logging.WARNING)

    if not args.fol_model:
        logging.error("Error: --fol_model is required")
        return
    if not args.initial_template:
        logging.error("Error: --initial_template is required")
        return
    if not args.dataset:
        logging.error("Error: --dataset is required")
        return

    budget_options = [args.gepa_auto, args.max_full_evals, args.max_metric_calls]
    budget_provided = sum(opt is not None for opt in budget_options)
    if budget_provided == 0:
        logging.error("Error: Must provide exactly one of: --gepa_auto, --max_full_evals, or --max_metric_calls")
        return
    elif budget_provided > 1:
        logging.error("Error: Cannot provide multiple budget options. Choose only one of: --gepa_auto, --max_full_evals, or --max_metric_calls")
        return

    logging.info("="*80)
    logging.info("GEPA PROMPT OPTIMIZATION FOR FOL PROOF GENERATION")
    logging.info("="*80)
    logging.info(f"FOL Model: {args.fol_model}")
    logging.info(f"Judge Model: {args.judge_model}")
    logging.info(f"Initial Template: {args.initial_template}")
    logging.info(f"Datasets: {args.dataset}")
    logging.info(f"Downsample: {args.downsample}")
    logging.info(f"Split: {args.split}")
    logging.info(f"Train samples: {args.num_train_samples}")
    logging.info(f"Val samples: {args.num_val_samples}")
    logging.info(f"Standard weight: {args.standard_weight}")
    logging.info(f"Judge weight: {args.judge_weight}")

    if args.gepa_auto:
        logging.info(f"GEPA Budget: auto={args.gepa_auto}")
    elif args.max_full_evals:
        logging.info(f"GEPA Budget: max_full_evals={args.max_full_evals}")
    elif args.max_metric_calls:
        logging.info(f"GEPA Budget: max_metric_calls={args.max_metric_calls}")

    logging.info("="*80)

    logging.info(f"Loading template: {args.initial_template}")
    template_handler = TemplateHandler(
        template_name=args.initial_template,
        include_assistant_prompt=False
    )

    logging.info("Loading datasets...")
    all_examples = load_and_merge_datasets(
        dataset_names=args.dataset,
        split=args.split,
        downsample=args.downsample,
        seed=args.seed
    )

    random.seed(args.seed)
    random.shuffle(all_examples)

    train_examples = all_examples[:args.num_train_samples]
    val_examples = all_examples[args.num_train_samples:args.num_train_samples + args.num_val_samples]

    logging.info(f"Train examples: {len(train_examples)}")
    logging.info(f"Val examples: {len(val_examples)}")

    train_data = [
        dspy.Example(ex).with_inputs('question', 'contexts')
        for ex in train_examples
    ]
    val_data = [
        dspy.Example(ex).with_inputs('question', 'contexts')
        for ex in val_examples
    ]

    logging.info("Initializing DSPy language models...")
    fol_lm = dspy.LM(
        model=f"openai/{args.fol_model}",
        api_base=f'{args.vllm_url}:{args.fol_port}/v1',
        api_key='EMPTY',
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    judge_kwargs = {
        'model': f"openai/{args.judge_model}",
        'api_base': f'{args.vllm_url}:{args.judge_port}/v1',
        'api_key': 'EMPTY',
        'max_tokens': args.judge_max_tokens,
        'temperature': args.judge_temperature
    }

    if args.judge_enable_thinking:
        judge_kwargs['extra_body'] = {
            'chat_template_kwargs': {'enable_thinking': True}
        }
        logging.info("Thinking mode enabled for judge model")

    judge_lm = dspy.LM(**judge_kwargs)

    dspy.settings.configure(lm=fol_lm)

    logging.info("Creating metric functions...")
    standard_metric_fn = create_standard_metric(template_handler, weight=1.0)
    llm_judge_metric_fn = create_llm_judge_metric(judge_lm, facts_proof_divided=args.facts_proof_divided, weight=1.0)
    combined_metric_fn = create_combined_metric(
        standard_metric_fn,
        llm_judge_metric_fn,
        standard_weight=args.standard_weight,
        judge_weight=args.judge_weight
    )

    logging.info("Creating seed program...")
    seed_program = FOLProofGenerationModule(template_handler)

    logging.info("Evaluating seed program...")
    from dspy import Evaluate

    evaluate_seed = Evaluate(
        devset=val_data,
        metric=lambda example, pred, trace=None, pred_name=None, pred_trace=None: combined_metric_fn(example, pred, trace, pred_name, pred_trace).score,
        num_threads=args.num_threads,
        display_progress=True
    )
    seed_result = evaluate_seed(seed_program)
    seed_score = float(seed_result)
    logging.info(f"Seed program score: {seed_score:.2%}")

    output_dir = Path(OUTPUT_ROOT) / args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = output_dir / 'logs'
    os.makedirs(log_dir, exist_ok=True)

    logging.info("="*80)
    logging.info("STARTING GEPA OPTIMIZATION")
    logging.info("="*80)

    try:
        from dspy import GEPA
    except ImportError:
        logging.error("GEPA not available in dspy. Please update dspy.")
        return

    gepa_kwargs = {
        'metric': combined_metric_fn,
        'num_threads': args.num_threads,
        'track_stats': True,
        'track_best_outputs': False,
        'reflection_lm': judge_lm
    }

    if args.gepa_auto:
        gepa_kwargs['auto'] = args.gepa_auto
    elif args.max_full_evals:
        gepa_kwargs['max_full_evals'] = args.max_full_evals
    elif args.max_metric_calls:
        gepa_kwargs['max_metric_calls'] = args.max_metric_calls

    gepa_kwargs['log_dir'] = log_dir

    if args.use_wandb:
        gepa_kwargs['use_wandb'] = True
        gepa_kwargs['wandb_init_kwargs'] = {
            'project': args.wandb_project,
            'name': args.wandb_name if args.wandb_name else f'gepa_fol_{timestamp}'
        }

    optimizer = GEPA(**gepa_kwargs)

    optimized_program = optimizer.compile(
        seed_program,
        trainset=train_data,
        valset=val_data
    )

    logging.info("="*80)
    logging.info("OPTIMIZATION COMPLETE")
    logging.info("="*80)

    logging.info("Evaluating optimized program...")
    evaluate_opt = Evaluate(
        devset=val_data,
        metric=lambda example, pred, trace=None, pred_name=None, pred_trace=None: combined_metric_fn(example, pred, trace, pred_name, pred_trace).score,
        num_threads=args.num_threads,
        display_progress=True
    )
    optimized_result = evaluate_opt(optimized_program)
    optimized_score = float(optimized_result)

    logging.info(f"Seed program score: {seed_score:.2%}")
    logging.info(f"Optimized program score: {optimized_score:.2%}")
    logging.info(f"Improvement: {(optimized_score - seed_score):.2%}")

    optimized_instructions = "Not available"
    try:
        if hasattr(optimized_program, 'generate'):
            if hasattr(optimized_program.generate, 'predict'):
                optimized_instructions = optimized_program.generate.predict.signature.instructions
            elif hasattr(optimized_program.generate, 'signature'):
                optimized_instructions = optimized_program.generate.signature.instructions
    except Exception as e:
        logging.warning(f"Could not extract optimized instructions: {e}")

    program_path = output_dir / f"optimized_program_{timestamp}.json"
    try:
        optimized_program.save(program_path)
        logging.info(f"Optimized program saved: {program_path}")
    except Exception as e:
        logging.warning(f"Could not save program: {e}")
        try:
            state_dict_path = output_dir / f"optimized_program_state_{timestamp}.json"
            with open(state_dict_path, 'w') as f:
                json.dump(optimized_program.dump_state(), f, indent=2)
            logging.info(f"Saved program state dict: {state_dict_path}")
            program_path = state_dict_path
        except Exception as e2:
            logging.warning(f"Could not save state dict either: {e2}")

    instructions_path = output_dir / f"optimized_instructions_{timestamp}.txt"
    with open(instructions_path, 'w') as f:
        f.write("OPTIMIZED INSTRUCTIONS\n")
        f.write("="*80 + "\n\n")
        f.write(optimized_instructions)
        f.write("\n\n" + "="*80 + "\n")
        f.write("OPTIMIZATION INFO\n")
        f.write("="*80 + "\n")
        f.write(f"FOL Model: {args.fol_model}\n")
        f.write(f"Judge Model: {args.judge_model}\n")
        f.write(f"Initial Template: {args.initial_template}\n")
        f.write(f"Datasets: {', '.join(args.dataset)}\n")
        f.write(f"Seed Score: {seed_score:.4f}\n")
        f.write(f"Optimized Score: {optimized_score:.4f}\n")
        f.write(f"Improvement: {(optimized_score - seed_score):.4f}\n")

    logging.info(f"Optimized instructions saved: {instructions_path}")

    stats = {
        'timestamp': timestamp,
        'fol_model': args.fol_model,
        'judge_model': args.judge_model,
        'initial_template': args.initial_template,
        'datasets': args.dataset,
        'downsample': args.downsample,
        'num_train_samples': len(train_data),
        'num_val_samples': len(val_data),
        'seed_score': float(seed_score),
        'optimized_score': float(optimized_score),
        'improvement': float(optimized_score - seed_score),
        'standard_weight': args.standard_weight,
        'judge_weight': args.judge_weight
    }

    stats_path = output_dir / "gepa_optimization_stats.csv"
    stats_df = pd.DataFrame([stats])

    if stats_path.exists():
        stats_df.to_csv(stats_path, mode='a', header=False, index=False)
        logging.info(f"Appended stats to: {stats_path}")
    else:
        stats_df.to_csv(stats_path, mode='w', header=True, index=False)
        logging.info(f"Created stats CSV: {stats_path}")

    results_path = output_dir / f"optimization_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump({
            'stats': stats,
            'optimized_instructions': optimized_instructions
        }, f, indent=2)

    logging.info(f"Full results saved: {results_path}")

    logging.info("\n" + "="*80)
    logging.info("GEPA OPTIMIZATION COMPLETE!")
    logging.info("="*80)
    logging.info("\nOutput files:")
    logging.info(f"  - Optimized instructions: {instructions_path}")
    if program_path:
        logging.info(f"  - Optimized program: {program_path}")
    logging.info(f"  - Statistics: {stats_path}")
    logging.info(f"  - Full results: {results_path}")


if __name__ == "__main__":
    main()
