from datasets import DatasetDict, load_dataset
import random
import json
import re
import unicodedata
from typing import Optional

class DatasetHandler:
    """Class to handle loading and preprocessing of different datasets into a standardized format.

    The standardized format is a DatasetDict with 'train' and 'test' splits, both containing fields:
        - 'question': str
        - 'contexts': List[str]
        - 'answer': str

    For logic datasets, additional fields may be present:
        - 'facts': List[str] (optional, FOL formulas)
        - 'proof': List[str] (proof chain)
    """
    def __init__(
            self,
            dataset_name: str,
            train_downsample: float = 1.0,
            test_downsample: float = 1.0,
            sample_seed: int = 42,
            proof_type: str = "extended",
            yesno_ratio: float = 0.5,
            max_facts: Optional[int] = None,
            include_facts_field: bool = False,
            filter_non_ascii: bool = True,
            # Index-based slicing (takes priority over percentage)
            train_start_index: Optional[int] = None,
            train_stop_index: Optional[int] = None,
            # Percentage-based slicing (used if index not provided)
            train_start_perc: Optional[float] = None,
            train_stop_perc: Optional[float] = None
        ):
            self.dataset_name = dataset_name
            self.train_downsample = train_downsample
            self.test_downsample = test_downsample
            self.sample_seed = sample_seed
            self.proof_type = proof_type
            self.yesno_ratio = yesno_ratio
            self.max_facts = max_facts
            self.include_facts_field = include_facts_field
            self.filter_non_ascii = filter_non_ascii
            # Index-based slicing
            self.train_start_index = train_start_index
            self.train_stop_index = train_stop_index
            # Percentage-based slicing
            self.train_start_perc = train_start_perc
            self.train_stop_perc = train_stop_perc

            # Map logic dataset names to their HuggingFace repository paths
            self.logic_dataset_map = {
                "logic_translations_q14B": "disi-unibo-nlp/logic_translations_q14B",
                # Add more logic datasets here with their HuggingFace paths
            }

            self.method_map = {
                  "2wiki": self._load_2wiki_dataset,
                  "hotpotqa": self._load_hotpotqa_dataset,
                  "folio": self._load_folio_dataset,
                  "prontoqa": self._load_prontoqa_dataset,
                  # Logic datasets
                  "logic_translations_q14B": self._load_logic_dataset,
                  # Add more datasets and their corresponding methods here
            }

            # Validate proof_type only for logic datasets
            logic_datasets = ["logic_translations_q14B"]
            if self.dataset_name in logic_datasets:
                if self.proof_type not in ["standard", "extended"]:
                    raise ValueError(f"proof_type must be one of ['standard', 'extended'], got '{self.proof_type}'")

    def load_dataset(self) -> DatasetDict:
        if self.dataset_name not in self.method_map:
            raise ValueError(f"Dataset {self.dataset_name} not supported. Either use a supported dataset or implement a custom loader method.")
        ds = self.method_map[self.dataset_name]()
        ds = self.apply_downsampling(ds)
        ds = self.apply_index_slicing(ds)

        # Apply preprocessing and filtering for datasets that will be used with Prover9
        if self.filter_non_ascii and self.dataset_name in ['hotpotqa', '2wiki']:
            # First, normalize Unicode characters to ASCII equivalents where possible
            ds = self._preprocess_dataset_text(ds)
            # Then filter out any remaining instances with non-ASCII characters
            ds = self._filter_non_ascii_instances(ds)

        return ds

    def apply_downsampling(self, ds: DatasetDict) -> DatasetDict:
        if self.train_downsample < 1.0:
            ds['train'] = ds['train'].shuffle(seed=self.sample_seed).select(
                range(int(len(ds['train']) * self.train_downsample))
            )
        if self.test_downsample < 1.0:
            ds['test'] = ds['test'].shuffle(seed=self.sample_seed).select(
                range(int(len(ds['test']) * self.test_downsample))
            )
        return ds

    def apply_index_slicing(self, ds: DatasetDict) -> DatasetDict:
        """Apply index slicing to train split.

        Slicing is applied AFTER downsampling and shuffling, so indices
        refer to positions in the already-processed dataset.

        Priority: index > perc > default (full range)
        """
        # Compute train slice bounds
        train_len = len(ds['train'])
        if self.train_start_index is not None:
            train_start = self.train_start_index
        elif self.train_start_perc is not None:
            train_start = int(train_len * self.train_start_perc)
        else:
            train_start = 0

        if self.train_stop_index is not None:
            train_stop = self.train_stop_index
        elif self.train_stop_perc is not None:
            train_stop = int(train_len * self.train_stop_perc)
        else:
            train_stop = train_len

        # Apply slicing only if bounds differ from full range
        if train_start != 0 or train_stop != train_len:
            ds['train'] = ds['train'].select(range(train_start, train_stop))

        return ds

    @staticmethod
    def _normalize_to_ascii(text: str) -> str:
        """Normalize Unicode text to ASCII by replacing common characters.

        Applies the following transformations:
        1. Replace common punctuation variants (dashes, quotes, etc.) with ASCII equivalents
        2. Decompose accented characters and strip diacritics
        3. Replace special spaces with regular spaces

        Args:
            text: Input text that may contain Unicode characters

        Returns:
            Text with Unicode characters replaced by ASCII equivalents where possible
        """
        # Define character replacements (Unicode -> ASCII)
        replacements = {
            # Dashes
            '\u2013': '-',  # en dash –
            '\u2014': '-',  # em dash —
            '\u2212': '-',  # minus sign −
            '\u2010': '-',  # hyphen ‐
            '\u2011': '-',  # non-breaking hyphen ‑

            # Quotes
            '\u201C': '"',  # left double quote "
            '\u201D': '"',  # right double quote "
            '\u201E': '"',  # double low quote „
            '\u201F': '"',  # double high-reversed quote ‟
            '\u2018': "'",  # left single quote '
            '\u2019': "'",  # right single quote ' (also apostrophe)
            '\u201A': "'",  # single low quote ‚
            '\u201B': "'",  # single high-reversed quote ‛
            '\u00AB': '"',  # left guillemet «
            '\u00BB': '"',  # right guillemet »
            '\u2039': "'",  # left single guillemet ‹
            '\u203A': "'",  # right single guillemet ›

            # Ellipsis
            '\u2026': '...',  # horizontal ellipsis …

            # Spaces
            '\u00A0': ' ',  # non-breaking space
            '\u2000': ' ',  # en quad
            '\u2001': ' ',  # em quad
            '\u2002': ' ',  # en space
            '\u2003': ' ',  # em space
            '\u2004': ' ',  # three-per-em space
            '\u2005': ' ',  # four-per-em space
            '\u2006': ' ',  # six-per-em space
            '\u2007': ' ',  # figure space
            '\u2008': ' ',  # punctuation space
            '\u2009': ' ',  # thin space
            '\u200A': ' ',  # hair space
            '\u202F': ' ',  # narrow no-break space
            '\u205F': ' ',  # medium mathematical space

            # Bullets
            '\u2022': '*',  # bullet •
            '\u2023': '>',  # triangular bullet ‣
            '\u2043': '-',  # hyphen bullet ⁃

            # Other common symbols
            '\u00D7': 'x',  # multiplication sign ×
            '\u00F7': '/',  # division sign ÷
            '\u2032': "'",  # prime ′ (minutes, feet)
            '\u2033': '"',  # double prime ″ (seconds, inches)
        }

        # Apply character replacements
        translation_table = str.maketrans(replacements)
        text = text.translate(translation_table)

        # Decompose accented characters using NFKD normalization
        # This separates base characters from combining diacritical marks
        # Example: 'é' -> 'e' + combining acute accent
        text = unicodedata.normalize('NFKD', text)

        # Remove combining diacritical marks (keep only ASCII-range characters)
        # This effectively strips accents: ū -> u, é -> e, etc.
        text = ''.join(char for char in text if ord(char) < 128)

        return text

    def _preprocess_dataset_text(self, ds: DatasetDict) -> DatasetDict:
        """Apply ASCII normalization to all text fields in the dataset.

        Args:
            ds: Input dataset

        Returns:
            Dataset with normalized text
        """
        import logging
        logger = logging.getLogger(__name__)

        def normalize_example(example):
            """Normalize all text fields in an example."""
            # Normalize question
            example['question'] = self._normalize_to_ascii(example['question'])

            # Normalize answer
            example['answer'] = self._normalize_to_ascii(example['answer'])

            # Normalize contexts (list of strings)
            example['contexts'] = [self._normalize_to_ascii(ctx) for ctx in example['contexts']]

            # Normalize logic-specific fields if present
            if 'facts' in example and example['facts']:
                if isinstance(example['facts'], list):
                    example['facts'] = [self._normalize_to_ascii(fact) for fact in example['facts']]
                else:
                    example['facts'] = self._normalize_to_ascii(example['facts'])

            if 'proof' in example and example['proof']:
                if isinstance(example['proof'], list):
                    example['proof'] = [self._normalize_to_ascii(step) for step in example['proof']]
                else:
                    example['proof'] = self._normalize_to_ascii(example['proof'])

            return example

        logger.info(f"Applying ASCII normalization to {self.dataset_name} dataset...")

        # Apply normalization to both splits
        ds['train'] = ds['train'].map(normalize_example)
        ds['test'] = ds['test'].map(normalize_example)

        return ds

    def _filter_non_ascii_instances(self, ds: DatasetDict) -> DatasetDict:
        """Filter dataset to keep only ASCII-compatible instances.

        This is necessary for compatibility with Prover9, which only supports ASCII characters.
        Instances with non-ASCII characters in question, answer, or contexts will be removed.

        Args:
            ds: Input dataset

        Returns:
            Filtered dataset with only ASCII-safe instances
        """
        import logging
        logger = logging.getLogger(__name__)

        def is_ascii_safe_instance(example) -> bool:
            """Check if all text in instance is ASCII-compatible."""
            # Check question
            if not example['question'].isascii():
                return False

            # Check answer
            if not example['answer'].isascii():
                return False

            # Check contexts (list of strings)
            if not all(ctx.isascii() for ctx in example['contexts']):
                return False

            # Check logic-specific fields if present
            if 'facts' in example and example['facts']:
                if isinstance(example['facts'], list):
                    if not all(fact.isascii() for fact in example['facts']):
                        return False
                elif not example['facts'].isascii():
                    return False

            if 'proof' in example and example['proof']:
                if isinstance(example['proof'], list):
                    if not all(step.isascii() for step in example['proof']):
                        return False
                elif not example['proof'].isascii():
                    return False

            return True

        # Store original sizes
        orig_train = len(ds['train'])
        orig_test = len(ds['test'])

        # Apply filtering
        ds['train'] = ds['train'].filter(is_ascii_safe_instance)
        ds['test'] = ds['test'].filter(is_ascii_safe_instance)

        # Log statistics
        train_kept = len(ds['train'])
        test_kept = len(ds['test'])
        train_filtered = orig_train - train_kept
        test_filtered = orig_test - test_kept

        logger.info(f"ASCII filtering results for {self.dataset_name}:")
        logger.info(f"  Train: {train_kept}/{orig_train} kept ({train_filtered} filtered, "
                   f"{train_filtered/orig_train*100:.1f}%)")
        logger.info(f"  Test: {test_kept}/{orig_test} kept ({test_filtered} filtered, "
                   f"{test_filtered/orig_test*100:.1f}%)")

        return ds

    def _load_2wiki_dataset(self) -> DatasetDict:
        ds = load_dataset("disi-unibo-nlp/2wikimultihop-processed")
        # Standardize column names
        ds = ds.rename_column("supporting_sentences", "contexts")
        ds = ds.remove_columns(["paragraph", "supporting_words"])
        ds = ds.map(lambda x: {
            "answer": x["answer"][0]
        })
        # Standardize splits
        test_ds = ds.pop("validation")
        ds["test"] = test_ds
        return ds

    def _load_hotpotqa_dataset(self) -> DatasetDict:
        ds = load_dataset("disi-unibo-nlp/hotpotqa-processed")
        # Standardize column names
        ds = ds.rename_column("supporting_sentences", "contexts")
        ds = ds.remove_columns(["paragraph", "supporting_words"])
        ds = ds.map(lambda x: {
            "answer": x["answer"][0]
        })
        # Standardize splits
        test_ds = ds.pop("validation")
        ds["test"] = test_ds
        return ds

    def _load_folio_dataset(self) -> DatasetDict:
        """Load and preprocess the FOLIO (First Order Logic) reasoning dataset.

        FOLIO contains natural language premises and conclusions with formal logic annotations.
        Each instance requires determining if a conclusion is True or False given a set of premises.
        Instances with "Uncertain" labels are filtered out.

        Returns:
            DatasetDict with fields:
                - question: str (the conclusion statement)
                - answer: str (True or False label)
                - contexts: List[str] (individual premise statements)
        """
        # Load dataset from HuggingFace
        ds = load_dataset("tasksource/folio")

        # Filter out instances with "Uncertain" label
        ds = ds.filter(lambda x: x["label"] != "Uncertain")

        # Transform fields to standardized format
        ds = ds.map(lambda x: {
            "question": x["conclusion"],
            "answer": x["label"],
            "contexts": [line.strip() for line in x["premises"].split('\n') if line.strip()]
        })

        # Remove original columns, keeping only standardized fields
        ds = ds.remove_columns(["premises", "premises-FOL", "conclusion", "conclusion-FOL",
                                "story_id", "example_id", "label"])

        # Standardize splits: rename validation to test
        test_ds = ds.pop("validation")
        ds["test"] = test_ds

        return ds

    def _load_prontoqa_dataset(self) -> DatasetDict:
        """Load and preprocess the ProntoQA-OOD reasoning dataset.

        ProntoQA is a synthetic reasoning dataset where all original examples are True.
        This method augments the dataset by creating False examples (30%) through
        text replacement: "is a" -> "is not a".

        Note: Only queries containing " is a " can be negated, so the actual percentage
        of False examples may be lower than 30% depending on the query distribution.

        Returns:
            DatasetDict with fields:
                - question: str (the query with "Prove: " removed and "... True or False?" added)
                - answer: str ("True" or "False")
                - contexts: List[str] (individual context statements)
        """
        # Load dataset from HuggingFace
        ds = load_dataset("disi-unibo-nlp-students/prontoqa-ood")

        # Transform and augment each example
        def process_example(example, idx):
            # Parse contexts from question field (split by newlines)
            contexts = [line.strip() for line in example["question"].split('\n') if line.strip()]
            contexts = [c.strip() + "." for cont in contexts for c in cont.split('.') if c.strip()]

            # Parse query: remove "Prove: " and add "... True or False?"
            query = example["query"]
            if query.startswith("Prove: "):
                query = query[7:]  # Remove "Prove: " (7 characters)

            # Currently, all queries start with "A is a ..." or "A is ..."
            # Move the is (first occurrence) at the start, and replace the final "." with "?"
            parts = query.split(' is ', 1)  # Split on first occurrence of " is "
            subject = parts[0]
            predicate = parts[1].rstrip('.')  # Remove trailing period

            # Create true and false examples with 50% probability
            rng = random.Random(self.sample_seed + idx)
            if rng.random() < 0.5 and re.match(r'^\w+\s+is\b', query):
                # Create False examples from the True ones
                # The question format is currently "Is A ...?" or "Is A a ...?"
                # These should be negated, so that they become "Is A not ...?" or "Is A not a ...?"
                answer = "No"
                # Check if predicate already starts with "not" to avoid double negation
                if predicate.startswith("not "):
                    # Remove the "not" instead of adding another one
                    question = f"Is {subject} {predicate[4:]}?"  # Skip "not " (4 characters)
                else:
                    # Add "not" before the predicate
                    question = f"Is {subject} not {predicate}?"
            else:
                # Keep as True example
                answer = "Yes"
                question = f"Is {subject} {predicate}?"

            return {
                "question": question,
                "answer": answer,
                "contexts": contexts
            }

        # Apply transformation with index for deterministic randomness
        ds = ds.map(process_example, with_indices=True)

        # Remove original columns, keeping only standardized fields
        columns_to_remove = [col for col in ds['train'].column_names
                             if col not in ['question', 'answer', 'contexts']]
        ds = ds.remove_columns(columns_to_remove)

        return ds

    def _load_logic_dataset(self) -> DatasetDict:
        """Load and preprocess logic translation dataset from HuggingFace.

        Returns:
            DatasetDict with fields:
                - question: str
                - answer: str
                - contexts: List[str] (NL facts)
                - facts: List[str] (FOL formulas, optional based on include_facts_field)
                - proof: List[str] (proof chain)
        """
        # Get HuggingFace repo path from the logic_dataset_map
        if self.dataset_name not in self.logic_dataset_map:
            raise ValueError(f"Logic dataset {self.dataset_name} not found in logic_dataset_map")

        hf_repo_path = self.logic_dataset_map[self.dataset_name]
        ds = load_dataset(hf_repo_path)

        # Smart split detection: check if dataset has 'test' split
        if 'test' in ds:
            # Use existing test split
            # Keep only train and test splits
            ds = DatasetDict({
                'train': ds['train'],
                'test': ds['test']
            })
        else:
            # Create train/test split from 'train' (90/10 split)
            train_test_split = ds['train'].train_test_split(test_size=0.1, seed=self.sample_seed)
            ds = DatasetDict({
                'train': train_test_split['train'],
                'test': train_test_split['test']
            })

        # Process each example to extract standardized fields
        ds = ds.map(
            lambda x: self._preprocess_logic_instance(x),
            remove_columns=ds['train'].column_names  # Remove all original columns
        )

        return ds

    def _preprocess_logic_instance(self, instance):
        """Preprocess a single logic dataset instance.

        Args:
            instance: Raw instance from the logic translation dataset

        Returns:
            Dictionary with standardized fields
        """
        # Set random seed for reproducibility (combine with instance_id if available)
        rng = random.Random(self.sample_seed)
        if 'instance_id' in instance:
            rng = random.Random(hash(instance['instance_id']) + self.sample_seed)

        # Extract Q&A based on yesno_ratio (fields are directly in instance)
        use_yesno = rng.random() < self.yesno_ratio

        if use_yesno and 'yesno_question' in instance:
            question = instance['yesno_question']
            answer = instance['yesno_answer']
        elif 'shortform_question' in instance:
            question = instance['shortform_question']
            answer = instance['shortform_answer']
        else:
            # Fallback if structure is unexpected
            question = "No question available"
            answer = "No answer available"

        # Extract proof chain (fields are directly in instance with _proof suffix)
        # proof_field_map = {
        #     'extended': 'extended_proof',
        #     'reduced': 'reduced_proof',
        #     'combined': 'combined_proof'
        # }
        # proof_field = proof_field_map.get(self.proof_type, 'combined_proof')
        # proof = instance.get(proof_field, [])

        # Extract required fields for building proof dynamically
        proof_structure_str = instance.get('proof_structure', '[]')
        try:
            proof_structure = json.loads(proof_structure_str) if isinstance(proof_structure_str, str) else proof_structure_str
        except json.JSONDecodeError:
            proof_structure = []

        proof_elements_dict_str = instance.get('proof_elements_dict', '{}')
        try:
            proof_elements_dict = json.loads(proof_elements_dict_str) if isinstance(proof_elements_dict_str, str) else proof_elements_dict_str
        except json.JSONDecodeError:
            proof_elements_dict = {}

        # Extract facts (stored as JSON string)
        facts_dict_str = instance.get('facts_dict', '{}')
        try:
            facts_dict = json.loads(facts_dict_str) if isinstance(facts_dict_str, str) else facts_dict_str
        except json.JSONDecodeError:
            facts_dict = {}

        # Extract natural language descriptions of facts
        facts_nl_str = instance.get('facts_nl', '{}')
        try:
            facts_nl = json.loads(facts_nl_str) if isinstance(facts_nl_str, str) else facts_nl_str
        except json.JSONDecodeError:
            facts_nl = {}

        # Get existing extended_proof and hypothesis for building new proof types
        extended_proof = instance.get('extended_proof', [])
        hypothesis = instance.get('hypothesis', '')
        is_contradiction = instance.get('is_proof_by_contradiction', False)

        # Build proof based on type
        if self.proof_type == 'standard':
            # Standard: extended_proof + hypothesis at end
            proof = extended_proof.copy() if isinstance(extended_proof, list) else []
            final_formula = f"¬({hypothesis})" if is_contradiction else hypothesis
            proof.append(final_formula)

        elif self.proof_type == 'extended':
            # Extended: Add first-use elements before implications, then hypothesis at end
            proof = self._build_extended_proof(
                extended_proof, proof_structure, facts_dict,
                proof_elements_dict, hypothesis, is_contradiction
            )

        else:
            # Fallback for old proof types
            proof = []

        # Determine which facts to include based on max_facts and proof presence
        if self.max_facts is not None:
            fact_keys = self._prioritize_facts(facts_dict, proof, self.max_facts)
        else:
            fact_keys = sorted(facts_dict.keys())

        # Build contexts (NL) and facts (FOL) lists
        contexts = [facts_nl.get(key, '') for key in fact_keys]  # Natural language descriptions
        facts_list = [facts_dict[key] for key in fact_keys]  # Translated FOL formulas

        # Build result dictionary
        result = {
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'proof': proof
        }

        # Add facts field if requested
        if self.include_facts_field:
            result['facts'] = facts_list

        return result

    def _prioritize_facts(self, facts_dict, proof, max_facts):
        """Prioritize facts that appear in the proof, then add others up to max_facts.

        Args:
            facts_dict: Dictionary mapping FOL formulas to NL representations
            proof: List of proof steps
            max_facts: Maximum number of facts to include

        Returns:
            List of fact keys (FOL formulas) to include
        """
        proof_str = ' '.join(proof) if proof else ''
        fact_keys = list(facts_dict.keys())

        # Separate facts into those in proof and those not in proof
        in_proof = []
        not_in_proof = []

        for fact_key in fact_keys:
            # Check if fact appears in proof (exact match or as substring)
            if fact_key in proof_str:
                in_proof.append(fact_key)
            else:
                not_in_proof.append(fact_key)

        # Sort for deterministic behavior
        in_proof.sort()
        not_in_proof.sort()

        # Prioritize facts in proof, then add others
        prioritized = in_proof + not_in_proof

        # Limit to max_facts
        return prioritized[:max_facts]

    def _build_extended_proof(self, extended_proof, proof_structure, facts_dict,
                              proof_elements_dict, hypothesis, is_contradiction):
        """Build extended proof with first-use elements before implications.

        This takes the extended_proof (which has implications like "(A ∧ B) → C") and
        reconstructs it by adding individual source elements before each implication
        the first time they're used. Formulas are wrapped in parentheses for clarity.

        Args:
            extended_proof: List of implication strings from dataset
            proof_structure: List of proof step structures like ["fact1 & fact2 -> int1"]
            facts_dict: Dictionary of fact formulas
            proof_elements_dict: Dictionary of proof element formulas
            hypothesis: Hypothesis formula
            is_contradiction: Whether this is proof by contradiction

        Returns:
            List with first-use elements (parenthesized), implications, and hypothesis
        """
        import re

        proof_steps = []
        seen_elements = set()

        # Build combined formula dictionary
        all_formulas = {}
        all_formulas.update(facts_dict)
        all_formulas.update(proof_elements_dict)
        all_formulas["hypothesis"] = hypothesis

        def add_parentheses_if_needed(formula):
            """Add parentheses around formula if it contains operators and isn't already parenthesized."""
            formula = formula.strip()
            # Check if already fully parenthesized
            if formula.startswith('(') and formula.endswith(')'):
                return formula
            # Check if contains logical operators (needs parentheses)
            if any(op in formula for op in ['∧', '∨', '→', '↔', '¬']):
                return f"({formula})"
            return formula

        # Process each proof step from proof_structure
        for i, step_structure in enumerate(proof_structure):
            match = re.match(r'(.+?)\s*->\s*(.+)', step_structure)
            if not match:
                continue

            sources_str, target = match.groups()
            target = target.strip()

            # Parse source elements
            source_parts = [s.strip() for s in sources_str.split('&')]

            # Add each source formula if first time (with parentheses)
            for part in source_parts:
                part_clean = part.strip('[]').strip()
                if part_clean.lower() == 'void':
                    continue

                if part_clean not in seen_elements:
                    formula = all_formulas.get(part_clean, None)
                    if formula and formula.strip() != "False":
                        proof_steps.append(add_parentheses_if_needed(formula))
                        seen_elements.add(part_clean)

            # Add the corresponding implication from extended_proof
            if i < len(extended_proof):
                proof_steps.append(extended_proof[i])

            seen_elements.add(target)

        # Add hypothesis at end (with parentheses if needed)
        final_formula = f"¬({hypothesis})" if is_contradiction else hypothesis
        proof_steps.append(add_parentheses_if_needed(final_formula))

        return proof_steps


if __name__ == "__main__":
    # Print dataset sizes for all supported datasets (no downsampling)
    datasets = ["hotpotqa", "2wiki", "folio", "prontoqa", "logic_translations_q14B"]

    print("=" * 60)
    print("Dataset Instance Counts (no downsampling)")
    print("=" * 60)

    for ds_name in datasets:
        try:
            handler = DatasetHandler(ds_name, filter_non_ascii=False)
            ds = handler.load_dataset()
            print(f"\n{ds_name}:")
            print(f"  train: {len(ds['train']):,} instances")
            print(f"  test:  {len(ds['test']):,} instances")
        except Exception as e:
            print(f"\n{ds_name}: Error loading - {e}")

    print("\n" + "=" * 60)