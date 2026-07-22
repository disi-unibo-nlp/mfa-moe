from typing import List, Optional, Tuple
from pathlib import Path
import yaml
import os
import random
from dataclasses import dataclass


class PromptConfig:
    """
    Configuration class for prompt templates.

    Attributes:
        system_prompt: Optional system prompt string.
        user_prompt: User prompt string with {question} and {context} placeholders.
        context_mode: "first" to use the first context (after optional shuffling), "all" to use all contexts.
        context_sep: Separator string to use when context_mode is "all".
        context_shuffle: Whether to shuffle contexts before selection/application.
        assistant_prompt: Optional assistant prompt string. Will be used if include_assistant_prompt is True. Can include {answer}, {facts} and {proof} placeholders.
    """
    def __init__(
            self, 
            system_prompt: Optional[str], 
            user_prompt: str, 
            context_mode: str,
            context_sep: Optional[str],
            context_shuffle: bool = False,
            assistant_prompt: Optional[str] = None
        ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.context_mode = context_mode
        self.context_sep = context_sep
        self.context_shuffle = context_shuffle
        self.assistant_prompt = assistant_prompt
        self.check_config()

    def check_config(self):
        """Validate the template configuration."""
        if self.context_mode not in ["first", "all"]:
            raise ValueError(f"Invalid context_mode: {self.context_mode}. Must be 'first' or 'all'.")
        if self.user_prompt is None:
            raise ValueError("user_prompt must be provided in the template.")
        if self.context_mode == "all" and not self.context_sep:
            raise ValueError("context_sep must be provided when context_mode is 'all'.")
        if self.context_shuffle is None:
            raise ValueError("context_shuffle must be specified in the template.")
        if "{question}" not in self.user_prompt:
            raise ValueError("user_prompt must contain a {question} placeholder.")
        if "{context}" not in self.user_prompt:
            raise ValueError("user_prompt must contain a {context} placeholder.")
    
    @staticmethod
    def from_dict(config_dict: dict) -> 'PromptConfig':
        """Create a PromptConfig instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters.
            
        Returns:
            PromptConfig instance.
        """
        return PromptConfig(
            system_prompt=config_dict.get('system_prompt'),
            user_prompt=config_dict['user_prompt'],
            context_mode=config_dict['context_mode'],
            context_sep=config_dict.get('context_sep'),
            context_shuffle=config_dict.get('context_shuffle', False),
            assistant_prompt=config_dict.get('assistant_prompt')
        )


class ParsingConfig:
    """
        Args:
            logic_present: Whether a logic block is expected in the output.
            logic_start: Substring indicating the start of the logic block. If None, starts at the beginning.
            logic_end: Substring indicating the end of the logic block. If None, goes until the end (must be present if answer_present is True and logic_present is True).
            facts_proof_divided: Whether facts and proof steps are divided in separate sections. If True, facts_end must be specified. If False, everything between logic_start and logic_end is considered proof.
            facts_start: Substring indicating the start of the facts section (if None, starts at the beginning of logic block).
            facts_end: Substring indicating the end of the facts section (must be specified if facts_proof_divided is True).
            proof_start: Substring indicating the start of the proof section (if None, it's set to facts_end).
            proof_end: Substring indicating the end of the proof section (if None, proof goes until the end of logic block).
            answer_present: Whether an answer is expected in the output.
            answer_start: Substring indicating the separator before the answer (ignored if answer_present is False). If None and logic_end is specified, this is set to logic_end.
    """
    def __init__(
        self,
        logic_present: bool,
        logic_start: Optional[str],
        logic_end: Optional[str],
        facts_proof_divided: bool,
        facts_start: Optional[str],
        facts_end: Optional[str],
        proof_start: Optional[str],
        proof_end: Optional[str],
        answer_present: bool,
        answer_start: Optional[str]
        ):
        self.logic_present = logic_present
        self.logic_start = logic_start
        self.logic_end = logic_end
        self.facts_proof_divided = facts_proof_divided
        self.facts_start = facts_start
        self.facts_end = facts_end
        self.proof_start = proof_start
        self.proof_end = proof_end
        self.answer_present = answer_present
        self.answer_start = answer_start
        self.check_config()
        
    def check_config(self):
        """
        Validate configuration
        """
        if not self.answer_present and not self.logic_present:
            raise ValueError("At least one of logic_present or answer_present must be True.")
        if not self.logic_present and (self.logic_start is not None or self.logic_end is not None):
            raise ValueError("logic_start and logic_end should be None when logic_present is False.")
        if not self.answer_present and self.answer_start is not None:
            raise ValueError("answer_start should be None when answer_present is False.")
        if self.logic_present and self.answer_present: 
            if self.logic_end is None:
                raise ValueError("logic_end must be specified when answer and logic are expected.")
            if self.answer_start is None:
                self.answer_start = self.logic_end # set default answer_start to logic_end if not specified
        if self.facts_proof_divided and not self.logic_present:
            raise ValueError("facts_proof_divided can only be True when logic_present is also True.")
        if not self.facts_proof_divided:
            if self.facts_start is not None or self.facts_end is not None or self.proof_start is not None or self.proof_end is not None:
                raise ValueError("facts_start, facts_end, proof_start, proof_end should be None when facts_proof_divided is False.")
        else:
            if self.facts_end is None:
                raise ValueError("facts_end must be specified when facts_proof_divided is True.")
            if self.proof_start is None:
                self.proof_start = self.facts_end # set default proof_start to facts_end if not specified

    def from_dict(config_dict: dict) -> 'ParsingConfig':
        """Create a ParsingConfig instance from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.
        Returns:
            ParsingConfig instance.
        """
        answer_start = config_dict.get('answer_start')
        if answer_start is not None:
            answer_start = answer_start.lower()
        return ParsingConfig(
            logic_present=config_dict['logic_present'],
            logic_start=config_dict.get('logic_start'),
            logic_end=config_dict.get('logic_end'),
            facts_proof_divided=config_dict.get('facts_proof_divided', False),
            facts_start=config_dict.get('facts_start'),
            facts_end=config_dict.get('facts_end'),
            proof_start=config_dict.get('proof_start'),
            proof_end=config_dict.get('proof_end'),
            answer_present=config_dict['answer_present'],
            answer_start=answer_start
        )


@dataclass
class ParsingOutput:
    success: bool
    format_quality: float
    facts: Optional[str]
    proof: Optional[str]
    answer: Optional[str]


class TemplateHandler:
    """
    Handler class for loading and managing format templates.

    Loads the specified template file from the templates directory.

    Handles prompt formatting and output parsing according to the loaded template.
    """
    def __init__(self,
            template_name: str = "default",
            include_assistant_prompt: bool = False,
            templates_root: Path = Path("./templates")
        ):
        """
        Args:
            template_name: Name of the template file (without .yaml extension). Defaults to "default".
            include_assistant_prompt: Whether to include assistant prompt in the messages.
            templates_root: Root directory containing template files.
        """
        self.templates_root = templates_root
        self.template_name = template_name
        self.include_assistant_prompt = include_assistant_prompt
        self.template = self.load_template()
        self.prompt_config = PromptConfig.from_dict(self.template["prompt"])
        self.parsing_config = ParsingConfig.from_dict(self.template["parsing"])

    def load_template(self) -> dict:
        """Load the specified template YAML file.

        Returns:
            Loaded template as a dictionary.

        Raises:
            FileNotFoundError: If the template file does not exist.
        """
        template_path = os.path.join(self.templates_root, f"{self.template_name}.yaml")

        if not os.path.exists(template_path):
            raise FileNotFoundError(
                f"Template file not found: {template_path}. "
                f"Please ensure {self.template_name}.yaml exists in {self.templates_root}/"
            )

        with open(template_path, 'r') as f:
            template_content = yaml.safe_load(f)

        return template_content
    
    def format_context(self, contexts: List[str]) -> str:
        """Format the context(s) according to the context_mode and context_shuffle settings.

        Args:
            contexts: List of context strings.

        Returns:
            Formatted context string.
        """
        if self.prompt_config.context_shuffle:
            random.shuffle(contexts)
        if self.prompt_config.context_mode == "first":
            return contexts[0]
        return self.prompt_config.context_sep.join(contexts)
    
    def format_user_prompt(self, question: str, contexts: List[str], **kwargs) -> str:
        """Format the user prompt with question and contexts, and any additional parameters.

        Args:
            question: The question string.
            contexts: List of context strings.
            **kwargs: Additional parameters to pass to the template (e.g., proof, facts).

        Returns:
            Formatted user prompt string.
        """
        formatted_context = self.format_context(contexts)
        # Prepare format kwargs with question and context
        format_kwargs = {"question": question, "context": formatted_context}

        # Add any additional kwargs (e.g., proof, facts)
        # Convert lists to strings if needed
        for key, value in kwargs.items():
            if isinstance(value, list):
                format_kwargs[key] = "\n".join(value)
            else:
                format_kwargs[key] = value

        return self.prompt_config.user_prompt.format(**format_kwargs)
    
    def format_assistant_prompt(
        self,
        answer: str,
        target_logic: Optional[str] = None,
        proof: Optional[List[str]] = None,
        facts: Optional[List[str]] = None
    ) -> str:
        """Format the assistant prompt with the answer and optionally the target logic/proof/facts.

        Args:
            answer: The answer string.
            target_logic: The target logic string (if any). Deprecated, use proof instead.
            proof: Optional list of proof steps (will be joined with newlines).
            facts: Optional list of fact formulas (will be joined with newlines).

        Returns:
            Formatted assistant prompt string.
        """
        if self.prompt_config.assistant_prompt is None:
            raise ValueError("No assistant_prompt defined in the template.")

        # Prepare format kwargs
        format_kwargs = {"answer": answer}

        # Handle target_logic (backward compatibility)
        if target_logic is not None:
            format_kwargs["target_logic"] = target_logic

        # Handle proof (convert list to string)
        if proof is not None:
            format_kwargs["proof"] = "\n".join(proof) if isinstance(proof, list) else proof

        # Handle facts (convert list to string)
        if facts is not None:
            format_kwargs["facts"] = "\n".join(facts) if isinstance(facts, list) else facts

        return self.prompt_config.assistant_prompt.format(**format_kwargs)

    def create_messages(
        self,
        question: str,
        contexts: List[str],
        answer: Optional[str] = None,
        proof: Optional[List[str]] = None,
        facts: Optional[List[str]] = None
    ) -> List[dict]:
        """Format the full prompt with question and contexts.

        Args:
            question: The question string.
            contexts: List of context strings.
            answer: Optional answer string for SFT training.
            proof: Optional list of proof steps for SFT training (also used in user prompt if template includes {proof}).
            facts: Optional list of fact formulas for SFT training (also used in user prompt if template includes {facts}).

        Returns:
            List of message dictionaries for the model input.
        """
        messages = []
        if self.prompt_config.system_prompt:
            messages.append({"role": "system", "content": self.prompt_config.system_prompt})

        # Build kwargs for user prompt formatting
        user_prompt_kwargs = {}
        if proof is not None:
            user_prompt_kwargs['proof'] = proof
        if facts is not None:
            user_prompt_kwargs['facts'] = facts

        user_content = self.format_user_prompt(question, contexts, **user_prompt_kwargs)
        messages.append({"role": "user", "content": user_content})

        if self.include_assistant_prompt and answer is not None:
            assistant_content = self.format_assistant_prompt(
                answer=answer,
                proof=proof,
                facts=facts
            )
            messages.append({"role": "assistant", "content": assistant_content})
        return messages

    def _split_check(self, string: str, sep: str) -> Tuple[bool, str, str]:
        """Split the string at the first occurrence of sep.

        Args:
            string: The string to split.
            sep: The separator string.

        Returns: tuple of
            - bool indicating whether sep was found
            - the part before sep (or full string if not found)
            - the part after sep (or full string if not found)
        """
        if sep in string:
            before, after = string.split(sep, 1)
            return True, before, after
        else:
            return False, string, string

    def _split_check_last(self, string: str, sep: str) -> Tuple[bool, str, str]:
        """Split the string at the LAST occurrence of sep.
        Useful when models echo prompts and we want the final occurrence.

        Args:
            string: The string to split.
            sep: The separator string.

        Returns: tuple of
            - bool indicating whether sep was found
            - the part before sep (or full string if not found)
            - the part after sep (or full string if not found)
        """
        if sep in string:
            before, after = string.rsplit(sep, 1)
            return True, before, after
        else:
            return False, string, string

    def _extract_logic(self, output: str) -> Tuple[Optional[str], int, int]:
        """Extract logic block from the output string and evaluate formatting score. Raise error if logic block not expected. Return none on extraction failure.
        Also returns a "max_score" and "logic_formatting_score" for logic extraction.
        """
        if not self.parsing_config.logic_present:
            raise ValueError("Logic block extraction requested but logic_present is False in parsing config.")

        max_score = 0
        score = 0

        # Remove everything before logic_start, if specified
        current_output = output
        if self.parsing_config.logic_start is not None:
            max_score += 1
            found, _, current_output = self._split_check(current_output, self.parsing_config.logic_start)
            if found:
                score += 1
        # Extract logic block
        if self.parsing_config.logic_end is not None:
            max_score += 1
            found, logic, _ = self._split_check(current_output, self.parsing_config.logic_end)
            if found:
                score += 1
        elif not self.parsing_config.answer_present:
            logic = current_output
        else:
            raise ValueError("logic_end must be specified when answer_present is True.")

        logic = logic.strip()

        if score < max_score:
            return None, max_score, score

        return logic, max_score, score
    
    def _extract_facts(self, logic_block: str) -> Tuple[Optional[str], int, int]:
        """Extract facts section from the logic block and evaluate formatting score. Raise error if facts section not expected. Return none on extraction failure.
        Also returns a "max_score" and "facts_formatting_score" for facts extraction.
        """
        if not self.parsing_config.facts_proof_divided:
            raise ValueError("Facts extraction requested but facts_proof_divided is False in parsing config.")
        
        max_score = 0
        score = 0

        current_output = logic_block
        if self.parsing_config.facts_start is not None:
            max_score += 1
            found, _, current_output = self._split_check(current_output, self.parsing_config.facts_start)
            if found:
                score += 1
        # Extract facts section
        max_score += 1
        found, facts, _ = self._split_check(current_output, self.parsing_config.facts_end)
        if found:
            score += 1
        
        facts = facts.strip()

        if score < max_score:
            return None, max_score, score
        
        return facts, max_score, score

    def _extract_proof(self, logic_block: str) -> Tuple[Optional[str], int, int]:
        """Extract proof section from the logic block and evaluate formatting score. Raise error if proof section not expected. Return none on extraction failure.
        Also returns a "max_score" and "proof_formatting_score" for proof extraction.
        """
        if not self.parsing_config.facts_proof_divided:
            raise ValueError("Proof extraction requested but facts_proof_divided is False in parsing config.")
        
        max_score = 0
        score = 0

        current_output = logic_block
        # Remove everything before proof_start (only track score if proof_start != facts_end)
        found, _, current_output = self._split_check(current_output, self.parsing_config.proof_start)
        if self.parsing_config.proof_start != self.parsing_config.facts_end:
            max_score += 1
            if found:
                score += 1
        if self.parsing_config.proof_end is not None:
            max_score += 1
            found, proof, _ = self._split_check(current_output, self.parsing_config.proof_end)
            if found:
                score += 1
        else:
            proof = current_output
        proof = proof.strip()

        if score < max_score:
            return None, max_score, score

        return proof, max_score, score

    def _extract_answer(self, output: str) -> Tuple[Optional[str], int, int]:
        """Extract answer from the output string and evaluate formatting score. Raise error if answer not expected. Return none on extraction failure.
        Also returns a "max_score" and "answer_formatting_score" for answer extraction.
        """
        if not self.parsing_config.answer_present:
            raise ValueError("Answer extraction requested but answer_present is False in parsing config.")
        
        output = output.strip().lower()

        max_score = 0
        score = 0

        current_output = output
        if self.parsing_config.answer_start is None:
            return output.strip(), 1, 1

        # Use LAST occurrence to handle models that echo prompts
        # This ensures we extract the actual answer, not echoed instruction text
        found, _, answer = self._split_check_last(current_output, self.parsing_config.answer_start)
        # only track scores if answer_start != logic_end (already tracked in logic extraction)
        if self.parsing_config.answer_start != self.parsing_config.logic_end:
            max_score += 1
            if found:
                score += 1
        answer = answer.strip()

        if not found:
            answer = None

        return answer, max_score, score

    def parse_output(self, output: str) -> ParsingOutput:
        """
        Parse the model output to extract logic and answer.
        Also returns a "format_quality" score indicating how well the output matched the expected format,
        between 0.0 (no match) and 1.0 (perfect match).

        Args:
            output: The full model output string.

        Returns:
            ParsingOutput object containing:
            - success: Whether parsing was successful.
            - format_quality: Float score between 0.0 and 1.0 indicating format match quality.
            - logic: Extracted logic string, or None if not present or parsing failed.
            - answer: Extracted answer string, or None if not present or parsing failed.
        """
        assert isinstance(output, str), "Output argument must be a string."

        max_score = 0
        score = 0
        success = True
        has_multiple_blocks = False

        # Check for multiple logic blocks (multiple occurrences of separators)
        if self.parsing_config.logic_present:
            if self.parsing_config.logic_start is not None:
                logic_start_count = output.count(self.parsing_config.logic_start)
                if logic_start_count > 1:
                    has_multiple_blocks = True
            if self.parsing_config.logic_end is not None:
                logic_end_count = output.count(self.parsing_config.logic_end)
                if logic_end_count > 1:
                    has_multiple_blocks = True

        logic = None
        facts = None
        proof = None

        if self.parsing_config.logic_present:
            logic, logic_max, logic_score = self._extract_logic(output)
            max_score += logic_max
            score += logic_score
            if logic is None:
                success = False
                logic = ""

            if self.parsing_config.facts_proof_divided:
                facts, facts_max, facts_score = self._extract_facts(logic)
                max_score += facts_max
                score += facts_score
                if facts is None:
                    success = False

                proof, proof_max, proof_score = self._extract_proof(logic)
                max_score += proof_max
                score += proof_score
                if proof is None:
                    success = False
            else:
                facts = None
                proof = logic

        answer = None
        if self.parsing_config.answer_present:
            answer, answer_max, answer_score = self._extract_answer(output)
            max_score += answer_max
            score += answer_score
            if answer is None:
                success = False

        format_quality = score / max_score if max_score > 0 else 1.0

        # If multiple logic blocks detected, cap format_quality at 0.9
        if has_multiple_blocks and format_quality > 0.9:
            format_quality = 0.9

        if not success:
            return ParsingOutput(
                success=False,
                format_quality=format_quality,
                facts=None,
                proof=None,
                answer=None
            )

        return ParsingOutput(
            success=True,
            format_quality=format_quality,
            facts=facts,
            proof=proof,
            answer=answer
        )