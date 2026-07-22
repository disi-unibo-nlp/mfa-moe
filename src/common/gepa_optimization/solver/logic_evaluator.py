from nltk.inference.prover9 import Expression, Prover9Command
from solver.P9_formula import P9Formula
from solver.fol_formula import FOLFormula
from typing import List, Optional
import logging

class LogicEvaluator:
    """Class to use as interface for evaluating the FOL syntax and logic correctness of a set of formulas.

    Attributes:
        facts (Optional[str | List[str]]): The string or list of strings containing the facts.
        proof_steps (str | List[str]): The string or list of strings containing the proof steps.

    The two inputs can be either strings (with each formula separated by new lines) or lists of strings (one formula per list element).
    If facts is None, they are mixed in with the proof steps (no prior separation).

    Methods:
        is_valid(): Returns True if all formulas are valid FOL syntax, False otherwise.
        is_consistent(timeout): Returns True if the set of formulas (facts + proof_steps) is consistent.
        is_entailed(timeout): Returns True if the final proof_step is entailed by everything before.

    If logic correctness methods are called when the formulas are not valid, a RuntimeError is raised.
    """
    def __init__(self, facts: Optional[str | List[str]], proof_steps: str | List[str], timeout: int = 1) -> None:
        self.facts = facts
        self.proof_steps = proof_steps
        self.invalid_fact_indices = []
        self.invalid_proof_indices = []

        try:
            self.is_fol_valid = self._parse_formulas()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.debug(f"[EVALUATOR] Formula parsing failed: {type(e).__name__}: {e}")
            self.is_fol_valid = False
            self.solver_proof_steps = []
            self.solver_facts = []

        # Check if we have minimum valid formulas to proceed with logic checks
        self.has_minimum = (len(self.solver_proof_steps) >= 2 or
                       (len(self.solver_facts) >= 1 and len(self.solver_proof_steps) >= 1))

        if self.has_minimum:
            try:
                self.is_consistent_flag = self._is_consistent(timeout=timeout)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.debug(f"[EVALUATOR] Consistency check failed: {type(e).__name__}: {e}")
                self.is_fol_valid = False
                self.has_minimum = False
                self.is_consistent_flag = False

            if self.is_consistent_flag:
                try:
                    self.is_entailed_flag = self._is_entailed(timeout=timeout)
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.debug(f"[EVALUATOR] Entailment check failed: {type(e).__name__}: {e}")
                    self.is_fol_valid = False
                    self.has_minimum = False
                    self.is_entailed_flag = False
            else:
                self.is_entailed_flag = False
        else:
            self.is_consistent_flag = False
            self.is_entailed_flag = False

    def _split_formulas(self, formulas: str | List[str]) -> List[str]:
        """Split formulas string into list of formulas, or return the list as is. Remove empty formulas."""
        if isinstance(formulas, str):
            formulas = formulas.strip().split('\n')
        return [formula.strip() for formula in formulas if formula.strip()] # remove empty formulas
    
    def _convert_to_prover9(self, formulas: List[str]) -> tuple[List[Expression], List[int]]:
        solver_formulas = []
        invalid_indices = []
        for i, fact in enumerate(formulas):
            fol_rule = FOLFormula(fact)
            if fol_rule.is_valid:
                p9_rule = P9Formula(fol_rule)
                solver_formulas.append(Expression.fromstring(p9_rule.formula))
            else:
                invalid_indices.append(i)
        return solver_formulas, invalid_indices

    def _parse_formulas(self):
        is_valid = True

        # Uniform inputs to lists of formulas
        if self.facts is not None:
            self.facts = self._split_formulas(self.facts)
        self.proof_steps = self._split_formulas(self.proof_steps)

        # convert facts to prover9 format if present
        self.solver_facts = []
        if self.facts is not None:
            self.solver_facts, self.invalid_fact_indices = self._convert_to_prover9(self.facts) 
            if self.invalid_fact_indices != []:
                is_valid = False

        # convert proof steps to prover9 format
        self.solver_proof_steps, self.invalid_proof_indices = self._convert_to_prover9(self.proof_steps)
        if self.invalid_proof_indices != []:
            is_valid = False

        return is_valid
    
    def _is_entailed(self, timeout: int = 1) -> bool:
        assumptions = self.solver_facts + self.solver_proof_steps[:-1]
        goal = self.solver_proof_steps[-1]
        prover = Prover9Command(goal, assumptions, timeout=timeout)
        return prover.prove()

    def _is_consistent(self, timeout: int = 1) -> bool:
        all_formulas = self.solver_facts + self.solver_proof_steps
        false_formula = Expression.fromstring('False')
        is_contradiction = Prover9Command(false_formula, all_formulas, timeout=timeout).prove()
        return not is_contradiction
    
    def is_valid(self) -> bool:
        # Check if the logic program has minimum valid formulas to proceed
        return self.has_minimum
    
    def is_entailed(self) -> bool:
        """Check if the final proof step is entailed by the facts and previous proof steps."""
        if not self.is_valid():
            raise RuntimeError("Cannot check entailment: logic program contains invalid formulas.")
        return self.is_entailed_flag
    
    def is_consistent(self) -> bool:
        """Check if the set of formulas (facts + proof steps) is consistent."""
        if not self.is_valid():
            raise RuntimeError("Cannot check consistency: logic program contains invalid formulas.")
        return self.is_consistent_flag

    def get_valid_rate(self) -> float:
        """Return the rate of valid FOL formulas (between 0 and 1).

        Returns:
            float: The proportion of valid formulas out of total formulas.
                   Returns 0.0 if there are no formulas.
        """
        if not self.has_minimum:
            return 0.0

        # Count total formulas
        total_formulas = 0
        if self.facts is not None:
            total_formulas += len(self.facts)
        total_formulas += len(self.proof_steps)

        if total_formulas == 0:
            return 0.0

        # Count invalid formulas
        invalid_count = len(self.invalid_fact_indices) + len(self.invalid_proof_indices)

        # Calculate valid rate
        valid_count = total_formulas - invalid_count
        return valid_count / total_formulas

    def get_valid_formula_count(self) -> dict:
        """Return the count of valid FOL formulas by category.

        Returns:
            dict: Dictionary with keys:
                - 'facts': Number of valid fact formulas
                - 'proof_steps': Number of valid proof step formulas
                - 'total': Total number of valid formulas
        """
        valid_fact_count = len(self.solver_facts)
        valid_proof_count = len(self.solver_proof_steps)

        return {
            'facts': valid_fact_count,
            'proof_steps': valid_proof_count,
            'total': valid_fact_count + valid_proof_count
        }

    def has_minimum_valid_formulas(self) -> bool:
        """Check if there are enough valid formulas to proceed with logic checks.

        Returns:
            bool: True if there are at least 2 valid proof steps OR
                  at least 1 valid proof step and at least 1 valid fact.
        """
        return self.has_minimum

if __name__ == "__main__":
    # should be consistent but not entail the final step
    facts = """Swag(Davide)
    Married(Davide, Alice)"""
    proof_steps = """∀x ∀y ((Married(x, y)) → (Swag(x) → Swag(y)))
    ¬Swag(Alice)"""

    evaluator = LogicEvaluator(facts, proof_steps)
    print("Example 1")
    print("Is valid:", evaluator.is_valid())
    print("Is consistent:", evaluator.is_consistent())
    print("Is final step entailed:", evaluator.is_entailed())

    # # ground-truth: True
    # logic_program = """Premises:
    # Czech(miroslav) ∧ ChoralConductor(miroslav) ∧ Specialize(miroslav, renaissance) ∧ Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    # ∀x (ChoralConductor(x) → Musician(x)) ::: Any choral conductor is a musician.
    # ∃x (Musician(x) ∧ Love(x, music)) ::: Some musicians love music.
    # Book(methodOfStudyingGregorianChant) ∧ Author(miroslav, methodOfStudyingGregorianChant) ∧ Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    # Conclusion:
    # ∃y ∃x (Czech(x) ∧ Author(x, y) ∧ Book(y) ∧ Publish(y, year1946)) ::: A Czech person wrote a book in 1946.
    # """

    # # ground-truth: False
    # logic_program = """Premises:
    # MusicPiece(symphonyNo9) ::: Symphony No. 9 is a music piece.
    # ∀x ∃z (¬Composer(x) ∨ (Write(x,z) ∧ MusicPiece(z))) ::: Composers write music pieces.
    # Write(beethoven, symphonyNo9) ::: Beethoven wrote Symphony No. 9.
    # Lead(beethoven, viennaMusicSociety) ∧ Orchestra(viennaMusicSociety) ::: Vienna Music Society is an orchestra and Beethoven leads the Vienna Music Society.
    # ∀x ∃z (¬Orchestra(x) ∨ (Lead(z,x) ∧ Conductor(z))) ::: Orchestras are led by conductors.
    # Conclusion:
    # ¬Conductor(beethoven) ::: Beethoven is not a conductor."""

    # # ground-truth: True
    # logic_program = """Predicates:
    # JapaneseCompany(x) ::: x is a Japanese game company.
    # Create(x, y) ::: x created the game y.
    # Top10(x) ::: x is in the Top 10 list.
    # Sell(x, y) ::: x sold more than y copies.
    # Premises:
    # ∃x (JapaneseCompany(x) ∧ Create(x, legendOfZelda)) ::: A Japanese game company created the game the Legend of Zelda.
    # ∀x ∃z (¬Top10(x) ∨ (JapaneseCompany(z) ∧ Create(z,x))) ::: All games in the Top 10 list are made by Japanese game companies.
    # ∀x (Sell(x, oneMillion) → Top10(x)) ::: If a game sells more than one million copies, then it will be selected into the Top 10 list.
    # Sell(legendOfZelda, oneMillion) ::: The Legend of Zelda sold more than one million copies.
    # Conclusion:
    # Top10(legendOfZelda) ::: The Legend of Zelda is in the Top 10 list."""

    # logic_program = """Premises:
    # ∀x (Listed(x) → ¬NegativeReviews(x)) ::: If the restaurant is listed in Yelp’s recommendations, then the restaurant does not receive many negative reviews.
    # ∀x (GreaterThanNine(x) → Listed(x)) ::: All restaurants with a rating greater than 9 are listed in Yelp’s recommendations.
    # ∃x (¬TakeOut(x) ∧ NegativeReviews(x)) ::: Some restaurants that do not provide take-out service receive many negative reviews.
    # ∀x (Popular(x) → GreaterThanNine(x)) ::: All restaurants that are popular among local residents have ratings greater than 9.
    # GreaterThanNine(subway) ∨ Popular(subway) ::: Subway has a rating greater than 9 or is popular among local residents.
    # Conclusion:
    # TakeOut(subway) ∧ ¬NegativeReviews(subway) ::: Subway provides take-out service and does not receive many negative reviews."""