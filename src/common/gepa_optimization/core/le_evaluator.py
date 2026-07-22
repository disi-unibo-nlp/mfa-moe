"""
Logical Equivalence (LE) Evaluator for FOL formulas.

The LE metric compares two FOL formulas by their truth tables:
1. Parse both formulas to trees
2. Extract unique literals (predicates)
3. Generate all 2^n boolean assignments
4. Find optimal literal binding between formulas
5. Compare truth table outputs
6. Return fraction of matching rows

Based on the MALLS paper (Yang et al., ACL 2024).
"""

import re
import nltk
import numpy as np
from copy import deepcopy
from itertools import product, permutations
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass
from Levenshtein import distance as edit_dist
import logging

# Simple implementation of edit distance without Levenshtein package
# def edit_dist(s1: str, s2: str) -> int:
#     """Simple Levenshtein distance implementation."""
#     if len(s1) < len(s2):
#         return edit_dist(s2, s1)
#     if len(s2) == 0:
#         return len(s1)
#     prev_row = range(len(s2) + 1)
#     for i, c1 in enumerate(s1):
#         curr_row = [i + 1]
#         for j, c2 in enumerate(s2):
#             insertions = prev_row[j + 1] + 1
#             deletions = curr_row[j] + 1
#             substitutions = prev_row[j] + (c1 != c2)
#             curr_row.append(min(insertions, deletions, substitutions))
#         prev_row = curr_row
#     return prev_row[-1]


# =============================================================================
# FOL Parser
# =============================================================================

OP_LIST = ['⊕', '∨', '∧', '→', '↔', '∀', '∃', '¬', '(', ')', ',']
SYM_REG = re.compile(r'[^⊕∨∧→↔∀∃¬(),]+')

# Extended CFG to support SynFOL grammar (adds '¬' S and '(' S ')' at S level)
CFG_TEMPLATE = """
S -> F | Q F | '¬' S | '(' S ')'
Q -> QUANT VAR | QUANT VAR Q
F -> '¬' '(' F ')' | '(' F ')' | F OP F | L
OP -> '⊕' | '∨' | '∧' | '→' | '↔'
L -> '¬' PRED '(' TERMS ')' | PRED '(' TERMS ')'
TERMS -> TERM | TERM ',' TERMS
TERM -> CONST | VAR
QUANT -> '∀' | '∃'
"""


def _reorder_quantifiers(rule_str: str) -> str:
    """Move quantifiers to the front of the formula."""
    matches = re.findall(r'[∃∀]\w', rule_str)
    for match in matches[::-1]:
        rule_str = '%s ' % match + rule_str.replace(match, '', 1)
    return rule_str


def _tokenize_fol(s: str) -> Tuple[List[str], str]:
    """Tokenize FOL string into list of tokens."""
    # Add spaces around operators
    for op in OP_LIST:
        s = s.replace(op, ' %s ' % op)

    tokens = [e.strip() for e in s.split()]
    tokens = [e.replace("'", '') for e in tokens]  # Remove quotes
    tokens = [e for e in tokens if e != '']

    # Handle multi-word symbols (e.g., "dc universe" -> "DcUniverse")
    result = []
    cur_str_ls = []
    for e in tokens:
        if (len(e) > 1) and SYM_REG.match(e):
            cur_str_ls.append(e[0].upper() + e[1:])
        else:
            if len(cur_str_ls) > 0:
                result.extend([''.join(cur_str_ls), e])
            else:
                result.extend([e])
            cur_str_ls = []
    if len(cur_str_ls) > 0:
        result.append(''.join(cur_str_ls))

    return result, s


def _make_cfg_str(token_ls: List[str]) -> str:
    """Create CFG string with dynamic symbol definitions."""
    sym_ls = list(set([e for e in token_ls if SYM_REG.match(e)]))
    if not sym_ls:
        sym_ls = ['DUMMY']  # Need at least one symbol
    sym_str = ' | '.join(["'%s'" % s for s in sym_ls])
    cfg_str = CFG_TEMPLATE + 'VAR -> %s\nPRED -> %s\nCONST -> %s' % (sym_str, sym_str, sym_str)
    return cfg_str


def parse_fol_to_tree(rule_str: str) -> Optional[nltk.Tree]:
    """Parse FOL string to NLTK tree. Returns None if parsing fails."""
    try:
        rule_str = _reorder_quantifiers(rule_str)
        tokens, _ = _tokenize_fol(rule_str)
        cfg_str = _make_cfg_str(tokens)

        grammar = nltk.CFG.fromstring(cfg_str)
        parser = nltk.ChartParser(grammar)
        tree = parser.parse_one(tokens)
        return tree
    except Exception:
        return None


# =============================================================================
# LE Metric Implementation
# =============================================================================

@dataclass
class LEResult:
    """Result of LE score computation."""
    score: float  # 0.0 to 1.0
    success: bool
    error_message: Optional[str] = None
    true_inputs: Optional[List[str]] = None
    pred_inputs: Optional[List[str]] = None


class LEEvaluator:
    """Logical Equivalence evaluator based on truth table comparison."""

    DUMMY_INPUT_STR = '#DUMMY'
    DUMMY_DISTANCE = 10000

    def __init__(
        self,
        max_literals: int = 12,
        soft_binding: bool = True,
        greedy_match: bool = True,
        top_n: int = 100
    ):
        """
        Initialize LE Evaluator.

        Args:
            max_literals: Maximum number of literals before returning 0 (computational limit)
            soft_binding: If True, handle different literal counts via dummy variables
            greedy_match: If True, use greedy matching for literal binding (faster)
            top_n: Maximum number of bindings to evaluate
        """
        self.max_literals = max_literals
        self.soft_binding = soft_binding
        self.greedy_match = greedy_match
        self.top_n = top_n

        self.logger = logging.getLogger(__name__)

    def compute_le_score(self, predicted_fol: str, reference_fol: str) -> LEResult:
        """
        Compute LE score between predicted and reference FOL strings.

        Args:
            predicted_fol: Predicted FOL formula string
            reference_fol: Ground truth FOL formula string

        Returns:
            LEResult with score, success status, and optional details
        """
        # Parse both formulas
        ref_tree = parse_fol_to_tree(reference_fol)
        if ref_tree is None:
            return LEResult(
                score=0.0,
                success=False,
                error_message=f"Failed to parse reference FOL: {reference_fol}"
            )

        pred_tree = parse_fol_to_tree(predicted_fol)
        if pred_tree is None:
            return LEResult(
                score=0.0,
                success=False,
                error_message=f"Failed to parse predicted FOL: {predicted_fol}"
            )

        try:
            score, true_inputs, pred_inputs = self._find_best_le_score(ref_tree, pred_tree)
            return LEResult(
                score=score,
                success=True,
                true_inputs=true_inputs,
                pred_inputs=pred_inputs
            )
        except Exception as e:
            return LEResult(
                score=0.0,
                success=False,
                error_message=f"Error computing LE: {str(e)}"
            )

    def _find_best_le_score(
        self,
        true_root: nltk.Tree,
        pred_root: nltk.Tree
    ) -> Tuple[float, List[str], List[str]]:
        """Find the best LE score over all literal bindings."""

        # Extract unique literals from each tree
        true_inputs: Set[str] = set()
        pred_inputs: Set[str] = set()
        self._find_inputs(true_root, true_inputs)
        self._find_inputs(pred_root, pred_inputs)
        true_inputs_list = list(true_inputs)
        pred_inputs_list = list(pred_inputs)

        n_true = len(true_inputs_list)
        n_pred = len(pred_inputs_list)

        # Check computational limit
        max_n = max(n_true, n_pred)
        if max_n > self.max_literals:
            self.logger.warning(f"Number of literals {max_n} exceeds max_literals ({self.max_literals}). Returning 0 score.")
            return 0.0, true_inputs_list, pred_inputs_list

        best_score = 0.0
        best_pred_inputs = None

        # Handle different literal counts
        if n_true != n_pred:
            if self.soft_binding:
                # Extend shorter list with dummy inputs
                diff = abs(n_true - n_pred)
                if n_true < max_n:
                    true_inputs_list.extend([f'{self.DUMMY_INPUT_STR}_{i}' for i in range(diff)])
                else:
                    pred_inputs_list.extend([f'{self.DUMMY_INPUT_STR}_{i}' for i in range(diff)])
            else:
                return 0.0, true_inputs_list, pred_inputs_list

        # Generate all 2^n boolean assignments
        input_vecs = self._gen_input_vecs(len(true_inputs_list))

        # Compute truth table for reference
        true_name2ind = {e: i for i, e in enumerate(true_inputs_list)}
        true_res_vec = self._eval_tree(true_root, true_name2ind, input_vecs)

        # Try different literal bindings
        if self.greedy_match:
            binding_iter = self._enumerate_bindings_greedy(true_inputs_list, pred_inputs_list, self.top_n)
        else:
            binding_iter = permutations(range(len(pred_inputs_list)))

        for cnt, binding_inds in enumerate(binding_iter):
            if isinstance(binding_inds, list):
                binded_pred = [pred_inputs_list[i] for i in binding_inds]
            else:
                binded_pred = [pred_inputs_list[i] for i in binding_inds]

            pred_name2ind = {e: i for i, e in enumerate(binded_pred)}
            pred_res_vec = self._eval_tree(pred_root, pred_name2ind, input_vecs)

            # Compute matching fraction
            score = float(np.mean(pred_res_vec == true_res_vec))

            if score > best_score:
                best_score = score
                best_pred_inputs = binded_pred

            if cnt + 1 >= self.top_n:
                break

        return best_score, true_inputs_list, best_pred_inputs or pred_inputs_list

    def _find_inputs(self, root, input_set: Set[str]) -> None:
        """Extract unique literals (predicates with arguments) from parse tree."""
        if isinstance(root, str):
            return

        label = root.label()

        if label == 'L':
            # Literal node - extract the predicate string
            literal_str = ''.join(root.leaves())
            # Remove leading negation if present
            if literal_str.startswith('¬'):
                literal_str = literal_str[1:]
            input_set.add(literal_str)
        else:
            for child in root:
                self._find_inputs(child, input_set)

    def _gen_input_vecs(self, num_inputs: int) -> np.ndarray:
        """Generate all 2^n boolean truth table assignments."""
        return np.array(list(product([False, True], repeat=num_inputs)))

    def _eval_tree(self, root, name2ind: dict, input_vecs: np.ndarray) -> np.ndarray:
        """Evaluate parse tree to boolean vector over all truth assignments."""
        if isinstance(root, str):
            raise ValueError("Should not reach leaf string directly")

        label = root.label()

        if label == 'S':
            # Handle extended grammar: S -> '¬' S | '(' S ')' | F | Q F
            if len(root) >= 2 and isinstance(root[0], str) and root[0] == '¬':
                # S -> '¬' S
                return ~self._eval_tree(root[1], name2ind, input_vecs)
            elif len(root) >= 3 and isinstance(root[0], str) and root[0] == '(':
                # S -> '(' S ')'
                return self._eval_tree(root[1], name2ind, input_vecs)
            else:
                # S -> F | Q F (quantifier ignored for propositional evaluation)
                return self._eval_tree(root[-1], name2ind, input_vecs)

        elif label == 'F':
            if len(root) == 1 and root[0].label() == 'L':
                # F -> L
                return self._eval_tree(root[0], name2ind, input_vecs)
            elif root[-2].label() == 'F':
                # F -> '¬' '(' F ')' | '(' F ')'
                is_negated = isinstance(root[0], str) and root[0] == '¬'
                result = self._eval_tree(root[-2], name2ind, input_vecs)
                return ~result if is_negated else result
            elif root[-2].label() == 'OP':
                # F -> F OP F
                p = self._eval_tree(root[0], name2ind, input_vecs)
                q = self._eval_tree(root[-1], name2ind, input_vecs)
                op = root[1][0]  # Get operator string

                if op == '⊕':
                    return np.logical_xor(p, q)
                elif op == '∨':
                    return np.logical_or(p, q)
                elif op == '∧':
                    return np.logical_and(p, q)
                elif op == '→':
                    return np.logical_or(~p, q)  # p → q ≡ ¬p ∨ q
                elif op == '↔':
                    return np.logical_or(np.logical_and(p, q), np.logical_and(~p, ~q))
                else:
                    raise ValueError(f"Unknown operator: {op}")
            else:
                raise ValueError(f"Unexpected F structure: {root}")

        elif label == 'L':
            # Literal node
            literal_str = ''.join(root.leaves())
            is_negated = literal_str.startswith('¬')
            if is_negated:
                literal_str = literal_str[1:]

            if literal_str not in name2ind:
                # Dummy literal - always False
                vec = np.zeros(len(input_vecs), dtype=bool)
            else:
                vec = input_vecs[:, name2ind[literal_str]]

            return ~vec if is_negated else vec

        else:
            raise ValueError(f"Unexpected label: {label}")

    def _enumerate_bindings_greedy(self, ls1: List[str], ls2: List[str], top_n: int):
        """Enumerate literal bindings using greedy edit distance matching."""
        used_inds = []

        def _similarity(e1: str, e2: str) -> int:
            if e1.startswith(self.DUMMY_INPUT_STR) or e2.startswith(self.DUMMY_INPUT_STR):
                return self.DUMMY_DISTANCE
            return edit_dist(e1, e2)

        def _enum(ind1: int):
            if ind1 == len(ls1):
                yield deepcopy(used_inds)
                return

            e1 = ls1[ind1]
            matches = [(i, _similarity(e1, ls2[i])) for i in range(len(ls2)) if i not in used_inds]
            matches.sort(key=lambda x: x[1])

            for i, _ in matches:
                used_inds.append(i)
                yield from _enum(ind1 + 1)
                used_inds.pop()

        for cnt, binding in enumerate(_enum(0)):
            yield binding
            if cnt + 1 >= top_n:
                break
