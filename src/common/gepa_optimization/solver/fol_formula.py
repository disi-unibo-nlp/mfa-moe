import multiprocessing
from queue import Empty
from solver.fol_parser import FOLParser


def _parse_with_timeout_worker(normalized_fol: str, result_queue) -> None:
    """Worker function for parsing FOL in subprocess to avoid GC-related timeout issues."""
    try:
        parser = FOLParser()
        tree = parser.parse_text_FOL_to_tree(normalized_fol)
        result_queue.put(('success', tree))
    except Exception as e:
        result_queue.put(('error', f"{type(e).__name__}: {str(e)}"))


class FOLFormula:
    # Symbol mapping for normalization - allows flexible parsing of alternative operator representations
    SYMBOL_NORMALIZATION_MAP = {
        # Implication
        '->': '→',
        '=>': '→',
        'implies': '→',

        # Biconditional
        '<->': '↔',
        '<=>': '↔',
        'iff': '↔',

        # Conjunction
        '&': '∧',
        '&&': '∧',
        '^': '∧',
        'and': '∧',

        # Disjunction
        '|': '∨',
        '||': '∨',
        'or': '∨',

        # Negation
        '!': '¬',
        '~': '¬',
        'not': '¬',

        # Universal quantifier
        'forall': '∀',
        'all': '∀',

        # Existential quantifier
        'exists': '∃',
        'some': '∃',

        # XOR
        'xor': '⊕',
        'XOR': '⊕',
    }

    @staticmethod
    def _normalize_symbols(formula_str: str) -> str:
        """
        Convert alternative symbol representations to canonical forms.

        Args:
            formula_str: Formula string possibly containing alternative symbols

        Returns:
            Normalized formula string with canonical symbols
        """
        import re

        # Separate word-based operators (need word boundaries) from symbol-based operators
        word_operators = {
            'and', 'or', 'not', 'forall', 'all', 'exists', 'some',
            'implies', 'iff', 'xor', 'XOR'
        }

        # Sort by length (longest first) to handle multi-char operators correctly
        # This ensures '<->' is replaced before '->', preventing incorrect splits
        sorted_mappings = sorted(
            FOLFormula.SYMBOL_NORMALIZATION_MAP.items(),
            key=lambda x: -len(x[0])
        )

        for alt_symbol, canonical_symbol in sorted_mappings:
            if alt_symbol in word_operators:
                # Use word boundaries for word-based operators to avoid matching substrings
                # \b matches word boundaries (before/after alphanumeric characters)
                pattern = r'\b' + re.escape(alt_symbol) + r'\b'
                formula_str = re.sub(pattern, canonical_symbol, formula_str)
            else:
                # Direct replacement for symbol-based operators
                formula_str = formula_str.replace(alt_symbol, canonical_symbol)

        return formula_str

    def __init__(self, str_fol, timeout: int = 2) -> None:
        self.parser = FOLParser()

        # Normalize symbols before parsing to support alternative representations
        normalized_fol = self._normalize_symbols(str_fol)

        # Parse with multiprocessing-based timeout protection
        # NLTK chart parser can enter infinite loops on complex/ambiguous formulas
        # Using multiprocessing instead of signal.alarm() to avoid GC conflicts
        try:
            result_queue = multiprocessing.Queue()
            parse_process = multiprocessing.Process(
                target=_parse_with_timeout_worker,
                args=(normalized_fol, result_queue)
            )

            parse_process.start()
            parse_process.join(timeout=timeout)

            if parse_process.is_alive():
                # Timeout: process still running, kill it
                parse_process.terminate()
                parse_process.join()
                tree = None
                self.is_valid = False
                return

            # Process finished: get result
            try:
                status, data = result_queue.get(timeout=0.1)
                if status == 'success':
                    tree = data
                else:
                    # Parse error in subprocess
                    tree = None
                    self.is_valid = False
                    return
            except Empty:
                # Process died without result
                tree = None
                self.is_valid = False
                return
        except Exception:
            # Multiprocessing setup error
            tree = None
            self.is_valid = False
            return

        self.tree = tree
        if tree is None:
            self.is_valid = False
        else:
            self.is_valid = True
            self.variables, self.constants, self.predicates = self.parser.symbol_resolution(tree)
    
    def __str__(self) -> str:
        _, rule_str = self.parser.msplit(''.join(self.tree.leaves()))
        return rule_str
    
    def is_valid(self):
        return self.is_valid

    def _get_formula_template(self, tree, name_mapping):
        for i, subtree in enumerate(tree):
            if isinstance(subtree, str):
                # Modify the leaf node label
                if subtree in name_mapping:
                    new_label = name_mapping[subtree]
                    tree[i] = new_label
            else:
                # Recursive call to process this subtree
                self._get_formula_template(subtree, name_mapping)

    def get_formula_template(self):
        template = self.tree.copy(deep=True)
        name_mapping = {}
        for i, f in enumerate(self.predicates):
            name_mapping[f] = 'F%d' % i
        for i, f in enumerate(self.constants):
            name_mapping[f] = 'C%d' % i

        self._get_formula_template(template, name_mapping)
        self.template = template
        _, self.template_str = self.parser.msplit(''.join(self.template.leaves()))
        return name_mapping, self.template_str
        
    
if __name__ == '__main__':
    # str_fol = '\u2200x (Dog(x) \u2227 WellTrained(x) \u2227 Gentle(x) \u2192 TherapyAnimal(x))'
    # str_fol = '\u2200x (Athlete(x) \u2227 WinsGold(x, olympics) \u2192 OlympicChampion(x))'
    str_fol = '\u2203x \u2203y (Czech(x) \u2227 Book(y) \u2227 Author(x, y) \u2227 Publish(y, year1946))'
    #str_fol = '∀x (InThisClub(x) ∧ PerformOftenIn(x, schoolTalentShow) → Attend(x, schoolEvent) ∧ VeryEngagedWith(x, schoolEvent))'
    #str_fol = 'PerformInTalentShows(bonnie)'
    fol_rule = FOLFormula(str_fol)
    if fol_rule.is_valid:
        print(fol_rule)
        #(fol_rule.isFOL)
        # print(fol_rule.variables)
        # print(fol_rule.constants)
        # print(fol_rule.predicates)
        name_mapping, template = fol_rule.get_formula_template()
        print(template)
        print(name_mapping)