from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .data import SENTENCE_LABELS, EpisodeDocument

SEED_INSTRUCTIONS = r"""You are an expert reviewer of mathematical reasoning traces. Classify
the CURRENT UNIT by its primary local function in the adapted Schoenfeld Episode Theory.

Use the problem statement and neighboring units to resolve where information came from and what
the current unit is doing. Classify the current unit only; neighboring text is evidence, not the
target.

Choose exactly one label:
- Read: passively restates information, choices, or the goal explicitly present in the problem.
- Analyze: recalls a concept, interprets information, introduces notation, identifies structure,
  or derives a relationship without mechanically carrying out a chosen computation.
- Plan: commits to a concrete next mathematical operation or strategy before executing it.
- Implement: executes a procedure, substitution, calculation, enumeration, or directly states
  the result of that execution.
- Explore: tentatively proposes a substantive hypothesis, option, guess, or alternative path.
- Verify: checks or judges an existing candidate, calculation, conclusion, or consistency claim.
- Monitor: a content-light hesitation, self-regulation comment, transition, or structural marker.

Apply this decision sequence:
1. Is it only a structural marker or content-light self-regulation? Monitor.
2. Is it checking an already available result or candidate? Verify.
3. Is it tentatively proposing a substantive possibility? Explore.
4. Is it committing to a concrete future mathematical action? Plan.
5. Is it carrying out an operation or giving its direct result? Implement.
6. Is it merely repeating the problem statement? Read.
7. Otherwise, if it interprets or derives mathematical structure, choose Analyze.

Critical boundaries:
- Read versus Analyze: use the supplied problem statement. A given fact is Read; an interpretation,
  identification, notation choice, or derived relation is Analyze.
- Analyze versus Verify: a new inference is Analyze; testing or confirming an existing claim is
  Verify.
- Analyze versus Implement: explaining or deriving a relationship is Analyze; mechanically
  applying a selected operation is Implement.
- Plan versus Monitor: a concrete future operation is Plan; "let me think", "okay", or a transition
  with no mathematical action is Monitor.
- Explore versus Analyze: an uncertain possibility is Explore; a committed deduction is Analyze.

The corpus sometimes stores a multi-line structural marker and boxed answer in one annotated unit.
For a unit beginning with "**Final Answer**", "Answer:", or a think-control tag, follow the corpus
precedence and choose Monitor when the unit is primarily a structural wrapper. Otherwise classify
the dominant function of the whole unit.

Return only the label name, with no Markdown, explanation, or additional text."""


@dataclass(frozen=True)
class FewShotSentence:
    example_id: str
    problem_statement: str
    previous_sentence: str
    sentence: str
    next_sentence: str
    label: str

    @property
    def question_id(self) -> str:
        return f"curated:{self.example_id}"

    @property
    def unit_id(self) -> int:
        return 0

    def metadata(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "source": "hand-curated contrastive example",
            "label": self.label,
        }


# Three deliberately unambiguous examples per class. The examples form contrastive
# neighborhoods around the recurrent Read/Analyze, Analyze/Verify,
# Analyze/Implement, Plan/Monitor, and Explore/Analyze errors.
CURATED_CONTRASTIVE_EXAMPLES = (
    FewShotSentence(
        "read-given-value",
        "A tank initially contains 40 liters of water. How much remains after 7 liters are removed?",
        "<START OF RESPONSE>",
        "The tank starts with 40 liters.",
        "We need to account for the 7 liters removed.",
        "Read",
    ),
    FewShotSentence(
        "read-goal",
        "If 3x + 2 = 20, what is x?",
        "The equation is 3x + 2 = 20.",
        "The question asks for the value of x.",
        "I should isolate x.",
        "Read",
    ),
    FewShotSentence(
        "read-choice",
        "Which value equals 2 + 3? A. 4 B. 5 C. 6 D. 7",
        "There are four choices.",
        "Choice B is 5.",
        "Now I will calculate 2 + 3.",
        "Read",
    ),
    FewShotSentence(
        "analyze-identify",
        "If 3x + 2 = 20, what is x?",
        "The problem gives 3x + 2 = 20.",
        "This is a linear equation in one variable.",
        "I will isolate x.",
        "Analyze",
    ),
    FewShotSentence(
        "analyze-relation",
        "A rectangle has length x + 2 and width x. Express its perimeter.",
        "A rectangle has two equal lengths and two equal widths.",
        "Therefore the perimeter can be represented as 2(x + 2) + 2x.",
        "Next I will simplify the expression.",
        "Analyze",
    ),
    FewShotSentence(
        "analyze-concept",
        "Two nonvertical lines have slopes 2 and m. When are they perpendicular?",
        "I need the perpendicular-slope condition.",
        "Perpendicular slopes are negative reciprocals.",
        "Thus m must be -1/2.",
        "Analyze",
    ),
    FewShotSentence(
        "plan-operation",
        "If 3x + 2 = 20, what is x?",
        "I need to isolate x.",
        "Next I will subtract 2 from both sides.",
        "That gives 3x = 18.",
        "Plan",
    ),
    FewShotSentence(
        "plan-enumerate",
        "How many two-digit numbers are divisible by 15?",
        "The valid numbers must lie between 10 and 99.",
        "I will list the multiples of 15 in that interval.",
        "They are 15, 30, 45, 60, 75, and 90.",
        "Plan",
    ),
    FewShotSentence(
        "plan-substitute",
        "Check whether x = 4 solves 2x + 1 = 9.",
        "The candidate value is x = 4.",
        "I will substitute 4 for x in the left-hand side.",
        "The left-hand side becomes 2(4) + 1.",
        "Plan",
    ),
    FewShotSentence(
        "implement-operation",
        "If 3x + 2 = 20, what is x?",
        "Next I will subtract 2 from both sides.",
        "Subtracting 2 from both sides gives 3x = 18.",
        "Now I will divide by 3.",
        "Implement",
    ),
    FewShotSentence(
        "implement-result",
        "What is 24 divided by 6?",
        "I will divide 24 by 6.",
        "24 / 6 = 4.",
        "The quotient is therefore 4.",
        "Implement",
    ),
    FewShotSentence(
        "implement-substitution",
        "Evaluate 2x + 1 when x = 4.",
        "Substitute 4 for x.",
        "2(4) + 1 = 9.",
        "So the expression has value 9.",
        "Implement",
    ),
    FewShotSentence(
        "explore-hypothesis",
        "Find a useful way to factor x^4 - 1.",
        "Direct expansion may be cumbersome.",
        "Maybe I can treat it as a difference of squares.",
        "That would give (x^2 - 1)(x^2 + 1).",
        "Explore",
    ),
    FewShotSentence(
        "explore-option",
        "Determine which graph matches the equation.",
        "The first graph does not seem to fit.",
        "Perhaps option C has the correct intercept.",
        "I need to test its slope as well.",
        "Explore",
    ),
    FewShotSentence(
        "explore-alternative",
        "Solve a geometry problem that may admit similar triangles.",
        "The angle information is not yet enough.",
        "Alternatively, there might be a useful cyclic quadrilateral.",
        "I will inspect the opposite angles.",
        "Explore",
    ),
    FewShotSentence(
        "verify-substitution",
        "Check whether x = 4 solves 2x + 1 = 9.",
        "Substitution gives 2(4) + 1 = 9.",
        "The two sides match, so x = 4 is valid.",
        "The check is complete.",
        "Verify",
    ),
    FewShotSentence(
        "verify-choice",
        "Which value equals 2 + 3? A. 4 B. 5 C. 6 D. 7",
        "The calculation produced 5.",
        "Since choice B is 5, choice B is correct.",
        "I can give the final answer.",
        "Verify",
    ),
    FewShotSentence(
        "verify-reasonableness",
        "A bag costs $8 and three bags are purchased.",
        "The computed total is $24.",
        "$24 is reasonable because it is three times $8.",
        "Therefore the calculation checks out.",
        "Verify",
    ),
    FewShotSentence(
        "monitor-hesitation",
        "If 3x + 2 = 20, what is x?",
        "I need to solve the equation.",
        "Hmm, let me think.",
        "I should isolate x.",
        "Monitor",
    ),
    FewShotSentence(
        "monitor-transition",
        "Find the area of a triangle with base 6 and height 4.",
        "The area formula is one half times base times height.",
        "Okay, moving on.",
        "I will substitute 6 and 4.",
        "Monitor",
    ),
    FewShotSentence(
        "monitor-structural",
        "Select the correct answer after solving the problem.",
        "The reasoning is complete.",
        "**Final Answer**\n\\boxed{B}</think>",
        "<END OF RESPONSE>",
        "Monitor",
    ),
)


def select_few_shot_sentences(
    documents: Sequence[EpisodeDocument],
    *,
    count: int = 21,
) -> tuple[FewShotSentence, ...]:
    """Return a balanced prefix of the audited, synthetic contrastive bank."""
    del documents  # The fixed bank cannot leak validation or test annotations.
    if count <= 0:
        raise ValueError("Few-shot example count must be positive")
    if count > len(CURATED_CONTRASTIVE_EXAMPLES):
        raise ValueError(
            f"At most {len(CURATED_CONTRASTIVE_EXAMPLES)} curated examples are available"
        )

    by_label = {
        label: [example for example in CURATED_CONTRASTIVE_EXAMPLES if example.label == label]
        for label in SENTENCE_LABELS
    }
    selected: list[FewShotSentence] = []
    round_index = 0
    while len(selected) < count:
        for label in SENTENCE_LABELS:
            if len(selected) >= count:
                break
            candidates = by_label[label]
            if round_index < len(candidates):
                selected.append(candidates[round_index])
        round_index += 1
    return tuple(selected)


def build_few_shot_instructions(examples: Sequence[FewShotSentence]) -> str:
    if not examples:
        raise ValueError("At least one few-shot example is required")

    sections = [SEED_INSTRUCTIONS, "\n\nCurated contrastive examples:"]
    for index, example in enumerate(examples, start=1):
        sections.append(
            "\n\n"
            f"Example {index}\n"
            f"Problem: {example.problem_statement}\n"
            f"Previous unit: {example.previous_sentence}\n"
            f"CURRENT UNIT: {example.sentence}\n"
            f"Next unit: {example.next_sentence}\n"
            f"Correct label: {example.label}"
        )
    sections.append("\n\nNow classify the supplied CURRENT UNIT. Return only its label.")
    return "".join(sections)
