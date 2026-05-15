from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class StepLabels(BaseModel):
    first_error_step: Optional[int] = None
    contradiction_steps: list[int] = Field(default_factory=list)
    backtracking_steps: list[int] = Field(default_factory=list)
    self_correction_steps: list[int] = Field(default_factory=list)
    final_answer_reversal: bool = False


class ModelLogs(BaseModel):
    """Populated in Experiment 2+. Empty for Experiment 1."""

    hidden_states: Optional[str] = None       # path to saved tensor file
    router_logits: Optional[str] = None
    selected_experts: Optional[str] = None
    expert_weights: Optional[str] = None
    attention_maps_optional: Optional[str] = None


class TraceRecord(BaseModel):
    dataset: str
    problem_id: str
    prompt: str
    gold_answer: str
    model_id: str
    model_answer: str
    is_correct: Optional[bool] = None
    cot_text: str
    steps: list[str] = Field(default_factory=list)
    step_labels: StepLabels = Field(default_factory=StepLabels)
    model_logs: ModelLogs = Field(default_factory=ModelLogs)
