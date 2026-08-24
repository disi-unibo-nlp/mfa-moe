from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class StepLabels(BaseModel):
    # Gold first-error step from datasets like ProcessBench/PRM800K.
    # None when no ground-truth is available.
    first_error_step: int | None = None
    # Earliest heuristic-detected reasoning event (backtracking or contradiction).
    # This is where the model SIGNALS an issue, not necessarily where the error occurs.
    first_reasoning_event_step: int | None = None
    contradiction_steps: list[int] = Field(default_factory=list)
    backtracking_steps: list[int] = Field(default_factory=list)
    self_correction_steps: list[int] = Field(default_factory=list)
    # Exact lexical triggers used for automatic labels.  This is empty in
    # legacy result files and makes new heuristic annotations auditable.
    backtracking_evidence: list[str] = Field(default_factory=list)
    contradiction_evidence: list[str] = Field(default_factory=list)
    final_answer_reversal: bool = False


class ModelLogs(BaseModel):
    """Populated in Experiment 2+. Empty for Experiment 1."""

    hidden_states: str | None = None       # path to saved tensor file
    router_logits: str | None = None
    selected_experts: str | None = None    # path to saved tensor (num_layers, seq_len, top_k)
    expert_weights: str | None = None      # path to saved tensor (num_layers, seq_len, top_k)
    # Original model-layer indices represented by dimension 0 of the saved
    # tensors. None means the tensors contain every layer in natural order.
    layer_indices: list[int] | None = None
    attention_maps_optional: str | None = None


class TraceRecord(BaseModel):
    dataset: str
    problem_id: str
    source_problem_id: str | None = None
    sample_id: int = 0
    prompt: str
    # System prompt used at generation time (None = the default SYSTEM_PROMPT).
    # Downstream teacher-forced passes (Exp2/Exp3/event_routing) must rebuild the
    # prompt with this same value or the routing is conditioned on the wrong context.
    system_prompt: str | None = None
    # Exact OpenAI-compatible messages used by external generators such as
    # llama.cpp.  When present, teacher-forced extraction replays these instead
    # of reconstructing a legacy system/user prompt pair.
    generation_messages: list[dict[str, str]] | None = None
    gold_answer: str
    model_id: str
    model_answer: str
    is_correct: bool | None = None
    cot_text: str
    steps: list[str] = Field(default_factory=list)
    step_labels: StepLabels = Field(default_factory=StepLabels)
    model_logs: ModelLogs = Field(default_factory=ModelLogs)
    scoring_method: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Distinguishes direct math reasoning from meta-reasoning (e.g. ProcessBench)
    task_type: Literal["reasoning", "meta_reasoning"] = "reasoning"
