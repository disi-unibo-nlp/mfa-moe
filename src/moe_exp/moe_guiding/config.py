from __future__ import annotations

import math
from dataclasses import asdict, dataclass

CONDITIONS = ("baseline", "selected", "transfer")
ARCHITECTURE = "MoEGuidingMixtralForCausalLM"


def parse_layers(value: str) -> tuple[int, ...] | None:
    """Parse zero-based layer indices; None means every MoE layer."""
    if value.strip().lower() == "all":
        return None
    try:
        layers = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise ValueError("Layers must be 'all' or comma-separated integers, e.g. 0,12.") from error
    if not layers or any(layer < 0 for layer in layers) or len(set(layers)) != len(layers):
        raise ValueError("Layers must be distinct nonnegative integers.")
    return tuple(sorted(layers))


@dataclass(frozen=True)
class RoutingConfig:
    condition: str = "selected"
    threshold: float = 0.4
    layers: tuple[int, ...] | None = (12,)

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"Condition must be one of {CONDITIONS}.")
        if not math.isfinite(self.threshold) or not -1 <= self.threshold <= 1:
            raise ValueError("The margin threshold must be finite and between -1 and 1.")
        if self.layers is not None:
            layers = tuple(self.layers)
            if (
                not layers
                or any(type(layer) is not int or layer < 0 for layer in layers)
                or len(set(layers)) != len(layers)
            ):
                raise ValueError("Layers must be distinct nonnegative integers, or null for all.")
            object.__setattr__(self, "layers", tuple(sorted(layers)))

    def selected_layers(self, num_layers: int) -> tuple[int, ...]:
        layers = tuple(range(num_layers)) if self.layers is None else self.layers
        if not layers or max(layers) >= num_layers:
            raise ValueError(f"Selected layers {layers} are outside a {num_layers}-layer model.")
        return layers

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["layers"] = None if self.layers is None else list(self.layers)
        return payload
