"""Bind the native signature while invoking the installed masking operator."""
import inspect
from .contracts import require


class RouterBinding:
    def __init__(self, router, required=("router_logits",)):
        from moe_exp.expert_memory_study.loader import RouterMask
        self.installed = router._compute_routing
        owner = getattr(self.installed, "__self__", None)
        if isinstance(owner, RouterMask):
            require(type(owner) is RouterMask and owner.router is router,
                    "unsupported router mask ownership")
            require(getattr(owner.original, "__self__", None) is router,
                    "unsupported nested router wrapper")
            self.signature = inspect.signature(owner.original)
            require(owner.signature == self.signature, "preserved native signature drift")
        else:
            self.signature = inspect.signature(self.installed)
            require(not any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                            for p in self.signature.parameters.values()),
                    "unsupported variadic router wrapper")
        require(all(n in self.signature.parameters for n in required),
                "required native router parameters unavailable")

    def bind(self, *args, **kwargs):
        return self.signature.bind(*args, **kwargs)

    def invoke(self, bound):
        return self.installed(*bound.args, **bound.kwargs)

    def selector(self, hidden, logits, indices_type):
        bound = self.signature.bind(hidden_states=hidden, router_logits=logits,
                                    indices_type=indices_type)
        return self.invoke(bound)
