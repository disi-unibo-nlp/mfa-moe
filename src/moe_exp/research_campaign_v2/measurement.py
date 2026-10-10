"""One saved-sequence pass collects output-aware saliency and native dose inputs."""
from __future__ import annotations

import inspect
from .contracts import digest, require
from .binding import RouterBinding
from .lifecycle import restore, guard, cleanup
from moe_exp.verification_study.adapters import PredictionClock, discover_routers


class ForwardObserver:
    def __init__(self, model, contract, prompt, completion, closure, reasoning_positions, bank_positions):
        import torch
        self.clock = PredictionClock(prompt, completion, closure)
        self.prompt_length = len(prompt)
        self.reasoning = set(reasoning_positions)
        self.bank = set(bank_positions)
        require(len(self.bank) <= 64, "calibration position limit exceeded")
        self.originals, self.handle, self.data = [], None, {}
        routers = discover_routers(model)
        require(set(routers) == set(contract["layers"]), "measurement routed layer map drift")
        try:
            for l, (name, router, monolithic) in sorted(routers.items()):
                require(not monolithic and router.top_k == contract["native_k"], "output-aware observer requires modular native kernel")
                runner = model.get_submodule(name)
                binding = RouterBinding(router, ("hidden_states", "router_logits", "indices_type"))
                original = binding.installed
                self.originals.append((router, original))
                E = contract["experts"]
                self.data[str(l)] = {"weighted_norm_sum": torch.zeros(E, dtype=torch.float64),
                    "selected_count": torch.zeros(E, dtype=torch.int64), "positions": [], "scores": [],
                    "native_ids": [], "native_weights": [], "reconstruction_max_abs": 0.,"reap_supported":True}
                def compute(*args, _l=l, _runner=runner, _binding=binding, **kwargs):
                    bound = _binding.bind(*args, **kwargs)
                    x, logits = bound.arguments["hidden_states"], bound.arguments["router_logits"]
                    nw, ni = _binding.invoke(bound)
                    contexts = self.clock.contexts
                    require(len(contexts) == len(x), "measurement token attribution mismatch")
                    # output_position is the NEXT completion index. Saliency is
                    # attributed to the actual saved input token, one position back.
                    positions = [c["output_position"] - 1 for c in contexts]
                    chosen = [j for j, p in enumerate(positions) if p in self.reasoning]
                    cell = self.data[str(_l)]
                    cell["router_logits_dtype"]=str(logits.dtype)
                    cell["hidden_states_dtype"]=str(x.dtype)
                    if chosen:
                        ix = torch.tensor(chosen, device=x.device)
                        xx, ids, native_weights = x[ix].clone(), ni[ix], nw[ix]
                        gate = torch.softmax(logits[ix].float(), dim=-1).gather(1, ids.long())
                        gate /= gate.sum(-1, keepdim=True).clamp_min(1e-30)
                        combined = _runner.routed_experts.forward_modular(xx.clone(), native_weights, ids)
                        if not _runner._fused_output_is_reduced and _runner.moe_config.tp_size > 1:
                            from vllm.distributed import tensor_model_parallel_all_reduce
                            combined = tensor_model_parallel_all_reduce(combined)
                        combined = combined[..., :xx.shape[-1]]
                        reconstruction = torch.zeros_like(combined)
                        for slot in range(ids.shape[1]):
                            unit = torch.zeros_like(native_weights)
                            unit[:, slot] = 1.
                            output = _runner.routed_experts.forward_modular(xx.clone(), unit, ids)
                            require(isinstance(output, torch.Tensor), "expert output is not a native tensor")
                            # vLLM kernels can reduce inside the kernel; never sum twice.
                            if not _runner._fused_output_is_reduced and _runner.moe_config.tp_size > 1:
                                from vllm.distributed import tensor_model_parallel_all_reduce
                                output = tensor_model_parallel_all_reduce(output)
                            output = output[..., :xx.shape[-1]]
                            norm = output.float().norm(dim=-1)
                            identity = ids[:, slot].long()
                            cell["weighted_norm_sum"].scatter_add_(0, identity.cpu(), (norm * gate[:, slot]).double().cpu())
                            cell["selected_count"].scatter_add_(0, identity.cpu(), torch.ones(len(identity), dtype=torch.int64))
                            reconstruction += output * native_weights[:, slot:slot+1]
                        error = float((reconstruction.float() - combined.float()).abs().max())
                        cell["reconstruction_max_abs"] = max(cell["reconstruction_max_abs"], error)
                        cell["reap_supported"] &= bool(torch.allclose(reconstruction.float(), combined.float(),atol=.02,rtol=.02))
                        # A native operator whose input-side weighting prevents
                        # output reconstruction leaves REAP explicitly unsupported.
                        # Its saved native router bank still supports the other
                        # pruning methods and causal calibration; no rule changes.
                    selected = [j for j, p in enumerate(positions) if p in self.bank]
                    if selected:
                        ix = torch.tensor(selected, device=logits.device)
                        cell["positions"] += [positions[j] for j in selected]
                        cell["scores"] += logits[ix].detach().float().cpu().tolist()
                        cell["native_ids"] += ni[ix].detach().cpu().tolist()
                        cell["native_weights"] += nw[ix].detach().float().cpu().tolist()
                    return nw, ni
                router._compute_routing = guard(self, compute, "measurement_router")
            def before(module, args, kwargs):
                bound = inspect.signature(module.forward).bind(*args, **kwargs)
                self.clock.observe(bound.arguments["input_ids"].detach().cpu().tolist(),
                                   bound.arguments["positions"].detach().cpu().tolist())
            self.handle = model.register_forward_pre_hook(guard(self, before, "prediction_clock"), with_kwargs=True)
        except BaseException:
            self.restore()
            raise

    def restore(self):
        return restore(self)

    def finish(self):
        self.restore()
        result = {}
        for l, cell in self.data.items():
            require(len(cell["positions"]) == len(set(cell["positions"])), "duplicate native calibration position")
            require(set(cell["positions"]) == self.bank, "missing native calibration position")
            result[l] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in cell.items()}
        return {"layers": result, "measurement_sha256": digest(result), "natural_generations": 0,
                "reap_definition": "mean over selected tokens of norm(expert output) times selected-normalized softmax gate weight",
                "source": "CerebrasResearch/reap src/reap/pruning_metrics.py; normalization correction 2026-03-11",
                "reference_commit":"1970473c51ca3caeb98c10392f15b3a08a672974"}


class MeasurementWorker:
    def measurement_begin(self, **kwargs):
        require(getattr(self, "_measurement", None) is None, "measurement already active")
        self._measurement = ForwardObserver(self.model_runner.get_model(), **kwargs)
        return {"status": "attached"}

    def measurement_abort(self):
        observer=getattr(self,"_measurement",None)
        if observer is None:return {"status":"detached","failure":None,"cleanup_errors":[]}
        failure=getattr(observer,"failure",None)
        marker=RuntimeError("original execution failed")
        cleanup([("detach_measurement",observer.restore)],marker)
        self._measurement=None
        return {"status":"detached","failure":failure,"cleanup_errors":marker.campaign_cleanup_errors,
            "validation_skipped_after_original_failure":True}

    def measurement_end(self):
        try:
            return self._measurement.finish()
        finally:
            self._measurement = None

    def calibrate(self, contract, banks, target, control, fractions, sign):
        """Native selector at the same saved states; a common attainable dose."""
        import torch
        routers = discover_routers(self.model_runner.get_model())
        layer = target["layer"]
        router = routers[layer][1]
        binding = RouterBinding(router, ("hidden_states", "router_logits", "indices_type"))
        function = binding.selector
        require(sign in (-1, 1) and banks, "invalid calibration population")
        from .design import family_weights
        bank_weights=family_weights(banks)
        logits_types={b["router_logits_dtype"] for b in banks}
        hidden_types={b["hidden_states_dtype"] for b in banks}
        require(len(logits_types)==len(hidden_types)==1,"native calibration dtype changes across saved inputs")
        logits_dtype=getattr(torch,next(iter(logits_types)).removeprefix("torch."))
        hidden_dtype=getattr(torch,next(iter(hidden_types)).removeprefix("torch."))
        saved_scores=torch.cat([torch.tensor(b["scores"],dtype=torch.float32) for b in banks])
        saved_ids=torch.cat([torch.tensor(b["native_ids"],dtype=torch.int32) for b in banks])
        saved_weights=torch.cat([torch.tensor(b["native_weights"],dtype=torch.float32) for b in banks])
        row_weights=torch.cat([torch.full((len(b["scores"]),),float(w)/len(b["scores"]),dtype=torch.float64)
            for w,b in zip(bank_weights,banks)])
        width=self.model_runner.get_model().get_submodule(routers[layer][0]).moe_config.hidden_dim
        # Qualification prices the complete selector role. Vectorize saved rows
        # in bounded chunks instead of allocating each family's CUDA tensors for
        # every bisection step; the estimand retains exact family weights.
        chunks=[]
        for start in range(0,len(saved_scores),8192):
            logits=saved_scores[start:start+8192].to(device="cuda",dtype=logits_dtype)
            hidden=torch.zeros(len(logits),width,device="cuda",dtype=hidden_dtype)
            native_w,native_i=function(hidden,logits,torch.int32)
            require(torch.equal(native_i.cpu(),saved_ids[start:start+8192]) and
                torch.allclose(native_w.float().cpu(),saved_weights[start:start+8192],atol=1e-6,rtol=1e-5),"saved native selector fidelity fails")
            native_vector=torch.zeros_like(logits,dtype=torch.float32).scatter_(1,native_i.long(),native_w.float())
            chunks.append((logits,native_vector,row_weights[start:start+8192].cuda()))
            del hidden
        def evaluate(ids, offset):
            value=0.
            for logits,a,w in chunks:
                hidden=torch.zeros(len(logits),width,device="cuda",dtype=hidden_dtype)
                changed = logits.clone()
                changed[:, ids] += sign * offset
                action_w, action_i = function(hidden, changed, torch.int32)
                b = torch.zeros_like(logits,dtype=torch.float32).scatter_(1, action_i.long(), action_w.float())
                value+=float((((a-b).abs().sum(-1)/a.abs().sum(-1).clamp_min(1e-12))*w).sum())
            return value
        # The range is a recorded native-score scale, not an arbitrary fixed dose.
        scales = [float(torch.tensor(b["scores"]).std(-1).median()) for b in banks]
        scale = max(float(torch.tensor(scales).median()), 1e-6)
        offsets = [scale * x for x in (0., .125, .25, .5, 1., 2., 4., 8.)]
        curves = [[evaluate(ids, o) for o in offsets] for ids in (target["experts"], control)]
        attainable = min(max(curve) for curve in curves)
        if attainable <= 0:
            return {"status": "insufficient_support", "curves": curves, "offsets": offsets}
        selected = []
        for curve in curves:
            doses = []
            for fraction in fractions:
                goal = fraction * attainable
                candidates = [i for i,v in enumerate(curve) if v >= goal]
                if not candidates:
                    return {"status": "insufficient_support", "curves": curves, "offsets": offsets}
                lo, hi = (offsets[min(candidates)-1] if min(candidates) else 0.), offsets[min(candidates)]
                ids = target["experts"] if curve is curves[0] else control
                for _ in range(16):
                    mid = (lo + hi) / 2
                    if evaluate(ids, mid) >= goal:
                        hi = mid
                    else:
                        lo = mid
                doses.append(sign * hi)
            selected.append(doses)
        return {"status": "complete", "target_doses": selected[0], "control_doses": selected[1],
                "common_attainable_displacement": attainable, "fractions": fractions,
                "offset_range": offsets, "target_curve": curves[0], "control_curve": curves[1],
                "families": len({b["family"] for b in banks}), "same_state_native_selector_fidelity": True}
