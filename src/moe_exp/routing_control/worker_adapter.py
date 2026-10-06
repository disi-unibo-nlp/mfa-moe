"""Ordered-action adapter over the sealed vLLM worker; GPU qualification is separate.

No torch/vLLM imports occur until install() is called on an allocated GPU worker.
The base worker owns token-row mapping, closure detection, recompute and native top-k.
This adapter changes only per-row policy indices and masks before buffer upload.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Mapping

import numpy as np

from .design import digest

VERSION = 'ordered-worker-v2'
FIELDS = frozenset({'action_policy_names', 'slots', 'horizon', 'sha256'})


@dataclass(frozen=True)
class OrderedPulse:
    names: tuple[str, ...]
    indices: tuple[int, ...]
    slots: tuple[int, ...]
    sha256: str

    @classmethod
    def load(cls, value: Mapping, table, first_policy: str):
        if set(value) != FIELDS or value['horizon'] != 1024:
            raise ValueError('unknown ordered-action fields or horizon')
        content={k:v for k,v in value.items() if k!='sha256'}
        if digest(content) != value['sha256']:
            raise ValueError('ordered-action digest mismatch')
        names,slots=tuple(value['action_policy_names']),tuple(value['slots'])
        if not 1<=len(names)<=2 or slots!=(0,512)[:len(names)] or names[0]!=first_policy:
            raise ValueError('freeze one or two ordered actions at positions 0 and 512')
        if any(type(s) is not int for s in slots) or any(not isinstance(n,str) for n in names):
            raise ValueError('invalid ordered pulse metadata')
        indices=[]
        for name in names:
            index=table.index_of(name); policy=table.policies[index]
            op=policy.operator
            targets=policy.targets
            if op.kind!='bias' or op.sign!=1 or op.magnitude not in (.5,1.) or policy.schedule.kind!='always':
                raise ValueError('ordered actions require positive .5/1 bias and closure-aware always schedule')
            if targets is None:
                raise ValueError('ordered action has no expert set')
            layers=[layer for layer,_ in targets.experts]
            if not 1<=len(layers)<=4 or layers!=list(range(min(layers),max(layers)+1)):
                raise ValueError('ordered action requires at most four adjacent layers')
            if any(not 1<=len(ids)<=2 for _,ids in targets.experts):
                raise ValueError('ordered action requires at most two experts per layer')
            indices.append(index)
        return cls(names,tuple(indices),slots,value['sha256'])

    def rows(self, positions, native_active):
        positions=np.asarray(positions)
        active=np.asarray(native_active,bool)
        if positions.ndim!=1 or active.shape!=positions.shape or positions.dtype.kind not in 'iu':
            raise ValueError('invalid branch position/mask arrays')
        mask=np.zeros(len(positions),np.bool_)
        indices=np.zeros(len(positions),np.int32)
        for slot,index in zip(self.slots,self.indices):
            take=active & (positions>=slot) & (positions<slot+256)
            mask[take]=True;indices[take]=index
        return mask,indices


def install():
    """Extend this process's sealed hook objects; never change files or installed vLLM.

    Call before model loading, using this module's WorkerExtension FQCN. Qualification
    must compare inactive native rows, both ranks, batching, pulse order and recovery.
    """
    from moe_steer import vllm_ext as E
    if getattr(E.install_steering,'routing_control_version',None)==VERSION:
        return E
    original_install=E.install_steering
    original_accumulate=E._accumulate
    dose_buffers={}

    def accumulate(tele,slot_idx,h,act,pidx,tmember,native_w,native_i,actual_w,actual_i):
        original_accumulate(tele,slot_idx,h,act,pidx,tmember,native_w,native_i,actual_w,actual_i)
        buffers=dose_buffers.get(id(tele))
        if buffers is None:return
        by_policy,inactive,n_policy=buffers
        original_accumulate(by_policy,slot_idx*n_policy+pidx,h,act,pidx,tmember,
            native_w,native_i,actual_w,actual_i)
        # Compare actual and native routing for every inactive row at the same hidden
        # state. Fixed-size device accumulation remains safe under CUDA graphs.
        import torch
        off=~act
        values=torch.stack([off.float(),(off & (actual_i!=native_i).any(dim=1)).float(),
            (off & (actual_w!=native_w).any(dim=1)).float()],dim=1)
        inactive[:,h].index_add_(0,slot_idx.long(),values)
    E._accumulate=accumulate

    def extended_install(*args,**kwargs):
        ctl=original_install(*args,**kwargs)
        import torch
        n_policy=len(ctl.table.policies)
        with torch.inference_mode(False):
            ctl.ordered_dose=torch.zeros(((ctl.n_slots+1)*n_policy,len(ctl.layers),len(E.TELE_FIELDS)),
                dtype=torch.float32,device=ctl.tele.device)
            ctl.ordered_inactive=torch.zeros((ctl.n_slots+1,len(ctl.layers),3),
                dtype=torch.float32,device=ctl.tele.device)
        dose_buffers[id(ctl.tele)]=(ctl.ordered_dose,ctl.ordered_inactive,n_policy)
        original_new,original_plan,original_record,original_read=ctl._new_state,ctl._plan_rows,ctl._record,ctl._read_slots

        def new_state(self,rid,req,start):
            st=original_new(rid,req,start)
            extra=getattr(getattr(req,'sampling_params',None),'extra_args',None) or {}
            meta=(extra.get('steer') or {}).get('meta') or {}
            ordered=meta.get('routing_control')
            st.ordered=None if ordered is None else OrderedPulse.load(ordered,self.table,st.policy_name)
            st.ordered_active_rows={name:0 for name in (st.ordered.names if st.ordered else ())}
            st.ordered_segments={name:[] for name in (st.ordered.names if st.ordered else ())}
            st.ordered_dose_acc=np.zeros((n_policy,len(self.layers),len(E.TELE_FIELDS)),np.float64)
            st.ordered_inactive_acc=np.zeros((len(self.layers),3),np.float64)
            # Native and sham requests also receive a telemetry slot so neighboring
            # inactive rows can be checked, without editing their routing mask.
            st.needs_slot=True
            return st

        def plan_rows(self,st,req,a,b,s,mask,pidx,slot):
            old_hwm=st.hwm;old_pulse=st.cpu_pulse_rows;old_active=st.cpu_active_rows
            old_policy=self.policy_active_rows[st.policy_name]
            original_plan(st,req,a,b,s,mask,pidx,slot)
            if st.ordered is None:return
            # Base mapping: row at prompt_len-1 predicts branch output zero.
            positions=np.arange(s,s+b-a,dtype=np.int64)-st.prompt_len+1
            new_mask,new_indices=st.ordered.rows(positions,mask[a:b])
            mask[a:b]=new_mask;pidx[a:b]=new_indices
            fresh=np.arange(s,s+b-a)>=old_hwm
            count=int(np.count_nonzero(new_mask & fresh))
            st.cpu_pulse_rows=old_pulse+count;st.cpu_active_rows=old_active+count
            self.policy_active_rows[st.policy_name]=old_policy+count
            for name,index in dict(zip(st.ordered.names,st.ordered.indices)).items():
                used=positions[new_mask & fresh & (new_indices==index)]
                st.ordered_active_rows[name]+=len(used)
                for position in used.tolist():
                    segments=st.ordered_segments[name]
                    if segments and segments[-1][1]==position:segments[-1][1]=position+1
                    else:segments.append([position,position+1])

        def record(self,st,*args,**kwargs):
            value=original_record(st,*args,**kwargs)
            value['inactive_native_checks']={str(layer):dict(zip(
                ('rows','expert_identity_mismatches','weight_mismatches'),
                [int(round(x)) for x in st.ordered_inactive_acc[h]])) for h,layer in enumerate(self.layers)}
            if st.ordered is not None:
                value.update(ordered_worker_version=VERSION,ordered_template_sha256=st.ordered.sha256,
                    ordered_action_rows=st.ordered_active_rows,
                    ordered_segments=st.ordered_segments,
                    ordered_action_dose={name:{str(layer):dict(zip(E.TELE_FIELDS,
                        [float(x) for x in st.ordered_dose_acc[index,h]])) for h,layer in enumerate(self.layers)}
                        for name,index in dict(zip(st.ordered.names,st.ordered.indices)).items()},
                    routing_dose_scope='per-action/per-layer sparse top-k gate L1 displacement, turnover and target hits; TV=L1/2')
            return value

        def read_slots(self,states):
            slotted=[st for st in states if st.slot is not None]
            if slotted:
                with E._sync_allowed():
                    slots=torch.tensor([st.slot for st in slotted],dtype=torch.long,device=self.tele.device)
                    indices=(slots[:,None]*n_policy+torch.arange(n_policy,device=slots.device)[None,:]).flatten()
                    dose=self.ordered_dose.index_select(0,indices).cpu().numpy().reshape(
                        len(slotted),n_policy,len(self.layers),len(E.TELE_FIELDS))
                    inactive=self.ordered_inactive.index_select(0,slots).cpu().numpy()
                    self.ordered_dose.index_fill_(0,indices,0);self.ordered_inactive.index_fill_(0,slots,0)
                for i,st in enumerate(slotted):
                    st.ordered_dose_acc+=dose[i];st.ordered_inactive_acc+=inactive[i]
            return original_read(states)
        ctl._new_state=MethodType(new_state,ctl)
        ctl._plan_rows=MethodType(plan_rows,ctl)
        ctl._record=MethodType(record,ctl)
        ctl._read_slots=MethodType(read_slots,ctl)
        return ctl
    extended_install.routing_control_version=VERSION
    E.install_steering=extended_install
    return E


# This file is imported as a worker extension only on allocated compute nodes.
# Importing the CPU row planner does not install hooks.
def worker_extension_class():
    E=install()
    class OrderedWorkerExtension(E.SteerWorkerExtension):
        def steer_ordered_status(self):
            return {'version':VERSION,'status':self.steer_status(),
                    'qualification':'must be established by separate saved GPU results'}
    return OrderedWorkerExtension
