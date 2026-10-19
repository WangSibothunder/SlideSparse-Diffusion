"""Preallocated active-bank storage. Any mode transfer runs inside timed select."""
import torch

KEYS=('packed','metadata','scales','bias')


def load_resident_banks(model):
    """Resident packed modes with the library's existing recommended dispatch."""
    from pathlib import Path
    from .io import sha
    from .library import profile_for,prepare_selected
    result={};configured=set()
    for unit in model.units:
        uid=unit['unit'];bias=torch.cat([model.native.get_submodule(n).bias for n in unit['targets']]).float();result[uid]={}
        for mode in [4,6,8]:
            path=Path(model.config['bank_directory'])/str(mode)/(str(uid)+'.pt');blob=torch.load(path,map_location='cpu',weights_only=True)
            if blob['unit']!=uid or blob['group']!=mode:raise ValueError('prepared bank belongs to another unit')
            cfg,front=profile_for(model.library,mode,unit,recommended=model.config.get('projection_execution')=='recommended')
            op=model.library.PackedLinear(blob['compact'].cuda(),blob['metadata'].cuda(),blob['scales'].cuda(),bias,
                unit['k'],unit['n'],mode,config=cfg,frontend=front,identity=dict(qweight_sha=blob['q_sha'],packing_payload_source=str(path),packing_payload_sha=sha(path)))
            prepare_selected(model.library,op,unit['m'],configured);result[uid][mode]=op
    return result


class ResidentBankSlot:
    def __init__(self,payloads,device):
        self.device=torch.device(device);self.host={};self.current_mode=None
        if not payloads or any(type(g)is not int or g not in (4,6,8) for g in payloads):raise ValueError('frozen sparse modes required')
        for mode,values in payloads.items():
            if set(values)!=set(KEYS):raise ValueError('complete packed payload required')
            rows={}
            for key,value in values.items():
                if value.device.type!='cpu' or not value.is_contiguous():raise ValueError('offline contiguous CPU payload required')
                if key=='packed' and value.dtype!=torch.int8:raise ValueError('packed INT8 payload required')
                if key=='metadata' and value.dtype!=torch.int32:raise ValueError('metadata INT32 required')
                if key in ('scales','bias') and (value.dtype!=torch.float32 or not torch.isfinite(value).all()):raise ValueError('finite FP32 scale/bias required')
                rows[key]=value.pin_memory() if self.device.type=='cuda' and not value.is_pinned() else value
            self.host[mode]=rows
        self.buffers={};self.views={};self.transfer_bytes=0;self.transfers=0
        for key in KEYS:
            values=[row[key] for row in self.host.values()]
            if len({v.dtype for v in values})!=1:raise ValueError('mode buffer dtype differs')
            self.buffers[key]=torch.empty(max(v.numel() for v in values),dtype=values[0].dtype,device=self.device)
        for mode,values in self.host.items():
            self.views[mode]={key:self.buffers[key][:value.numel()].view(value.shape) for key,value in values.items()}
        self.identity=dict(frozen_payload_shapes={str(g):{k:list(v.shape) for k,v in row.items()} for g,row in self.host.items()},
            GPU_persistent_bytes=sum(v.numel()*v.element_size() for v in self.buffers.values()),
            host_pinned_when_CUDA=self.device.type=='cuda',one_active_mode_per_unit=True,
            all_transfers_require_timed_selector=True,forward_disk_IO_or_packing=False)

    def activate(self,mode,include_parameters=True):
        if type(mode)is not int or mode not in self.host:raise ValueError('unknown prepared bank mode')
        if mode==self.current_mode:return 0
        copied=0
        for key in KEYS if include_parameters else ('packed','metadata'):
            source=self.host[mode][key];self.views[mode][key].copy_(source,non_blocking=self.device.type=='cuda')
            copied+=source.numel()*source.element_size()
        self.current_mode=mode;self.transfer_bytes+=copied;self.transfers+=1
        return copied

def arena_bytes(payloads):
    if not payloads or any(set(row)!=set(KEYS) for row in payloads.values()):raise ValueError('all four active-slot tensors required')
    return sum(max(row[key].numel()*row[key].element_size() for row in payloads.values()) for key in KEYS)

def initial_residency_budget(slot_bytes,free_bytes,dense_peak_reserved,current_reserved,transient_lower_bound,safety_bytes=4*2**30):
    values=(slot_bytes,free_bytes,dense_peak_reserved,current_reserved,transient_lower_bound,safety_bytes)
    if any(type(v)is not int or v<0 for v in values):raise ValueError('invalid measured residency components')
    reserve=max(0,dense_peak_reserved-current_reserved)
    required=slot_bytes+reserve+transient_lower_bound+safety_bytes
    if required>free_bytes:raise ValueError('active slots plus measured Dense workspace/margin exceed instantaneous free memory')
    return dict(required_incremental_bytes=required,free_bytes=free_bytes,slot_tensor_bytes=slot_bytes,
        measured_Dense_peak_extra_reserve_bytes=reserve,safety_bytes=safety_bytes,
        Sparse_peak_measured=False,full_Sparse_memory_acceptance=False,required_H2D_must_be_timed=True)



def load_flux_banks(model):
    """Existing packed banks and recorded configurations; no repacking/search."""
    from pathlib import Path
    from .io import read,sha
    from .resources import host_available
    config=model.config;plan=read(config['slot_plan']);packed=read(config['packed_manifest']);foundation=read(config['memory_plan'])
    entries={(row['unit'],row['group']):row for row in packed['entries']}
    cases={(row['unit'],row['group']):row for row in plan['deployments']};ops={};configured=set()
    if host_available()<plan['all_mode_host_payload_bytes']*2+32*2**30:raise RuntimeError('insufficient host memory for pinned banks')
    dense_peak=int(plan['authenticated_Dense_workspace_record135']['maximum_peak_reserved_bytes'])
    initial_residency_budget(int(plan['slot_tensor_bytes']),int(torch.cuda.mem_get_info()[0]),dense_peak,int(torch.cuda.memory_reserved()),int(foundation['single_projection_transient_lower_bound_bytes']))
    for unit in model.units:
        uid=unit['unit'];payloads={};attributes={}
        for mode in [4,6,8]:
            entry=entries[uid,mode]
            if sha(entry['path'])!=entry['sha']:raise ValueError('packed bank changed')
            blob=torch.load(entry['path'],map_location='cpu',weights_only=True)
            if blob['unit']!=unit or blob['members']!=unit['targets'] or blob['group']!=mode:raise ValueError('bank belongs to another projection')
            p=blob['payload'];payloads[mode]={key:p[key].contiguous() for key in KEYS};attributes[mode]=dict(strategy=p['strategy'],identity=p['identity'])
        slot=ResidentBankSlot(payloads,'cuda');model.bank_slots[uid]=slot;ops[uid]={}
        for mode in [4,6,8]:
            case=cases[uid,mode];cfg=case['config'];front=case['frontend']
            if front.startswith(('graph','direct2_')):raise ValueError('this slot layout does not support prepared graph frontends')
            op=model.library.PackedLinear(**slot.views[mode],k=unit['k'],n=unit['n'],mode=mode,config=cfg,frontend=front,**attributes[mode])
            resource=(mode,unit['k'],unit['n'],cfg,front,bool(unit['layout']['slice_start']))
            if resource not in configured:torch.ops.sdacc_dualscale_sm80.configure(op.packed,cfg,resource[-1]);configured.add(resource)
            ops[uid][mode]=op
    return ops


def coalesce_parameters(model,ops):
    """One stable scale/bias arena for the memory-bounded FLUX bank slots.

    Weight/metadata slots retain their existing shapes and modes. The caller
    fills the arena inside timed selection; kernels keep the same tensor views.
    """
    ranges={};size=0
    for uid,slot in model.bank_slots.items():
        n=model.units[uid]['n']
        if any(row[key].numel()!=n for row in slot.host.values() for key in ('scales','bias')):
            raise ValueError('coalesced parameters require the original output channels')
        ranges[uid]=(size,n);size+=2*n
    model.coefficient_arena=torch.empty(size,dtype=torch.float32,device='cuda')
    model.coefficient_ranges=ranges;model.raw_parameters_cache={}
    for uid,slot in model.bank_slots.items():
        offset,n=ranges[uid]
        for key,start in [('scales',offset),('bias',offset+n)]:
            slot.buffers[key]=model.coefficient_arena[start:start+n]
            for mode,row in slot.host.items():
                view=slot.buffers[key].view(row[key].shape);slot.views[mode][key]=view
                setattr(ops[uid][mode],key,view)
