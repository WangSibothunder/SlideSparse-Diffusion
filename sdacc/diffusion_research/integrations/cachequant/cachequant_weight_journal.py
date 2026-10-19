"""Own native unpruned weight grids; immutable per-module CPU prefix.

No activation grid, Hessian, OBS compensation, integer kernel, fitted affine,
caller cost or model quality is admitted. Values remain on the original
native dtype/grid, including Conv before any flatten/unfold conversion.
"""
import copy, os
from pathlib import Path
import torch
from .evidence import atomic_json, load, sha
from .storage import atomic_save
from .storage import tensor_sha


def validate_weight_grid(grid, declaration):
    required={'codes','delta','zero','native_dtype','original_shape'}
    if not required<=set(grid):raise ValueError('incomplete native weight grid')
    q,s,z=(grid[k] for k in ('codes','delta','zero'))
    shape=tuple(declaration['shape']);dtype=declaration['dtype']
    if len(shape) not in (2,4) or not all(type(n)==int and n>0 for n in shape):
        raise ValueError('actual original Linear or Conv shape required')
    if (q.dtype!=torch.uint8 or tuple(q.shape)!=shape or grid['original_shape']!=list(shape)
        or grid['native_dtype']!=dtype or str(s.dtype)!=dtype or s.dtype!=z.dtype
        or tuple(s.shape)!=(shape[0],*([1]*(len(shape)-1))) or tuple(z.shape)!=tuple(s.shape)):
        raise ValueError('native original dtype/Conv rank/per-output grid changed')
    if any(v.device.type!='cpu' or v.requires_grad for v in (q,s,z)):
        raise ValueError('offline CPU frozen native grid only')
    if (not torch.isfinite(s).all() or not (s>0).all() or not torch.isfinite(z).all()
        or not torch.equal(z,z.round()) or not ((z>=0)&(z<=255)).all()):
        raise ValueError('unchanged asymmetric uint8 format cannot represent this native grid')
    return dict(codes_sha=tensor_sha(q),delta_sha=tensor_sha(s),zero_sha=tensor_sha(z),
        actual_original_shape=list(shape),native_dtype=dtype,bytes=sum(v.numel()*v.element_size() for v in (q,s,z)),
        own_weight_grid_only=True,actual_INT8_OBS_activation_numeric_cost_quality_pending=True)


class WeightGridJournal:
    def __init__(self,directory,identity,sites,validator=validate_weight_grid):
        self.root=Path(directory).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.identity=copy.deepcopy(identity);self.sites=copy.deepcopy(sites);self.validator=validator
        if not sites or len(set(sites))!=len(sites):raise ValueError('unique declared original weight modules required')
        self.order=tuple(sorted(sites))
        if identity.get('module_order')!=list(self.order):raise ValueError('explicit actual module order required')
        self.index_path=self.root/'index.json'

    def _path(self,name):
        p=(self.root/name).resolve()
        if not p.is_relative_to(self.root):raise ValueError('weight receipt escapes own cloud directory')
        return p

    def _entry(self,site,entry):
        declaration=self.sites[site]
        if (entry['site']!=site or entry['weight_sha']!=declaration['weight_sha']
            or entry['bias_sha']!=declaration['bias_sha'] or entry['identity']!=self.identity):
            raise ValueError('mixed original weight/quantizer/module source')
        path=self._path(entry['path'])
        if sha(path)!=entry['sha']:raise ValueError('changed native grid payload')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if (payload['identity']!=self.identity or payload['site']!=site
            or payload['pid']!=entry['pid'] or payload['runtime_source_sha']!=entry['runtime_source_sha']
            or payload['declaration']!=declaration or payload['native_oracle_bitwise_checked'] is not True):
            raise ValueError('unproven or mixed native weight grid producer')
        if self.validator(payload['grid'],declaration)!=entry['format']:
            raise ValueError('actual native grid format changed')
        return payload['grid']

    def index(self,check_payloads=True):
        result=load(self.index_path) if self.index_path.exists() else dict(identity=self.identity,order=list(self.order),entries=[])
        if result['identity']!=self.identity or result['order']!=list(self.order) or len(result['entries'])>len(self.order):
            raise ValueError('mixed or extra native weight prefix')
        for site,entry in zip(self.order,result['entries']):
            if (entry['site']!=site or entry['identity']!=self.identity
                or entry['weight_sha']!=self.sites[site]['weight_sha'] or entry['bias_sha']!=self.sites[site]['bias_sha']):
                raise ValueError('mixed/reordered original weight source')
            if check_payloads:self._entry(site,entry)
        return result

    def recover(self):
        result=self.index()
        # Exact completed module may be adopted after interrupted index rename.
        # A partial TMP payload is never a successful module.
        while len(result['entries'])<len(self.order):
            site=self.order[len(result['entries'])];tag='module_'+str(len(result['entries'])).zfill(4)
            path=self.root/'grids'/(tag+'.pt');receipt=self.root/'receipts'/(tag+'.json')
            if not receipt.exists():
                if path.exists():raise ValueError('grid payload without committed receipt requires independent audit')
                break
            entry=load(receipt);self._entry(site,entry);result['entries'].append(entry)
        if self.index_path.exists() and result!=load(self.index_path):atomic_json(self.index_path,result)
        elif result['entries'] and not self.index_path.exists():atomic_json(self.index_path,result)
        return result

    def commit(self,site,grid,runtime_source_sha):
        # Existing grids were checked at recover()/their original commit;
        # check all payloads again at complete(), avoiding O(N^2) large I/O.
        index=self.index(check_payloads=False)
        if len(index['entries'])==len(self.order) or site!=self.order[len(index['entries'])]:
            raise ValueError('successful weight grid repeated, skipped or reordered')
        declaration=self.sites[site];fmt=self.validator(grid,declaration)
        tag='module_'+str(len(index['entries'])).zfill(4);path=self.root/'grids'/(tag+'.pt');receipt=self.root/'receipts'/(tag+'.json')
        if path.exists() or receipt.exists():raise ValueError('saved native module must recover, never overwrite')
        atomic_save(path,dict(identity=self.identity,site=site,grid=grid,pid=os.getpid(),runtime_source_sha=runtime_source_sha,
            declaration=declaration,native_oracle_bitwise_checked=True))
        entry=dict(identity=self.identity,site=site,path=str(path.relative_to(self.root)),sha=sha(path),pid=os.getpid(),
            runtime_source_sha=runtime_source_sha,weight_sha=declaration['weight_sha'],bias_sha=declaration['bias_sha'],format=fmt)
        atomic_json(receipt,entry,immutable=True);self._entry(site,entry)
        index['entries'].append(entry);atomic_json(self.index_path,index)
        return entry

    def complete(self):
        result=self.index()
        if len(result['entries'])!=len(self.order):raise ValueError('all actual native weight modules required')
        return result
