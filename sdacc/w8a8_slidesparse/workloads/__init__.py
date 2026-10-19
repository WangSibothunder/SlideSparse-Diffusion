import json
from pathlib import Path
from ..runtime import PackedLinear
from ..packing import group_size
from ..composite import SharedProjection,ProtectedFFNOutput
_INV=json.loads((Path(__file__).parent/'inventory.json').read_text())
_CHOICES=json.loads((Path(__file__).parent.parent/'selected_profiles.json').read_text())['entries']
def prepare_for(workload,module,qweight,scales,bias=None,mode='2:4',attention_weight=None,execution='default'):
 alias=next(a for a in _INV['aliases'] if a['workload']==workload and a['module']==module)
 g=next(g for g in _INV['groups'] if g['id']==alias['group_id']);l=group_size(mode)
 choice=next(c for c in _CHOICES if c['group_id']==g['id'] and c['mode']==l)
 if execution not in ('default','recommended'):raise ValueError('unknown execution recipe')
 if execution=='recommended':
  recipes=json.loads((Path(__file__).parent.parent/'compiler_profiles.json').read_text())['entries']
  match=next((c for c in recipes if c['group_id']==g['id'] and c['mode']==l),None)
  if match is not None:choice=match
 if not choice['selectable']:raise ValueError('This complete projection did not pass the acceleration CI gate; keep Dense')
 if tuple(qweight.shape)!=(g['n'],g['k']):raise ValueError('weight shape differs from frozen workload')
 p=PackedLinear.prepare(qweight,scales,bias,l,config=choice['config'],frontend=choice['frontend'])
 if g['recipe']=='protected_ffn':
  if attention_weight is None:raise ValueError('BF16 protected attention weight required')
  return ProtectedFFNOutput(p,attention_weight)
 return SharedProjection(p,g['members']) if len(g['members'])>1 else p
