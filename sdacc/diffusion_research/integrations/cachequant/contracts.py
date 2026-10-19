from dataclasses import dataclass
from pathlib import Path
from .evidence import load, sha

FORCED_REFRESH=(0,7,14,19)
PHASES=((0,7),(7,14),(14,20))
RATES=(15,25,35,45)

def phase(step):
 if not 0<=step<20:raise ValueError('step outside Sigma20')
 return 0 if step<7 else 1 if step<14 else 2

def validate_policy(policy):
 if len(policy)!=20 or any(len(row)!=140 for row in policy):raise ValueError('policy must be20x140')
 for lo,hi in PHASES:
  if any(row!=policy[lo] for row in policy[lo:hi]):raise ValueError('not complete three-stage policy')
 if any(type(g)!=int or g not in (0,4,6,8) for row in policy for g in row):raise ValueError('invalid sparse mode')

def splits(book):
 cal=book['calibration'];pilot=book['expansion'][:4];evaluation=book['evaluation'][:64]
 if tuple(map(len,(cal,pilot,evaluation)))!=(100,4,64):raise ValueError('100/4/64 contract')
 sets=[{c['id'] for c in v} for v in (cal,pilot,evaluation)]
 if tuple(map(len,sets))!=(100,4,64) or any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):raise ValueError('cohort overlap')
 return cal,pilot,evaluation

@dataclass(frozen=True)
class Inputs:
 checkpoint: Path
 conditions: Path
 units: Path
 bank_root: Path
 library_snapshot: Path
 frozen_schedules: Path
 output: Path
 baseline_package: Path
 baseline_delivery: Path
 checkpoint_identity: Path
 @classmethod
 def from_config(cls,config):
  data=config['inputs']
  return cls(**{k:Path(data[k]).expanduser().resolve() for k in cls.__dataclass_fields__})
 def validate(self):
  for name in self.__dataclass_fields__:
   p=getattr(self,name)
   if name!='output' and not p.exists():raise FileNotFoundError(p)
  cal,pilot,ev=splits(load(self.conditions))
  schedules=load(self.frozen_schedules)
  for rate in RATES:validate_policy(schedules[f'priority_s{rate}_affine100']['policy'])
  return dict(conditions_sha=sha(self.conditions),schedules_sha=sha(self.frozen_schedules),
              counts=[len(cal),len(pilot),len(ev)])
