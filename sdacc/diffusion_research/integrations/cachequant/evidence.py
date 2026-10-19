import hashlib, json, math, os
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda: stream.read(8<<20), b''): h.update(block)
 return h.hexdigest()

def atomic_json(path, value, immutable=False):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 if immutable and path.exists():
  if path.read_bytes()!=data: raise ValueError(f'refuse changed evidence: {path}')
  return
 temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
 with temp.open('wb') as s:s.write(data);s.flush();os.fsync(s.fileno())
 temp.replace(path)

def load(path): return json.loads(Path(path).read_text())
def verify_manifest(root, manifest):
 for name,digest in manifest['files'].items():
  p=(Path(root)/name).resolve()
  if not p.is_relative_to(Path(root).resolve()): raise ValueError('manifest traversal')
  if sha(p)!=digest: raise ValueError(f'SHA mismatch: {name}')

def receipt(path, required=()):
 value=load(path)
 if not value.get('complete'): raise ValueError(f'incomplete receipt: {path}')
 for key in required:
  if not value.get(key): raise ValueError(f'missing proof: {key}')
 return value

COST_FIELDS=('logical_layouts','raw_layouts','members','member_order','addends',
 'math','frontend_source','backend_source','backend_binary','torch','cuda','abi',
 'compiler','configuration','cache_operation','padding_operation','input_quantization')
def cost_key(identity):
 missing=set(COST_FIELDS)-identity.keys()
 if missing: raise ValueError(f'incomplete execution identity: {sorted(missing)}')
 value={k:identity[k] for k in COST_FIELDS}
 if any(value[k] is None for k in COST_FIELDS):raise ValueError('unknown execution identity')
 return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def finite_rows(rows, fields):
 for row in rows:
  if any(not math.isfinite(float(row[k])) for k in fields): raise ValueError('nonfinite metrics')
