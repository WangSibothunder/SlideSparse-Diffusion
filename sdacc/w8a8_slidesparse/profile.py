"""Frozen shape dispatch: no profiling, search, packing, or file I/O in forward."""
import json
from pathlib import Path
_DATA=json.loads((Path(__file__).parent/'profiles.json').read_text())
_PROFILES={(r['mode'],r['m'],r['k'],r['n']): (r['config'],r['frontend']) for r in _DATA['entries']}
LEGACY_CORE_SHA="e4af641f13175c73225ed6cc8e15a50c88de5d7454811151811081fa85551c61"
def execution_profile(mode,m,k,n):
    return _PROFILES.get((mode,m,k,n),(5,'triton'))
def known_configurations(mode,k,n):
    return sorted({5}|{c for (l,m,kk,nn),(c,f) in _PROFILES.items() if l==mode and kk==k and nn==n})
