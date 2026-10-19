"""Print integration metadata only; this is not an experiment launcher."""
import json
from . import catalog

print(json.dumps(catalog(), indent=2, ensure_ascii=False))
