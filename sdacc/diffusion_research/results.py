from pathlib import Path
from .io import read
from .registry import definition

def catalog(model=None,index=None):
    index=Path(index) if index else Path(__file__).with_name('DATA_INDEX.json')
    data=read(index)
    names=[definition(model).name] if model else list(data['models'])
    result={name:dict(data['models'][name]) for name in names if name in data['models']}
    for row in result.values():
        if row.get('complete_marker') and Path(row['complete_marker']).exists():row['status']='complete' if read(row['complete_marker']).get('complete') else 'incomplete'
    return result
