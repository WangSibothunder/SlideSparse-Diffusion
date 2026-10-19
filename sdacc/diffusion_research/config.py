from pathlib import Path
import os
from .io import read
from .registry import definition

def load(path,model=None):
    path=Path(path).resolve();config=read(path)
    item=definition(config['model'])
    if model and definition(model).name!=item.name:raise ValueError('model and config differ')
    if config.get('status')=='planned':raise NotImplementedError(item.notes)
    for key in ['library','checkpoint','conditions','conditioning_source','units','conditioning_directory','bank_directory','packed_manifest','slot_plan','memory_plan','policy_file','coefficients_directory','dense_results','dense_generation','dense_decode','metric_assets','pipeline_source','study_directory']:
        value=config.get(key)
        if value:
            target=Path(os.path.expandvars(value)).expanduser()
            config[key]=str(target if target.is_absolute() else path.parent/target)
    config['config_path']=str(path)
    return config
