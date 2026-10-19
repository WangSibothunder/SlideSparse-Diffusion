from .io import sha
"""Pinned metric loaders extracted unchanged from the accepted original study."""
import json,hashlib,sys,types,importlib
from pathlib import Path
import numpy as np
import torch

IR_SHA='d71dc8d4912bc633c06ab8ef1ef6a81e8199d4369f14b47b29baf1d19de561f6'

CLIP_SHA='52bdba61c0cb19e034fc891b15e6f15291d4c06dcceae69b772776c7bc9f8b35'


def load_ir(root,device):
    base=root/'obs_rank_schedule_20261006/diagnostics/sigma_imagereward_cpu_20261008'
    sys.path.insert(0,str(base/'deps'))
    manifest=json.loads((base/'official_source_manifest.json').read_text())
    for name,digest in manifest['python_files'].items():
        if sha(base/'official'/name)!=digest:raise ValueError('official ImageReward source differs')
    for name,sub in [('ImageReward','ImageReward'),('ImageReward.models','ImageReward/models'),('ImageReward.models.BLIP','ImageReward/models/BLIP')]:
        module=types.ModuleType(name);module.__path__=[str(base/'official'/sub)];sys.modules[name]=module
    pretrain=importlib.import_module('ImageReward.models.BLIP.blip_pretrain')
    from transformers import BertTokenizer
    def tokenizer():
        t=BertTokenizer.from_pretrained(str(base/'assets/tokenizer'),local_files_only=True)
        t.add_special_tokens({'bos_token':'[DEC]'});t.add_special_tokens({'additional_special_tokens':['[ENC]']})
        t.enc_token_id=t.additional_special_tokens_ids[0];return t
    pretrain.init_tokenizer=tokenizer
    cls=importlib.import_module('ImageReward.ImageReward').ImageReward
    weight=base/'assets/ImageReward.pt'
    if sha(weight)!=IR_SHA:raise ValueError('ImageReward weight identity differs')
    model=cls(med_config=str(base/'assets/med_config.json'),device=device)
    result=model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=False)
    if result.missing_keys or result.unexpected_keys:raise ValueError('ImageReward checkpoint keys mismatch')
    return model.to(device).float().eval(),dict(weights_sha256=IR_SHA,official_commit='78f8051059b34fef8e0032ccb40561c23a2ddbbb',source_root=str(base/'official'))

def interval(values):
    from scipy.stats import t
    v=np.asarray(values,dtype=float);mean=float(v.mean());half=float(t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0
    return dict(mean=mean,ci95=[mean-half,mean+half],n=len(v))

