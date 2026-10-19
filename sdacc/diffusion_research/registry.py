from dataclasses import dataclass
from importlib import import_module

@dataclass(frozen=True)
class ModelDefinition:
    name:str
    module:str
    model_class:str
    status:str
    steps:int
    units:int|None
    notes:str

MODELS={
    'sigma':ModelDefinition('sigma','.models.sigma','SigmaModel','implemented',20,140,'PixArt-Sigma 1024; original fused QKV/KV routing'),
    'flux':ModelDefinition('flux','.models.flux','FluxModel','implemented',50,190,'FLUX.1-dev 512; dual/single blocks and protected attention'),
    'pixart-alpha':ModelDefinition('pixart-alpha','.models.pixart_alpha','PixArtAlphaModel','implemented',20,140,'PixArt-Alpha native256; DPM-Solver++20, PixArt shared projection routing'),
    'sd3-medium':ModelDefinition('sd3-medium','.models.sd3','SD3Model','implemented',25,142,'SD3-Medium 512; native FlowMatch, CLIP-L/G plus T5, joint image/text projections'),
    'dit-xl':ModelDefinition('dit-xl','.models.dit_xl','DiTXLModel','implemented',50,84,'DiT-XL/2 256 or 512; native class conditioning/CFG and shared self-QKV/FFN routing'),
    'sd3.5-large':ModelDefinition('sd3.5-large','.models.sd3','SD35LargeModel','implemented',25,226,'SD3.5-Large 512; shared MMDiT, 38 layers and native RMS Q/K normalization'),
}
ALIASES={'pixart-sigma':'sigma','flux-dev':'flux','pixart-α':'pixart-alpha','ditxl':'dit-xl','dit':'dit-xl','sd3':'sd3-medium','SD3-Medium':'sd3-medium',
         'sd35-large':'sd3.5-large','sd3.5large':'sd3.5-large','sd3.5-large':'sd3.5-large'}
def definition(name):
    name=name.lower()
    key=ALIASES.get(name,name)
    if key not in MODELS:raise ValueError('unknown model: '+name)
    return MODELS[key]
def build(name,config,output):
    item=definition(name)
    if item.status!='implemented':raise NotImplementedError(item.notes)
    module=import_module(item.module,__package__)
    return getattr(module,item.model_class)(config,output)

def inventory(name,model,resolution,text_length):
    item=definition(name);module=import_module(item.module,__package__)
    return getattr(module,item.model_class).inventory(model,resolution,text_length)
