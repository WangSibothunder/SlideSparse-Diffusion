"""Class-conditioned DiT-XL: reuse PixArt self-QKV/FFN and the native CFG loop."""
from .sigma import SigmaModel


def projection_inventory(model,resolution,text_length=0):
    units=[]
    tokens=(resolution//8//model.config.patch_size)**2
    for block in range(len(model.transformer_blocks)):
        prefix=f'transformer_blocks.{block}'
        for role,names in [('QKV',[prefix+'.attn1.to_'+n for n in ('q','k','v')]),
                           ('FFN-in',[prefix+'.ff.net.0.proj']),('FFN-out',[prefix+'.ff.net.2'])]:
            members=[model.get_submodule(n) for n in names]
            units.append(dict(unit=len(units),block=block,module=role,targets=names,
                m=2*tokens,k=members[0].in_features,n=sum(m.out_features for m in members)))
    return units


class DiTXLModel(SigmaModel):
    spec_name='dit512'
    inventory=staticmethod(projection_inventory)
