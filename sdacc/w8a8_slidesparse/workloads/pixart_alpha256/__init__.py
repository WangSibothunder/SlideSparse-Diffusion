from .. import prepare_for
WORKLOAD='pixart_alpha256'
def prepare(module,qweight,scales,bias=None,mode='2:4',attention_weight=None,execution='default'):
 return prepare_for(WORKLOAD,module,qweight,scales,bias,mode,attention_weight,execution)
