from .. import prepare_for
WORKLOAD='sd35_large512'
def prepare(module,qweight,scales,bias=None,mode='2:4',attention_weight=None,execution='default'):
 return prepare_for(WORKLOAD,module,qweight,scales,bias,mode,attention_weight,execution)
