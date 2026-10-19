"""Independent FP64 diagonal OLS, without optimization or heldout data."""
import numpy as np
def fit(count,values):
    assert count>1 and values.ndim==2 and values.shape[0]==5 and np.isfinite(values).all()
    s,d,ss,sd,dd=values;ms=s/count;md=d/count
    vs=np.maximum(ss/count-ms*ms,0);vd=np.maximum(dd/count-md*md,0)
    covariance=sd/count-ms*md
    a=np.divide(covariance,vs,out=np.ones_like(vs),where=vs>1e-12);b=md-a*ms
    mse=np.maximum(a*a*vs+vd-2*a*covariance+(a*ms+b-md)**2,0)
    raw=np.maximum(vs+vd-2*covariance+(ms-md)**2,0);valid=vd>1e-12
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return dict(a=a.astype(np.float32),b=b.astype(np.float32),diagnostic=dict(count=int(count),
        raw_mse_sum=float(count*raw.sum()),affine_mse_sum=float(count*mse.sum()),
        valid_channels=int(valid.sum()),excluded_constant_channels=int((~valid).sum()),
        r2_median=float(np.median(1-mse[valid]/vd[valid])) if valid.any() else None))
def fold(a,b,scales,bias):
    assert a.shape==b.shape==scales.shape==bias.shape and all(np.isfinite(v).all() for v in (a,b,scales,bias))
    sw=(a.astype(np.float32)*scales.astype(np.float32)).astype(np.float32)
    bv=(a.astype(np.float32)*bias.astype(np.float32)+b.astype(np.float32)).astype(np.float32)
    assert np.isfinite(sw).all() and np.isfinite(bv).all();return sw,bv
def positions(policy):return [f'{t}:{u}' for t,row in enumerate(policy) for u,m in enumerate(row) if m]
