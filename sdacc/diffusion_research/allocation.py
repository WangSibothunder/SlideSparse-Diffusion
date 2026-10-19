"""Mode priority first, then low incremental error; exact integer budget."""
from math import gcd, isfinite
from functools import reduce
PHASES=((0,7),(7,14),(14,20))
TARGETS=(15,25,35,45)

def feasible_priority_subset(items, cap):
    # The bitset answers feasibility only. It does not replace the error order.
    if not items:return [],0
    ordered=sorted(items,key=lambda x:(x['delta_error'],x['unit']))
    step=reduce(gcd,(x['increment'] for x in ordered));limit=cap//step
    weights=[x['increment']//step for x in ordered]
    suffix=[0]*(len(weights)+1);suffix[-1]=1
    for i in range(len(weights)-1,-1,-1):suffix[i]=suffix[i+1]|(suffix[i+1]<<weights[i])
    reachable=suffix[0]&((1<<(limit+1))-1)
    goal=reachable.bit_length()-1;remaining=goal;selected=[]
    for i,x in enumerate(ordered):
        w=weights[i]
        if remaining>=w and ((suffix[i+1]>>(remaining-w))&1):
            selected.append(x);remaining-=w
    assert remaining==0
    return selected,goal*step

def allocate_phase(rows, percent):
    assert 0<=percent<=50 and len({r['unit'] for r in rows})==len(rows)
    total=sum(r['parameters'] for r in rows);cap=total*percent//100
    policy={r['unit']:0 for r in rows};used=0;waves=[]
    for mode in (8,6,4):
        if used>=cap:break
        items=[]
        for r in rows:
            uid=r['unit'];previous=policy[uid]
            assert r['parameters']>0 and all(isfinite(v) and v>=0 for v in r['errors'].values())
            if not r['admitted'][str(mode)]:continue
            old=r['structural_zeros'][str(previous)] if previous else 0
            delta=r['structural_zeros'][str(mode)]-old
            assert delta>0
            items.append(dict(unit=uid,increment=delta,
                delta_error=r['errors'][str(mode)]-(r['errors'][str(previous)] if previous else 0)))
        remaining=cap-used
        if sum(x['increment'] for x in items)<=remaining:
            selected=sorted(items,key=lambda x:(x['delta_error'],x['unit']));increment=sum(x['increment'] for x in selected);finished_wave=True
        else:
            selected,increment=feasible_priority_subset(items,remaining);finished_wave=False
        for x in selected:policy[x['unit']]=mode
        used+=increment;waves.append(dict(mode=mode,selected=selected,increment=increment,finished_wave=finished_wave))
        # Rounding a partially filled preferred wave never opens the next wave.
        if not finished_wave:break
    assert used<=cap
    maximum=sum(max((r['structural_zeros'][str(m)] for m in (4,6,8) if r['admitted'][str(m)]),default=0) for r in rows)
    assert cap<=maximum,'target infeasible under accepted modes'
    return dict(policy=[policy[r['unit']] for r in sorted(rows,key=lambda r:r['unit'])],
        total_parameters=total,requested_percent=percent,requested_integer_zero_cap=cap,
        structural_zeros=used,actual_percent=100*used/total,rounding_shortfall=cap-used,waves=waves)
