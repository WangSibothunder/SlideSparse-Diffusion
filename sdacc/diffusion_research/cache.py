"""Whole-prediction hold outside compilation; CEM schedule from an untimed warmup.

No DuCa internals, activation collector, fitting, or cross-sample reuse. The
pipeline still performs every scheduler/CFG/variance operation at every step.
"""
import math
import torch


def cem_schedule(priors, count, steps, forced):
    """Existing cumulative-prior quadratic CEM recurrence, arbitrary step count."""
    base = round(steps / count)
    gaps = range(max(1, base - 1), min(10, base + 1) + 1)
    costs = {}
    for gap in gaps:
        lag = gap - 1
        if not lag:
            costs.update({(t-1, t): 0. for t in range(1, steps)})
            continue
        values = priors[lag]
        acc = 0.
        for t, value in enumerate(values):
            acc += value
            if t >= gap:
                costs[t-gap, t] = acc * lag * lag
    dp = {(0, 1): (0., (0,))}
    for k in range(2, count + 1):
        for t in range(1, steps):
            choices = []
            for gap in gaps:
                previous = t-gap
                if previous < 0 or any(previous < a < t for a in forced):
                    continue
                if (previous, k-1) in dp and (previous, t) in costs:
                    old, path = dp[previous, k-1]
                    choices.append((old+costs[previous, t], path+(t,)))
            if choices:
                dp[t, k] = min(choices)
    if (steps-1, count) not in dp:
        raise ValueError('CEM budget cannot satisfy frozen policy boundaries')
    objective, refresh = dp[steps-1, count]
    return dict(refresh=list(refresh), objective=objective, count=count,
                forced=sorted(forced), weighter='quadratic', interval=1)


def warmup_prior(predictions):
    """Only raw outputs of one already required Dense warmup; scalar CPU export."""
    vectors = torch.stack([p.float().reshape(-1) for p in predictions])
    norms = vectors.norm(dim=1).clamp_min(1e-8)
    priors = {}
    for lag in (1, 2):
        similarity = (vectors[lag:]*vectors[:-lag]).sum(dim=1)/(norms[lag:]*norms[:-lag])
        values = (1-similarity.clamp(-1, 1)).clamp_min(0).cpu().tolist()
        priors[lag] = [0.] * lag + values
    if not all(math.isfinite(x) for values in priors.values() for x in values):
        raise ValueError('nonfinite warmup prior')
    return priors


def execution_variant(variant, refresh):
    """Diff against the previous EXECUTED row, never a skipped logical row."""
    policy = variant['policy']
    updates = [[] for _ in policy]
    previous = [-1]*len(policy[0])
    for step in refresh:
        row = policy[step]
        updates[step] = [(uid, int(mode)) for uid, mode in enumerate(row)
                         if previous[uid] != mode]
        previous = row
    return dict(variant, updates=updates)


class PredictionHold(torch.nn.Module):
    def __init__(self, core, steps, select=None):
        super().__init__()
        self.core, self.steps, self.select = core, steps, select
        self.reset(False)

    def reset(self, enabled, refresh=(), boundaries=(), observe=False):
        self.enabled = enabled
        self.refresh = frozenset(refresh)
        self.boundaries = frozenset(boundaries)
        if enabled and not ({0, self.steps-1} | self.boundaries) <= self.refresh:
            raise ValueError('cache reset boundary must refresh')
        self.step, self.cached, self.executed = 0, None, []
        self.observe, self.predictions = observe, []

    def forward(self, *args, **kwargs):
        step = self.step
        if step >= self.steps:
            raise ValueError('extra transformer call')
        self.step += 1
        if step in self.boundaries:
            self.cached = None
        if not self.enabled or step in self.refresh:
            if self.select is not None:
                self.select(step)
            result = self.core(*args, **kwargs)
            self.executed.append(step)
            if self.enabled or self.observe:
                if kwargs.get('return_dict', True) or not isinstance(result, tuple) or len(result) != 1:
                    raise ValueError('raw transformer prediction tuple required')
                # Capture outputs may alias a replay buffer: retain an owned tensor.
                retained = result[0].detach().clone()
                if self.enabled:
                    self.cached = retained
                if self.observe:
                    self.predictions.append(retained)
            return result
        if self.cached is None:
            raise ValueError('cache hit without producing prediction')
        return (self.cached,)
