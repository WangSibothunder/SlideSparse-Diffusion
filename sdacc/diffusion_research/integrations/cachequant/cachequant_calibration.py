"""Pinned native MSE estimates, exact offline activation-state replay.

Collect estimates on the complete actual tensor before any CPU conversion.
Replaying the original time-major calibration order avoids storing all feature
maps. The upstream running scale/zero are shared across times: keep that state
and the original dtype/operation order, rather than silently using twenty
independent EMA accumulators. This module does not collect model trajectories,
admit execution classes, or fit Sparse/DEC coefficients.
"""
import importlib.util
from pathlib import Path
import torch
from .evidence import load, sha, verify_manifest


def native_quantizer(vendor):
    vendor = Path(vendor)
    verify_manifest(vendor, load(vendor/'SOURCE_MANIFEST.json'))
    source = vendor/'quant/quant_layer.py'
    spec = importlib.util.spec_from_file_location('sigma_native_cachequant_quant_layer', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UniformAffineQuantizer


def make_quantizer(kind, native_class):
    if kind not in ('weight', 'activation'):
        raise ValueError('explicit native weight/activation quantizer required')
    return native_class(n_bits=8, symmetric=False, channel_wise=kind=='weight',
                        scale_method='mse', leaf_param=kind=='activation',
                        num_timesteps=20, prob=1., always_zero=False)


def _grid(delta, zero):
    if not torch.isfinite(delta).all() or not (delta > 0).all() or not torch.isfinite(zero).all():
        raise ValueError('native grid degenerate; preserve failure, do not invent a symmetric grid or epsilon')
    if delta.dtype != zero.dtype or delta.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('native scale/zero dtype differs')


@torch.no_grad()
def weight_grid(weight, native_class):
    if weight.ndim not in (2,4) or not torch.isfinite(weight).all():
        raise ValueError('finite original Linear or Conv2d weight required')
    quantizer = make_quantizer('weight', native_class)
    original_native_output = quantizer(weight)
    delta, zero = quantizer.delta.detach(), quantizer.zero_point.detach()
    _grid(delta, zero)
    if delta.shape != zero.shape or delta.shape != (weight.shape[0], *([1]*(weight.ndim-1))):
        raise ValueError('native per-output weight grid required')
    rounded_zero = zero.round()
    if not ((rounded_zero >= 0) & (rounded_zero <= 255)).all():
        raise ValueError('native weight zero cannot be encoded on unchanged unsigned grid; independent format proof required')
    codes = (torch.round(weight/delta)+rounded_zero).clamp(0,255)
    if not torch.equal((codes-rounded_zero)*delta, original_native_output):
        raise ValueError('native weight division/rounding/dequantization boundary changed')
    return dict(codes=codes.to(torch.uint8).cpu(), delta=delta.cpu().clone(), zero=zero.cpu().clone(),
                native_dtype=str(weight.dtype), original_shape=list(weight.shape),
                source='actual original weight, pinned native MSE before optional format flattening',
                gradients=False, optional_reconstruction=False,
                real_INT8_output_and_OBS_numeric_pending=True)


@torch.no_grad()
def activation_estimate(value, native_class):
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError('finite complete actual activation tensor required')
    quantizer = make_quantizer('activation', native_class)
    delta, zero = quantizer.init_quantization_scale_2(value, channel_wise=False)
    _grid(delta, zero)
    if delta.numel() != 1 or zero.numel() != 1:
        raise ValueError('native per-tensor activation grid required')
    return dict(delta=delta.detach().cpu().clone(), zero=zero.detach().cpu().clone(),
                shape=list(value.shape), numel=value.numel(), dtype=str(value.dtype),
                source_device=str(value.device), full_tensor_before_sampling=True)


@torch.no_grad()
def replay_activation_estimates(samples, executed_steps, native_class):
    """Samples must be ordered complete sample transactions from one own fit.

    This numerical helper also supports small CPU contracts; its return is never
    an own100cal receipt. A model producer must separately verify source/cohort,
    100 complete samples, original inputs, CFG order and collector bitwise.
    """
    steps = tuple(executed_steps)
    if not samples or not steps or steps != tuple(sorted(set(steps))) or steps[0] != 0 or steps[-1] >= 20:
        raise ValueError('ordered real executed step set beginning at zero required')
    expected = set(steps)
    dtype = None
    for packet in samples:
        if set(packet) != expected:
            raise ValueError('missing executed activation is not an unexecuted cache slot')
        for step, value in packet.items():
            _grid(value['delta'], value['zero'])
            if (value['delta'].numel()!=1 or value['zero'].numel()!=1
                or not value.get('full_tensor_before_sampling') or value['numel'] <= 0):
                raise ValueError('complete per-tensor native MSE estimate required')
            current = value['delta'].dtype
            if dtype is None:
                dtype = current
            if current != dtype:
                raise ValueError('native activation dtype changed between calibration samples')
    quantizer = make_quantizer('activation', native_class)
    for step in steps:
        for packet in samples:
            value = packet[step]
            delta, zero = quantizer.update_quantize_range(value['delta'], value['zero'])
            _grid(delta, zero)
            if quantizer.delta is None:
                quantizer.delta = torch.nn.Parameter(torch.stack([delta]*20))
                quantizer.zero_point = torch.nn.Parameter(torch.stack([zero]*20))
            else:
                quantizer.delta[step] = delta
                quantizer.zero_point[step] = zero
    return dict(delta=quantizer.delta.detach().clone(), zero=quantizer.zero_point.detach().clone(),
                executed_steps=list(steps), samples=len(samples),
                ordering='native time-major, then original calibration condition order',
                native_running_state='one shared 0.1 estimate + 0.9 running accumulator across steps',
                unexecuted_slots_not_measurements=True, own100cal_scientific_receipt=False,
                gradients=False, optional_reconstruction=False)


def activation_slot(grid, step):
    if step not in grid['executed_steps']:
        raise ValueError('unexecuted native initialization slot cannot authorize a new executed caller')
    return grid['delta'][step], grid['zero'][step]
