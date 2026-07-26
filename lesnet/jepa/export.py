"""Export the context encoder for a light 512 MB inference runtime (see docs/jepa-world-model.md).

The deployable artifact is ONNX (fp32 + onnxruntime dynamic int8), served by onnxruntime's
CPUExecutionProvider — NOT the torch runtime, whose ~300-500 MB CPU RSS is the real constraint
against a 512 MB box. The budget gate MEASURES peak inference RSS in a fresh process, rather than
asserting on-disk weight size. The torch `.pt` checkpoint is kept only for research/fine-tuning.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

from lesnet.jepa.config import JEPAConfig
from lesnet.jepa.vision_transformer import build_encoder, set_attention_impl

BUDGET_MB = 512.0


def load_encoder(checkpoint_path, image_size=None, map_location='cpu'):
    """Rebuild the context encoder from a checkpoint; optionally at a different resolution.

    Passing `image_size` rebuilds the encoder at the transfer/deploy resolution — the sin-cos
    position grid regenerates analytically, so no interpolation is needed.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    config = JEPAConfig(**checkpoint['config'])
    if image_size is not None:
        config.image_size = image_size
    encoder = build_encoder(config)
    encoder.load_state_dict(checkpoint['state_dict'])
    encoder.eval()
    return encoder, config


def export_onnx(encoder, out_path, image_size, opset=18):
    """Export the fp32 encoder to ONNX with eager attention (portable) and a dynamic batch axis."""
    set_attention_impl(encoder, use_sdpa=False)
    try:
        dummy = torch.randn(1, 3, image_size, image_size)
        torch.onnx.export(
            encoder, dummy, str(out_path), opset_version=opset,
            input_names=['image'], output_names=['tokens'],
            dynamic_axes={'image': {0: 'batch'}, 'tokens': {0: 'batch'}},
            verbose=False,  # also avoids a Windows cp1252 crash on the exporter's emoji progress print
        )
    finally:
        set_attention_impl(encoder, use_sdpa=True)
    return out_path


def quantize_onnx_int8(fp32_path, int8_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    return int8_path


def _size_mb(path):
    """Total MB of an ONNX file including any external-data sidecars in its directory."""
    path = Path(path)
    total = os.path.getsize(path)
    for sibling in path.parent.glob(path.name + '*'):
        if sibling != path:
            total += os.path.getsize(sibling)
    return total / 1e6


# Measures TRUE peak RSS across session build + forward (the high-water mark), pinned to a
# single intra-op thread + sequential execution to approximate a small deploy container.
_RSS_PROBE = textwrap.dedent('''
    import sys, os, platform, numpy as np, onnxruntime as ort, psutil
    onnx_path, image_size = sys.argv[1], int(sys.argv[2])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(onnx_path, sess_options=options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    session.run(None, {name: np.random.rand(1, 3, image_size, image_size).astype("float32")})
    system = platform.system()
    if system == "Windows":
        peak = psutil.Process(os.getpid()).memory_info().peak_wset
    else:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # bytes on macOS, KB on Linux
        peak = ru if system == "Darwin" else ru * 1024
    print(int(peak))
''')


def measure_peak_rss(onnx_path, image_size, python_exe=None):
    """Peak RSS (MB) of a fresh process that loads onnxruntime + the model and runs one forward."""
    python_exe = python_exe or sys.executable
    result = subprocess.run(
        [python_exe, '-c', _RSS_PROBE, str(onnx_path), str(image_size)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'RSS probe failed: {result.stderr.strip()}')
    return int(result.stdout.strip().splitlines()[-1]) / 1e6


def _onnx_parity(encoder, onnx_path, image_size):
    """Max abs diff between torch and onnxruntime outputs on one input (export sanity check)."""
    import onnxruntime as ort
    example = torch.randn(1, 3, image_size, image_size)
    set_attention_impl(encoder, use_sdpa=False)
    try:
        with torch.no_grad():
            torch_out = encoder(example).numpy()
    finally:
        set_attention_impl(encoder, use_sdpa=True)
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    onnx_out = session.run(None, {session.get_inputs()[0].name: example.numpy()})[0]
    return float(np.abs(torch_out - onnx_out).max())


def _tier_report(name, path, encoder, size, budget_mb, measure_rss):
    report = {'level': name, 'onnx_mb': round(_size_mb(path), 2),
              'parity_max_abs_diff': round(_onnx_parity(encoder, path, size), 6), 'path': str(path)}
    if measure_rss:
        rss = measure_peak_rss(path, size)
        report['peak_rss_mb'] = round(rss, 1)
        report['fits_budget'] = rss < budget_mb
    else:
        report['peak_rss_mb'] = None
        report['fits_budget'] = None
    return report


def export_tiers(checkpoint_path, out_dir, image_size=None, budget_mb=BUDGET_MB, measure_rss=True):
    """Produce multiple precision tiers (fp32, int8, and fp16 if available) from ONE trained encoder.

    Strategy for the size family: train the biggest model once, then ship precision tiers. Each tier
    is exported, parity-checked vs torch, and its 512 MB fit MEASURED. fp16 is best-effort (skipped
    with a note if onnxconverter-common is absent) so a missing optional dep never breaks the run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder, config = load_encoder(checkpoint_path, image_size=image_size)
    size = image_size or config.image_size

    fp32_onnx = out_dir / 'encoder_fp32.onnx'
    export_onnx(encoder, fp32_onnx, size)
    tiers = [_tier_report('fp32', fp32_onnx, encoder, size, budget_mb, measure_rss)]

    int8_onnx = out_dir / 'encoder_int8.onnx'
    quantize_onnx_int8(fp32_onnx, int8_onnx)
    tiers.append(_tier_report('int8', int8_onnx, encoder, size, budget_mb, measure_rss))

    fp16_onnx = out_dir / 'encoder_fp16.onnx'
    try:
        import onnx
        from onnxconverter_common import float16
        model16 = float16.convert_float_to_float16(onnx.load(str(fp32_onnx)), keep_io_types=True)
        onnx.save(model16, str(fp16_onnx))
        tiers.append(_tier_report('fp16', fp16_onnx, encoder, size, budget_mb, measure_rss))
    except Exception as error:  # noqa: BLE001 - optional tier
        print(f'  (fp16 tier skipped: {error})')

    return {'encoder': config.encoder, 'image_size': size, 'budget_mb': budget_mb, 'tiers': tiers}


def export(checkpoint_path, out_dir, image_size=None, budget_mb=BUDGET_MB, measure_rss=True,
           fp32_tol=1e-3, int8_tol=0.5):
    """Export ONNX fp32 + int8, GATE on parity (fp32 tight, int8 loose) and MEASURED peak RSS.

    Raises ValueError if either exported artifact is numerically broken — a non-portable or badly
    quantized encoder must not ship silently just because it fits the RSS budget.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder, config = load_encoder(checkpoint_path, image_size=image_size)
    size = image_size or config.image_size

    fp32_onnx = out_dir / 'encoder_fp32.onnx'
    int8_onnx = out_dir / 'encoder_int8.onnx'
    export_onnx(encoder, fp32_onnx, size)
    quantize_onnx_int8(fp32_onnx, int8_onnx)

    fp32_parity = _onnx_parity(encoder, fp32_onnx, size)        # torch vs fp32 onnx (must be tight)
    int8_parity = _onnx_parity(encoder, int8_onnx, size)        # torch vs the SHIPPED int8 artifact
    report = {
        'encoder': config.encoder,
        'image_size': size,
        'fp32_onnx_mb': round(_size_mb(fp32_onnx), 2),
        'int8_onnx_mb': round(_size_mb(int8_onnx), 2),
        'onnx_parity_max_abs_diff': round(fp32_parity, 6),
        'int8_parity_max_abs_diff': round(int8_parity, 6),
        'budget_mb': budget_mb,
        'fp32_onnx_path': str(fp32_onnx),
        'int8_onnx_path': str(int8_onnx),
    }
    if fp32_parity > fp32_tol:
        raise ValueError(f'ONNX fp32 export is not faithful: max abs diff {fp32_parity:.4g} > {fp32_tol}.')
    if int8_parity > int8_tol:
        raise ValueError(f'int8 ONNX degrades outputs: max abs diff {int8_parity:.4g} > {int8_tol}. '
                         f'Deploy the fp32 ONNX or a larger/less-aggressively-quantised variant.')
    if measure_rss:
        rss = measure_peak_rss(int8_onnx, size)
        report['peak_rss_mb'] = round(rss, 1)
        report['fits_budget'] = rss < budget_mb
    else:
        report['peak_rss_mb'] = None
        report['fits_budget'] = None
    return report
