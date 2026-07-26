"""Pretrain the I-JEPA world model, then optionally probe + export (see docs/jepa-world-model.md).

Smoke (CPU, synthetic):  python commands/run_pretrain_jepa.py --smoke
Real  (GPU, manifest):   python commands/run_pretrain_jepa.py --manifest data/dataset/manifest.csv \
                             --encoder vit_small --epochs 300 --batch-size 128 --export
Real  (GPU, raw glob):   python commands/run_pretrain_jepa.py --data-dir <archive> --probe-dir <labeled>

Requires the optional torch extra:  pip install -e ".[jepa]"
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from lesnet.jepa.config import JEPAConfig
from lesnet.jepa.data import build_dataloader
from lesnet.jepa.engine import stage_batching, train
from lesnet.jepa.export import export, export_tiers
from lesnet.jepa.probe import run_probe


def _make_synthetic_class_folder(image_size, per_class=6, classes=('benign', 'malignant')):
    """Tiny labelled class-folder tree of random images, to exercise the probe path in smoke."""
    root = Path(tempfile.mkdtemp(prefix='lesnet_jepa_probe_'))
    rng = np.random.default_rng(0)
    for label, name in enumerate(classes):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            base = rng.integers(0, 128, size=(image_size, image_size, 3), dtype=np.uint8)
            base[..., label] = np.clip(base[..., label].astype(int) + 100, 0, 255).astype(np.uint8)
            Image.fromarray(base).save(folder / f'{i}.png')
    return root


def _parse_schedule(text):
    """'0:128,0.7:224' -> ((0.0, 128), (0.7, 224)); '' -> () (single-resolution training)."""
    if not text:
        return ()
    stages = []
    for chunk in text.split(','):
        fraction, _, size = chunk.partition(':')
        stages.append((float(fraction), int(size)))
    return tuple(sorted(stages))


def _log_extras(config, probe_metrics, export_report):
    """Log probe + export scalars/text to the same TensorBoard run."""
    if not config.tensorboard:
        return
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return
    writer = SummaryWriter(log_dir=str(Path(config.artifacts_dir) / 'tb'))
    if probe_metrics:
        for key in ('sensitivity', 'specificity', 'roc_auc', 'worst_group_gap', 'accuracy'):
            value = probe_metrics.get(key)
            if isinstance(value, (int, float)):
                writer.add_scalar(f'probe/{key}', value, 0)
        for group, sens in (probe_metrics.get('fitzpatrick_sensitivity') or {}).items():
            writer.add_scalar(f'probe/fitz_{group}_sensitivity', sens, 0)
        writer.add_text('probe/metrics', json.dumps(probe_metrics, indent=2))
    if export_report:
        for key in ('peak_rss_mb', 'int8_onnx_mb', 'fp32_onnx_mb', 'onnx_parity_max_abs_diff'):
            value = export_report.get(key)
            if isinstance(value, (int, float)):
                writer.add_scalar(f'export/{key}', value, 0)
        writer.add_text('export/report', json.dumps(export_report, indent=2))
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Pretrain the LesNet I-JEPA world model.")
    parser.add_argument('--data-dir', default='data/dataset', help="Unlabeled image root (recursive).")
    parser.add_argument('--manifest', help="manifest.csv: pretrain on train/val splits, probe on test.")
    parser.add_argument('--artifacts', default='artifacts/jepa')
    from lesnet.jepa.vision_transformer import ENCODER_CHOICES
    parser.add_argument('--encoder', default='vit_small', choices=ENCODER_CHOICES,
                        help="vit_tiny|small|base|large|huge or aliases tiny|small|medium|large|xlarge.")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--warmup-epochs', type=int, default=None,
                        help="LR warmup length. Defaults to 8%% of --epochs: set --epochs to the "
                             "budget you will actually run, or training never leaves warmup.")
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--patch-size', type=int, default=16,
                        help="16 for from-scratch; DINOv2 checkpoints require 14.")
    parser.add_argument('--pretrained', default='',
                        help="Openly-licensed init, e.g. dinov2_vitb14 (Apache-2.0). "
                             "Non-commercial weights are refused.")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--mixed-precision', default='bf16', choices=['bf16', 'fp16', 'no'],
                        help="Use fp16 on Kaggle P100/T4 (no bf16 support); 'no' is the safe fallback.")
    parser.add_argument('--no-hair-removal', action='store_true',
                        help="Skip DullRazor hair removal (much faster input pipeline at scale).")
    parser.add_argument('--patience', type=int, default=15,
                        help="Early-stopping patience in epochs (--epochs is the upper bound).")
    parser.add_argument('--no-early-stopping', action='store_true', help="Train the full --epochs.")
    parser.add_argument('--max-train-seconds', type=int, default=0,
                        help="Wall-clock budget; checkpoints + stops before a session kill (0=off).")
    parser.add_argument('--hdf5', help="Full-archive HDF5 bundle (id -> encoded JPEG) to pretrain on.")
    parser.add_argument('--exclude-manifest',
                        help="Manifest whose held-out test images are kept OUT of HDF5 pretraining.")
    parser.add_argument('--grad-accum', type=int, default=1,
                        help="Micro-batches per optimiser step (effective batch = batch-size * this).")
    parser.add_argument('--max-batch-size', type=int, default=0,
                        help="Cap the low-res stage's scaled-up micro-batch (0=uncapped).")
    parser.add_argument('--resize-schedule', default='',
                        help="Low-res-first schedule 'frac:size,frac:size' e.g. '0:128,0.7:224'.")
    parser.add_argument('--predictor-dim', type=int, default=192, help="Predictor width (I-JEPA: 384).")
    parser.add_argument('--predictor-depth', type=int, default=6, help="Predictor depth (I-JEPA: 12).")
    parser.add_argument('--drop-path', type=float, default=0.0, help="Stochastic depth (ViT-L: 0.1-0.2).")
    parser.add_argument('--layerscale', type=float, default=0.0, help="LayerScale init (deep ViTs: 1e-4).")
    parser.add_argument('--compile', action='store_true', help="torch.compile the model.")
    parser.add_argument('--probe-every', type=int, default=0,
                        help="Epochs between in-training probes (0=off). Needs --manifest/--probe-manifest.")
    parser.add_argument('--probe-manifest', help="Labelled manifest for the in-training probe.")
    parser.add_argument('--probe-subset', type=int, default=3000,
                        help="Images per split for the in-training probe.")
    parser.add_argument('--select-on', default='loss', choices=['loss', 'probe'],
                        help="Checkpoint + early-stop on SSL loss or on downstream probe AUC.")
    parser.add_argument('--probe-dir', help="Class-subfolder root for a fallback probe (if no --manifest).")
    parser.add_argument('--resume', help="Path to a training_state.pt to resume from.")
    parser.add_argument('--export', action='store_true', help="Export ONNX + int8 and measure 512 MB RSS.")
    parser.add_argument('--quant-tiers', action='store_true',
                        help="Export multiple precision tiers (fp32/int8/fp16) from the one encoder.")
    parser.add_argument('--no-rss', action='store_true', help="Skip the RSS measurement in export.")
    parser.add_argument('--smoke', action='store_true', help="Tiny synthetic CPU end-to-end run.")
    args = parser.parse_args()

    factory = None
    if args.smoke:
        config = JEPAConfig.smoke_config()
        config.artifacts_dir = args.artifacts
        dataloader = build_dataloader(config, synthetic_samples=16)
    else:
        config = JEPAConfig(
            data_dir=args.data_dir, artifacts_dir=args.artifacts, encoder=args.encoder,
            epochs=args.epochs, image_size=args.image_size, batch_size=args.batch_size,
            patch_size=args.patch_size, pretrained=args.pretrained,
            warmup_epochs=(args.warmup_epochs if args.warmup_epochs is not None
                           else max(round(args.epochs * 0.08), 1)),
            probe_subset=args.probe_subset,
            num_workers=args.num_workers, mixed_precision=args.mixed_precision,
            remove_hair=not args.no_hair_removal,
            early_stopping=not args.no_early_stopping, early_stop_patience=args.patience,
            max_train_seconds=args.max_train_seconds,
            hdf5_path=args.hdf5 or '', exclude_manifest=args.exclude_manifest or args.manifest or '',
            grad_accum_steps=args.grad_accum, max_batch_size=args.max_batch_size,
            resize_schedule=_parse_schedule(args.resize_schedule),
            predictor_embed_dim=args.predictor_dim, predictor_depth=args.predictor_depth,
            predictor_num_heads=max(args.predictor_dim // 64, 1),
            drop_path_rate=args.drop_path, layerscale_init=args.layerscale,
            compile_model=args.compile, probe_every=args.probe_every,
            probe_manifest=args.probe_manifest or args.manifest or '', select_on=args.select_on,
        )

        # Glob the image root ONCE: enumerating half a million files costs minutes, and the
        # resolution schedule rebuilds the loader at every stage change.
        cached_paths = None
        if not args.manifest and not args.hdf5:
            from lesnet.jepa.data import _glob_images
            cached_paths = _glob_images(args.data_dir)
            print(f'indexed {len(cached_paths)} images under {args.data_dir}', flush=True)

        def factory(image_size, batch_size=None, _config=config, _args=args, _paths=cached_paths):
            return build_dataloader(_config, root=_args.data_dir, manifest_path=_args.manifest,
                                    image_size=image_size, batch_size=batch_size, paths=_paths)

        first_size = _parse_schedule(args.resize_schedule)[0][1] if args.resize_schedule \
            else config.image_size
        dataloader = factory(first_size, stage_batching(config, first_size)[0])

    _, history, checkpoint = train(config, dataloader, resume=args.resume, dataloader_factory=factory)
    print(f"Pretrained. Encoder checkpoint: {checkpoint}")
    print(f"  final epoch loss: {history[-1]:.4f}")

    probe_source = args.manifest or args.probe_dir
    probe_root = None
    if args.smoke and not probe_source:
        probe_root = _make_synthetic_class_folder(config.image_size)
        probe_source = str(probe_root)

    artifacts = Path(config.artifacts_dir)
    probe_metrics = None
    if probe_source:
        from lesnet.jepa.export import load_encoder
        model, _ = load_encoder(checkpoint)
        probe_metrics = run_probe(model, probe_source, config, batch_size=config.batch_size)
        print(f"  linear probe: {json.dumps(probe_metrics)}")
        (artifacts / 'probe_metrics.json').write_text(json.dumps(probe_metrics, indent=2))

    export_report = None
    if args.quant_tiers:
        export_dir = artifacts / 'export'
        tiers = export_tiers(checkpoint, export_dir, measure_rss=not args.no_rss)
        (export_dir / 'tiers.json').write_text(json.dumps(tiers, indent=2))
        print(f"  quant tiers: {json.dumps(tiers)}")
    elif args.export or args.smoke:
        export_dir = artifacts / 'export'
        export_report = export(checkpoint, export_dir, measure_rss=not args.no_rss)
        (export_dir / 'report.json').write_text(json.dumps(export_report, indent=2))
        print(f"  export: {json.dumps(export_report)}")
        if export_report.get('fits_budget') is False:
            print(f"  WARNING: measured RSS {export_report['peak_rss_mb']} MB exceeds "
                  f"the {export_report['budget_mb']} MB budget.")

    _log_extras(config, probe_metrics, export_report)


if __name__ == '__main__':
    main()
