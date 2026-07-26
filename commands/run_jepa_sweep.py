"""Cheap local hyperparameter sweep for I-JEPA, scored on downstream probe AUC.

The recipe in JEPAConfig is I-JEPA's ImageNet recipe. Dermoscopy is not ImageNet: lesions are
centred and roughly rotation-invariant, colour carries real signal (and real device bias), and
most of the ISIC archive is low-resolution total-body tiles. Committing hundreds of dollars of
H100 time to an unvalidated recipe is how you buy an expensive bad model.

So: many SHORT runs of a SMALL encoder on a SUBSET, ranked by the same held-out probe AUC the
real run selects on. Each config costs minutes on a laptop GPU. What transfers from this to the
full run is the *ordering* of choices (LR, mask geometry, augmentation), not the absolute AUC —
that is the standard assumption behind scaled-down sweeps and it is worth stating plainly.

    python commands/run_jepa_sweep.py --data-dir data/isic_384/images \
        --probe-manifest data/isic_dx/manifest.csv --out artifacts/sweep

Resumable: configs already present in results.json are skipped.
"""
import argparse
import copy
import json
import time
from pathlib import Path

# name -> overrides applied on top of the baseline. One knob at a time so the winner is readable.
SWEEP = {
    'baseline': {},
    'lr_half': {'lr': 5e-4},
    'lr_double': {'lr': 2e-3},
    'lr_quad': {'lr': 4e-3},
    'wd_high': {'weight_decay': 0.1},
    'ema_slow': {'ema_momentum': (0.999, 1.0)},
    'ema_fast': {'ema_momentum': (0.99, 1.0)},
    'pred_shallow': {'predictor_depth': 6},
    'pred_wide': {'predictor_embed_dim': 512, 'predictor_num_heads': 8},
    'mask_wide_targets': {'pred_mask_scale': (0.10, 0.25)},
    'mask_big_targets': {'pred_mask_scale': (0.20, 0.30)},
    'mask_6_targets': {'num_pred_masks': 6},
    'mask_2_targets': {'num_pred_masks': 2},
    'ctx_smaller': {'enc_mask_scale': (0.60, 0.90)},
    'crop_gentle': {'rrc_min_scale': 0.7},
    'crop_aggressive': {'rrc_min_scale': 0.25},
    'no_colour_constancy': {'colour_constancy': False},
    'droppath_off': {'drop_path_rate': 0.0},
    'layerscale_off': {'layerscale_init': 0.0},
    'loss_l2': {'loss': 'l2'},
}


def build_config(base_kwargs, overrides):
    from lesnet.jepa.config import JEPAConfig
    kwargs = copy.deepcopy(base_kwargs)
    kwargs.update(overrides)
    return JEPAConfig(**kwargs)


def run_one(name, overrides, args, paths):
    from lesnet.jepa.data import build_dataloader
    from lesnet.jepa.engine import train
    from lesnet.jepa.probe import quick_probe

    base = dict(
        encoder=args.encoder, image_size=args.image_size, patch_size=16,
        batch_size=args.batch_size, grad_accum_steps=args.grad_accum,
        predictor_embed_dim=384, predictor_depth=12, predictor_num_heads=6,
        drop_path_rate=0.1, layerscale_init=1e-4,
        # min_keep is an absolute patch count tuned for a 14x14 grid; on the smaller sweep grid a
        # target block cannot exceed it and mask sampling fails outright. Scale it to the grid.
        min_keep=max(round(10 * (args.image_size // 16) ** 2 / 196), 2),
        epochs=args.epochs, warmup_epochs=max(round(args.epochs * 0.1), 1),
        num_workers=args.num_workers, mixed_precision=args.mixed_precision,
        remove_hair=False, colour_constancy=True,
        early_stopping=False, tensorboard=False, checkpoint_every=0, log_every=200,
        artifacts_dir=str(Path(args.out) / name),
    )
    config = build_config(base, overrides)

    def factory(image_size, batch_size=None, _c=config, _p=paths):
        return build_dataloader(_c, root='sweep', image_size=image_size,
                                batch_size=batch_size, paths=_p)

    started = time.time()
    model, history, _ = train(config, factory(config.image_size), dataloader_factory=factory)
    metrics = quick_probe(model.context_encoder, args.probe_manifest, config,
                          image_size=config.image_size, subset=args.probe_subset,
                          batch_size=max(config.batch_size, 32), num_workers=args.num_workers)
    return {
        'name': name, 'overrides': {k: list(v) if isinstance(v, tuple) else v
                                    for k, v in overrides.items()},
        'roc_auc': metrics.get('roc_auc'), 'sensitivity': metrics.get('sensitivity'),
        'specificity': metrics.get('specificity'), 'final_loss': round(history[-1], 5),
        'minutes': round((time.time() - started) / 60, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Local I-JEPA recipe sweep, ranked by probe AUC.")
    parser.add_argument('--data-dir', default='data/isic_384/images')
    parser.add_argument('--probe-manifest', default='data/isic_dx/manifest.csv')
    parser.add_argument('--out', default='artifacts/sweep')
    parser.add_argument('--encoder', default='vit_small')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--subset', type=int, default=60000, help="Images per short run.")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--probe-subset', type=int, default=1500)
    parser.add_argument('--mixed-precision', default='fp16')
    parser.add_argument('--only', nargs='*', help="Run only these config names.")
    args = parser.parse_args()

    from lesnet.jepa.data import _glob_images, manifest_image_ids

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / 'results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {r['name'] for r in results}

    # Same leakage discipline as the real run: held-out test images never enter SSL.
    held_out = manifest_image_ids(args.probe_manifest, ('test',))
    paths = [p for p in _glob_images(args.data_dir) if p.stem not in held_out][:args.subset]
    print(f'sweep set: {len(paths)} images ({len(held_out)} held-out ids excluded)', flush=True)

    names = args.only or list(SWEEP)
    for index, name in enumerate(names, 1):
        if name in done:
            print(f'[{index}/{len(names)}] {name}: cached', flush=True)
            continue
        print(f'[{index}/{len(names)}] {name}: {SWEEP[name] or "(baseline)"}', flush=True)
        try:
            record = run_one(name, SWEEP[name], args, paths)
        except Exception as error:                 # a bad config must not sink the sweep
            record = {'name': name, 'overrides': SWEEP[name], 'roc_auc': None,
                      'error': repr(error)[:300]}
        results.append(record)
        results_path.write_text(json.dumps(results, indent=2))
        print(f'    -> auc {record.get("roc_auc")} ({record.get("minutes")} min) '
              f'{record.get("error", "")}', flush=True)

    ranked = sorted([r for r in results if r.get('roc_auc') is not None],
                    key=lambda r: -r['roc_auc'])
    lines = ['| rank | config | probe AUC | sens | spec | final loss |',
             '|------|--------|-----------|------|------|------------|']
    for position, record in enumerate(ranked, 1):
        lines.append(f"| {position} | `{record['name']}` | **{record['roc_auc']:.4f}** | "
                     f"{record.get('sensitivity')} | {record.get('specificity')} | "
                     f"{record.get('final_loss')} |")
    (out / 'results.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)
    if ranked:
        print(f"\nbest: {ranked[0]['name']} -> {ranked[0]['overrides']}", flush=True)


if __name__ == '__main__':
    main()
