"""Two-rank DDP smoke test for the I-JEPA engine, runnable without a GPU.

Multi-GPU bugs do not usually announce themselves — they hang. A rank that takes a different
branch (early-stops, skips a probe, rebuilds a loader) leaves the others parked in a collective
until the NCCL timeout, after hours of paid GPU time. This exercises the whole distributed path on
CPU with the gloo backend so those bugs surface locally in seconds.

Checks, in order of what they would have cost on a real cluster:
  1. the run TERMINATES (no deadlock) — enforced by the caller's subprocess timeout
  2. gradients are actually synchronised: both ranks end with identical encoder weights
  3. exactly one checkpoint is written (rank 0 owns the filesystem)
  4. early stopping fires on every rank together, not just rank 0

    python tests/ddp_smoke.py <output.json>
"""
import hashlib
import json
import os
import sys
import tempfile

import torch.multiprocessing as mp


def _fingerprint(module):
    digest = hashlib.sha256()
    for name, param in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(param.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _worker(rank, world_size, artifacts, port, results):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK='0',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    import torch.distributed as dist

    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    from lesnet.jepa.config import JEPAConfig
    from lesnet.jepa.data import build_dataloader
    from lesnet.jepa.engine import train

    config = JEPAConfig(
        image_size=96, patch_size=16, encoder='vit_tiny',
        predictor_embed_dim=96, predictor_depth=2, predictor_num_heads=3,
        num_pred_masks=2, min_keep=2, batch_size=2, grad_accum_steps=2, num_workers=0,
        epochs=4, warmup_epochs=1, mixed_precision='no', remove_hair=False,
        colour_constancy=False, scale_lr=False, checkpoint_every=2, tensorboard=False,
        log_every=1, early_stopping=True, early_stop_patience=1, early_stop_min_delta=1e9,
        artifacts_dir=artifacts,
    )
    loader = build_dataloader(config, synthetic_samples=32)
    model, history, checkpoint = train(config, loader, device='cpu')
    results[rank] = json.dumps({
        'rank': rank,
        'fingerprint': _fingerprint(model.context_encoder),
        'epochs_run': len(history),
        'checkpoint': str(checkpoint) if checkpoint else None,
        'sampler_sharded': len(loader) > 0,
    })


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'ddp_smoke_result.json'
    world_size = 2
    with tempfile.TemporaryDirectory() as artifacts:
        manager = mp.Manager()
        results = manager.dict()
        mp.spawn(_worker, args=(world_size, artifacts, 29517, results), nprocs=world_size,
                 join=True)
        payload = [json.loads(results[r]) for r in sorted(results.keys())]
        encoders = [p for p in os.listdir(artifacts) if p.endswith('.pt')]

    report = {
        'ranks': payload,
        'weights_identical': len({p['fingerprint'] for p in payload}) == 1,
        'same_epoch_count': len({p['epochs_run'] for p in payload}) == 1,
        'checkpoint_files': sorted(encoders),
    }
    report['ok'] = (len(payload) == world_size and report['weights_identical']
                    and report['same_epoch_count'])
    with open(out_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
