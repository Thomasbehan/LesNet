"""Append-only training-progress stream for the live dashboard.

TensorBoard event files need TensorBoard to read them. This writes a plain JSONL line per logged
step and per epoch, so `commands/train_dashboard.py` (or anything else) can tail it with no
dependencies and no parsing of protobufs. One line per event, flushed immediately — a dashboard
that lags behind the run is worse than no dashboard.
"""
import json
import time
from pathlib import Path


class ProgressWriter:
    """Writes <artifacts_dir>/progress.jsonl. Safe to construct when tensorboard is absent."""

    def __init__(self, artifacts_dir, run_meta=None):
        self.path = Path(artifacts_dir) / 'progress.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self._seen = 0
        if run_meta:
            self.write('run', **run_meta)

    def write(self, kind, **fields):
        record = {'kind': kind, 'wall': round(time.time() - self.started, 2), **fields}
        try:
            with open(self.path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
        except OSError:                       # a dashboard must never be able to kill a long run
            pass
        return record

    def step(self, epoch, step, update, total_updates, loss, lr, target_std, images_seen):
        """Per-log-step scalars plus a throughput estimate the dashboard can show as img/s."""
        elapsed = max(time.time() - self.started, 1e-6)
        self._seen = images_seen
        return self.write('step', epoch=epoch, step=step, update=update,
                          total_updates=total_updates, loss=round(float(loss), 6),
                          lr=float(lr), target_std=round(float(target_std), 5),
                          images_seen=images_seen, imgs_per_sec=round(images_seen / elapsed, 1))

    def epoch(self, epoch, epoch_loss, best, no_improve, probe=None):
        return self.write('epoch', epoch=epoch, epoch_loss=round(float(epoch_loss), 6),
                          best=None if best is None else round(float(best), 6),
                          no_improve=no_improve, probe=probe)

    def finish(self, reason, checkpoint=None):
        return self.write('finish', reason=reason, checkpoint=str(checkpoint) if checkpoint else None)
