"""Multi-GPU helpers for I-JEPA pretraining (see docs/jepa-world-model.md).

Single-process training must keep working untouched: every helper degrades to a no-op when
torch.distributed is not initialised, so the same `train()` runs on a laptop and on 8xH100.

The failure mode this module exists to prevent is a *hang*, not a crash. When ranks disagree about
control flow — one decides to early-stop, one runs an extra probe, one rebuilds a dataloader — the
others block forever in a collective and the run silently burns money. So every control-flow
decision that ranks could disagree about is funnelled through `broadcast_flag`.
"""
import datetime
import os

import torch
import torch.distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if is_distributed() else 0


def world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main():
    return rank() == 0


def local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))


def setup(backend=None, timeout_minutes=30):
    """Initialise the process group from torchrun's env vars. Returns the device to train on.

    No-op (returns cuda/cpu) when not launched under torchrun, so single-GPU runs are unaffected.
    """
    if is_distributed():
        return current_device()
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    backend = backend or ('nccl' if torch.cuda.is_available() else 'gloo')
    dist.init_process_group(backend=backend,
                            timeout=datetime.timedelta(minutes=timeout_minutes))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
    return current_device()


def current_device():
    if torch.cuda.is_available():
        return f'cuda:{local_rank()}' if is_distributed() else 'cuda'
    return 'cpu'


def barrier():
    if is_distributed():
        dist.barrier()


def all_reduce_mean(value, device):
    """Mean of a python scalar across ranks (identity when single-process)."""
    if not is_distributed():
        return float(value)
    tensor = torch.tensor([float(value)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / world_size())


def broadcast_flag(value, device, src=0):
    """Broadcast a bool from `src` so every rank takes the SAME branch.

    Early stopping, checkpointing and probe scheduling are decided on rank 0 (it owns the probe and
    the filesystem). Letting each rank decide independently is how multi-GPU jobs deadlock: rank 0
    breaks out of the epoch loop while the rest wait in the next all-reduce until the timeout.
    """
    if not is_distributed():
        return bool(value)
    tensor = torch.tensor([1 if value else 0], device=device, dtype=torch.int32)
    dist.broadcast(tensor, src=src)
    return bool(tensor.item())


def broadcast_int(value, device, src=0):
    """Broadcast an int from `src` — used for tri-state control flow (improved / not / no-probe)."""
    if not is_distributed():
        return int(value)
    tensor = torch.tensor([int(value)], device=device, dtype=torch.int64)
    dist.broadcast(tensor, src=src)
    return int(tensor.item())


def unwrap(model):
    """The underlying module, whether or not it is wrapped in DistributedDataParallel."""
    return model.module if hasattr(model, 'module') else model


def cleanup():
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()
