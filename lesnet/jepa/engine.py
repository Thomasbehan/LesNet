"""I-JEPA pretraining loop (see docs/jepa-world-model.md).

AdamW (weight decay excluded from norms/biases/mask-token) with cosine LR + weight-decay
schedules (linear warmup), a linearly-ramped target-encoder EMA momentum, LR scaled to the
effective batch size, optional mixed precision, gradient clipping, TensorBoard logging with a
representation-collapse monitor, and resumable checkpoints. Only the context encoder is needed
downstream; save_encoder writes the deploy checkpoint, save_training_state the resume state.
"""
import contextlib
import json
import math
import os
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from lesnet.jepa import distributed as dist_utils
from lesnet.jepa.modeling import IJEPA
from lesnet.jepa.progress import ProgressWriter

_AMP_DTYPES = {'bf16': torch.bfloat16, 'fp16': torch.float16}


def _make_writer(config):
    """TensorBoard SummaryWriter at <artifacts_dir>/tb, or None if unavailable/disabled."""
    if not config.tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        warnings.warn("tensorboard not installed; skipping logging. `pip install tensorboard`.")
        return None
    return SummaryWriter(log_dir=str(Path(config.artifacts_dir) / 'tb'))


def cosine_value(step, total_steps, base, final, warmup_steps=0, start=0.0):
    """Linear warmup start->base over `warmup_steps`, then cosine base->final."""
    if warmup_steps and step < warmup_steps:
        return start + (base - start) * step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * progress))


def _param_groups(model):
    """Split trainable params: weight decay on matrices only, none on norms/biases/mask-token."""
    decay, no_decay = [], []
    named = list(model.context_encoder.named_parameters()) + list(model.predictor.named_parameters())
    for name, param in named:
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or 'mask_token' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def save_encoder(model, config, directory, tag='context_encoder'):
    """Persist the context encoder weights + config (the artifact consumed by export/transfer)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{tag}.pt'
    torch.save({'state_dict': model.context_encoder.state_dict(), 'config': asdict(config)}, path)
    with open(directory / 'jepa_config.json', 'w', encoding='utf-8') as handle:
        json.dump(asdict(config), handle, indent=2)
    return path


def save_training_state(model, optimizer, scaler, step, epoch, config, directory,
                        best_loss=None, best_probe=None):
    """Atomically persist full state for --resume (encoders, predictor, optimizer, RNG, bests)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'training_state.pt'
    tmp = directory / 'training_state.pt.tmp'
    torch.save({
        'best_loss': best_loss, 'best_probe': best_probe,
        'context_encoder': model.context_encoder.state_dict(),
        'target_encoder': model.target_encoder.state_dict(),
        'predictor': model.predictor.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'step': step, 'epoch': epoch,
        'torch_rng': torch.get_rng_state(),
        'numpy_rng': np.random.get_state(),
        'config': asdict(config),
    }, tmp)
    os.replace(tmp, path)
    return path


def _load_training_state(model, optimizer, scaler, resume_path):
    state = torch.load(resume_path, map_location='cpu', weights_only=False)
    model.context_encoder.load_state_dict(state['context_encoder'])
    model.target_encoder.load_state_dict(state['target_encoder'])
    model.predictor.load_state_dict(state['predictor'])
    optimizer.load_state_dict(state['optimizer'])
    scaler.load_state_dict(state['scaler'])
    torch.set_rng_state(state['torch_rng'])
    np.random.set_state(state['numpy_rng'])
    # carry the bests across sessions, else the first post-resume score always "improves" and
    # overwrites a better saved encoder
    best_loss = state.get('best_loss')
    best_probe = state.get('best_probe')
    return (state['step'], state['epoch'] + 1,        # resume at the next epoch
            float('inf') if best_loss is None else best_loss,
            -float('inf') if best_probe is None else best_probe)


def resolution_for_epoch(config, epoch):
    """Image size for `epoch` under config.resize_schedule (empty schedule -> config.image_size)."""
    if not config.resize_schedule:
        return config.image_size
    progress = epoch / max(config.epochs, 1)
    size = config.image_size
    for fraction, value in config.resize_schedule:
        if progress >= float(fraction):
            size = int(value)
    return size


def stage_batching(config, image_size):
    """(micro_batch, grad_accum) for `image_size`, holding the *token* budget constant.

    config.batch_size/grad_accum_steps describe the reference resolution (config.image_size). A
    128px stage costs ~1/3 the memory per image of a 224px one, so reusing the 224px batch there
    leaves the GPU idle — measured 45 -> 156 img/s for ViT-L on an 8 GB card by scaling the batch
    up. Accumulation is scaled down in step so the *effective* batch (and thus the LR the linear
    scaling rule implies) stays put.
    """
    reference_tokens = max((config.image_size // config.patch_size) ** 2, 1)
    tokens = max((image_size // config.patch_size) ** 2, 1)
    effective = config.batch_size * max(config.grad_accum_steps, 1)
    batch = max(round(config.batch_size * reference_tokens / tokens), 1)
    if config.max_batch_size:
        batch = min(batch, config.max_batch_size)
    batch = min(batch, effective)
    return batch, max(round(effective / batch), 1)


def _run_probe(model, config, device, image_size, epoch, writer):
    """In-training linear probe on the frozen context encoder; returns metrics or None."""
    from lesnet.jepa.probe import quick_probe
    try:
        metrics = quick_probe(model.context_encoder, config.probe_manifest, config, device=device,
                              image_size=image_size, subset=config.probe_subset,
                              batch_size=max(config.batch_size, 32), num_workers=config.num_workers)
    except Exception as error:                    # a probe failure must never kill a long run
        warnings.warn(f'in-training probe failed at epoch {epoch}: {error!r}')
        return None
    print(f'[epoch {epoch}] probe auc {metrics.get("roc_auc")} '
          f'sens {metrics.get("sensitivity")} spec {metrics.get("specificity")}', flush=True)
    if writer is not None:
        for key in ('roc_auc', 'sensitivity', 'specificity'):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                writer.add_scalar(f'probe/{key}', value, epoch)
    return metrics


def train(config, dataloader, device=None, resume=None, dataloader_factory=None):
    """Pretrain I-JEPA. `dataloader_factory(image_size)` enables the resolution schedule; without
    it the fixed `dataloader` is used at config.image_size throughout."""
    device = device or dist_utils.setup()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    main = dist_utils.is_main()

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True   # no-op pre-Ampere, free speed on Ampere+
        torch.backends.cudnn.allow_tf32 = True

    model = IJEPA(config).to(device)
    core = model                                  # unwrapped: EMA, stats and saving use this
    if dist_utils.is_distributed():
        # Only the context encoder + predictor require grad, so the frozen target encoder is
        # naturally excluded from the gradient buckets.
        # device_ids must reflect where the MODULE lives, not merely whether CUDA exists: a
        # CPU-placed module with device_ids set is rejected outright by DDP.
        placement = torch.device(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[placement.index or 0] if placement.type == 'cuda' else None,
            find_unused_parameters=False)
        print(f'[rank {dist_utils.rank()}/{dist_utils.world_size()}] DDP ready on {device}',
              flush=True)
    if config.compile_model:
        try:
            model = torch.compile(model)
        except Exception as error:                     # compile is an optimisation, never a blocker
            warnings.warn(f'torch.compile unavailable ({error!r}); running eager.')
    decay, no_decay = _param_groups(core)
    optimizer = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': config.weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=config.start_lr,
    )
    params = decay + no_decay

    steps_per_epoch = max(len(dataloader), 1)
    # Schedules advance per OPTIMISER UPDATE, not per micro-batch: the resolution schedule changes
    # the micro-batch (and therefore the accumulation count) mid-run, and a micro-step-keyed cosine
    # would silently change shape when it did. Updates per epoch depend only on the effective batch.
    effective_batch = config.batch_size * max(config.grad_accum_steps, 1)
    dataset_size = len(getattr(dataloader, 'dataset', [])) or steps_per_epoch * config.batch_size
    updates_per_epoch = max(dataset_size // effective_batch, 1)
    total_updates = updates_per_epoch * config.epochs
    warmup_updates = updates_per_epoch * config.warmup_epochs
    if config.warmup_epochs >= config.epochs * 0.5:
        # Guard rail learned the hard way: a 300-epoch schedule with 40 warmup epochs that only
        # gets 21 epochs of wall-clock never leaves warmup — LR keeps rising, the SSL loss keeps
        # rising with it, and loss-based selection then ships the epoch-0 encoder.
        raise ValueError(
            f'warmup_epochs={config.warmup_epochs} is >= half of epochs={config.epochs}. Set the '
            f'schedule to the budget you will actually run (warmup ~5-10% of epochs), otherwise '
            f'training never reaches peak LR and never anneals.')
    amp_dtype = _AMP_DTYPES.get(config.mixed_precision) if device == 'cuda' else None
    use_scaler = config.mixed_precision == 'fp16' and device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    ema_start, ema_end = config.ema_momentum
    writer = _make_writer(config) if main else None
    progress_log = ProgressWriter(config.artifacts_dir, {
        'encoder': config.encoder, 'image_size': config.image_size,
        'epochs': config.epochs, 'pretrained': getattr(config, 'pretrained', ''),
    }) if main else None
    log_every = max(config.log_every, 1)

    # LR scaled to the effective batch size (linear scaling rule) — effective, not micro
    lr_scale = effective_batch / config.lr_reference_batch if config.scale_lr else 1.0
    peak_lr, start_lr, final_lr = config.lr * lr_scale, config.start_lr * lr_scale, config.final_lr * lr_scale

    step, start_epoch = 0, 0
    best_loss, best_probe = float('inf'), -float('inf')
    if resume:
        step, start_epoch, best_loss, best_probe = _load_training_state(
            model, optimizer, scaler, resume)
        print(f'Resumed from {resume} at epoch {start_epoch}, step {step} '
              f'(best loss {best_loss:.4f}, best probe {best_probe:.4f}).')
    update = start_epoch * updates_per_epoch

    history = []
    epochs_no_improve, checkpoint_path, probe_metrics = 0, None, None
    select_on_probe = config.select_on == 'probe' and bool(config.probe_manifest)
    # The caller builds the first loader at the first scheduled resolution, so start there rather
    # than at config.image_size — otherwise epoch 0 rebuilds an identical loader (and re-lists the
    # whole HDF5) for nothing.
    images_per_step = config.batch_size * dist_utils.world_size()
    current_size = resolution_for_epoch(config, start_epoch)
    _, accum = stage_batching(config, current_size)
    if current_size != config.image_size:      # model is built at config.image_size; retarget it
        from lesnet.jepa.vision_transformer import set_image_size
        set_image_size(core, current_size, config.patch_size)
        if main:
            print(f'starting at {current_size}px '
                  f'({(current_size // config.patch_size) ** 2} tokens), accum {accum}', flush=True)
    start_wall = time.time()
    for epoch in range(start_epoch, config.epochs):
        # resolution schedule: retarget the sin-cos grids and rebuild the input pipeline
        epoch_size = resolution_for_epoch(config, epoch)
        if epoch_size != current_size and dataloader_factory is not None:
            from lesnet.jepa.vision_transformer import set_image_size
            set_image_size(core, epoch_size, config.patch_size)
            stage_batch, accum = stage_batching(config, epoch_size)
            dataloader = dataloader_factory(epoch_size, stage_batch)
            steps_per_epoch = max(len(dataloader), 1)
            current_size = epoch_size
            if main:
                print(f'[epoch {epoch}] resolution -> {epoch_size}px '
                      f'({(epoch_size // config.patch_size) ** 2} tokens) '
                      f'batch {stage_batch} x accum {accum}', flush=True)
        # reshuffle each rank's shard differently every epoch
        if hasattr(getattr(dataloader, 'sampler', None), 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

        model.train()
        running, micro = 0.0, 0
        grad_norm, momentum = 0.0, ema_start
        optimizer.zero_grad(set_to_none=True)
        for images, masks_enc, masks_pred in dataloader:
            images = images.to(device, non_blocking=True)
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]

            lr = cosine_value(update, total_updates, peak_lr, final_lr, warmup_updates, start_lr)
            weight_decay = cosine_value(update, total_updates, config.weight_decay,
                                        config.final_weight_decay)
            for index, group in enumerate(optimizer.param_groups):
                group['lr'] = lr
                if index == 0:                       # decay group only; no_decay stays 0.0
                    group['weight_decay'] = weight_decay

            # .item() on the collapse stats syncs the GPU; only pay it on logging steps
            core.stats_enabled = (step % log_every == 0)
            # DDP all-reduces gradients on every backward by default. During accumulation only
            # the LAST micro-step needs syncing; no_sync() on the others cuts the interconnect
            # traffic by `accum` and is the difference between scaling to 8 GPUs and not.
            syncing = (micro + 1) % accum == 0
            sync_ctx = (model.no_sync() if (not syncing and hasattr(model, 'no_sync'))
                        else contextlib.nullcontext())
            with sync_ctx:
                if amp_dtype is not None:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        loss = model(images, masks_enc, masks_pred)
                else:
                    loss = model(images, masks_enc, masks_pred)

                scaled = loss / accum
                if use_scaler:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

            micro += 1
            if micro % accum == 0:                   # one optimiser update per accumulation window
                if use_scaler:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                progress = min(update / max(total_updates - 1, 1), 1.0)  # clamp against overshoot
                momentum = ema_start + (ema_end - ema_start) * progress
                core.update_target(momentum)

            running += loss.item()
            if step % log_every == 0:
                stats = core.last_stats
                if main:
                    print(f'epoch {epoch} step {step} update {update}/{total_updates} '
                          f'loss {loss.item():.4f} lr {lr:.2e} '
                          f'target_std {stats.get("target_std", 0):.4f}', flush=True)
                if progress_log is not None:
                    progress_log.step(epoch, step, update, total_updates, loss.item(), lr,
                                  stats.get('target_std', 0.0),
                                  images_seen=step * images_per_step)
                if writer is not None and main:
                    writer.add_scalar('train/loss', loss.item(), step)
                    writer.add_scalar('train/lr', lr, step)
                    writer.add_scalar('train/weight_decay', weight_decay, step)
                    writer.add_scalar('train/ema_momentum', momentum, step)
                    writer.add_scalar('train/grad_norm', float(grad_norm), step)
                    writer.add_scalar('collapse/target_std', stats.get('target_std', 0.0), step)
                    writer.add_scalar('collapse/pred_std', stats.get('pred_std', 0.0), step)
            step += 1

        # every rank's shard contributes to the epoch mean, else ranks disagree on "improved"
        epoch_loss = dist_utils.all_reduce_mean(running / steps_per_epoch, device)
        history.append(epoch_loss)
        elapsed = time.time() - start_wall

        # optional downstream read-out; when select_on='probe' it drives checkpointing + patience.
        # Rank 0 alone runs it (it owns the probe data and the filesystem) and the resulting
        # decision is broadcast, so no rank can branch differently and strand the others.
        if config.probe_every and config.probe_manifest and (epoch + 1) % config.probe_every == 0:
            probe_metrics = (_run_probe(core, config, device, current_size, epoch, writer)
                             if main else None)

        if select_on_probe:
            score = (probe_metrics or {}).get('roc_auc')
            if score is None:
                improved = None               # no fresh probe this epoch: hold patience steady
            else:
                improved = score > best_probe + config.early_stop_min_delta
                if improved:
                    best_probe = score
                probe_metrics = None          # consume, so the next epoch does not re-count it
        else:
            improved = epoch_loss < best_loss - config.early_stop_min_delta

        # Tri-state, decided on rank 0 and broadcast: 1 improved, 0 not, 2 no fresh probe.
        code = 2 if improved is None else (1 if improved else 0)
        improved = {1: True, 0: False, 2: None}[dist_utils.broadcast_int(code, device)]
        if improved:                          # keep the BEST encoder, not merely the last
            epochs_no_improve = 0
            if main:
                checkpoint_path = save_encoder(core, config, config.artifacts_dir)
        elif improved is not None:
            epochs_no_improve += 1
        best_loss = min(best_loss, epoch_loss)
        selected = f'probe_auc {best_probe:.4f}' if select_on_probe else f'best {best_loss:.4f}'
        if main:
            print(f'[epoch {epoch}] mean loss {epoch_loss:.4f} {selected} '
                  f'no_improve {epochs_no_improve} elapsed {elapsed:.0f}s', flush=True)
        if progress_log is not None:
            progress_log.epoch(epoch, epoch_loss, best_probe if select_on_probe else best_loss,
                           epochs_no_improve,
                           probe={'roc_auc': (probe_metrics or {}).get('roc_auc')}
                           if probe_metrics else None)
        if writer is not None and main:
            writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
            writer.add_scalar('train/best_loss', best_loss, epoch)
        if config.checkpoint_every and (epoch + 1) % config.checkpoint_every == 0 and main:
            save_training_state(core, optimizer, scaler, step, epoch, config,
                                config.artifacts_dir, best_loss, best_probe)

        # never early-stop mid-schedule: an encoder stopped in the 128px stage has never seen the
        # deployment resolution. The wall-clock budget still applies (it is a hard session limit).
        at_final_resolution = current_size == config.image_size
        stop_reason = None
        if (config.early_stopping and at_final_resolution
                and epochs_no_improve >= config.early_stop_patience):
            stop_reason = f'early stop: no improvement for {epochs_no_improve} epochs'
        elif config.max_train_seconds and elapsed > config.max_train_seconds:
            stop_reason = f'time budget reached ({elapsed:.0f}s > {config.max_train_seconds}s)'
        # THE deadlock guard: all ranks must leave the epoch loop together. Rank 0 decides.
        stopping = dist_utils.broadcast_flag(stop_reason is not None, device)
        if stopping:
            if main:
                print(stop_reason or 'stopping', flush=True)
                save_training_state(core, optimizer, scaler, step, epoch, config,
                                    config.artifacts_dir, best_loss, best_probe)
            dist_utils.barrier()
            break

    if main:
        if checkpoint_path is None:           # never improved (degenerate) -> save last
            checkpoint_path = save_encoder(core, config, config.artifacts_dir)
        save_training_state(core, optimizer, scaler, step, epoch, config,
                            config.artifacts_dir, best_loss, best_probe)
    if progress_log is not None:
        progress_log.finish(stop_reason or 'completed', checkpoint_path)
    if writer is not None and main:
        writer.close()
    dist_utils.cleanup()
    return core, history, checkpoint_path
