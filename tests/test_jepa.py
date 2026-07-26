"""CPU smoke tests for the JEPA world-model subsystem.

Skipped entirely when torch is absent (e.g. CI, which does not install the optional [jepa]
extra), so they never break the default test matrix.
"""
import pickle

import pytest

torch = pytest.importorskip("torch")

from lesnet.jepa.config import JEPAConfig  # noqa: E402
from lesnet.jepa.data import build_dataloader  # noqa: E402
from lesnet.jepa.masking import MultiBlockMaskCollator  # noqa: E402
from lesnet.jepa.modeling import IJEPA  # noqa: E402
from lesnet.jepa.preprocessing import build_transform  # noqa: E402
from lesnet.jepa.vision_transformer import build_encoder  # noqa: E402


def _tiny_config(**overrides):
    base = dict(
        image_size=96, patch_size=16, encoder="vit_tiny",
        predictor_embed_dim=96, predictor_depth=2, predictor_num_heads=3,
        num_pred_masks=2, min_keep=2, batch_size=4, num_workers=0,
        remove_hair=False, colour_constancy=False,
    )
    base.update(overrides)
    return JEPAConfig(**base)


def _batch(config):
    collator = MultiBlockMaskCollator(config)
    images = [torch.rand(3, config.image_size, config.image_size) for _ in range(config.batch_size)]
    return collator(images)


def test_mask_collator_shapes():
    config = _tiny_config()
    batch, masks_enc, masks_pred = _batch(config)
    assert batch.shape == (config.batch_size, 3, config.image_size, config.image_size)
    assert len(masks_enc) == config.num_enc_masks
    assert len(masks_pred) == config.num_pred_masks
    for mask in list(masks_enc) + list(masks_pred):
        assert mask.shape[0] == config.batch_size


def test_impossible_mask_config_raises_not_hangs():
    config = _tiny_config(image_size=64, min_keep=999)
    with pytest.raises(ValueError):
        _batch(config)


def test_collator_is_picklable_for_spawn_workers():
    # regression: no multiprocessing.Value, so it pickles cleanly to spawned dataloader workers
    pickle.dumps(MultiBlockMaskCollator(_tiny_config()))


def test_ijepa_forward_produces_scalar_loss_and_collapse_stats():
    config = _tiny_config()
    model = IJEPA(config)
    batch, masks_enc, masks_pred = _batch(config)
    loss = model(batch, masks_enc, masks_pred)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert model.last_stats["target_std"] >= 0.0 and "pred_std" in model.last_stats


def test_target_encoder_frozen_and_ema_tracks():
    config = _tiny_config()
    model = IJEPA(config)
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
    before = next(model.target_encoder.parameters()).clone()
    with torch.no_grad():
        for param in model.context_encoder.parameters():
            param.add_(1.0)
    model.update_target(0.9)
    assert not torch.equal(before, next(model.target_encoder.parameters()))


def test_weight_decay_param_groups_exclude_norms_and_mask_token():
    from lesnet.jepa.engine import _param_groups
    model = IJEPA(_tiny_config())
    decay, no_decay = _param_groups(model)
    assert all(p.ndim >= 2 for p in decay)
    # identity check: `tensor in list` would trigger elementwise == across mismatched shapes
    assert any(p is model.predictor.mask_token for p in no_decay)   # excluded despite ndim == 3
    assert any(p.ndim == 1 for p in no_decay)              # norms / biases


def test_size_family_aliases_resolve_and_build():
    from lesnet.jepa.vision_transformer import ENCODER_CHOICES, SIZE_ALIASES, resolve_encoder
    assert resolve_encoder("medium") == "vit_base" and resolve_encoder("xlarge") == "vit_huge"
    assert resolve_encoder("vit_small") == "vit_small"          # vit_* passes through
    for alias in SIZE_ALIASES:
        assert alias in ENCODER_CHOICES
    # a friendly alias builds the same encoder as its vit_* key, and config.encoder_dims agrees
    tiny_alias = build_encoder(_tiny_config(encoder="tiny", image_size=32))
    tiny_key = build_encoder(_tiny_config(encoder="vit_tiny", image_size=32))
    assert sum(p.numel() for p in tiny_alias.parameters()) == sum(p.numel() for p in tiny_key.parameters())
    assert JEPAConfig(encoder="medium").encoder_dims() == (768, 12, 12)


def test_encoder_forward_guard_on_wrong_resolution():
    encoder = build_encoder(_tiny_config(image_size=96))
    encoder(torch.rand(1, 3, 96, 96))                      # correct resolution is fine
    with pytest.raises(ValueError):
        encoder(torch.rand(1, 3, 128, 128))                # mismatched grid -> loud error


def test_dataloader_synthetic_batch():
    config = _tiny_config()
    loader = build_dataloader(config, synthetic_samples=8)
    batch, masks_enc, masks_pred = next(iter(loader))
    assert batch.shape[0] == config.batch_size
    assert len(masks_pred) == config.num_pred_masks


def test_shared_preprocessing_transform_shape():
    from PIL import Image
    import numpy as np
    config = _tiny_config(colour_constancy=True)           # numpy-only, runs without opencv
    image = Image.fromarray(np.random.randint(0, 255, (120, 100, 3), dtype=np.uint8))
    tensor = build_transform(config, train=True)(image)
    assert tensor.shape == (3, config.image_size, config.image_size)
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0   # [0,1], not ImageNet-normed


def test_resolution_schedule_retargets_encoder_exactly():
    """Low-res-first training: the sin-cos grids regenerate, weights carry over, forward works."""
    from lesnet.jepa.vision_transformer import set_image_size
    config = _tiny_config(image_size=96)
    model = IJEPA(config)
    before = model.context_encoder.blocks[0].attn.qkv.weight.clone()

    num_patches = set_image_size(model, 64, config.patch_size)
    assert num_patches == (64 // 16) ** 2
    tokens = model.context_encoder(torch.rand(1, 3, 64, 64))
    assert tokens.shape[1] == num_patches
    assert torch.equal(before, model.context_encoder.blocks[0].attn.qkv.weight)  # weights untouched

    set_image_size(model, 96, config.patch_size)                   # and back again
    assert model.context_encoder(torch.rand(1, 3, 96, 96)).shape[1] == (96 // 16) ** 2


def test_min_keep_scales_with_the_grid_so_low_res_stages_are_maskable():
    """min_keep is an absolute patch count; the mask scales are fractional. Without scaling it
    with the grid, the 128px stage of the schedule cannot sample a legal target block at all."""
    config = _tiny_config(image_size=224, min_keep=10)
    assert MultiBlockMaskCollator(config).min_keep == 10                 # unchanged at config res
    low = MultiBlockMaskCollator(config, image_size=128)
    assert low.min_keep == round(10 * 64 / 196)
    smallest_target_block = int(config.pred_mask_scale[0] * low.height * low.width)
    assert smallest_target_block > low.min_keep                          # a legal block exists
    images = [torch.rand(3, 128, 128) for _ in range(2)]
    batch, masks_enc, masks_pred = low(images)                           # and sampling succeeds
    assert batch.shape == (2, 3, 128, 128)
    assert len(masks_pred) == config.num_pred_masks


def test_resolution_for_epoch_follows_schedule():
    from lesnet.jepa.engine import resolution_for_epoch
    config = _tiny_config(image_size=224, epochs=10, resize_schedule=((0.0, 128), (0.6, 224)))
    assert resolution_for_epoch(config, 0) == 128
    assert resolution_for_epoch(config, 5) == 128
    assert resolution_for_epoch(config, 6) == 224
    assert resolution_for_epoch(_tiny_config(image_size=224), 0) == 224   # empty schedule


def test_parse_resize_schedule():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'commands' / 'run_pretrain_jepa.py'
    spec = importlib.util.spec_from_file_location('run_pretrain_jepa', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._parse_schedule('0:128,0.7:224') == ((0.0, 128), (0.7, 224))
    assert module._parse_schedule('') == ()


def test_grad_accumulation_matches_large_batch_step():
    """Accumulating N micro-batches must equal one N-times-larger batch, or the effective batch
    size (and therefore the scaled LR) is a lie."""
    from lesnet.jepa.engine import _param_groups
    config = _tiny_config(batch_size=2)
    torch.manual_seed(0)
    model = IJEPA(config)
    decay, no_decay = _param_groups(model)
    params = decay + no_decay
    batches = [_batch(config), _batch(config)]

    def grads(accumulate):
        model.zero_grad(set_to_none=True)
        for images, enc, pred in batches:
            loss = model(images, enc, pred)
            (loss / len(batches) if accumulate else loss).backward()
            if not accumulate:
                break
        return [p.grad.clone() for p in params if p.grad is not None]

    accumulated = grads(True)
    model.zero_grad(set_to_none=True)
    total = sum(model(i, e, p) for i, e, p in batches) / len(batches)
    total.backward()
    single = [p.grad.clone() for p in params if p.grad is not None]
    assert len(accumulated) == len(single)
    for a, b in zip(accumulated, single):
        assert torch.allclose(a, b, atol=1e-5)


def test_warmup_longer_than_half_the_run_is_rejected():
    """The failure this guards against actually shipped: a 300-epoch schedule with 40 warmup
    epochs that only got 21 epochs of wall-clock never left warmup, so the SSL loss rose the
    whole run and loss-based selection saved the epoch-0 encoder."""
    from lesnet.jepa.engine import train
    config = _tiny_config(epochs=4, warmup_epochs=4)
    loader = build_dataloader(config, synthetic_samples=8)
    with pytest.raises(ValueError, match='warmup_epochs'):
        train(config, loader, device='cpu')


def test_training_state_round_trips_best_metrics(tmp_path):
    """Bests must survive a resume, or the first score after resuming always 'improves' and
    overwrites a better encoder — the chained-session failure mode."""
    from lesnet.jepa.engine import _load_training_state, _param_groups, save_training_state
    config = _tiny_config(artifacts_dir=str(tmp_path))
    model = IJEPA(config)
    decay, no_decay = _param_groups(model)
    optimizer = torch.optim.AdamW([{'params': decay}, {'params': no_decay}], lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    save_training_state(model, optimizer, scaler, 7, 3, config, tmp_path,
                        best_loss=0.25, best_probe=0.91)

    step, epoch, best_loss, best_probe = _load_training_state(
        IJEPA(config), optimizer, scaler, tmp_path / 'training_state.pt')
    assert (step, epoch) == (7, 4)
    assert best_loss == pytest.approx(0.25) and best_probe == pytest.approx(0.91)


def test_manifest_ids_cover_source_prefixed_filenames(tmp_path):
    """build_dataset writes '<source>_<archive id>.jpg'; archives key on the bare id. Both forms
    must land in the exclusion set or the held-out test images leak into pretraining."""
    from lesnet.jepa.data import manifest_image_ids
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text(
        'image_path,source_dataset,raw_label,group_id,fitzpatrick,anatomical_site,age,sex,split,'
        'triage_bucket,diagnosis\n'
        'imgs/isic_ISIC_0000001.jpg,isic,mel,g1,,torso,50,male,test,malignant,melanoma\n'
        'imgs/isic_ISIC_0000002.jpg,isic,nv,g2,,torso,50,male,train,benign,nevus\n',
        encoding='utf-8')
    ids = manifest_image_ids(manifest, ('test',))
    assert 'isic_ISIC_0000001' in ids and 'ISIC_0000001' in ids
    assert 'ISIC_0000002' not in ids                                # train rows stay in SSL


def test_directory_loader_excludes_held_out_ids(tmp_path):
    """A raw image directory carries no splits, so held-out ids must be subtracted by filename —
    and a list that matches nothing must fail loudly rather than leak the test set into SSL."""
    import numpy as np
    from PIL import Image
    images = tmp_path / 'images'
    images.mkdir()
    for name in ('ISIC_0000001', 'ISIC_0000002', 'ISIC_0000003'):
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            images / f'{name}.jpg')
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text(
        'image_path,source_dataset,raw_label,group_id,fitzpatrick,anatomical_site,age,sex,split,'
        'triage_bucket,diagnosis\n'
        'x/isic_ISIC_0000001.jpg,isic,mel,g1,,torso,50,male,test,malignant,melanoma\n'
        'x/isic_ISIC_0000002.jpg,isic,nv,g2,,torso,50,male,train,benign,nevus\n', encoding='utf-8')

    config = _tiny_config(image_size=32, batch_size=2)
    loader = build_dataloader(config, root=str(images), exclude_manifest=str(manifest))
    assert len(loader.dataset) == 2                       # the one test id dropped

    unmatched = tmp_path / 'unmatched.csv'
    unmatched.write_text(
        'image_path,source_dataset,raw_label,group_id,fitzpatrick,anatomical_site,age,sex,split,'
        'triage_bucket,diagnosis\n'
        'x/isic_NOT_AN_ID.jpg,isic,mel,g1,,torso,50,male,test,malignant,melanoma\n', encoding='utf-8')
    with pytest.raises(ValueError, match='none matched'):
        build_dataloader(config, root=str(images), exclude_manifest=str(unmatched))


def test_hdf5_dataset_reads_and_excludes(tmp_path):
    h5py = pytest.importorskip("h5py")
    import io
    import numpy as np
    from PIL import Image
    from lesnet.jepa.data import Hdf5ImageDataset

    path = tmp_path / 'images.hdf5'
    with h5py.File(path, 'w') as handle:
        for name in ('ISIC_0000001', 'ISIC_0000002', 'ISIC_0000003'):
            buffer = io.BytesIO()
            Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(
                buffer, format='JPEG')
            handle.create_dataset(name, data=np.frombuffer(buffer.getvalue(), dtype=np.uint8))

    config = _tiny_config(image_size=32)
    dataset = Hdf5ImageDataset(path, config, exclude_ids={'ISIC_0000001'})
    assert len(dataset) == 2 and dataset.num_excluded == 1
    assert dataset[0].shape == (3, 32, 32)

    with pytest.raises(ValueError, match='none matched'):           # silent-leak guard
        Hdf5ImageDataset(path, config, exclude_ids={'NOT_AN_ID'})


def test_pretrained_loader_refuses_non_commercial_weights():
    """LesNet is MPL-2.0 and must stay commercially usable, so the loader is an allow-list, not a
    convenience wrapper — a non-commercial checkpoint would poison every redistribution."""
    from lesnet.jepa.pretrained import ALLOWED, load_pretrained
    for banned in ('panderm_vitl16', 'DermLIP_PanDerm', 'dinov3_vitb16', 'ijepa_vith14',
                   'celldino_vitb'):
        with pytest.raises(ValueError, match='Refusing to load'):
            load_pretrained(None, banned)
    with pytest.raises(ValueError, match='Unknown pretrained weights'):
        load_pretrained(None, 'some_random_checkpoint')
    assert all(spec[3] == 'Apache-2.0' for spec in ALLOWED.values())


def test_progress_stream_round_trips_into_the_dashboard(tmp_path):
    """The dashboard tails progress.jsonl while training writes it, so a half-written trailing
    line must be skipped rather than crash the reader."""
    import importlib.util
    from pathlib import Path

    from lesnet.jepa.progress import ProgressWriter

    writer = ProgressWriter(tmp_path, {'encoder': 'vit_small', 'epochs': 3})
    writer.step(0, 10, 1, 100, loss=0.5, lr=1e-4, target_std=0.3, images_seen=320)
    writer.epoch(0, 0.42, best=0.42, no_improve=0, probe={'roc_auc': 0.91})
    writer.finish('completed', tmp_path / 'context_encoder.pt')
    with open(tmp_path / 'progress.jsonl', 'a', encoding='utf-8') as handle:
        handle.write('{"kind": "step", "loss": 0.4')      # torn line, mid-write

    path = Path(__file__).resolve().parents[1] / 'commands' / 'train_dashboard.py'
    spec = importlib.util.spec_from_file_location('train_dashboard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = module._read_progress(tmp_path / 'progress.jsonl')
    assert payload['run']['encoder'] == 'vit_small'
    assert len(payload['steps']) == 1 and payload['steps'][0]['imgs_per_sec'] > 0
    assert payload['epochs'][0]['probe']['roc_auc'] == 0.91
    assert payload['finished'] == 'completed'
    assert module._resolve(tmp_path) == tmp_path


def test_position_embedding_survives_a_state_dict_round_trip():
    """A LEARNED position embedding must round-trip through state_dict(). It used to be a
    non-persistent buffer, so saving dropped it, reloading regenerated a sin-cos grid, and the
    exported encoder produced embeddings with cosine 0.085 to the model we thought we had saved —
    pretrained weights paired with the wrong positional signal."""
    config = _tiny_config(image_size=96)
    encoder = build_encoder(config)
    learned = torch.randn_like(encoder.pos_embed)
    encoder.register_buffer('pos_embed', learned, persistent=True)
    encoder.pos_embed_pretrained = True

    assert 'pos_embed' in encoder.state_dict(), 'pos_embed missing from state_dict'
    restored = build_encoder(config)
    restored.load_state_dict(encoder.state_dict())
    assert torch.equal(restored.pos_embed, learned)

    encoder.eval()
    restored.eval()
    image = torch.rand(1, 3, 96, 96)
    with torch.no_grad():
        assert torch.allclose(encoder(image), restored(image), atol=1e-6)


def test_pretrained_position_embedding_resamples_to_our_grid():
    """DINOv2 learns its position embedding and uses patch 14 + a cls token; ours is sin-cos with
    neither. The learned grid must be resampled (not regenerated) or the pretrained blocks get a
    position signal they were never trained against."""
    from lesnet.jepa.pretrained import _resize_pos_embed
    source = torch.randn(1, 16 * 16, 384)
    resized = _resize_pos_embed(source, 14)
    assert resized.shape == (1, 14 * 14, 384)
    assert torch.equal(_resize_pos_embed(source, 16), source)      # no-op when already correct


def test_ddp_two_ranks_stay_in_lockstep(tmp_path):
    """Multi-GPU bugs hang rather than crash, and only above one rank — so run two real ranks
    (gloo/CPU) and assert the three things that cost money on a cluster: the job terminates,
    gradients actually sync, and only rank 0 writes to disk."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).with_name('ddp_smoke.py')
    result_path = tmp_path / 'ddp.json'
    proc = subprocess.run(
        [sys.executable, str(script), str(result_path)],
        cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True,
        timeout=600)                       # a deadlock shows up here as a timeout, which is a fail
    assert result_path.exists(), f'ddp smoke produced no report\n{proc.stdout}\n{proc.stderr}'
    report = json.loads(result_path.read_text())
    assert report['weights_identical'], 'ranks diverged: gradients are not being all-reduced'
    assert report['same_epoch_count'], 'ranks ran different epoch counts: control flow desynced'
    assert [r['checkpoint'] for r in report['ranks']].count(None) == len(report['ranks']) - 1
    assert report['ok']


def test_onnx_export_parity(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from lesnet.jepa.engine import save_encoder
    from lesnet.jepa.export import export
    config = _tiny_config()
    config.artifacts_dir = str(tmp_path)
    checkpoint = save_encoder(IJEPA(config), config, tmp_path)
    report = export(checkpoint, tmp_path / "export", measure_rss=False)
    assert (tmp_path / "export" / "encoder_int8.onnx").exists()
    assert report["onnx_parity_max_abs_diff"] < 1e-3      # torch vs onnxruntime agree


def _write_probe_manifest(tmp_path, config, per_group=4):
    from PIL import Image
    import numpy as np
    from lesnet.data.records import LesionRecord, save_manifest
    images_dir = tmp_path / "img"
    images_dir.mkdir()
    records, idx = [], 0
    for split in ("train", "test"):
        for bucket in ("benign", "malignant"):
            for _ in range(per_group):
                path = images_dir / f"{idx}.png"
                Image.fromarray(
                    np.random.randint(0, 255, (config.image_size, config.image_size, 3), dtype=np.uint8)
                ).save(path)
                records.append(LesionRecord(
                    image_path=str(path), source_dataset="synthetic", raw_label=bucket,
                    group_id=f"patient-{idx}", fitzpatrick=(idx % 3) + 1,
                    split=split, triage_bucket=bucket,
                ))
                idx += 1
    manifest = tmp_path / "manifest.csv"
    save_manifest(records, str(manifest))
    return str(manifest)


def test_probe_manifest_leakage_free_metrics(tmp_path):
    pytest.importorskip("sklearn")
    from lesnet.jepa.probe import run_probe
    config = _tiny_config()
    manifest = _write_probe_manifest(tmp_path, config)
    encoder = build_encoder(config)
    metrics = run_probe(encoder, manifest, config, batch_size=4)
    assert metrics["source"] == "manifest"
    for key in ("sensitivity", "specificity", "roc_auc", "operating_threshold", "fitzpatrick_sensitivity"):
        assert key in metrics
    assert 0.0 <= metrics["sensitivity"] <= 1.0
