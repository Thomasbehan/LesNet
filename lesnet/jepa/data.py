"""Unlabeled dermoscopy dataset + loader for I-JEPA pretraining (see docs/jepa-world-model.md).

Sources images from a recursive glob, a manifest.csv, or an HDF5 archive bundle. The manifest and
HDF5 paths both support EXCLUDING the held-out 'test' images from self-supervised pretraining —
otherwise probe/eval images leak into SSL and the reported numbers are transductive.
Preprocessing follows the shared transfer contract (jepa/preprocessing.py).
"""
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from lesnet.jepa.masking import MultiBlockMaskCollator
from lesnet.jepa.preprocessing import build_transform

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _glob_images(root):
    paths = sorted(p for p in Path(root).rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f'No images found under {root} (extensions: {sorted(IMAGE_EXTENSIONS)}).')
    return paths


def _manifest_images(manifest_path, splits):
    """Image paths from a manifest, restricted to the given splits.

    Requires every row to carry a split: a None split would otherwise be pulled into SSL, and the
    probe could then evaluate on images the encoder saw unlabeled (a transductive leak). Run
    build_dataset.py, which always emits leakage-free grouped splits.
    """
    from lesnet.data.records import load_manifest
    records = load_manifest(manifest_path)
    missing = sum(1 for r in records if r.split is None)
    if missing:
        raise ValueError(
            f'{missing}/{len(records)} manifest rows lack a split. SSL requires grouped splits so '
            f'the held-out test set stays out of pretraining — regenerate the manifest with '
            f'build_dataset.py.'
        )
    keep = [r.image_path for r in records if r.split in splits]
    if not keep:
        raise ValueError(f'No manifest rows in splits {splits} of {manifest_path}.')
    return [Path(p) for p in keep]


class ImagePathDataset(Dataset):
    """Images from an explicit path list, transformed per the shared contract."""

    def __init__(self, paths, config, train=True, image_size=None):
        self.paths = list(paths)
        self.transform = build_transform(config, train=train, image_size=image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image)


def manifest_image_ids(manifest_path, splits):
    """Archive IDs of the manifest rows in `splits` — the SSL exclusion list.

    build_dataset.py writes files as '<source>_<archive id>.jpg' (e.g. 'isic_ISIC_0031712.jpg'),
    so emit both the full stem and the source-stripped id: HDF5 archives key on the bare id, and
    an exclusion list that silently matches nothing would leak the test set into pretraining.
    """
    from lesnet.data.records import load_manifest
    ids = set()
    for record in load_manifest(manifest_path):
        if record.split not in splits:
            continue
        stem = Path(record.image_path).stem
        ids.add(stem)
        _, _, bare = stem.partition('_')
        if bare:
            ids.add(bare)
    return ids


class Hdf5ImageDataset(Dataset):
    """Images from an HDF5 archive bundle: one dataset per image id, holding encoded JPEG bytes.

    This is how the full ISIC archive (~500k images) is consumed — a single 4 GB file of 256px
    crops instead of half a million small files, which no notebook filesystem enjoys. The handle
    is opened lazily *per worker*: h5py handles are not fork-safe, so opening in __init__ and
    then forking dataloader workers yields silent garbage.
    """

    def __init__(self, hdf5_path, config, exclude_ids=(), train=True, image_size=None):
        import h5py
        self.path = str(hdf5_path)
        self._handle = None
        with h5py.File(self.path, 'r') as handle:
            keys = list(handle.keys())
        excluded = {str(i) for i in exclude_ids}
        self.keys = [k for k in keys if k not in excluded]
        self.num_excluded = len(keys) - len(self.keys)
        if excluded and not self.num_excluded:
            raise ValueError(
                f'{len(excluded)} ids were meant to be held out of pretraining but none matched a '
                f'key in {self.path} (keys look like {keys[:3]}). Refusing to pretrain: an '
                f'exclusion list that matches nothing silently leaks the test set into SSL.')
        if not self.keys:
            raise ValueError(f'No usable images in {self.path} after excluding {len(excluded)} ids.')
        self.transform = build_transform(config, train=train, image_size=image_size)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self._handle is None:
            import h5py
            self._handle = h5py.File(self.path, 'r')
        raw = self._handle[self.keys[index]][()]
        data = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
        with Image.open(io.BytesIO(data)) as image:
            return self.transform(image)


class SyntheticImageDataset(Dataset):
    """Random images for the CPU smoke path (no files needed)."""

    def __init__(self, num_samples, image_size):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return torch.rand(3, self.image_size, self.image_size, generator=generator)


def _worker_init_fn(worker_id):
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)


def build_dataloader(config, root=None, manifest_path=None, synthetic_samples=None,
                     hdf5_path=None, exclude_manifest=None, image_size=None,
                     batch_size=None, paths=None):
    """DataLoader yielding (images, enc_masks, pred_masks) via the multi-block collator.

    Precedence: synthetic_samples > hdf5_path > manifest_path (pretrain splits) > root glob.
    `image_size` overrides config.image_size for this loader (used by the resolution schedule);
    `exclude_manifest` keeps that manifest's held-out test images out of pretraining. Pass `paths`
    to reuse an already-globbed file list — enumerating a 550k-image directory takes minutes, and
    the resolution schedule rebuilds this loader mid-run.
    """
    hdf5_path = hdf5_path or (config.hdf5_path or None)
    size = image_size or config.image_size
    if synthetic_samples is not None:
        dataset = SyntheticImageDataset(synthetic_samples, size)
    elif hdf5_path is not None:
        exclude_manifest = exclude_manifest or config.exclude_manifest or None
        held_out = tuple(s for s in ('train', 'val', 'test') if s not in config.pretrain_splits)
        exclude = manifest_image_ids(exclude_manifest, held_out) if exclude_manifest else ()
        dataset = Hdf5ImageDataset(hdf5_path, config, exclude_ids=exclude, image_size=size)
        print(f'HDF5 pretraining set: {len(dataset)} images '
              f'({dataset.num_excluded} held-out {held_out} images excluded)', flush=True)
    elif manifest_path is not None:
        dataset = ImagePathDataset(_manifest_images(manifest_path, config.pretrain_splits), config,
                                   image_size=size)
    else:
        paths = list(paths) if paths is not None else _glob_images(root or config.data_dir)
        exclude_manifest = exclude_manifest or config.exclude_manifest or None
        if exclude_manifest:
            # Same leakage guard as the HDF5 path: a raw image directory carries no splits, so the
            # labelled manifest's held-out ids have to be subtracted by filename or the encoder
            # pretrains on the very images the probe scores.
            held_out = tuple(s for s in ('train', 'val', 'test') if s not in config.pretrain_splits)
            exclude = manifest_image_ids(exclude_manifest, held_out)
            kept = [p for p in paths if p.stem not in exclude]
            if len(kept) == len(paths):
                raise ValueError(
                    f'{len(exclude)} ids were meant to be held out of pretraining but none matched '
                    f'a file under {root or config.data_dir} (files look like '
                    f'{[p.name for p in paths[:3]]}). Refusing to pretrain: an exclusion list that '
                    f'matches nothing silently leaks the test set into SSL.')
            print(f'directory pretraining set: {len(kept)} images '
                  f'({len(paths) - len(kept)} held-out {held_out} images excluded)', flush=True)
            paths = kept
        dataset = ImagePathDataset(paths, config, image_size=size)

    collator = MultiBlockMaskCollator(config, image_size=size)
    generator = torch.Generator().manual_seed(config.seed)
    # Under torchrun each rank must see a DISJOINT shard, otherwise N GPUs just recompute the same
    # gradient N times. drop_last keeps every rank's step count identical — a rank that runs one
    # step fewer leaves the others blocked in the next all-reduce.
    from lesnet.jepa.distributed import is_distributed, rank, world_size
    sampler = None
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size(), rank=rank(),
                                     shuffle=True, seed=config.seed, drop_last=True)
    return DataLoader(
        dataset,
        batch_size=batch_size or config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        generator=generator,
        worker_init_fn=_worker_init_fn,
    )
