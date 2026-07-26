"""Download the whole ISIC archive at a fixed resolution, multithreaded and resumable.

The archive publishes only two renditions — `full` (multi-megapixel) and `thumbnail_256` — so a
512 px dataset means fetching `full` and downsampling in the worker. Metadata paging is
cursor-based and capped at 100 rows/page (~5.5k serial requests for the ~553k-image archive), so
it runs as a producer feeding the download pool instead of as a separate phase.

Re-running resumes: the crawl restarts from the saved cursor, and images already on disk are
skipped. Ctrl-C is safe — images are written to a .partial file and renamed.

    python commands/download_isic_archive.py --dest data/isic_512 --size 512 --workers 48
    python commands/download_isic_archive.py --dest data/isic_smoke --size 512 --limit 200

Output layout (consumable by commands/build_dataset.py):
    <dest>/metadata.csv          isic_id, url, patient_id, diagnosis, age, sex, site
    <dest>/images/<isic_id>.jpg  downsampled image
"""
import argparse
import json
import time
from pathlib import Path

from lesnet.data.sources import isic


def main():
    parser = argparse.ArgumentParser(description="Download the ISIC archive at a fixed resolution.")
    parser.add_argument('--dest', default='data/isic_512', help="Output root.")
    parser.add_argument('--size', type=int, default=512,
                        help="Shorter-side pixels (0 keeps the original full-resolution bytes).")
    parser.add_argument('--square', action='store_true',
                        help="Centre-crop to size x size instead of preserving aspect ratio.")
    parser.add_argument('--quality', type=int, default=90, help="Output JPEG quality.")
    parser.add_argument('--workers', type=int, default=32, help="Concurrent downloads.")
    parser.add_argument('--limit', type=int, help="Stop after N images (smoke testing).")
    parser.add_argument('--progress-every', type=int, default=1000, help="Images between log lines.")
    args = parser.parse_args()

    dest = Path(args.dest)
    print(f"ISIC -> {dest} at {args.size or 'full'}px, {args.workers} workers", flush=True)
    started = time.time()
    counters = isic.download_archive(
        str(dest), size=args.size or None, square=args.square, quality=args.quality,
        workers=args.workers, limit=args.limit, progress_every=args.progress_every)

    on_disk = len(list((dest / 'images').glob('*.jpg')))
    counters['images_on_disk'] = on_disk
    counters['bytes_on_disk'] = sum(p.stat().st_size for p in (dest / 'images').glob('*.jpg'))
    (dest / 'download_report.json').write_text(json.dumps(counters, indent=2), encoding='utf-8')

    print(f"\ndone in {time.time() - started:.0f}s: {counters['ok']} ok, {counters['failed']} failed, "
          f"{on_disk} images on disk ({counters['bytes_on_disk'] / 1e9:.1f} GB), "
          f"{counters['bytes'] / 1e9:.1f} GB fetched", flush=True)
    if counters['failed']:
        print("re-run the same command to retry the failures (existing images are skipped).")


if __name__ == '__main__':
    main()
