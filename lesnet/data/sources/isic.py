"""ISIC Archive v2 source — resumable, threaded metadata collection + selective image fetch.

Two phases so we can do "malignant-maximal" sourcing cheaply: page the whole archive's
metadata (no images), then download images only for a chosen subset (e.g. all malignant +
suspicious + a balanced benign sample). ``parse`` reads the written metadata.csv into records.
"""
import csv
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock, Thread

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lesnet.data.records import LesionRecord, read_csv_rows, to_float

API_URL = "https://api.isic-archive.com/api/v2/images/"
METADATA_FIELDS = ['isic_id', 'url', 'patient_id', 'diagnosis', 'diagnosis_1',
                   'age_approx', 'sex', 'anatom_site_general']


def make_session(pool=32):
    session = requests.Session()
    retries = Retry(
        total=8, connect=5, read=5, backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504, 520, 522, 524],
        respect_retry_after_header=True, raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=pool, pool_connections=pool)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def _specific_diagnosis(clinical):
    for key in ('diagnosis_3', 'diagnosis_2', 'diagnosis_1'):
        if clinical.get(key):
            return clinical[key]
    return clinical.get('diagnosis') or ''


def records_from_page(payload, resolution):
    """Extract metadata rows (incl. a downloadable image url) from one API page."""
    rows = []
    for image in payload['results']:
        clinical = image.get('metadata', {}).get('clinical', {})
        files = image.get('files', {})
        chosen = files.get(resolution) or files.get('full') or files.get('thumbnail_256')
        url = (chosen or {}).get('url')
        if not url:
            continue
        rows.append({
            'isic_id': image['isic_id'], 'url': url,
            'patient_id': clinical.get('patient_id') or image.get('patient_id') or '',
            'diagnosis': _specific_diagnosis(clinical),
            'diagnosis_1': clinical.get('diagnosis_1') or '',
            'age_approx': clinical.get('age_approx', ''),
            'sex': clinical.get('sex', ''),
            'anatom_site_general': clinical.get('anatom_site_1', ''),
        })
    return rows


def url_map(root):
    """{isic_id: image url} read back from a written metadata.csv (for selective download)."""
    return {row['isic_id']: row['url']
            for row in read_csv_rows(os.path.join(root, 'metadata.csv'))
            if row.get('isic_id') and row.get('url')}


def collect_metadata(root, resolution='full', page_size=100, limit=None, session=None):
    """Page the archive's metadata into <root>/metadata.csv — incremental + resumable.

    Each page is appended immediately and the cursor saved to metadata_state.json, so a long
    crawl survives interruption (re-run resumes) and is observable while running. Returns the
    rows collected *this call*; use ``url_map(root)`` for the full id->url map after completion.
    """
    return list(iter_metadata(root, resolution, page_size, limit, session))


def iter_metadata(root, resolution='full', page_size=100, limit=None, session=None):
    """Streaming form of collect_metadata: yields each new row as its page arrives.

    The archive caps `limit` at 100 rows per page, so a full crawl is ~5.5k strictly-serial
    requests. Yielding lets a caller start downloading images off page 1 instead of waiting
    ~40 minutes for the crawl to finish with the network otherwise idle.
    """
    os.makedirs(root, exist_ok=True)
    session = session or make_session()
    metadata_path = os.path.join(root, 'metadata.csv')
    state_path = os.path.join(root, 'metadata_state.json')

    next_url, params, seen = API_URL, {'limit': page_size}, set()
    if os.path.exists(state_path) and os.path.exists(metadata_path):
        state = json.load(open(state_path, encoding='utf-8'))
        next_url = state.get('next_url') or API_URL
        params = None if state.get('next_url') else {'limit': page_size}
        seen = {row['isic_id'] for row in read_csv_rows(metadata_path)}

    handle = open(metadata_path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
    if not seen:
        writer.writeheader()

    try:
        while next_url and (limit is None or len(seen) < limit):
            response = session.get(next_url, params=params, timeout=120)
            params = None
            if response.status_code != 200:
                break
            payload = response.json()
            for row in records_from_page(payload, resolution):
                if row['isic_id'] in seen:
                    continue
                writer.writerow({field: row.get(field, '') for field in METADATA_FIELDS})
                seen.add(row['isic_id'])
                yield row
            handle.flush()
            next_url = payload.get('next')
            json.dump({'next_url': next_url, 'collected': len(seen)},
                      open(state_path, 'w', encoding='utf-8'))
    finally:
        handle.close()


def download_images(root, url_by_id, workers=32, session=None):
    """Fetch images for the given {isic_id: url} into <root>/<isic_id>.jpg (skip existing)."""
    os.makedirs(root, exist_ok=True)
    session = session or make_session(workers)
    downloaded = failed = 0

    def _fetch(item):
        isic_id, url = item
        path = os.path.join(root, f"{isic_id}.jpg")
        if os.path.exists(path):
            return True
        response = session.get(url, timeout=120)
        if response.status_code == 200:
            with open(path, 'wb') as handle:
                handle.write(response.content)
            return True
        return False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch, item): item for item in url_by_id.items()}
        for future in as_completed(futures):
            if future.result():
                downloaded += 1
            else:
                failed += 1
    return downloaded, failed


def resize_jpeg(raw, size, square=False, quality=90):
    """Full-resolution JPEG bytes -> `size`-px JPEG bytes (shorter side, or square if asked).

    The archive publishes only `full` (multi-megapixel) and `thumbnail_256`; there is no 512
    rendition, so a 512 dataset means downsampling here. PIL's `draft()` does the bulk of the
    shrink inside the JPEG DCT decode, which is several times faster than decoding at full size
    and resizing afterwards — the difference between a few hours and most of a day over 550k images.
    """
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as image:
        image.draft('RGB', (size, size))            # DCT-domain prescale to >= the target
        image = image.convert('RGB')
        width, height = image.size
        scale = size / min(width, height)
        if scale < 1.0:
            image = image.resize((max(round(width * scale), size), max(round(height * scale), size)),
                                 Image.BICUBIC)
        if square:
            width, height = image.size
            left, top = (width - size) // 2, (height - size) // 2
            image = image.crop((left, top, left + size, top + size))
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        return buffer.getvalue()


def _download_one(session, isic_id, url, images_dir, size, square, quality):
    """Fetch + downsample one image. Returns (ok, bytes_downloaded); skips work already done."""
    path = os.path.join(images_dir, f'{isic_id}.jpg')
    if os.path.exists(path):
        return True, 0
    response = session.get(url, timeout=120)
    if response.status_code != 200:
        return False, 0
    raw = response.content
    data = resize_jpeg(raw, size, square=square, quality=quality) if size else raw
    tmp = f'{path}.partial'                          # write-then-rename: a killed run leaves no
    with open(tmp, 'wb') as handle:                  # half-written jpeg for the resume to skip
        handle.write(data)
    os.replace(tmp, path)
    return True, len(raw)


def _log(message):
    print(message, flush=True)      # unbuffered: a multi-hour run is usually watched via a log file


def download_archive(root, size=512, square=False, quality=90, workers=32, limit=None,
                     page_size=100, session=None, progress_every=1000, log=_log):
    """Stream the whole ISIC archive to <root>/images/<isic_id>.jpg at `size` px, resumably.

    Metadata paging is cursor-based and therefore strictly serial (~5.5k requests for the full
    archive), so it runs as a producer feeding a download pool rather than as a phase of its own —
    otherwise ~40 minutes of paging happens with the network otherwise idle. Rows are appended to
    <root>/metadata.csv as they arrive, so build_dataset.py can consume the result, and re-running
    resumes: the pager restarts from the saved cursor and existing images are skipped.

    `size=None` keeps the original full-resolution bytes.
    """
    images_dir = os.path.join(root, 'images')
    os.makedirs(images_dir, exist_ok=True)
    session = session or make_session(workers)
    queue = Queue(maxsize=workers * 8)               # bounded: don't page ahead of the downloads
    counters = {'ok': 0, 'failed': 0, 'bytes': 0, 'queued': 0}
    lock = Lock()
    start = time.time()

    def produce():
        try:
            # Resume: rows already crawled but never fetched (or fetched and lost) must be retried.
            # Without this, a run whose metadata crawl finished would enqueue nothing at all and
            # silently "succeed" with a partial image set.
            if os.path.exists(os.path.join(root, 'metadata.csv')):
                for isic_id, url in url_map(root).items():
                    if not os.path.exists(os.path.join(images_dir, f'{isic_id}.jpg')):
                        queue.put((isic_id, url))
                        counters['queued'] += 1
            for row in iter_metadata(root, resolution='full', page_size=page_size, limit=limit,
                                     session=session):
                queue.put((row['isic_id'], row['url']))
                counters['queued'] += 1
        finally:
            for _ in range(workers):
                queue.put(None)                      # one sentinel per consumer

    def consume():
        while True:
            item = queue.get()
            if item is None:
                return
            try:
                ok, nbytes = _download_one(session, item[0], item[1], images_dir, size, square,
                                           quality)
            except Exception:                        # one bad image must not sink a 550k-image run
                ok, nbytes = False, 0
            with lock:
                counters['ok' if ok else 'failed'] += 1
                counters['bytes'] += nbytes
                done = counters['ok'] + counters['failed']
                if progress_every and done % progress_every == 0:
                    elapsed = max(time.time() - start, 1e-9)
                    log(f'isic: {done} done ({counters["failed"]} failed) '
                        f'{done / elapsed:.1f} img/s {counters["bytes"] / 1e9:.1f} GB fetched')

    producer = Thread(target=produce, daemon=True)
    producer.start()
    consumers = [Thread(target=consume, daemon=True) for _ in range(workers)]
    for thread in consumers:
        thread.start()
    producer.join()
    for thread in consumers:
        thread.join()
    counters['seconds'] = round(time.time() - start, 1)
    return counters


def download(root, limit=None, resolution='full', workers=32):
    """Convenience full download: collect metadata then fetch every image."""
    collect_metadata(root, resolution=resolution, limit=limit)
    download_images(root, url_map(root), workers=workers)


def parse(root, limit=None):
    rows = read_csv_rows(os.path.join(root, 'metadata.csv'))
    rows = rows[:limit] if limit else rows
    records = []
    for row in rows:
        isic_id = row.get('isic_id')
        if not isic_id:
            continue
        records.append(LesionRecord(
            image_path=os.path.join(root, f"{isic_id}.jpg"),
            source_dataset='isic',
            raw_label=row.get('diagnosis') or row.get('diagnosis_1') or 'unknown',
            group_id=row.get('patient_id') or isic_id,
            anatomical_site=row.get('anatom_site_general') or None,
            age=to_float(row.get('age_approx')),
            sex=row.get('sex') or None,
        ))
    return records
