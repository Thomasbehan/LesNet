"""Upload decoding for /predict — the path that OOM-killed the 512 MB demo on real phone photos."""
import io

import numpy as np
import pytest
from PIL import Image

from lesnet.views.api import UploadTooLarge, load_upload


def _jpeg(width, height, exif_orientation=None):
    image = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        image.save(buffer, 'JPEG', exif=exif.tobytes())
    else:
        image.save(buffer, 'JPEG')
    buffer.seek(0)
    return buffer


def test_phone_photo_is_shrunk_to_model_input_size():
    array = load_upload(_jpeg(4032, 3024))           # 12 MP, the size that crashed the demo
    assert array.dtype == np.uint8 and array.shape[2] == 3
    assert max(array.shape[:2]) == 256
    assert array.shape[:2] == (192, 256)              # aspect ratio kept


def test_exif_rotation_is_honoured():
    array = load_upload(_jpeg(400, 300, exif_orientation=6))   # stored sideways, shown portrait
    assert array.shape[0] > array.shape[1]


def test_small_images_are_not_upscaled():
    assert load_upload(_jpeg(120, 90)).shape[:2] == (90, 120)


def test_absurd_pixel_counts_are_refused():
    with pytest.raises(UploadTooLarge):
        load_upload(_jpeg(400, 300), max_pixels=100_000)


def test_non_images_raise():
    with pytest.raises(Exception):
        load_upload(io.BytesIO(b'not an image'))


# --- /predict never answers with an HTML error page: the UI must always get JSON + a reason ---------

class _Upload:
    def __init__(self, data):
        self.file = io.BytesIO(data)
        self.filename = 'photo.jpg'


def _predict(post, predictor=None, monkeypatch=None):
    from pyramid import testing

    from lesnet.views import api
    if monkeypatch is not None:
        monkeypatch.setattr(api, '_get_predictor', lambda: predictor)
    request = testing.DummyRequest(post=post)
    body = api.predict_api(request)
    return request.response.status_int, body


def test_missing_image_is_json_400():
    status, body = _predict({})
    assert status == 400 and body['reason'] == 'no_image' and body['triage'] == 'abstain'


def test_unreadable_image_is_json_400():
    status, body = _predict({'image': _Upload(b'definitely not a jpeg')})
    assert status == 400 and body['reason'] == 'unreadable'


def test_predictor_crash_is_json_500(monkeypatch):
    class Broken:
        def predict(self, *_):
            raise RuntimeError('boom')
    status, body = _predict({'image': _Upload(_jpeg(64, 48).read())}, Broken(), monkeypatch)
    assert status == 500 and body['reason'] == 'internal' and 'boom' not in str(body)


def test_model_unavailable_is_json_503(monkeypatch):
    from lesnet.views import api

    def unavailable():
        raise FileNotFoundError('no model')
    monkeypatch.setattr(api, '_get_predictor', unavailable)
    status, body = _predict({'image': _Upload(_jpeg(64, 48).read())})
    assert status == 503 and body['reason'] == 'model_unavailable'
