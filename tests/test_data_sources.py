import csv
import io
import os
import zipfile

import pytest
import requests

from lesnet.data.sources import ddi, fitzpatrick17k, isic, pad_ufes
from lesnet.data.sources.base import SourceSpec
from lesnet.data.sources.registry import SOURCE_REGISTRY, get_source


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b''):
        self.status_code = status_code
        self._json = json_data
        self.content = content

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(self.status_code)


class FakeSession:
    def __init__(self, pages=None, by_url=None):
        self.pages = list(pages or [])
        self.by_url = by_url or {}

    def get(self, url, params=None, timeout=None):  # noqa: ARG002
        if self.by_url:
            return self.by_url.get(url, FakeResponse(404))
        return self.pages.pop(0)


def _write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- ISIC ---

def test_isic_specific_diagnosis_priority():
    assert isic._specific_diagnosis({'diagnosis_3': 'melanoma', 'diagnosis_1': 'x'}) == 'melanoma'
    assert isic._specific_diagnosis({'diagnosis_1': 'nevus'}) == 'nevus'
    assert isic._specific_diagnosis({'diagnosis': 'bcc'}) == 'bcc'
    assert isic._specific_diagnosis({}) == ''


def test_isic_records_from_page_resolution_and_skip():
    payload = {'results': [
        {'isic_id': 'I1', 'metadata': {'clinical': {'diagnosis_1': 'melanoma', 'age_approx': '60',
         'sex': 'male', 'anatom_site_1': 'torso', 'patient_id': 'P1'}},
         'files': {'thumbnail_256': {'url': 'thumb1'}}},
        {'isic_id': 'I2', 'metadata': {'clinical': {}}, 'files': {}},   # no url -> skipped
    ]}
    rows = isic.records_from_page(payload, 'full')   # falls back full->thumbnail_256
    assert len(rows) == 1
    assert rows[0]['isic_id'] == 'I1' and rows[0]['url'] == 'thumb1'
    assert rows[0]['diagnosis'] == 'melanoma' and rows[0]['patient_id'] == 'P1'


def test_isic_collect_metadata_pages_and_writes(tmp_path):
    page1 = FakeResponse(json_data={'results': [
        {'isic_id': 'I1', 'metadata': {'clinical': {'diagnosis_1': 'melanoma'}},
         'files': {'full': {'url': 'u1'}}}], 'next': 'page2'})
    page2 = FakeResponse(json_data={'results': [
        {'isic_id': 'I2', 'metadata': {'clinical': {'diagnosis_1': 'nevus'}},
         'files': {'full': {'url': 'u2'}}}], 'next': None})
    rows = isic.collect_metadata(str(tmp_path), session=FakeSession(pages=[page1, page2]))
    assert {row['isic_id'] for row in rows} == {'I1', 'I2'}
    assert os.path.exists(tmp_path / 'metadata.csv')


def test_isic_collect_metadata_stops_on_error(tmp_path):
    rows = isic.collect_metadata(str(tmp_path), session=FakeSession(pages=[FakeResponse(500)]))
    assert rows == []


def test_isic_download_images_success_and_failure(tmp_path):
    session = FakeSession(by_url={'u1': FakeResponse(200, content=b'img'), 'u2': FakeResponse(404)})
    downloaded, failed = isic.download_images(
        str(tmp_path), {'I1': 'u1', 'I2': 'u2'}, workers=2, session=session)
    assert downloaded == 1 and failed == 1
    assert (tmp_path / 'I1.jpg').read_bytes() == b'img'
    # existing file is skipped (still counts as downloaded)
    again, _ = isic.download_images(str(tmp_path), {'I1': 'u1'}, session=session)
    assert again == 1


def test_isic_collect_metadata_resumes(tmp_path):
    def page(isic_id, url, nxt):
        return FakeResponse(json_data={'results': [
            {'isic_id': isic_id, 'metadata': {'clinical': {'diagnosis_1': 'nevus'}},
             'files': {'full': {'url': url}}}], 'next': nxt})

    isic.collect_metadata(str(tmp_path), session=FakeSession(pages=[page('I1', 'u1', None)]))
    assert os.path.exists(tmp_path / 'metadata_state.json')
    # resume: state + metadata present -> I1 is 'seen' and skipped, only I2 is new.
    resume_page = FakeResponse(json_data={'results': [
        {'isic_id': 'I1', 'metadata': {'clinical': {}}, 'files': {'full': {'url': 'u1'}}},
        {'isic_id': 'I2', 'metadata': {'clinical': {}}, 'files': {'full': {'url': 'u2'}}}], 'next': None})
    collected = isic.collect_metadata(str(tmp_path), session=FakeSession(pages=[resume_page]))
    assert [row['isic_id'] for row in collected] == ['I2']
    assert set(isic.url_map(str(tmp_path))) == {'I1', 'I2'}


def test_isic_url_map_skips_blank_rows(tmp_path):
    _write_csv(tmp_path / 'metadata.csv', isic.METADATA_FIELDS, [
        {'isic_id': 'I1', 'url': 'u1', 'patient_id': '', 'diagnosis': 'nevus', 'diagnosis_1': '',
         'age_approx': '', 'sex': '', 'anatom_site_general': ''},
        {'isic_id': '', 'url': '', 'patient_id': '', 'diagnosis': '', 'diagnosis_1': '',
         'age_approx': '', 'sex': '', 'anatom_site_general': ''}])
    assert isic.url_map(str(tmp_path)) == {'I1': 'u1'}


def test_isic_download_orchestrates(tmp_path, monkeypatch):
    monkeypatch.setattr(isic, 'collect_metadata', lambda root, **kw: None)
    monkeypatch.setattr(isic, 'url_map', lambda root: {'I1': 'u1'})
    captured = {}
    monkeypatch.setattr(isic, 'download_images',
                        lambda root, url_by_id, **kw: captured.update(url_by_id))
    isic.download(str(tmp_path))
    assert captured == {'I1': 'u1'}


def test_isic_make_session_returns_session():
    assert isinstance(isic.make_session(), requests.Session)


def test_isic_parse(tmp_path):
    _write_csv(tmp_path / 'metadata.csv', isic.METADATA_FIELDS, [
        {'isic_id': 'I1', 'url': 'u1', 'patient_id': 'P1', 'diagnosis': 'melanoma',
         'diagnosis_1': '', 'age_approx': '60', 'sex': 'male', 'anatom_site_general': 'torso'},
        {'isic_id': '', 'url': '', 'patient_id': '', 'diagnosis': '', 'diagnosis_1': '',
         'age_approx': '', 'sex': '', 'anatom_site_general': ''},   # skipped (no id)
    ])
    records = isic.parse(str(tmp_path), limit=5)
    assert len(records) == 1
    assert records[0].group_id == 'P1' and records[0].age == 60.0


# --- PAD-UFES-20 ---

def test_pad_ufes_download_and_parse(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('metadata.csv',
                         'img_id,patient_id,lesion_id,diagnostic,fitspatrick,region,age,gender\n'
                         'PAT_1.png,PAD1,L1,BCC,3,arm,60,female\n'
                         ',,,,,,,\n')   # empty img_id -> skipped by parse
        archive.writestr('images/PAT_1.png', b'img-bytes')
    session = FakeSession(by_url={pad_ufes.MENDELEY_ZIP_URL: FakeResponse(200, content=buffer.getvalue())})

    pad_ufes.download(str(tmp_path), session=session)
    records = pad_ufes.parse(str(tmp_path))
    assert len(records) == 1
    assert records[0].source_dataset == 'pad_ufes_20'
    assert records[0].fitzpatrick == 3 and records[0].sex == 'female'


# --- Fitzpatrick17k ---

def test_fitzpatrick17k_download_handles_rotten_links(tmp_path):
    _write_csv(tmp_path / 'fitzpatrick17k.csv',
               ['md5hash', 'url', 'label', 'fitzpatrick_scale'],
               [{'md5hash': 'h1', 'url': 'good', 'label': 'melanoma', 'fitzpatrick_scale': '5'},
                {'md5hash': 'h2', 'url': 'bad', 'label': 'nevus', 'fitzpatrick_scale': '2'},
                {'md5hash': 'h3', 'url': 'good3', 'label': 'nevus', 'fitzpatrick_scale': '3'},
                {'md5hash': '', 'url': '', 'label': '', 'fitzpatrick_scale': ''}])   # skipped (no id)

    # h1 already on disk -> exercises the skip-existing branch.
    (tmp_path / 'images').mkdir()
    (tmp_path / 'images' / 'h1.jpg').write_bytes(b'already-here')

    class RottenSession:
        def get(self, url, timeout=None):  # noqa: ARG002
            if url == 'bad':
                raise requests.ConnectionError('dead link')
            return FakeResponse(200, content=b'img')

    fitzpatrick17k.download(str(tmp_path), session=RottenSession())
    assert (tmp_path / 'images' / 'h1.jpg').read_bytes() == b'already-here'   # not overwritten
    assert (tmp_path / 'images' / 'h3.jpg').read_bytes() == b'img'            # freshly fetched
    assert not (tmp_path / 'images' / 'h2.jpg').exists()                      # dead link skipped

    records = fitzpatrick17k.parse(str(tmp_path))
    assert len(records) == 3
    assert records[0].fitzpatrick == 5


# --- DDI ---

def test_ddi_parse_and_skin_tone(tmp_path):
    assert ddi.skin_tone_to_fitzpatrick('56') == 5
    assert ddi.skin_tone_to_fitzpatrick('99') is None
    _write_csv(tmp_path / 'ddi_metadata.csv', ['DDI_file', 'disease', 'skin_tone'],
               [{'DDI_file': 'x.png', 'disease': 'melanoma', 'skin_tone': '56'},
                {'DDI_file': '', 'disease': '', 'skin_tone': ''}])   # skipped
    records = ddi.parse(str(tmp_path))
    assert len(records) == 1 and records[0].fitzpatrick == 5


# --- registry ---

def test_registry_lookup():
    assert set(SOURCE_REGISTRY) == {'isic', 'pad_ufes_20', 'fitzpatrick17k', 'ddi'}
    assert isinstance(get_source('isic'), SourceSpec)
    assert get_source('ddi').requires_manual_download is True
    with pytest.raises(ValueError):
        get_source('nope')
