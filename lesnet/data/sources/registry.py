"""Registry of all data sources. Add a legally-usable dataset here to include it."""
from lesnet.data.sources import ddi, fitzpatrick17k, isic, pad_ufes
from lesnet.data.sources.base import SourceSpec

SOURCE_REGISTRY = {
    'isic': SourceSpec(
        name='isic', parse=isic.parse, download=isic.download,
        note='ISIC Archive v2 — API auto-download (aggregates HAM10000/BCN20000/challenge sets).'),
    'pad_ufes_20': SourceSpec(
        name='pad_ufes_20', parse=pad_ufes.parse, download=pad_ufes.download,
        note='PAD-UFES-20 (CC BY 4.0, Mendeley) — smartphone images with Fitzpatrick.'),
    'fitzpatrick17k': SourceSpec(
        name='fitzpatrick17k', parse=fitzpatrick17k.parse, download=fitzpatrick17k.download,
        note='Fitzpatrick17k — skin-type-labelled clinical images; many image links may rot.'),
    'ddi': SourceSpec(
        name='ddi', parse=ddi.parse, download=None, requires_manual_download=True,
        note='DDI — Stanford Research Use Agreement; place on disk manually.'),
}


def get_source(name):
    if name not in SOURCE_REGISTRY:
        raise ValueError(f"Unknown source '{name}'. Known: {sorted(SOURCE_REGISTRY)}")
    return SOURCE_REGISTRY[name]
