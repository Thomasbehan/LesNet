"""Source contract shared by every dataset loader.

A source maps a raw on-disk download into ``LesionRecord`` rows (``parse``); some can also
fetch that download over the network (``download``), others (licence-gated) must be placed
on disk manually and only parse.
"""
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class SourceSpec:
    name: str
    parse: Callable                     # (root, limit) -> list[LesionRecord]
    download: Optional[Callable] = None  # (root, limit) -> None ; None if not automatable
    requires_manual_download: bool = False
    note: str = ''
