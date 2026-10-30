"""Carga de corpus y verdad terreno (guion -> transcripcion de referencia)."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from .engine import ROOT

AUDIO_EXT = (".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".aac", ".mp4")


def reference_from_script(guion: str) -> str:
    lines = [re.sub(r"^[AB]:\s*", "", ln.strip()) for ln in guion.strip().splitlines() if ln.strip()]
    return " ".join(lines)


def load_clips(cfg: dict, task: str) -> list[dict]:
    """task: 'interviews' | 'calls'. Cada clip: id, reference, campos, trampas?, fecha_referencia?"""
    with open(ROOT / cfg["paths"]["ground_truth"][task], encoding="utf-8") as f:
        clips = yaml.safe_load(f)
    for c in clips:
        c["reference"] = reference_from_script(c["guion"])
        c.setdefault("fecha_referencia", cfg["reference_date"])
    return clips


def find_audio(cfg: dict, clip_id: str, group: str = "audio_dir") -> Path | None:
    d = ROOT / cfg["paths"][group]
    for ext in AUDIO_EXT:
        p = d / f"{clip_id}{ext}"
        if p.exists():
            return p
    return None
