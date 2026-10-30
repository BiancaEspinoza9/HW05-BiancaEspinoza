"""Genera audio SINTETICO (Edge TTS, sin API key) a partir de los guiones.

Es un grupo aparte (data/synthetic): sirve para probar la tuberia antes de grabar y para el
aumento opcional. NO reemplaza las grabaciones reales ni se mezcla con sus numeros.
  python scripts/make_synthetic.py --task interviews --locale es-PE
"""
import argparse
import asyncio
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_clips
from src.engine import ROOT, load_config
from src.ladder import _ff


async def tts(text: str, voice: str, path: str):
    import edge_tts
    for k in range(5):                      # el servicio rechaza a veces: reintento con espera
        try:
            await edge_tts.Communicate(text, voice).save(path)
            return
        except edge_tts.exceptions.NoAudioReceived:
            await asyncio.sleep(1.5 * (k + 1))
    raise RuntimeError(f"Edge TTS no devolvio audio para: {text[:40]}")


def build(clip: dict, voices: tuple[str, str], pause: float, out: Path):
    parts, sr = [], 16000
    with tempfile.TemporaryDirectory() as td:
        for i, ln in enumerate(l for l in clip["guion"].strip().splitlines() if l.strip()):
            spk = ln[0]
            mp3, wav = f"{td}/{i}.mp3", f"{td}/{i}.wav"
            asyncio.run(tts(re.sub(r"^[AB]:\s*", "", ln), voices[0] if spk == "A" else voices[1], mp3))
            _ff("-i", mp3, "-ac", "1", "-ar", str(sr), wav)
            x, _ = sf.read(wav, dtype="float32")
            parts += [x, np.zeros(int(pause * sr), dtype="float32")]
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), np.concatenate(parts), sr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["interviews", "calls"], required=True)
    ap.add_argument("--locale", default=None, help="clave de synthetic.accent_voices (por defecto voces A/B)")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    cfg = load_config()
    s = cfg["synthetic"]
    voices = tuple(s["accent_voices"][a.locale]) if a.locale else (s["voice_a"], s["voice_b"])
    sub = Path(ROOT / cfg["paths"]["synthetic_dir"]) / (a.locale or "default")
    for c in load_clips(cfg, a.task):
        if a.only and c["id"] not in a.only:
            continue
        out = sub / f"{c['id']}.wav"
        if not out.exists():
            build(c, voices, s["pause_s"], out)
        print("ok", out)
