"""Escalera de degradacion reproducible: L0 original, L1 nota de voz, L2 telefono, L3 ruido.

Se usa el ffmpeg que empaqueta imageio-ffmpeg (sin instalar nada del sistema).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .engine import ROOT


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ff(*args: str):
    r = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[-400:])


def _l1(src: str, dst: str, c: dict):
    with tempfile.TemporaryDirectory() as td:
        tmp = str(Path(td) / "x.ogg")
        try:
            _ff("-i", src, "-ac", str(c["channels"]), "-ar", str(c["sample_rate"]),
                "-c:a", c["codec"], "-b:a", c["bitrate"], tmp)
        except RuntimeError:                       # sin libopus: mp3 de bitrate bajo
            tmp = str(Path(td) / "x.mp3")
            _ff("-i", src, "-ac", "1", "-ar", str(c["sample_rate"]), "-c:a", "libmp3lame", "-b:a", "24k", tmp)
        _ff("-i", tmp, "-ac", "1", "-ar", str(c["sample_rate"]), dst)


def _l2(src: str, dst: str, c: dict):
    _ff("-i", src, "-ac", "1", "-af", f"highpass=f={c['highpass_hz']},lowpass=f={c['lowpass_hz']}",
        "-ar", str(c["sample_rate"]), dst)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Ruido rosa (1/f en potencia) por filtrado espectral de ruido blanco; semilla fija."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / np.sqrt(np.mean(x ** 2))


def speech_power(x: np.ndarray, sr: int, below_peak_db: float) -> float:
    """Potencia media de la voz medida SOLO en tramos activos (tramos de 20 ms a menos de
    `below_peak_db` dB del tramo mas fuerte). Evita que las pausas inflen el SNR."""
    n = int(0.02 * sr)
    frames = x[: len(x) // n * n].reshape(-1, n)
    p = np.mean(frames ** 2, axis=1) + 1e-12
    db = 10 * np.log10(p)
    active = db > db.max() - below_peak_db
    return float(np.mean(p[active]))


def add_noise(x: np.ndarray, sr: int, snr_db: float, seed: int, below_peak_db: float):
    rng = np.random.default_rng(seed)
    ps = speech_power(x, sr, below_peak_db)
    noise = pink_noise(len(x), rng) * np.sqrt(ps / 10 ** (snr_db / 10))
    y = x + noise
    peak = np.max(np.abs(y))
    if peak > 0.99:
        y = y * (0.99 / peak)                      # misma escala a ambos: el SNR no cambia
        scale = 0.99 / peak
    else:
        scale = 1.0
    measured = 10 * np.log10(speech_power(x * scale, sr, below_peak_db) / np.mean((noise * scale) ** 2))
    return y.astype(np.float32), float(measured)


def apply_level(src: str | Path, level: str, out_dir: str | Path, cfg: dict) -> dict:
    """-> {path, level, snr_target?, snr_measured?}. L0 devuelve el archivo original sin tocar."""
    src = str(src)
    if level == "L0":
        return {"path": src, "level": "L0"}
    lc = cfg["ladder"]
    out_dir = Path(out_dir) / level
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = str(out_dir / (Path(src).stem + ".wav"))
    if level == "L1":
        _l1(src, dst, lc["l1"])
        return {"path": dst, "level": level}
    if level == "L2":
        _l2(src, dst, lc["l2"])
        return {"path": dst, "level": level}
    if level == "L3":
        with tempfile.TemporaryDirectory() as td:
            base = str(Path(td) / "base.wav")
            _l1(src, base, lc["l1"])
            x, sr = sf.read(base, dtype="float32")
        l3 = lc["l3"]
        y, measured = add_noise(x, sr, l3["snr_db"], l3["seed"], l3["active_frame_db_below_peak"])
        sf.write(dst, y, sr)
        info = {"path": dst, "level": level, "snr_target": l3["snr_db"], "snr_measured": round(measured, 2)}
        Path(dst).with_suffix(".snr.json").write_text(json.dumps(info))
        return info
    raise ValueError(level)
