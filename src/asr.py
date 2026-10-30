"""ASR con faster-whisper. Mide dispositivo REAL, VRAM pico y tiempo (no lee la configuracion)."""
from __future__ import annotations

import inspect
import threading
import time

_MODELS: dict = {}


class VramMonitor:
    """Muestrea la memoria usada de la GPU (NVML) en un hilo; pico = max - base."""

    def __init__(self, interval: float = 0.05, index: int = 0):
        self.interval, self.index = interval, index
        self.available, self.base, self.peak = False, 0, 0
        self._stop = threading.Event()
        self._h = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.available = True
        except Exception:
            self.available = False

    def _used(self) -> int:
        return self._nv.nvmlDeviceGetMemoryInfo(self._h).used

    def __enter__(self):
        if self.available:
            self.base = self.peak = self._used()
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._used())
            time.sleep(self.interval)

    def __exit__(self, *a):
        if self.available:
            self._stop.set()
            self._t.join()
            self.peak = max(self.peak, self._used())

    @property
    def peak_mb(self) -> float | None:
        return round((self.peak - self.base) / 1024 ** 2, 1) if self.available else None


def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve(device: str, compute_type: str) -> tuple[str, str]:
    dev = ("cuda" if cuda_available() else "cpu") if device == "auto" else device
    if compute_type == "auto":
        compute_type = "float16" if dev == "cuda" else "int8"
    return dev, compute_type


def get_model(name: str, device: str, compute_type: str):
    key = (name, device, compute_type)
    if key not in _MODELS:
        from faster_whisper import WhisperModel
        t0 = time.perf_counter()
        _MODELS[key] = (WhisperModel(name, device=device, compute_type=compute_type),
                        time.perf_counter() - t0)
    return _MODELS[key]


def glossary_kwargs(model, glossary: list[str] | None, mode: str) -> tuple[dict, str]:
    """hotwords existe desde faster-whisper 1.0.3; si la version no lo trae, cae a initial_prompt."""
    if not glossary or mode == "none":
        return {}, "none"
    params = inspect.signature(model.transcribe).parameters
    if mode == "hotwords" and "hotwords" in params:
        return {"hotwords": ", ".join(glossary)}, "hotwords"
    return {"initial_prompt": "Glosario: " + ", ".join(glossary) + "."}, "initial_prompt"


def transcribe_file(path: str, cfg: dict, *, model: str | None = None, device: str | None = None,
                    compute_type: str | None = None, glossary: list[str] | None = None,
                    glossary_mode: str | None = None) -> dict:
    a = cfg["asr"]
    name = model or a["model"]
    dev, ct = resolve(device or a["device"], compute_type or a["compute_type"])
    mode = glossary_mode or a["glossary_mode"]
    with VramMonitor() as vram:
        m, load_s = get_model(name, dev, ct)
        kw, mode_used = glossary_kwargs(m, glossary, mode)
        t0 = time.perf_counter()
        segs, info = m.transcribe(path, language=a["language"], beam_size=a["beam_size"],
                                  vad_filter=a["vad_filter"], word_timestamps=a["word_timestamps"], **kw)
        text, words = [], []
        for s in segs:                      # el generador es perezoso: se mide al consumirlo
            text.append(s.text.strip())
            for w in (s.words or []):
                words.append({"w": w.word.strip(), "p": round(float(w.probability), 3),
                              "start": round(w.start, 2), "end": round(w.end, 2)})
        elapsed = time.perf_counter() - t0
    audio_s = float(info.duration)
    return {
        "transcript": " ".join(text), "words": words,
        "model": name, "device_requested": dev,
        "device": m.model.device,                  # leido de la libreria en ejecucion
        "compute_type": m.model.compute_type,
        "glossary_mode": mode_used,
        "elapsed_s": round(elapsed, 3), "load_s": round(load_s, 3), "audio_s": round(audio_s, 2),
        "sec_per_min": round(elapsed / (audio_s / 60), 2) if audio_s else None,
        "vram_peak_mb": vram.peak_mb,
        "cuda_device_count": __import__("ctranslate2").get_cuda_device_count(),
    }
