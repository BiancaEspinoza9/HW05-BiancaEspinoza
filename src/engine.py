"""Motor unico: audio -> transcripcion -> campos -> verificacion (-> validacion).

Una sola funcion publica, `process_audio`, atiende los dos esquemas (entrevista y pedido).
Este modulo NO importa ninguna libreria de interfaz (se prueba en tests/test_engine_purity.py).
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv

from . import asr
from .extract_llm import extract_llm
from .extract_rules import extract_rules
from .validate import validate_order
from .verify import verify_record

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    load_dotenv(ROOT / cfg["paths"]["env_file"])
    return cfg


def record_values(verified: dict, only_verified: bool = False, label: str = "sin_verificar") -> dict:
    """{campo: valor}. Con only_verified=True, lo no verificado queda en None."""
    return {k: (v["valor"] if (not only_verified or v["estado"] == "verificado") else None)
            for k, v in verified.items()}


def process_audio(audio_path: str | None, schema: str = "interview", extractor: str | None = None,
                  config: dict | None = None, *, transcript_result: dict | None = None,
                  reference_date: str | None = None, **asr_overrides) -> dict:
    """
    audio_path        ruta al audio (ignorada si se pasa transcript_result: modo cache/CI).
    schema            "interview" | "order".
    extractor         "rules" | "llm" (por defecto config.extraction.default_extractor).
    transcript_result salida previa de asr.transcribe_file -> evita re-transcribir.
    asr_overrides     model=, device=, compute_type=, glossary=, glossary_mode=.
    Devuelve: transcript, fields (valor+evidencia+estado por campo), record, verified,
              validation (solo pedido), device, elapsed_s, error ...
    """
    cfg = config or load_config()
    extractor = extractor or cfg["extraction"]["default_extractor"]
    ref = date.fromisoformat(reference_date or cfg["reference_date"])
    out = {"audio": str(audio_path) if audio_path else None, "schema": schema, "extractor": extractor,
           "transcript": "", "fields": {}, "record": {}, "verified": {}, "validation": None,
           "device": None, "compute_type": None, "model": None, "elapsed_s": None,
           "audio_s": None, "vram_peak_mb": None, "words": [], "glossary_mode": None, "error": None}
    t_total = time.perf_counter()
    try:
        tr = transcript_result
        if tr is None:
            tr = asr.transcribe_file(str(audio_path), cfg, **asr_overrides)
        for k in ("transcript", "words", "device", "compute_type", "model", "elapsed_s",
                  "audio_s", "vram_peak_mb", "glossary_mode"):
            out[k] = tr.get(k)
        if tr.get("device_requested") == "cuda" and tr.get("device") != "cuda":
            out["error"] = "GPU solicitada pero el modelo corrio en CPU"
    except Exception as e:                       # noqa: BLE001 - el motor reporta, no revienta
        out["error"] = f"asr: {type(e).__name__}: {e}"
        return out
    try:
        if extractor == "llm":
            extracted = extract_llm(schema, out["transcript"], cfg, ref)
        else:
            extracted = extract_rules(schema, out["transcript"], cfg, ref)
        out["fields"] = verify_record(schema, extracted, out["transcript"], cfg, ref)
        out["record"] = record_values(out["fields"])
        out["verified"] = {k: v["estado"] for k, v in out["fields"].items()}
        if schema == "order":
            out["validation"] = validate_order(record_values(out["fields"]), cfg, out["words"])
    except Exception as e:                       # noqa: BLE001
        out["error"] = ((out["error"] + " | ") if out["error"] else "") + f"extraccion: {type(e).__name__}: {e}"
    out["total_s"] = round(time.perf_counter() - t_total, 3)
    return out
