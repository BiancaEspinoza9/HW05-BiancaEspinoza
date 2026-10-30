"""Regla de verificacion: todo valor extraido debe existir LITERALMENTE en el transcript.

"Literal" se evalua en el espacio normalizado (misma regla que WER: sin tildes/puntuacion,
numeros en digitos), asi "seis mil quinientos" y 6500 son el mismo texto. Ademas se
recalcula el valor desde la evidencia: un modelo que "corrige" lo que oyo mal (valor
distinto de la evidencia) queda sin_verificar aunque acierte.
"""
from __future__ import annotations

from datetime import date

from .extract_rules import parse_availability
from .numbers_es import find_numbers
from .textnorm import norm_text


def _in(evidence: str, transcript_norm: str) -> bool:
    e = norm_text(evidence)
    return bool(e) and f" {e} " in f" {transcript_norm} "


def _nums(evidence: str) -> list[float]:
    toks = norm_text(evidence).split()
    return [v for _, _, v in find_numbers(toks, include_one=True)]


def _kw_hit(evidence: str, words: list[str]) -> bool:
    toks = set(norm_text(evidence).split())
    return any(w in toks for w in words)


def _check(field: str, ftype: str, entry: dict, T: str, schema_cfg: dict, ref: date) -> bool:
    val, ev = entry.get("valor"), entry.get("evidencia")
    if val is None or ev is None:
        return False
    if ftype == "numeric":
        ok = _in(ev, T)
        n = _nums(ev)
        return ok and (float(val) in n or (float(val) - 0.5 in n and "medio" in norm_text(ev).split()))
    if ftype == "numeric_currency":
        cur_words = {w for ws in schema_cfg["currencies"].values() for w in ws}
        cur_ok = any(w in norm_text(ev).split() for w in cur_words if w.isalpha()) or \
            norm_text(ev).split()[:1] == ["s"]
        want = schema_cfg["currencies"][val["moneda"]]
        cur_match = _kw_hit(ev, [w for w in want if w.isalpha()]) or (val["moneda"] == "PEN" and norm_text(ev).startswith("s "))
        return _in(ev, T) and float(val["monto"]) in _nums(ev) and cur_ok and cur_match
    if ftype == "text":
        return _in(ev, T) and _in(str(val), T)
    if ftype == "list":
        evs = ev if isinstance(ev, list) else [ev]
        return bool(val) and all(_in(e, T) for e in evs) and all(_in(v, T) for v in val)
    if ftype == "date":
        return _in(ev, T) and parse_availability(norm_text(ev), ref) == val
    if ftype == "categorical":
        kws = schema_cfg.get("english_keywords") or schema_cfg.get("payment_keywords") or {}
        cefr = {"a1": "basico", "a2": "basico", "b1": "intermedio", "b2": "intermedio",
                "c1": "avanzado", "c2": "avanzado"}
        words = list(kws.get(val, [])) + [val] + [c for c, k in cefr.items() if k == val and field == "nivel_ingles"]
        return val in (schema_cfg["fields"][field].get("classes") or []) and _in(ev, T) and _kw_hit(ev, words)
    if ftype == "promo":
        n = _nums(ev)
        tipo = "porcentaje" if "porciento" in norm_text(ev).split() else "monto"
        return _in(ev, T) and float(val["valor"]) in n and val["tipo"] == tipo
    if ftype == "items":
        if not val:
            return False
        for it in val:
            e = it.get("evidencia")
            if not e or not _in(e, T):
                return False
            n = _nums(e)
            if float(it["cantidad"]) not in n or float(it["precio_unitario"]) not in n:
                return False
        return True
    raise ValueError(ftype)


def verify_record(schema: str, extracted: dict, transcript: str, cfg: dict, ref: date) -> dict:
    """-> {campo: {valor, evidencia, estado}} con estado verificado|sin_verificar|no_encontrado."""
    sc = cfg["schemas"][schema]
    T = norm_text(transcript, cfg["normalization"])
    label = cfg["extraction"]["verification"]["unverified_label"]
    out = {}
    for field, spec in sc["fields"].items():
        entry = extracted.get(field)
        if not entry or entry.get("valor") in (None, [], ""):
            out[field] = {"valor": None, "evidencia": None, "estado": "no_encontrado"}
            continue
        try:
            ok = _check(field, spec["type"], entry, T, sc, ref)
        except Exception:                      # estructura malformada del LLM => no verificable
            ok = False
        out[field] = {**entry, "estado": "verificado" if ok else label}
    return out
