"""Camino 2 de extraccion: un LLM devuelve JSON con evidencia literal por campo.

El LLM NO decide que es verdad: verify.py comprueba despues que la evidencia exista en el
transcript y que el valor se recalcule desde ella.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date

from .extract_rules import parse_availability
from .textnorm import norm_text


def _client(cfg: dict):
    import anthropic
    key = os.environ.get(cfg["extraction"]["llm"]["api_key_env"])
    if not key:
        raise RuntimeError("falta ANTHROPIC_API_KEY en .env")
    return anthropic.Anthropic(api_key=key)


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text[text.find("{"): text.rfind("}") + 1])


def _wrap(d, key="valor"):
    if d is None:
        return None
    if isinstance(d, dict) and key in d:
        return {"valor": d[key], "evidencia": d.get("evidencia")}
    return None


def _to_internal(schema: str, raw: dict, ref: date) -> dict:
    if schema == "interview":
        out = {k: _wrap(raw.get(k)) for k in ("anios_experiencia", "ultimo_puesto", "pretension_salarial", "nivel_ingles")}
        t = _wrap(raw.get("tecnologias"))
        if t and t["valor"]:
            t["evidencia"] = [t["evidencia"]] if isinstance(t["evidencia"], str) else t["evidencia"]
        out["tecnologias"] = t
        d = _wrap(raw.get("disponibilidad"))
        if d and d["valor"]:
            phrase = norm_text(str(d["valor"]))
            iso = parse_availability(phrase, ref)
            d = {"valor": iso, "evidencia": d.get("evidencia") or d["valor"]} if iso else None
        out["disponibilidad"] = d
        return out
    items = raw.get("items") or []
    promo = raw.get("promocion")
    return {
        "items": {"valor": items, "evidencia": [i.get("evidencia") for i in items]} if items else None,
        "promocion": ({"valor": {"tipo": promo["tipo"], "valor": promo["valor"]}, "evidencia": promo.get("evidencia")}
                      if promo else None),
        "direccion_entrega": _wrap(raw.get("direccion_entrega")),
        "metodo_pago": _wrap(raw.get("metodo_pago")),
        "total_dicho": _wrap(raw.get("total_dicho")),
    }


def extract_llm(schema: str, transcript: str, cfg: dict, ref: date, client=None) -> dict:
    lc = cfg["extraction"]["llm"]
    client = client or _client(cfg)
    resp = client.messages.create(
        model=lc["model"], max_tokens=lc["max_tokens"], temperature=lc["temperature"],
        system=cfg["schemas"][schema]["llm_prompt"],
        messages=[{"role": "user", "content": f"Transcripcion:\n{transcript}"}])
    raw = _parse_json("".join(b.text for b in resp.content if getattr(b, "type", "") == "text"))
    return _to_internal(schema, raw, ref)
