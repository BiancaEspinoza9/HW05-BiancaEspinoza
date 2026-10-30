"""Capa de registro: compara el registro extraido contra la verdad terreno, clip por clip."""
from __future__ import annotations

from . import metrics as M
from .engine import record_values


def _truth_value(field: str, ftype: str, v):
    return v


def score_record(schema: str, result: dict, truth: dict, cfg: dict) -> dict:
    """Devuelve por campo: correcto (sobre todo lo extraido), correcto_verificado (solo si estado ==
    verificado), error absoluto en numericos, y estado de verificacion."""
    spec = cfg["schemas"][schema]["fields"]
    fields = result.get("fields") or {}
    out = {}
    for f, s in spec.items():
        pred = fields.get(f, {}).get("valor")
        estado = fields.get(f, {}).get("estado", "no_encontrado")
        t = truth.get(f)
        ok = M.field_correct(s["type"], t, pred, cfg) if (pred is not None or t is None) else False
        out[f] = {"ok": bool(ok), "estado": estado,
                  "ok_verificado": bool(ok and estado == "verificado"),
                  "abs_err": M.abs_error(s["type"], t, pred), "pred": pred, "truth": t}
    return out


def summarize_fields(scored: list[dict], schema: str, cfg: dict) -> dict:
    """scored: lista (por clip) de score_record. Exactitud por campo + macro F1 del categorico."""
    spec = cfg["schemas"][schema]["fields"]
    out = {}
    for f, s in spec.items():
        rows = [sc[f] for sc in scored if f in sc]
        n = len(rows)
        ver = [r for r in rows if r["estado"] == "verificado"]
        out[f] = {"n": n, "acc": sum(r["ok"] for r in rows) / n if n else None,
                  "cobertura_verificada": len(ver) / n if n else None,
                  "acc_entre_verificados": (sum(r["ok"] for r in ver) / len(ver)) if ver else None,
                  "mae": (sum(r["abs_err"] for r in rows if r["abs_err"] is not None) /
                          max(1, sum(r["abs_err"] is not None for r in rows))) if s["type"] in ("numeric", "numeric_currency") else None}
        if s["type"] == "categorical":
            labels = s["classes"]
            yt = [r["truth"] for r in rows]
            yp = [r["pred"] if r["pred"] in labels else "__none__" for r in rows]
            out[f]["macro_f1"] = M.macro_f1(yt, yp, labels)
    return out
