"""Validacion aritmetica del pedido y diagnostico del desajuste (ASR vs hablante).

Chequeo:  sum(cantidad * precio_unitario) - descuento  ==  total_dicho

Regla de diagnostico (documentada en README):
  D1  Si cambiar UN solo numero por su confusable acustico (15<->50, 60<->70, ...) hace
      cuadrar la cuenta  => causa "asr" (y se propone la reparacion).
  D2  Si no, y alguno de los numeros involucrados fue dicho con confianza < umbral
      => causa "asr".
  D3  Si no, los numeros se oyeron firmes y aun asi no cuadran => causa "hablante".
"""
from __future__ import annotations

import copy

from .numbers_es import find_numbers
from .textnorm import strip_accents


def compute_expected(items: list[dict], promo: dict | None) -> tuple[float, float, float]:
    subtotal = sum(i["cantidad"] * i["precio_unitario"] for i in items)
    disc = 0.0
    if promo:
        disc = promo["valor"] if promo["tipo"] == "monto" else subtotal * promo["valor"] / 100
    return subtotal, disc, subtotal - disc


def _alternatives(v: float, pairs: list[list[int]]) -> list[float]:
    alts = set()
    iv = int(v)
    if iv != v:
        return []
    for a, b in pairs:
        if iv == a:
            alts.add(b)
        if iv == b:
            alts.add(a)
        if iv >= 20 and (iv // 10) * 10 == a and a >= 40:
            alts.add(b + iv % 10)
        if iv >= 20 and (iv // 10) * 10 == b and b >= 40:
            alts.add(a + iv % 10)
    alts.discard(iv)
    return [float(x) for x in sorted(alts)]


def _slots(items, promo, total):
    s = []
    for k in range(len(items)):
        s.append((f"cantidad[{k}]", "cantidad", k))
        s.append((f"precio_unitario[{k}]", "precio", k))
    if promo:
        s.append(("promocion.valor", "promo", 0))
    s.append(("total_dicho", "total", 0))
    return s


def _apply(items, promo, total, slot, alt):
    items, promo = copy.deepcopy(items), copy.deepcopy(promo)
    _, t, k = slot
    if t == "cantidad":
        items[k]["cantidad"] = alt
    elif t == "precio":
        items[k]["precio_unitario"] = alt
    elif t == "promo":
        promo["valor"] = alt
    else:
        total = alt
    return items, promo, total


def _get(items, promo, total, slot):
    _, t, k = slot
    return {"cantidad": lambda: items[k]["cantidad"], "precio": lambda: items[k]["precio_unitario"],
            "promo": lambda: promo["valor"], "total": lambda: total}[t]()


def number_confidences(words: list[dict]) -> dict[float, float]:
    """Probabilidad minima de cada numero dicho (agrupando 'cuarenta y cinco' como un numero)."""
    toks = [strip_accents("".join(c for c in w["w"].lower() if c.isalnum() or c in ".,")) for w in words]
    out: dict[float, float] = {}
    for a, b, v in find_numbers(toks, include_one=False):
        p = min(words[i]["p"] for i in range(a, b))
        out[v] = min(p, out.get(v, 1.0))
    return out


def validate_order(rec: dict, cfg: dict, words: list[dict] | None = None) -> dict:
    """rec: valores del registro (items, promocion, total_dicho)."""
    sc, tol = cfg["schemas"]["order"], cfg["validation"]["tolerance"]
    items, promo, total = rec.get("items"), rec.get("promocion"), rec.get("total_dicho")
    if not items or total is None:
        return {"estado": "no_evaluable", "causa": None}
    sub, disc, exp = compute_expected(items, promo)
    res = {"estado": "ok", "subtotal": sub, "descuento": disc, "esperado": exp,
           "total_dicho": total, "diferencia": total - exp, "causa": None, "reparacion": None}
    if abs(total - exp) <= tol:
        return res
    res["estado"] = "desajuste"
    pairs = sc["confusable_numbers"]
    for slot in _slots(items, promo, total):
        cur = _get(items, promo, total, slot)
        for alt in _alternatives(cur, pairs):
            i2, p2, t2 = _apply(items, promo, total, slot, alt)
            if p2 is not None and p2["tipo"] == "porcentaje" and slot[1] == "promo" and alt > 100:
                continue
            if abs(t2 - compute_expected(i2, p2)[2]) <= tol:           # D1
                res.update(causa="asr", regla="D1", reparacion={"campo": slot[0], "dicho": cur, "probable": alt})
                return res
    thr = sc["diagnosis"]["low_confidence_threshold"]
    if words:
        conf = number_confidences(words)
        involved = {float(i["cantidad"]) for i in items} | {float(i["precio_unitario"]) for i in items} | {float(total)}
        if promo:
            involved.add(float(promo["valor"]))
        low = {v: p for v, p in conf.items() if v in involved and p < thr}
        if low:                                                         # D2
            res.update(causa="asr", regla="D2", reparacion={"numeros_dudosos": low})
            return res
    res.update(causa="hablante", regla="D3")                            # D3
    return res
