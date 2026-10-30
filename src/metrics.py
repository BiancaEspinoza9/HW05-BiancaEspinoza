"""Metricas de las dos capas: transcripcion (WER/CER) y registro (campos)."""
from __future__ import annotations

import difflib
import random
from collections import Counter

from .textnorm import norm_text


def _edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer_counts(ref: str, hyp: str, rules: dict | None = None) -> tuple[int, int]:
    r, h = norm_text(ref, rules).split(), norm_text(hyp, rules).split()
    return _edit_distance(r, h), len(r)


def cer_counts(ref: str, hyp: str, rules: dict | None = None) -> tuple[int, int]:
    r, h = list(norm_text(ref, rules)), list(norm_text(hyp, rules))
    return _edit_distance(r, h), len(r)


def wer(ref: str, hyp: str, rules: dict | None = None) -> float:
    e, n = wer_counts(ref, hyp, rules)
    return e / n if n else 0.0


def cer(ref: str, hyp: str, rules: dict | None = None) -> float:
    e, n = cer_counts(ref, hyp, rules)
    return e / n if n else 0.0


def bootstrap_ci(values: list[float], n: int = 2000, level: float = 0.95, seed: int = 7,
                 stat=lambda xs: sum(xs) / len(xs)) -> tuple[float, float, float]:
    """(estimado, bajo, alto). Remuestreo de clips (unidad = clip, no palabra)."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    est = stat(values)
    boots = sorted(stat([rng.choice(values) for _ in values]) for _ in range(n))
    lo = boots[int(((1 - level) / 2) * n)]
    hi = boots[min(n - 1, int((1 - (1 - level) / 2) * n))]
    return est, lo, hi


def paired_diff_ci(a: list[float], b: list[float], n: int = 2000, level: float = 0.95,
                   seed: int = 7) -> tuple[float, float, float]:
    """IC bootstrap de mean(a-b) para comparar dos configuraciones sobre los mismos clips."""
    return bootstrap_ci([x - y for x, y in zip(a, b)], n, level, seed)


def macro_f1(y_true: list, y_pred: list, labels: list) -> float:
    f1s = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def text_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


def _num(v):
    if isinstance(v, dict):
        v = v.get("monto", v.get("valor"))
    return None if v is None else float(v)


def field_correct(ftype: str, truth, pred, cfg: dict) -> bool:
    """Un campo cuenta como correcto segun su tipo. pred None => incorrecto (salvo truth None)."""
    m = cfg["metrics"]
    if truth is None:
        return pred is None
    if pred is None:
        return False
    if ftype == "numeric":
        return abs(_num(truth) - _num(pred)) <= m["numeric_tolerance"]
    if ftype == "numeric_currency":
        return (abs(_num(truth) - _num(pred)) <= m["numeric_tolerance"]
                and str(truth.get("moneda")) == str(pred.get("moneda")))
    if ftype == "text":
        return text_similarity(str(truth), str(pred)) >= m["text_match_threshold"]
    if ftype in ("categorical", "date"):
        return str(truth) == str(pred)
    if ftype == "list":
        return Counter(norm_text(x) for x in truth) == Counter(norm_text(x) for x in pred)
    if ftype == "promo":
        return truth.get("tipo") == pred.get("tipo") and abs(float(truth["valor"]) - float(pred["valor"])) <= m["numeric_tolerance"]
    if ftype == "items":
        def key(it):
            return (norm_text(it["producto"]), norm_text(it.get("tamano") or ""),
                    int(it["cantidad"]), round(float(it["precio_unitario"]), 2))
        return sorted(map(key, truth)) == sorted(map(key, pred))
    raise ValueError(ftype)


def abs_error(ftype: str, truth, pred) -> float | None:
    """Error absoluto para campos numericos (Capa de registro)."""
    if ftype in ("numeric", "numeric_currency") and truth is not None and pred is not None:
        return abs(_num(truth) - _num(pred))
    return None


def list_prf(truth: list[str], pred: list[str]) -> tuple[float, float, float]:
    t, p = Counter(norm_text(x) for x in truth), Counter(norm_text(x) for x in pred)
    tp = sum((t & p).values())
    prec = tp / sum(p.values()) if p else 0.0
    rec = tp / sum(t.values()) if t else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1
