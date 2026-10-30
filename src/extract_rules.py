"""Camino 1 de extraccion: reglas (regex) + capa de numeros. Sin modelos de lenguaje.

Cada campo devuelve {"valor": ..., "evidencia": <fragmento del transcript normalizado>} o None.
La verificacion literal vive en verify.py (se aplica igual a este camino y al LLM).
"""
from __future__ import annotations

import re
from datetime import date

from dateutil.relativedelta import relativedelta

from .textnorm import norm_text, strip_accents

MONTHS = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
          "agosto": 8, "setiembre": 9, "septiembre": 9, "octubre": 10, "noviembre": 11,
          "diciembre": 12}
_NUM = r"(\d+(?:\.\d+)?)"
_NUM1 = r"(\d+(?:\.\d+)?|un|una)"


def _n(tok: str) -> float:
    return 1.0 if tok in ("un", "una") else float(tok)


def _prefer_near(text: str, matches: list, keywords: list[str], window: int = 140):
    """Entre varias coincidencias, la que tiene una palabra clave justo antes."""
    for m in matches:
        if any(k in text[max(0, m.start() - window):m.start()] for k in keywords):
            return m
    return matches[0] if matches else None


# ----------------------------------------------------------------- entrevistas
def parse_availability(phrase: str, ref: date) -> str | None:
    """Frase (normalizada) -> fecha ISO absoluta. Compartido con la verificacion."""
    p = phrase
    if re.search(r"\b(inmediat\w*|ahora mismo|ya mismo)\b", p):
        return ref.isoformat()
    m = re.search(rf"\b(?:en|dentro de)\s+{_NUM1}\s+(dias?|semanas?|mes(?:es)?)\b", p)
    if m:
        n, unit = int(_n(m.group(1))), m.group(2)
        if unit.startswith("dia"):
            return (ref + relativedelta(days=n)).isoformat()
        if unit.startswith("semana"):
            return (ref + relativedelta(weeks=n)).isoformat()
        return (ref + relativedelta(months=n)).isoformat()
    if re.search(r"\bproxima semana\b", p):
        return (ref + relativedelta(weeks=1)).isoformat()
    if re.search(r"\bproximo mes\b", p):
        return (ref + relativedelta(months=1)).isoformat()
    m = re.search(rf"\b(\d+|primero)\s+de\s+({'|'.join(MONTHS)})\b", p)
    if m:
        day = 1 if m.group(1) == "primero" else int(m.group(1))
        d = date(ref.year, MONTHS[m.group(2)], day)
        if d < ref:
            d = d.replace(year=ref.year + 1)
        return d.isoformat()
    return None


_AVAIL_RE = re.compile(
    r"(?:\b(?:inmediat\w*|ahora mismo|ya mismo)\b"
    rf"|\b(?:en|dentro de)\s+{_NUM1}\s+(?:dias?|semanas?|mes(?:es)?)\b"
    r"|\bproxima semana\b|\bproximo mes\b"
    rf"|\b(?:\d+|primero)\s+de\s+(?:{'|'.join(MONTHS)})\b)")


def _years(t: str):
    pats = [rf"{_NUM1}\s+anos?(\s+y\s+medio)?\s+de\s+experiencia",
            rf"experiencia\s+(?:laboral\s+)?(?:de|como)?\s*{_NUM1}\s+anos?(\s+y\s+medio)?",
            rf"\b(?:tengo|cuento con|llevo)\s+{_NUM1}\s+anos?(\s+y\s+medio)?"]
    for p in pats:
        m = re.search(p, t)
        if m:
            v = _n(m.group(1)) + (0.5 if m.group(2) else 0)
            return {"valor": v, "evidencia": m.group(0)}
    return None


def _puesto(t: str):
    pats = [r"\bultimo (?:puesto|cargo)\s+(?:fue|era|ha sido)\s+(?:el de\s+)?(.+?)(?=\s+en\s+|\s+donde\s+|\s+para\s+|\s+durante\s+|\s*\||$)",
            r"\b(?:trabaje|trabajo|me desempene|me desempeno|estuve)\s+como\s+(.+?)(?=\s+en\s+|\s+donde\s+|\s+para\s+|\s+durante\s+|\s*\||$)",
            r"\bfui\s+(.+?)(?=\s+en\s+|\s+donde\s+|\s+para\s+|\s+durante\s+|\s*\||$)"]
    for p in pats:
        m = re.search(p, t)
        if m:
            return {"valor": m.group(1).strip(), "evidencia": m.group(0).strip()}
    return None


def _techs(t: str, glossary: list[str]):
    toks = t.split()
    found, ev = [], []
    for term in glossary:
        tt = norm_text(term).split()
        for i in range(len(toks) - len(tt) + 1):
            if toks[i:i + len(tt)] == tt:
                if term not in found:
                    found.append(term)
                    ev.append(" ".join(tt))
                break
    return {"valor": found, "evidencia": ev} if found else None


def _salary(t: str, cur_cfg: dict):
    cur_words = {w: c for c, ws in cur_cfg.items() for w in ws if w.isalpha()}
    alt = "|".join(sorted(cur_words, key=len, reverse=True))
    matches = list(re.finditer(rf"{_NUM}\s+(?:de\s+)?({alt})\b", t))
    m = _prefer_near(t, matches, ["pretension", "sueldo", "salario", "busco", "esperaria",
                                   "aspiro", "remuneracion", "economico"])
    if m:
        return {"valor": {"monto": float(m.group(1)), "moneda": cur_words[m.group(2)]},
                "evidencia": m.group(0)}
    m = re.search(rf"\bs\s+{_NUM}\b", t)       # "S/ 6500"
    if m:
        return {"valor": {"monto": float(m.group(1)), "moneda": "PEN"}, "evidencia": m.group(0)}
    return None


def _availability(t: str, ref: date):
    ms = list(_AVAIL_RE.finditer(t))
    m = _prefer_near(t, ms, ["disponib", "incorpor", "empez", "comenz", "iniciar", "preaviso", "podria"])
    if not m:
        return None
    # ventana minima desde la palabra clave para conservar la frase completa
    phrase = t[m.start():m.end()]
    v = parse_availability(phrase, ref)
    return {"valor": v, "evidencia": phrase} if v else None


def _english(t: str, kw: dict):
    cefr = {"a1": "basico", "a2": "basico", "b1": "intermedio", "b2": "intermedio",
            "c1": "avanzado", "c2": "avanzado"}
    for m in re.finditer(r"\bingles\b", t):
        win = t[max(0, m.start() - 60):m.end() + 60]
        hits = []
        for cls, words in kw.items():
            for w in words:
                for mm in re.finditer(rf"\b{w}\b", win):
                    hits.append((mm.start(), cls, w))
        for code, cls in cefr.items():
            for mm in re.finditer(rf"\b{code}\b", win):
                hits.append((mm.start(), cls, code))
        if hits:
            # el termino mas cercano a la palabra "ingles"
            pos = win.find("ingles")
            hits.sort(key=lambda h: abs(h[0] - pos))
            return {"valor": hits[0][1], "evidencia": hits[0][2]}
    return None


def extract_interview(transcript: str, cfg: dict, ref: date) -> dict:
    sc = cfg["schemas"]["interview"]
    t = norm_text(transcript, cfg["normalization"], keep_breaks=True)
    return {
        "anios_experiencia": _years(t),
        "ultimo_puesto": _puesto(t),
        "tecnologias": _techs(t, sc["glossary"]),
        "pretension_salarial": _salary(t, sc["currencies"]),
        "disponibilidad": _availability(t, ref),
        "nivel_ingles": _english(t, sc["english_keywords"]),
    }


# --------------------------------------------------------------------- pedidos
def _plural(word: str) -> str:
    return re.escape(word) + r"(?:s|es)?"


def _phrase_re(phrase: str) -> str:
    return r"\s+".join(_plural(w) for w in phrase.split())


def _items(t: str, menu: dict):
    prod_alt = "|".join(f"(?P<p{i}>{_phrase_re(p)})" for i, p in enumerate(menu))
    size_alt = "|".join(sorted({_phrase_re(s) for v in menu.values() for s in v["sizes"]}, key=len, reverse=True))
    pat = re.compile(rf"\b{_NUM1}\s+(?:{prod_alt})(?:\s+tamano\s+({size_alt}))?\s+a\s+{_NUM}\s+soles?\b")
    prods = list(menu)

    def run(seg: str):
        out = []
        for m in pat.finditer(seg):
            idx = next(i for i in range(len(prods)) if m.group(f"p{i}"))
            raw_size = m.group(len(prods) + 2)
            size = None
            if raw_size:
                for s in menu[prods[idx]]["sizes"]:
                    if re.fullmatch(_plural(s), raw_size):
                        size = s
            out.append({"producto": prods[idx], "tamano": size, "cantidad": int(_n(m.group(1))),
                        "precio_unitario": float(m.group(len(prods) + 3)), "evidencia": m.group(0)})
        return out

    cut = t.rfind("resumen")
    items = run(t[cut:]) if cut >= 0 else []
    if not items:                       # sin resumen: gana la ultima mencion de cada (producto,tamano)
        last = {}
        for it in run(t):
            last[(it["producto"], it["tamano"])] = it
        items = list(last.values())
    return items


def _promo(t: str):
    pats = [(rf"{_NUM}\s+(soles?|porciento)\s+de\s+descuento", 1, 2),
            (rf"descuento\s+(?:de\s+)?{_NUM}\s+(soles?|porciento)", 1, 2)]
    best = None
    for p, gv, gu in pats:
        for m in re.finditer(p, t):
            if best is None or m.start() > best[0].start():
                best = (m, gv, gu)
    if not best:
        return None
    m, gv, gu = best
    tipo = "porcentaje" if "porciento" in m.group(gu) else "monto"
    return {"valor": {"tipo": tipo, "valor": float(m.group(gv))}, "evidencia": m.group(0)}


def _address(raw: str):
    r = re.sub(r"\b(Jr|Av|Mz|Lt|Dpto|Nro|Urb|Psje|Cal)\.", r"\1", raw, flags=re.I)
    m = None
    for m in re.finditer(r"direcci[oó]n(?:\s+de\s+entrega)?\s+(?:es|ser[ií]a|ser[aá])\s+(.+?)(?:[.?!]\s|[.?!]$|$)",
                         r, flags=re.I | re.S):
        pass
    if not m:
        return None
    ev = norm_text(m.group(1))
    return {"valor": m.group(1).strip(), "evidencia": ev}


def _payment(t: str, kw: dict):
    best = None
    for cls, words in kw.items():
        for w in words:
            for m in re.finditer(rf"\b{w}\b", t):
                if best is None or m.start() > best[0]:
                    best = (m.start(), cls, w)
    return {"valor": best[1], "evidencia": best[2]} if best else None


def _total(t: str):
    ms = list(re.finditer(rf"\btotal\b\D{{0,40}}?{_NUM}\s+soles?\b", t))
    ms += list(re.finditer(rf"{_NUM}\s+soles?\s+en\s+total\b", t))
    if not ms:
        return None
    m = max(ms, key=lambda x: x.start())
    return {"valor": float(m.group(1)), "evidencia": m.group(0)}


def extract_order(transcript: str, cfg: dict, ref: date) -> dict:
    sc = cfg["schemas"]["order"]
    t = norm_text(transcript, cfg["normalization"], keep_breaks=True)
    items = _items(t, sc["menu"])
    return {
        "items": {"valor": items, "evidencia": [i["evidencia"] for i in items]} if items else None,
        "promocion": _promo(t),
        "direccion_entrega": _address(transcript),
        "metodo_pago": _payment(t, sc["payment_keywords"]),
        "total_dicho": _total(t),
    }


EXTRACTORS = {"interview": extract_interview, "order": extract_order}


def extract_rules(schema: str, transcript: str, cfg: dict, ref: date) -> dict:
    return EXTRACTORS[schema](transcript, cfg, ref)
