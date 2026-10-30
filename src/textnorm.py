"""Normalizacion de texto: una sola regla aplicada igual a referencia e hipotesis."""
from __future__ import annotations

import re
import unicodedata

from .numbers_es import find_numbers, fmt

_DEFAULT = {"lowercase": True, "strip_accents": True, "strip_punctuation": True,
            "numbers_to_digits": True, "collapse_whitespace": True}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_text(text: str, rules: dict | None = None, include_one: bool = False,
              keep_breaks: bool = False) -> str:
    """Regla (README): minusculas -> sin tildes -> sin puntuacion (se conservan digitos
    y el separador dentro de numeros) -> numeros en palabras a digitos -> espacios simples."""
    r = {**_DEFAULT, **(rules or {})}
    t = text.replace("%", " por ciento ")
    if r["lowercase"]:
        t = t.lower()
    if r["strip_accents"]:
        t = strip_accents(t)
    if r["strip_punctuation"]:
        t = re.sub(r"(?<=\d)[.,](?=\d)", "\u0001", t)      # protege 6.500 / 45,50
        t = re.sub(r"[^\w\s\u0001]", " | ", t)             # la puntuacion separa numeros
        t = t.replace("\u0001", ".")
        t = re.sub(r"_", " ", t)
    t = " ".join(t.split()).replace("por ciento", "porciento")   # "ciento" no es el numero 100 aqui
    toks = t.split()
    if r["numbers_to_digits"]:
        from .numbers_es import parse_digits
        out, last = [], 0
        for a, b, v in find_numbers(toks, include_one=include_one):
            out += toks[last:a]
            out.append(fmt(v))
            last = b
        out += toks[last:]
        toks = out
    if not keep_breaks:
        toks = [x for x in toks if x != "|"]
    return " ".join(toks)


def tokens(text: str, rules: dict | None = None) -> list[str]:
    return norm_text(text, rules).split()
