"""Numeros en espanol hablado -> valor. Capa de normalizacion de numeros.

Reglas (documentadas en README):
  R1  0-29 son palabras simples (uno ... veintinueve, incluye veintiuno/veintidos...).
  R2  30-90 son decenas; "y" + unidad (1-9) las completa ("cuarenta y cinco").
  R3  centenas: cien/ciento, doscientos ... novecientos; se suman a R1/R2 sin "y".
  R4  "mil" multiplica el grupo previo (1 si no hay) y admite un grupo <1000 despues.
  R5  "un/una" solo son numero si include_one=True (son articulos casi siempre).
  R6  Digitos: "6500", "6.500", "6,500" (3 digitos tras separador = miles), "45.50" (decimal).
  R7  Dos numeros que no se pueden combinar ("tres cincuenta") quedan separados.
Entrada ya en minusculas y sin tildes (ver textnorm.norm_text).
"""
from __future__ import annotations

import re

_BASE = {
    "cero": 0, "uno": 1, "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintiun": 21,
    "veintiuna": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24,
    "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28,
    "veintinueve": 29,
}
_TENS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
         "ochenta": 80, "noventa": 90}
_HUNDREDS = {"cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200,
             "trescientos": 300, "trescientas": 300, "cuatrocientos": 400,
             "cuatrocientas": 400, "quinientos": 500, "quinientas": 500,
             "seiscientos": 600, "seiscientas": 600, "setecientos": 700, "setecientas": 700,
             "ochocientos": 800, "ochocientas": 800, "novecientos": 900, "novecientas": 900}
_ONE_WORDS = {"un", "una"}
NUMBER_WORDS = set(_BASE) | set(_TENS) | set(_HUNDREDS) | {"mil", "y"}

_DIGITS = re.compile(r"^\d+(?:[.,]\d+)*$")


def parse_digits(tok: str) -> float | None:
    """R6: '6.500'->6500, '6,500'->6500, '45.50'->45.5, '45,5'->45.5."""
    if not _DIGITS.match(tok):
        return None
    for sep in ".,":
        if sep in tok:
            head, _, tail = tok.rpartition(sep)
            groups = tok.replace(",", ".").split(".")
            if len(groups) > 2 or (len(tail) == 3 and 1 <= len(groups[0]) <= 3):
                return float("".join(groups))      # separador de miles
            return float(head.replace(".", "").replace(",", "") + "." + tail)
    return float(tok)


def _below_1000(toks: list[str], i: int, include_one: bool):
    """Devuelve (valor, siguiente_i) o None."""
    n, start, val = len(toks), i, 0
    if i < n and toks[i] in _HUNDREDS:
        val += _HUNDREDS[toks[i]]
        i += 1
    if i < n and (toks[i] in _BASE) and (include_one or toks[i] not in _ONE_WORDS):
        if toks[i] == "cero" and i > start:
            return (val, i) if val else None
        val += _BASE[toks[i]]
        i += 1
    elif i < n and toks[i] in _TENS:
        val += _TENS[toks[i]]
        i += 1
        if i + 1 < n and toks[i] == "y" and toks[i + 1] in _BASE and 1 <= _BASE[toks[i + 1]] <= 9 \
                and (include_one or toks[i + 1] not in _ONE_WORDS):
            val += _BASE[toks[i + 1]]
            i += 2
    if i == start:
        return None
    return val, i


def parse_number_at(toks: list[str], i: int, include_one: bool = False):
    """Intenta leer un numero (palabras o digitos) en toks[i:]. -> (valor, siguiente_i) | None"""
    if i >= len(toks):
        return None
    d = parse_digits(toks[i])
    if d is not None:
        if d < 1000 and i + 1 < len(toks) and toks[i + 1] == "mil":     # "6 mil 500"
            total, j = d * 1000, i + 2
            nxt = parse_digits(toks[j]) if j < len(toks) else None
            if nxt is not None and nxt < 1000:
                return total + nxt, j + 1
            sub = _below_1000(toks, j, include_one)
            if sub:
                return total + sub[0], sub[1]
            return total, j
        return d, i + 1
    if toks[i] == "mil":                       # "mil doscientos"
        total, i = 1000, i + 1
        sub = _below_1000(toks, i, include_one)
        if sub:
            total, i = total + sub[0], sub[1]
        return float(total), i
    g = _below_1000(toks, i, include_one)
    if g is None:
        return None
    val, j = g
    if j < len(toks) and toks[j] == "mil":     # R4
        total, j = val * 1000, j + 1
        sub = _below_1000(toks, j, include_one)
        if sub:
            total, j = total + sub[0], sub[1]
        return float(total), j
    return float(val), j


def find_numbers(tokens: list[str], include_one: bool = False):
    """Todas las apariciones: lista de (inicio, fin_exclusivo, valor)."""
    out, i = [], 0
    while i < len(tokens):
        r = parse_number_at(tokens, i, include_one)
        if r:
            out.append((i, r[1], r[0]))
            i = r[1]
        else:
            i += 1
    return out


def fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def text_to_number(text: str, include_one: bool = True) -> float | None:
    """'seis mil quinientos' / '6500' / '6.500' -> 6500.0. None si no hay numero."""
    toks = text.lower().split()
    nums = find_numbers(toks, include_one=include_one)
    return nums[0][2] if nums else None
