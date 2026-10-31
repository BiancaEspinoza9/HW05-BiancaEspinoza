import subprocess
import sys

import numpy as np
import pytest

from src import metrics as M
from src.data import load_clips
from src.engine import ROOT, load_config, process_audio
from src.evaluate import score_record
from src.ladder import add_noise, speech_power
from src.numbers_es import text_to_number
from src.textnorm import norm_text
from src.validate import validate_order

cfg = load_config()


@pytest.mark.parametrize("s,v", [("seis mil quinientos", 6500), ("6500", 6500), ("6.500", 6500),
                                 ("cuarenta y cinco", 45), ("ciento veinte", 120), ("once mil trescientos", 11300),
                                 ("veintiuno", 21), ("mil doscientos", 1200), ("6 mil 500", 6500), ("45,50", 45.5)])
def test_numbers(s, v):
    assert text_to_number(s) == v


def test_norm_equivalence_and_porciento():
    assert norm_text("Seis mil quinientos soles.") == norm_text("6500 soles")
    assert "porciento" in norm_text("diez por ciento")
    assert norm_text("diez por ciento") == norm_text("10%")


def test_wer_cer():
    assert M.wer("hola mundo", "Hola, mundo.") == 0
    assert M.wer("a b c d", "a b x d") == 0.25
    assert M.cer("abc", "abd") == pytest.approx(1 / 3)


def test_macro_f1():
    assert M.macro_f1(["a", "b"], ["a", "b"], ["a", "b"]) == 1.0


def test_engine_has_no_ui_imports():
    code = ("import sys, src.engine; bad=[m for m in sys.modules if m.split('.')[0] in "
            "('streamlit','gradio','flask','tkinter','PyQt5')]; sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0


@pytest.mark.parametrize("task,schema", [("interviews", "interview"), ("calls", "order")])
def test_rules_perfect_on_reference_transcripts(task, schema):
    """Cota superior: con ASR perfecto, las reglas deben acertar y verificar todo."""
    for c in load_clips(cfg, task):
        r = process_audio(None, schema, config=cfg, reference_date=c["fecha_referencia"],
                          transcript_result={"transcript": c["reference"], "words": []})
        assert r["error"] is None
        for f, v in score_record(schema, r, c["campos"], cfg).items():
            assert v["ok"], (c["id"], f, v)


def test_unverified_when_value_not_in_transcript():
    from src.verify import verify_record
    from datetime import date
    out = verify_record("interview", {"anios_experiencia": {"valor": 9, "evidencia": "nueve anos de experiencia"}},
                        "tengo cuatro años de experiencia", cfg, date(2026, 10, 26))
    assert out["anios_experiencia"]["estado"] == "sin_verificar"


def test_validation_ok_and_diagnosis():
    items = [{"cantidad": 2, "precio_unitario": 45}, {"cantidad": 1, "precio_unitario": 18}]
    promo = {"tipo": "monto", "valor": 10}
    assert validate_order({"items": items, "promocion": promo, "total_dicho": 98}, cfg)["estado"] == "ok"
    asr = validate_order({"items": items, "promocion": promo, "total_dicho": 148}, cfg)   # 98 con 9 oido como 14?
    assert asr["estado"] == "desajuste"
    # 15 oido como 50: 2*45 + 50 - 10 = 130 -> real 2*45 + 15 - 10 = 95
    r = validate_order({"items": [{"cantidad": 2, "precio_unitario": 45}, {"cantidad": 1, "precio_unitario": 50}],
                        "promocion": promo, "total_dicho": 95}, cfg)
    assert r["causa"] == "asr" and r["regla"] == "D1"
    assert validate_order({"items": items, "promocion": promo, "total_dicho": 120}, cfg)["causa"] == "hablante"


def test_snr_is_the_requested_one():
    sr = 16000
    t = np.arange(sr * 3) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.7 * t) > 0)).astype("float32")
    _, measured = add_noise(x, sr, 10, 11, 30)
    assert measured == pytest.approx(10, abs=0.3)


def test_items_tolerate_asr_misspelled_products():
    """Whisper escribe 'pollas al abrazo' / 'gasezas': el producto se reconoce por parecido, la evidencia sigue literal."""
    tr = ("Le hago el resumen final. Dos pollas al abrazo, tamaño entero a 45 soles cada uno. "
          "Y dos gasezas, tamaño litro a 8 soles cada uno. El total a pagar es de 98 soles.")
    r = process_audio(None, "order", config=cfg, transcript_result={"transcript": tr, "words": []})
    items = r["record"]["items"]
    assert [(i["producto"], i["tamano"], i["cantidad"], i["precio_unitario"]) for i in items] == [
        ("pollo a la brasa", "entero", 2, 45.0), ("gaseosa", "litro", 2, 8.0)]
    assert r["verified"]["items"] == "verificado"
