"""App Streamlit. Carga resultados precomputados (results/summary.json); NO re-transcribe el corpus.
Solo la pestaña 'Probador' llama al motor, y solo con el audio que subes.
  streamlit run app.py
"""
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data import find_audio, load_clips
from src.engine import ROOT, load_config, process_audio

cfg = load_config()
st.set_page_config(page_title=cfg["app"]["title"], layout="wide")
st.title(cfg["app"]["title"])

sf = ROOT / cfg["app"]["results_file"]
S = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else None
runs = S["runs"] if S else []

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1 Problema y método", "2 Corpus", "3 Resultados",
                                        "4 Probador", "5 Pedidos (Tarea 2)"])

with tab1:
    st.markdown("""
**El problema.** Tu teléfono ya transcribe, pero el texto no es un registro: lo que una organización
necesita son *campos verificados* en una tabla y una medida de cuánto confiar en ellos.

**Método (un motor, dos tareas).** `process_audio(audio, schema)`:
1. *Una vez*: guiones → grabación → verdad terreno → transcripción cacheada.
2. *Por cada audio nuevo*: ASR (Whisper) → extracción (reglas o LLM) → **verificación literal**
   (todo valor debe existir en el transcript; si no, `sin_verificar`) → validación aritmética (pedidos).

**Dos capas de métricas, siempre separadas:** transcripción (WER, CER) y registro (exactitud por campo,
macro-F1 del categórico, error absoluto de los numéricos).

**Nunca debe usarse para** decidir contrataciones ni cobros sin revisión humana, ni con datos reales de
terceros sin consentimiento.
""")
    st.caption("Todos los datos del corpus son inventados; todas las personas que aparecen consintieron.")

with tab2:
    task = st.radio("Corpus", ["interviews", "calls"], horizontal=True,
                    format_func=lambda t: "Entrevistas" if t == "interviews" else "Llamadas de pedido")
    clips = load_clips(cfg, task)
    if S:
        d = {k: v for k, v in S["distribucion_verdad_terreno"].items() if k.startswith(task)}
        with st.expander("Distribución de cada campo (verdad terreno)"):
            st.json(d)
    cid = st.selectbox("Clip", [c["id"] for c in clips])
    clip = next(c for c in clips if c["id"] == cid)
    audio = find_audio(cfg, cid)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Audio")
        if audio:
            st.audio(str(audio))
        else:
            st.info("El audio no se versiona (ver README). Graba el clip siguiendo el guion.")
        st.write("Campos correctos:")
        st.json(clip["campos"])
    with c2:
        st.subheader("Transcripción")
        sel = [r for r in runs if r["task"] == task and r["group"] == "real" and r["level"] == "L0"]
        hyp = next((p["transcript"] for r in sel for p in r["per_clip"] if p["id"] == cid), None)
        st.text_area("Referencia (guion)", clip["reference"], height=180)
        st.text_area("Hipótesis del ASR (caché)", hyp or "— sin corrida real en caché —", height=180)

with tab3:
    if not S:
        st.warning("Aún no hay results/summary.json. Ejecuta scripts/run_asr.py y scripts/run_eval.py.")
    else:
        rows = [{"tarea": r["task"], "grupo": r["group"], "nivel": r["level"], "modelo": r["model"],
                 "device": r["device"], "precisión": r["compute_type"], "glosario": r["glossary_mode"],
                 "extractor": r["extractor"], "n": r["n_clips"],
                 "WER": round(r["wer"]["est"], 3), "WER IC95": f"[{r['wer']['lo']:.3f}, {r['wer']['hi']:.3f}]",
                 "CER": round(r["cer"]["est"], 3), "s/min": round(r["sec_per_min"], 1),
                 "VRAM pico MB": r["vram_peak_mb"], "exactitud campos": round(r["field_acc_mean"], 3)} for r in runs]
        df = pd.DataFrame(rows)
        st.caption("Los grupos `synthetic/*` se reportan aparte y nunca se mezclan con `real`.")
        st.subheader("Capa de transcripción + resumen del registro")
        st.dataframe(df, use_container_width=True)
        st.subheader("Capa de registro: por campo")
        pick = st.selectbox("Corrida", [f"{r['task']} | {r['group']} | {r['level']} | {r['run']} | {r['extractor']}" for r in runs])
        r = runs[[f"{x['task']} | {x['group']} | {x['level']} | {x['run']} | {x['extractor']}" for x in runs].index(pick)]
        st.dataframe(pd.DataFrame(r["fields"]).T, use_container_width=True)
        st.subheader("¿El WER predice si el registro sirve?")
        st.dataframe(pd.DataFrame(r["per_clip"])[["id", "wer", "cer", "campos_ok", "campos_n", "campos_ok_verificados"]],
                     use_container_width=True)
        if S["wer_vs_registro"]:
            st.write("Ejemplos extremos (corpus real):")
            st.json(S["wer_vs_registro"])
        if S["comparaciones_texto"]:
            st.subheader("¿Cambió el texto? (CPU vs GPU, precisión)")
            st.dataframe(pd.DataFrame(S["comparaciones_texto"]), use_container_width=True)
        if S["efecto_glosario"]:
            st.subheader("Efecto del glosario (WER sin − con, IC95 pareado)")
            st.json(S["efecto_glosario"])

with tab4:
    st.write("Sube un clip: se transcribe con el motor y se muestra el registro y qué campos se verificaron.")
    schema = st.radio("Esquema", ["interview", "order"], horizontal=True)
    extractor = st.radio("Extractor", ["rules", "llm"], horizontal=True)
    up = st.file_uploader("Audio", type=["wav", "mp3", "m4a", "ogg", "opus", "flac"])
    if up and st.button("Procesar"):
        with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as f:
            f.write(up.getvalue())
        with st.spinner("Transcribiendo…"):
            res = process_audio(f.name, schema, extractor, cfg)
        if res["error"]:
            st.error(res["error"])
        st.caption(f"dispositivo: **{res['device']}** · modelo {res['model']} · {res['compute_type']} · "
                   f"{res['elapsed_s']} s · VRAM pico {res['vram_peak_mb']} MB")
        st.text_area("Transcripción", res["transcript"], height=200)
        st.dataframe(pd.DataFrame([{"campo": k, "valor": json.dumps(v["valor"], ensure_ascii=False),
                                    "estado": v["estado"], "evidencia": json.dumps(v["evidencia"], ensure_ascii=False)}
                                   for k, v in res["fields"].items()]), use_container_width=True)
        if res["validation"]:
            st.subheader("Validación aritmética")
            st.json(res["validation"])

with tab5:
    st.subheader("Escalera de degradación, validación y atrapadas vs perdidas")
    calls = [r for r in runs if r["task"] == "calls" and r["group"] == "real" and r["extractor"] == "rules"]
    if not calls:
        st.info("Sin corridas reales de llamadas en caché. Ejecuta: "
                "`python scripts/run_asr.py --task calls --levels L0 L1 L2 L3` y luego `python scripts/run_eval.py`.")
        calls = [r for r in runs if r["task"] == "calls"]
        if calls:
            st.warning("Mostrando corridas sintéticas (grupo aparte).")
    if calls:
        lad = pd.DataFrame([{"nivel": r["level"], "grupo": r["group"], "WER": r["wer"]["est"],
                             "exactitud de campos": r["field_acc_mean"],
                             "SNR medido (dB)": next((p["snr_measured"] for p in r["per_clip"] if p["snr_measured"]), None)}
                            for r in calls]).sort_values("nivel")
        st.line_chart(lad.set_index("nivel")[["WER", "exactitud de campos"]])
        st.dataframe(lad, use_container_width=True)
        v = pd.DataFrame([{"nivel": r["level"], **r["validation"]} for r in calls]).sort_values("nivel")
        st.write("Trampas atrapadas vs perdidas por nivel, falsas alarmas y acierto del diagnóstico:")
        st.dataframe(v, use_container_width=True)
        lv = st.selectbox("Detalle por llamada", [r["level"] for r in calls])
        rr = next(r for r in calls if r["level"] == lv)
        st.dataframe(pd.DataFrame([{"id": p["id"], "trampa total": p["trampa_total"], "marcada": p["flagged"],
                                    "causa real": p.get("causa_real"),
                                    "diagnóstico": (p["validacion"] or {}).get("causa"),
                                    "regla": (p["validacion"] or {}).get("regla"),
                                    "esperado": (p["validacion"] or {}).get("esperado"),
                                    "total dicho": (p["validacion"] or {}).get("total_dicho")} for p in rr["per_clip"]]),
                     use_container_width=True)
