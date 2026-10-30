"""Pegar en Colab (Runtime > T4 GPU). Mide GPU vs CPU, float16 vs int8 y glosario sobre el MISMO corpus.
!git clone <tu-repo> && cd <tu-repo> && pip install -r requirements.txt
Sube tus audios a data/audio/ (E01..E04, V01..V04) y ejecuta:
"""
import subprocess, sys

def run(*a):
    print(">", *a); subprocess.run([sys.executable, *a], check=True)

for task in ("interviews", "calls"):
    for model in ("small", "large-v3"):                       # >= 2 tamanos
        for dev, ct in (("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")):
            run("scripts/run_asr.py", "--task", task, "--model", model, "--device", dev, "--compute-type", ct)
for mode in ("hotwords", "initial_prompt"):                   # efecto del glosario (solo entrevistas)
    run("scripts/run_asr.py", "--task", "interviews", "--model", "large-v3", "--device", "cuda",
        "--compute-type", "float16", "--glossary", mode)
run("scripts/run_asr.py", "--task", "calls", "--levels", "L0", "L1", "L2", "L3", "--model", "large-v3",
    "--device", "cuda", "--compute-type", "float16")          # escalera de degradacion
run("scripts/run_eval.py", "--llm")
# Evidencia de GPU: results/cache/**.json -> "device": "cuda" (leido de ctranslate2) y "vram_peak_mb" > 0
