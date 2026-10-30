# Guion del video (máx. 8 min, solo tú). Regla de oro: NADA de código antes del minuto 3.

Rellena los `[X]` con tus números de `results/summary.json` tras grabar y correr en Colab.

**0:00–1:00 · Problema** (slides)
"Mi teléfono ya transcribe, pero un texto no es un registro. Una organización necesita campos verificados en una tabla y saber cuánto confiar en ellos. Dictado, WhatsApp, YouTube, Zoom: ninguno guarda una fila en una base de datos; Otter cobra por minuto y manda el audio fuera."

**1:00–3:00 · Pipeline de las dos tareas** (tu Notion/Miro, sin código)
- Una vez: guiones → grabación con un compañero → verdad terreno → escalera L0–L3 → ASR en GPU → caché → métricas.
- Por audio nuevo: Whisper → extracción (reglas o LLM) → verificación literal → registro → (pedidos) validación aritmética y diagnóstico.
- Un solo motor, dos esquemas.

**3:00–5:00 · Decisiones con números** (tablas y gráficos)
- Verdad terreno: "salió gratis porque escribí el guion antes de grabar; el guion es la referencia" (no subtítulos de otro ASR: sería evaluación circular).
- GPU: "`device` leído de la librería = [cuda], VRAM pico [X] MB (float16) vs [Y] MB (int8); CPU [Z] s/min vs GPU [W] s/min; el texto [cambió / no cambió] (WER entre textos [X])."
- Glosario: "WER sin − con = [X] (IC95 [lo, hi]); recupera Python/Docker/Git/React" (en el piloto sintético, 'Python' salió 'paisan').
- Escalera: curva de WER y exactitud L0→L3, SNR medido [X] dB.

**5:00–7:00 · Demo en vivo** (app)
- Pestaña 4: sube un clip; muestra transcript, registro y `sin_verificar`.
- Pestaña 5: llamada V01 — dice 120, la cuenta da 114: el validador la marca; muestra D1/D2/D3. Menciona V03 como caso que el diagnóstico confunde.

**7:00–8:00 · Hallazgos y límites**
- WER no predice si el registro sirve: clip [id] con WER [alto] y registro perfecto; clip [id] con WER [bajo] y registro roto.
- Qué atrapa la aritmética: totales mal dichos ([X]/3); qué no: errores consistentes, transposiciones plausibles.
- Muestra chica (≈ 24 observaciones): IC anchos. Nunca usarlo para contratar o cobrar sin revisión humana.

**Checklist de lo que debes decir:** de dónde sale la verdad terreno · evidencia de GPU · por qué WER no predice el registro · qué atrapa/no atrapa la aritmética · para qué NO usarlo.
