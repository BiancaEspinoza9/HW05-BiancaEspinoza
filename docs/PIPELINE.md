# Pipeline (versión Mermaid de respaldo)

> La tarea exige documentarlo en **Notion o Miro, público y enlazado en el README**. Este archivo es el
> contenido listo para copiar a esa herramienta (cada bloque = una sección/diagrama). Mermaid solo es el
> fallback si la herramienta falla; si lo usas, explica por qué en el README.

## Qué corre UNA vez

```mermaid
flowchart LR
  A[Escribir guiones\ndata/ground_truth/*.yaml] --> B[Grabar con compañero\nteléfono, 2-4 min]
  B --> C[Verdad terreno\ntranscripción = guion + campos]
  B --> D[Escalera L0-L3\nsrc/ladder.py]
  D --> E[ASR en GPU Colab\nscripts/run_asr.py]
  E --> F[(results/cache\nJSON versionado)]
  F --> G[scripts/run_eval.py\nmétricas + IC]
  G --> H[(results/summary.json)]
  H --> I[app.py carga resultados]
  F --> J[GitHub Actions\npuerta de regresión sin GPU]
```

## Qué corre en CADA audio nuevo (`engine.process_audio`)

```mermaid
flowchart LR
  a[audio] --> b[Whisper faster-whisper\ndispositivo/VRAM medidos]
  b --> c[transcript + prob. por palabra]
  c --> d{extractor}
  d -->|rules| e[regex + capa de números]
  d -->|llm| f[LLM JSON con evidencia literal]
  e --> g[Verificación literal\nvalor ⊂ transcript]
  f --> g
  g -->|no está| h[sin_verificar]
  g --> i[registro]
  i -->|esquema order| j[Validación aritmética\nΣ cant×precio − promo = total]
  j -->|desajuste| k[Diagnóstico D1-D3\nASR vs hablante]
```

## Decisiones y descartes

| Paso | Decisión | Descartado y por qué |
|---|---|---|
| Verdad terreno | Guion escrito antes de grabar; el guion ES la referencia | Subtítulos de YouTube/ASR como referencia: evaluación circular |
| ASR | faster-whisper (CTranslate2), cuantización int8/float16 | openai-whisper: más VRAM; modelos solo-inglés |
| Dispositivo | Leído de `model.model.device` + VRAM por NVML | Confiar en `config.yaml`: el CPU fallback es silencioso |
| Normalización | Misma regla a referencia e hipótesis, números → dígitos | Comparar texto crudo: puntos y tildes inflan el WER |
| Números | Capa propia `numbers_es.py` | Dejar que Whisper decida dígitos/palabras: no es estable |
| Verificación | Evidencia literal + recálculo del valor desde la evidencia | Aceptar el JSON del LLM tal cual: puede "corregir" al ASR |
| Validación | Aritmética + reglas D1-D3 | Entrenar un clasificador con 4 llamadas: no hay datos |
| CI | Evaluar caché | Transcribir en Actions: los runners no tienen GPU |
| Motor | Un módulo, dos esquemas en `config.yaml` | Copiar el motor por tarea: costo al cambiar de modelo |
