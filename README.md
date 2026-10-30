# De la voz al registro — HW_05_202602

Entrevistas (Tarea 1) y llamadas de pedido (Tarea 2) convertidas en **registros con campos verificados**, con un
**único motor** (`src/engine.py`) y dos esquemas definidos en `config.yaml`.

> **Estado honesto del proyecto.** Todo el código, los guiones, la verdad terreno y la app están hechos y probados
> (`pytest`: 19 pruebas). Lo que **depende de ti** y por eso aún no tiene números reales: grabar los 8 audios,
> correr la comparación en GPU (Colab), el documento en Notion/Miro y el video. Las únicas cifras medidas hasta
> ahora provienen de **audio sintético (grupo aparte, ver §Resultados piloto)** y no cuentan como resultados finales.

- Dominio de la Tarea 2: **restaurante** (pollería inventada “El Sabor Norteño”).
- Documento del pipeline (Notion/Miro, público): **PEGAR ENLACE AQUÍ** — contenido listo en [docs/PIPELINE.md](docs/PIPELINE.md).
- Video (≤ 8 min): **PEGAR ENLACE AQUÍ** — guion en [docs/VIDEO_SCRIPT.md](docs/VIDEO_SCRIPT.md).

## Consentimiento y datos inventados
- Todo lo dicho en los audios es **inventado**: nombres, empresas, sueldos, direcciones, teléfonos. No hay datos reales de nadie.
- Todas las personas que aparecen en las grabaciones **aceptaron** participar; no se nombra a nadie que no lo haya aceptado.  *(Confírmalo tú antes de entregar.)*

## Instalación y ejecución
```bash
python -m venv .venv && .venv\Scripts\activate        # Python 3.12
pip install -r requirements.txt
copy .env.example .env                                 # solo si usarás el extractor LLM (ANTHROPIC_API_KEY)
python -m pytest -q tests                              # 19 pruebas
streamlit run app.py                                   # un solo comando
```
Pasos del flujo:
```bash
# 1) grabar siguiendo data/ground_truth/*.yaml  ->  guardar como data/audio/E01.m4a ... V04.wav (no se versiona)
# 2) transcribir y cachear (GPU en Colab: ver scripts/colab_gpu_benchmark.py)
python scripts/run_asr.py --task interviews --model small
python scripts/run_asr.py --task calls --levels L0 L1 L2 L3 --model small
python scripts/run_asr.py --task interviews --glossary hotwords       # efecto del glosario
# 3) evaluar (solo lee la caché; no transcribe)  ->  results/summary.json
python scripts/run_eval.py [--llm] [--gate]
```
**Audio y pesos no se versionan.** Para regenerar: grabar según los guiones (cada clip = el guion de su `.yaml`), o pedirlos al autor.
Los modelos Whisper se descargan solos al primer uso. Las **transcripciones cacheadas sí se versionan** (`results/cache`).

### Verificación de que el motor no importa UI
```bash
python -m pytest -q tests/test_core.py -k engine_has_no_ui
# equivalente manual:
python -c "import sys, src.engine; sys.exit(any(m.split('.')[0] in ('streamlit','gradio','flask','tkinter') for m in sys.modules))"
```
(`echo %errorlevel%` debe dar 0.)

## Un motor, dos esquemas
```python
from src.engine import process_audio
process_audio("data/audio/E01.m4a", schema="interview")   # 6 campos de candidato
process_audio("data/audio/V01.m4a", schema="order")       # items, promo, dirección, pago, total + validación
```
La función devuelve `transcript`, `fields` (valor + evidencia + estado), `record`, `verified`, `validation`,
`device` (leído de la librería), `elapsed_s`, `vram_peak_mb` y `error`. Lo único específico de cada tarea es el esquema
de `config.yaml` (campos, glosario, menú, prompts) y su función de reglas en `extract_rules.py`; la transcripción,
verificación y métricas son compartidas. Con `transcript_result=` se salta el ASR (modo caché/CI). No se copió ni editó
el motor: cambiar de modelo se hace en un solo lugar.

## Reglas clave (para defender en el video)
**Normalización (WER/CER y verificación)**: minúsculas → sin tildes → puntuación a separador (se conserva `6.500`/`45,50`) →
`%`→“por ciento” → números en palabras a dígitos → espacios simples. Se aplica idéntica a referencia e hipótesis.

**Capa de números** (`src/numbers_es.py`): 0–29 palabras simples; decenas + “y” + unidad; centenas; “mil” multiplica;
“un/una” no son número salvo petición (son artículos); dígitos con separador de miles (`6.500`, `6,500`) o decimal
(`45.50`); “6 mil 500”; “por ciento” no es 100. *Límite*: no resuelve céntimos hablados (“con cincuenta”).

**Verificación literal** (`src/verify.py`): cada valor trae una *evidencia*; debe estar en el transcript normalizado **y** el valor
debe recalcularse desde esa evidencia (fecha desde la frase, monto desde los números, categoría desde su palabra clave).
Si no, `sin_verificar`. Así un LLM que “corrige” al ASR queda marcado aunque acierte.

**Extracción**: (a) reglas/regex + glosario; (b) LLM → JSON con evidencia literal (`--llm`, `ANTHROPIC_API_KEY`; modelo en `config.yaml`).
En pedidos, el cliente puede corregirse: se lee el **resumen final** del operador (si no hay, gana la última mención de cada producto).

**Validación aritmética** (`src/validate.py`): `Σ cantidad×precio − promoción = total_dicho` (tolerancia 0.5 soles).
Diagnóstico del desajuste: **D1** un solo número intercambiado por su confundible acústico (15↔50, 60↔70…) cuadra la cuenta → *ASR*;
**D2** algún número involucrado se dijo con probabilidad < 0.60 → *ASR*; **D3** si no → *hablante*. Su acierto se mide contra una causa real
derivada de la verdad terreno (si todos los números extraídos coinciden con lo planeado, fue el hablante). **Limitación conocida**: V03 (92 dicho como 82)
cae en D1 aunque es error del hablante: una transposición plausible es indistinguible de una mala audición solo con aritmética.

**Escalera** (`src/ladder.py`): L0 original · L1 opus 16 kb/s mono (nota de voz) · L2 pasa-banda 300–3400 Hz a 8 kHz (teléfono) ·
L3 = L1 + ruido rosa sintético a **SNR 10 dB**. SNR medido, no supuesto: potencia media de la voz en tramos de 20 ms a menos de 30 dB del
pico, contra la potencia media del ruido sumado; se recalcula tras el mezclado y se guarda (`*.snr.json`, `snr_measured`; test: `test_snr_is_the_requested_one`).
10 dB ≈ calle/cafetería ruidosa; 0 dB sería voz = ruido. Si el registro sigue perfecto en L3, se baja el SNR en `config.yaml` y se re-ejecuta.

**GPU**: `asr.py` reporta `device` y `compute_type` leídos de `model.model` (ctranslate2) y la VRAM pico muestreada por NVML. Si se pidió
`cuda` y corrió en CPU, el motor devuelve `error`. **Glosario**: `hotwords` (faster-whisper ≥ 1.0.3, verificado con `inspect.signature`) o `initial_prompt`;
la corrida registra el modo realmente usado.

## Corpus (Fase 1)
| Tarea 1 | Campos en orden distinto por clip (variación) |
|---|---|
| E01 | años → puesto → tecnologías → inglés → sueldo → disponibilidad |
| E02 | disponibilidad → sueldo → años (“siete años y medio”) → puesto (“Fui…”) → tecnologías → inglés |
| E03 | tecnologías → puesto → años (“llevo dos años”, con distractor “un año en prácticas”) → inglés → sueldo → fecha absoluta |
| E04 | puesto (“me desempeño como”) → tecnologías → años (con distractor “dos años en la empresa actual”) → inglés → sueldo → “dentro de un mes” |

| Tarea 2 | corrección | total mal | dígitos | pago |
|---|---|---|---|---|
| V01 | sí | sí (dice 120, da 114) | no | efectivo |
| V02 | sí | sí (105 vs 99) | sí (342, 601) | tarjeta |
| V03 | sí | sí (82 vs 92) | no | yape |
| V04 | no | no | sí (mz. 8, lote 15) | tarjeta |

La verdad terreno es gratis porque **el guion se escribe antes de grabar**: la transcripción de referencia es el guion y los campos correctos salen de él.
Distribuciones de cada campo: `results/summary.json → distribucion_verdad_terreno` (pestaña 2 de la app).
**Si al grabar un clip dura < 2 min**, agrega pausas o más líneas al guion (y a su `.yaml`); el mínimo pedido es 2–4 min.

## Resultados piloto (audio SINTÉTICO — grupo aparte, NO son los resultados finales)
Edge TTS (voces es-PE), Whisper `small`, CPU int8, 4 entrevistas. Sirve para probar la tubería; el audio sintético es demasiado limpio (justo lo que advierte la tarea).

| Medida | Valor |
|---|---|
| WER | 0.7 % (IC95 0.2–1.1 %) |
| Exactitud de campos (5 de 6 campos al 100 %) | tecnologías 25 % |
| Qué falló | “Python”→“paisan”, “Docker”→“DAQ”, “Git”→“HEAP”, “React”→“Viacly”: el WER casi no se mueve, el registro sí |
| Glosario `hotwords` | recupera “Git” y “React” en algunos clips; WER −0.3 pp (IC95 0.0–0.6, n=4) — **n demasiado chico para concluir** |
| Cota superior | con la transcripción perfecta (guion), las reglas aciertan y verifican el 100 % (`test_rules_perfect_on_reference_transcripts`) |

Esto ya ilustra el punto de la tarea: **un WER de 0.7 % convive con un campo (tecnologías) roto** — reportar solo WER ocultaría el problema.

### Tablas finales (rellenar con TUS grabaciones)
Generadas por `python scripts/run_eval.py`; se ven en la pestaña 3 y 5 de la app. Pega aquí:
- Tabla 1 – transcripción (modelo × dispositivo × precisión × glosario): WER, CER, s/min, VRAM pico, IC95.
- Tabla 2 – registro (exactitud por campo, macro-F1 de `nivel_ingles`, error absoluto de años y sueldo).
- Tabla 3 – escalera L0–L3: WER, exactitud de campos, SNR medido, trampas atrapadas/perdidas/falsas alarmas, acierto del diagnóstico.
- Un clip de WER alto con registro perfecto y uno de WER bajo con registro roto (`summary.json → wer_vs_registro`).

## Limitaciones (decláralas en el video)
- ~4 clips × 6 campos ≈ 24 observaciones por tarea: sirve para ver un efecto, **no** para estimar con precisión; los IC son anchos.
- Las reglas de tecnologías usan el mismo glosario que se le da a Whisper: no detectan tecnologías fuera de la lista.
- La verificación detecta valores ausentes del transcript; **no** detecta un valor presente pero erróneo (p. ej. dos sueldos mencionados).
- La aritmética no detecta errores que la dejan consistente (ASR y hablante que se compensan) ni distingue transposiciones plausibles (V03).
- El ruido de L3 es sintético (rosa), no una cafetería real.
- **Nunca debe usarse** para decidir contrataciones, ni para cobrar un pedido sin confirmación humana, ni con datos reales de terceros sin consentimiento.

## Innovación incluida
- **Puerta de regresión en GitHub Actions** (`.github/workflows/regression.yml`): recalcula métricas desde la caché y falla bajo el umbral de `config.yaml → gate`. No transcribe: los runners no tienen GPU.
- **Eje de acento** (sintético): `scripts/make_synthetic.py --locale es-MX|es-AR|es-ES`; son voces sintéticas, no hablantes reales.
- Pendientes opcionales (no implementados): Notion por corrida, bot de Telegram, servidor MCP, separación de hablantes, despliegue público.

## Herramientas de IA declaradas
- **Claude Code (Claude Sonnet 5.5)**: escribió el código base, guiones y pruebas bajo mi dirección; yo debo poder explicar y modificar cada módulo.
- **MCP servers conectados**: *(completa los tuyos; ninguno fue necesario para el código de este repositorio)*.

## Estructura
```
config.yaml  .env.example  requirements.txt  app.py
src/  engine.py asr.py numbers_es.py textnorm.py extract_rules.py extract_llm.py verify.py validate.py metrics.py evaluate.py ladder.py data.py
scripts/  run_asr.py run_eval.py make_synthetic.py colab_gpu_benchmark.py
data/ground_truth/  interviews.yaml calls.yaml        results/cache/  results/summary.json
tests/  docs/PIPELINE.md  docs/VIDEO_SCRIPT.md  .github/workflows/regression.yml
```
