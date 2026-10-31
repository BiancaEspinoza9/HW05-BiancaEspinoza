# De la voz al registro — HW_05_202602

Entrevistas (Tarea 1) y llamadas de pedido (Tarea 2) convertidas en **registros con campos verificados**, con un
**único motor** (`src/engine.py`) y dos esquemas definidos en `config.yaml`.

> **Estado honesto del proyecto.** Código, guiones, verdad terreno y app están hechos y probados (`pytest`: 20 pruebas).
> Los 8 audios reales (E01–E04, V01–V04) ya están transcritos con Whisper `small` en **CPU** y evaluados (ver
> §Resultados con audio real). **Pendiente:** la comparación en GPU (Colab), el documento en Notion/Miro y el video.
> El audio sintético sigue como grupo aparte (§Resultados piloto) y no entra en los números principales.

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
python -m pytest -q tests                              # 20 pruebas
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

## Resultados con audio real (grabaciones propias, Whisper `small`, **CPU int8**)
Corpus: 4 entrevistas (E01–E04) + 4 llamadas (V01–V04). Generado con `python scripts/run_eval.py` sobre `results/cache/real/`
(2 000 remuestreos bootstrap, IC95). **Aún no hay corrida en GPU**: la columna VRAM está vacía a propósito (ver §Pendiente).

**Tabla 1 – Transcripción (L0, sin degradar)**

| Tarea | Modelo | Glosario | WER (IC95) | CER | s por min de audio | VRAM pico |
|---|---|---|---|---|---|---|
| Entrevistas | small | ninguno | 5.6 % (4.1–6.4) | 2.3 % | 30.6 | n/a (CPU) |
| Entrevistas | small | `hotwords` | 3.9 % (2.8–5.3) | 1.7 % | — | n/a (CPU) |
| Entrevistas | medium | ninguno | **2.1 %** (1.4–3.0) | 0.6 % | 64.6 | n/a (CPU) |
| Llamadas | small | ninguno | 15.4 % (9.5–21.4) | 5.9 % | 42.1 | n/a (CPU) |
| Llamadas | medium | ninguno | **13.1 %** (7.8–18.3) | 6.4 % | 65.5 | n/a (CPU) |

`medium` reduce el WER en entrevistas (2.1 % vs 5.6 %; los IC no se solapan) pero cuesta ~2× en tiempo; en llamadas la mejora (13.1 % vs 15.4 %)
cae dentro del ruido (los IC se solapan) y el CER sube un poco. Con `medium`, `tecnologias` llega a 50 % sin glosario y `direccion_entrega` de las llamadas baja de 100 % a 50 %:
un modelo más grande no mejora todos los campos a la vez. Todo con CPU int8; la comparación de precisión (float16/int8) requiere GPU.

Efecto del glosario (entrevistas, n = 4): WER sin − con = **+1.65 pp** (IC95 0.63–2.68). Mismo modelo, mismo dispositivo y misma precisión;
solo cambia `hotwords` (faster-whisper 1.2.1). Mejora `tecnologias` de 0 % a 50 % y `ultimo_puesto` de 75 % a 100 %.

**Tabla 2 – Registro (L0)**

| Entrevistas (sin glosario → con `hotwords`) | exactitud | | Llamadas (sin glosario) | exactitud |
|---|---|---|---|---|
| anios_experiencia | 100 % → 100 % | | items | 75 % |
| ultimo_puesto | 75 % → 100 % | | promocion | 100 % |
| tecnologias | **0 % → 50 %** | | direccion_entrega | 100 % |
| pretension_salarial | 100 % → 100 % | | metodo_pago | 100 % |
| disponibilidad | 100 % → 100 % | | total_dicho | 100 % |
| nivel_ingles | 100 % → 100 % | | | |

**Tabla 3 – Escalera de degradación (sin glosario, SNR de L3 medido = 10.0 dB en los 4 clips)**

| Nivel | WER entrevistas | WER llamadas | Exactitud media de campos (entr. / llam.) | Trampas atrapadas / perdidas | Diagnóstico correcto |
|---|---|---|---|---|---|
| L0 original | 5.6 % | 15.4 % | 79 % / 95 % | 3 / 0 | 1 de 3 |
| L1 voz WhatsApp | 5.1 % | 16.5 % | 83 % / 80 % | 2 / 1 | 0 de 2 |
| L2 teléfono | 6.7 % | 18.8 % | 75 % / 85 % | 3 / 0 | 2 de 3 |
| L3 ruido 10 dB | 12.5 % | 24.5 % | 83 % / 80 % | 3 / 0 | 3 de 3 |

El WER sube con la degradación (más en llamadas y en L3); la exactitud de campos **no** baja de forma monótona con n = 4 por celda: el ruido
de muestreo es del tamaño del efecto. No hubo falsas alarmas (la llamada sin trampa, V04, siempre cuadró).

**¿Se puede saber si el registro sirve mirando solo el WER? No.** Llamadas, L0: V03 tiene WER 17.6 % y los 5 campos correctos, mientras que
entrevistas, L0 sin glosario: E03 tiene el WER más bajo (3.3 %) y `tecnologias` roto (5 de 6 campos). El WER cuenta todas las palabras por igual;
al registro le importan unas pocas (Python, SQL, un precio) y ahí el ASR falla más. Detalle: `results/summary.json → wer_vs_registro`.

**Qué hace el extractor de ítems con un ASR imperfecto.** Whisper escribe “pollas al abrazo” o “gasezas”. `src/extract_rules.py` reconoce
producto y tamaño por parecido (`difflib`, umbral `schemas.order.fuzzy_match_threshold` en `config.yaml`, y exige margen sobre la segunda opción).
La evidencia sigue siendo el fragmento literal del transcript y cantidad/precio se recalculan desde ella, así que la regla de verificación no se relaja.
*Cuidado:* ese umbral se fijó mirando estos mismos audios; los números de `items` son optimistas.

**Arreglo explicado.** Con el extractor exacto original, `items` daba 0 % y el validador atrapaba 0 de 3 trampas (todo salía `no_evaluable`):
sin ítems no hay cuenta que cuadrar. Es un caso real de cómo un fallo de extracción oculta lo que la aritmética sí detectaría.

### Pendiente (no hay números todavía)
- **GPU:** `scripts/colab_gpu_benchmark.py` (Colab T4) para `float16` vs `int8_float16` vs CPU, `large-v3`, VRAM pico y la prueba de dispositivo.
- Si el **texto cambió** entre CPU/GPU y entre precisiones (`float16` vs `int8_float16`), y `large-v3` como tercer tamaño.
- Enlaces a Notion/Miro y al video.

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

Las tablas con audio real están arriba; se regeneran con `python scripts/run_eval.py` y se ven en las pestañas 3 y 5 de la app.

## Limitaciones (decláralas en el video)
- ~4 clips × 6 campos ≈ 24 observaciones por tarea: sirve para ver un efecto, **no** para estimar con precisión; los IC son anchos.
- Las reglas de tecnologías usan el mismo glosario que se le da a Whisper: no detectan tecnologías fuera de la lista.
- La verificación detecta valores ausentes del transcript; **no** detecta un valor presente pero erróneo (p. ej. dos sueldos mencionados).
- La aritmética no detecta errores que la dejan consistente (ASR y hablante que se compensan) ni distingue transposiciones plausibles (V03).
- El ruido de L3 es sintético (rosa), no una cafetería real.
- Los clips reales duran aprox. 1:15–2:05 (estimado por fragmentos del contenedor, sin ffprobe), por debajo del rango pedido de 2–4 min; el corpus real tiene 4 + 4 clips.
- Los umbrales de coincidencia aproximada de ítems se ajustaron viendo este mismo corpus (riesgo de sobreajuste).
- El diagnóstico D1–D3 se evalúa con 2–3 casos por nivel: sirve para ilustrar, no para estimar su precisión.
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
