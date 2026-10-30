"""Evalua TODO lo que haya en results/cache y escribe results/summary.json.

Solo lee transcripciones cacheadas (no transcribe): por eso corre en GitHub Actions sin GPU.
  python scripts/run_eval.py [--llm] [--gate]
--gate: sale con codigo 1 si el WER del grupo real L0 supera gate.max_wer (puerta de regresion).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import metrics as M
from src.data import load_clips
from src.engine import ROOT, load_config, process_audio
from src.evaluate import score_record, summarize_fields
from src.validate import compute_expected

SCHEMA = {"interviews": "interview", "calls": "order"}


def ci(vals, cfg):
    m = cfg["metrics"]
    e, lo, hi = M.bootstrap_ci(vals, m["bootstrap_resamples"], m["ci_level"], m["seed"])
    return {"est": e, "lo": lo, "hi": hi}


def true_cause(fields: dict, truth: dict) -> str:
    """'hablante' si todos los numeros extraidos coinciden con la verdad terreno; si no, 'asr'."""
    try:
        ex = {(i["producto"], i["tamano"]): (i["cantidad"], i["precio_unitario"]) for i in fields["items"]["valor"]}
        tr = {(i["producto"], i["tamano"]): (i["cantidad"], i["precio_unitario"]) for i in truth["items"]}
        promo_ok = (fields["promocion"]["valor"] if fields.get("promocion", {}).get("valor") else None) == \
                   ({"tipo": truth["promocion"]["tipo"], "valor": float(truth["promocion"]["valor"])} if truth["promocion"] else None)
        tot_ok = float(fields["total_dicho"]["valor"]) == float(truth["total_dicho"])
        return "hablante" if ex == tr and promo_ok and tot_ok else "asr"
    except Exception:
        return "asr"


def eval_run(task, group, level, key, cfg, clips, extractor):
    schema = SCHEMA[task]
    base = ROOT / cfg["paths"]["cache_dir"] / group / level / key
    per, scored, hyps = [], [], {}
    for c in clips:
        f = base / f"{c['id']}.json"
        if not f.exists():
            continue
        tr = json.loads(f.read_text(encoding="utf-8"))
        res = process_audio(None, schema, extractor=extractor, config=cfg, transcript_result=tr,
                            reference_date=c["fecha_referencia"])
        sc = score_record(schema, res, c["campos"], cfg)
        scored.append(sc)
        hyps[c["id"]] = tr["transcript"]
        we, wn = M.wer_counts(c["reference"], tr["transcript"], cfg["normalization"])
        ce, cn = M.cer_counts(c["reference"], tr["transcript"], cfg["normalization"])
        row = {"id": c["id"], "wer": we / wn, "cer": ce / cn, "words": wn,
               "campos_ok": sum(v["ok"] for v in sc.values()), "campos_n": len(sc),
               "campos_ok_verificados": sum(v["ok_verificado"] for v in sc.values()),
               "sec_per_min": tr.get("sec_per_min"), "error": res["error"], "estados": res["verified"],
               "transcript": tr["transcript"], "record": res["record"],
               "snr_measured": tr.get("snr_measured")}
        if schema == "order":
            v = res["validation"] or {}
            row["validacion"] = v
            row["trampa_total"] = bool(c["trampas"]["total_erroneo"])
            row["flagged"] = v.get("estado") == "desajuste"
            if row["flagged"]:
                row["causa_real"] = true_cause(res["fields"], c["campos"])
        per.append(row)
    if not per:
        return None
    meta = tr
    out = {"task": task, "group": group, "level": level, "run": key, "extractor": extractor,
           "model": meta["model"], "device": meta["device"], "compute_type": meta["compute_type"],
           "glossary_mode": meta.get("glossary_mode"), "n_clips": len(per),
           "wer": ci([r["wer"] for r in per], cfg), "cer": ci([r["cer"] for r in per], cfg),
           "wer_pooled": sum(r["wer"] * r["words"] for r in per) / sum(r["words"] for r in per),
           "sec_per_min": mean([r["sec_per_min"] for r in per if r["sec_per_min"]] or [0]),
           "vram_peak_mb": max([json.loads((base / f"{r['id']}.json").read_text())["vram_peak_mb"] or 0 for r in per]),
           "fields": summarize_fields(scored, schema, cfg), "per_clip": per}
    out["field_acc_mean"] = mean(v["acc"] for v in out["fields"].values() if v["acc"] is not None)
    if schema == "order":
        trap = [r for r in per if r["trampa_total"]]
        clean = [r for r in per if not r["trampa_total"]]
        fl = [r for r in per if r["flagged"]]
        out["validation"] = {
            "trampas": len(trap), "atrapadas": sum(r["flagged"] for r in trap),
            "perdidas": sum(not r["flagged"] for r in trap),
            "falsas_alarmas": sum(r["flagged"] for r in clean), "limpias": len(clean),
            "diagnosticos": len(fl),
            "diagnostico_correcto": sum(r["validacion"].get("causa") == r["causa_real"] for r in fl)}
    out["_hyps"] = hyps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    corpus = {t: load_clips(cfg, t) for t in SCHEMA}
    runs = []
    root = ROOT / cfg["paths"]["cache_dir"]
    for lv_dir in sorted(root.glob("**/L[0-3]")):
        group = lv_dir.parent.relative_to(root).as_posix()
        for run_dir in sorted(p for p in lv_dir.iterdir() if p.is_dir()):
            for task in SCHEMA:
                for ext in ["rules"] + (["llm"] if a.llm else []):
                    r = eval_run(task, group, lv_dir.name, run_dir.name, cfg, corpus[task], ext)
                    if r:
                        runs.append(r)

    # --- comparaciones: el texto cambio? (CPU vs GPU, precision) y efecto del glosario
    comps, gloss = [], []
    def k(r): return (r["task"], r["group"], r["level"], r["model"], r["extractor"])
    for r in runs:
        for b in runs:
            if k(r) == k(b) and r is not b:
                same_prec = (r["device"], r["compute_type"]) == (b["device"], b["compute_type"])
                if r["glossary_mode"] == b["glossary_mode"] and not same_prec and r["run"] < b["run"]:
                    ids = [x for x in r["_hyps"] if x in b["_hyps"]]
                    d = [M.wer(r["_hyps"][i], b["_hyps"][i], cfg["normalization"]) for i in ids]
                    comps.append({"a": r["run"], "b": b["run"], "task": r["task"], "group": r["group"],
                                  "level": r["level"], "wer_entre_textos": mean(d),
                                  "clips_identicos": sum(x == 0 for x in d), "n": len(d),
                                  "delta_sec_per_min": r["sec_per_min"] - b["sec_per_min"]})
                if (r["glossary_mode"] == "none" and b["glossary_mode"] != "none"
                        and (r["device"], r["compute_type"]) == (b["device"], b["compute_type"])):
                    ids = [p["id"] for p in r["per_clip"] if p["id"] in {q["id"] for q in b["per_clip"]}]
                    wa = {p["id"]: p["wer"] for p in r["per_clip"]}
                    wb = {p["id"]: p["wer"] for p in b["per_clip"]}
                    m = cfg["metrics"]
                    e, lo, hi = M.paired_diff_ci([wa[i] for i in ids], [wb[i] for i in ids],
                                                 m["bootstrap_resamples"], m["ci_level"], m["seed"])
                    gloss.append({"task": r["task"], "group": r["group"], "level": r["level"], "model": r["model"],
                                  "modo": b["glossary_mode"], "dif_wer_sin_menos_con": {"est": e, "lo": lo, "hi": hi},
                                  "n": len(ids)})

    # --- WER vs registro: ejemplos de que WER no predice si el registro sirve
    ex = {}
    for r in runs:
        if r["group"] != "real" or r["extractor"] != "rules":
            continue
        for p in r["per_clip"]:
            full = p["campos_ok"] == p["campos_n"]
            key = (r["task"], r["level"], r["run"])
            e = ex.setdefault(key, {"wer_alto_registro_perfecto": None, "wer_bajo_registro_roto": None})
            if full and (e["wer_alto_registro_perfecto"] is None or p["wer"] > e["wer_alto_registro_perfecto"]["wer"]):
                e["wer_alto_registro_perfecto"] = {"id": p["id"], "wer": p["wer"], "campos": f"{p['campos_ok']}/{p['campos_n']}"}
            if not full and (e["wer_bajo_registro_roto"] is None or p["wer"] < e["wer_bajo_registro_roto"]["wer"]):
                e["wer_bajo_registro_roto"] = {"id": p["id"], "wer": p["wer"], "campos": f"{p['campos_ok']}/{p['campos_n']}"}
    wer_vs = [{"task": k_[0], "level": k_[1], "run": k_[2], **v} for k_, v in ex.items()]

    # --- distribucion de la verdad terreno (Fase 1)
    dist = {}
    for t, clips in corpus.items():
        for fld in clips[0]["campos"]:
            vals = [c["campos"][fld] for c in clips]
            if isinstance(vals[0], (int, float)):
                dist[f"{t}.{fld}"] = {"min": min(vals), "mediana": median(vals), "max": max(vals)}
            elif isinstance(vals[0], list) and vals[0] and isinstance(vals[0][0], str):
                dist[f"{t}.{fld}"] = dict(Counter(x for v in vals for x in v))
            elif isinstance(vals[0], str):
                dist[f"{t}.{fld}"] = dict(Counter(vals))
    for r in runs:
        r.pop("_hyps")
    summary = {"runs": runs, "comparaciones_texto": comps, "efecto_glosario": gloss,
               "wer_vs_registro": wer_vs, "distribucion_verdad_terreno": dist,
               "trampas": {c["id"]: c["trampas"] for c in corpus["calls"]}}
    out = ROOT / cfg["app"]["results_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(runs)} corridas evaluadas -> {out}")
    for r in runs:
        print(f"{r['task']:<10} {r['group']:<18} {r['level']} {r['run']:<34} {r['extractor']:<5} "
              f"WER {r['wer']['est']:.3f} [{r['wer']['lo']:.3f},{r['wer']['hi']:.3f}] campos {r['field_acc_mean']:.2f}")
    if a.gate:
        thr = cfg["gate"]
        bad = [r for r in runs if r["group"] == "real" and r["level"] == "L0" and r["extractor"] == "rules"
               and (r["wer"]["est"] > thr["max_wer"] or r["field_acc_mean"] < thr["min_field_acc"])]
        if bad:
            print("GATE FALLIDO:", [(r["task"], r["run"]) for r in bad])
            sys.exit(1)
        print("GATE OK")


if __name__ == "__main__":
    main()
