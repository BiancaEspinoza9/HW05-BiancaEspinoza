"""Transcribe el corpus y guarda cada transcripcion en cache (results/cache).

Ejemplos:
  python scripts/run_asr.py --task calls --group real --levels L0 L1 L2 L3 --model small
  python scripts/run_asr.py --task interviews --group synthetic/default --model medium --device cuda --compute-type float16
  python scripts/run_asr.py --task interviews --glossary hotwords      # con vocabulario de dominio
Estructura: results/cache/<group>/<nivel>/<modelo>__<device>__<compute>__<glosario>/<clip>.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import asr
from src.data import AUDIO_EXT, load_clips
from src.engine import ROOT, load_config
from src.ladder import apply_level


def audio_for(cfg, group, clip_id):
    base = ROOT / (cfg["paths"]["audio_dir"] if group == "real" else Path(cfg["paths"]["synthetic_dir"]) / group.split("/", 1)[1])
    for ext in AUDIO_EXT:
        if (base / f"{clip_id}{ext}").exists():
            return base / f"{clip_id}{ext}"
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["interviews", "calls"], required=True)
    ap.add_argument("--group", default="real", help="real | synthetic/<locale>")
    ap.add_argument("--levels", nargs="+", default=["L0"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--compute-type", default=None)
    ap.add_argument("--glossary", default="none", choices=["none", "hotwords", "initial_prompt"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    glossary = cfg["schemas"]["interview"]["glossary"] if a.task == "interviews" else None
    dev, ct = asr.resolve(a.device or cfg["asr"]["device"], a.compute_type or cfg["asr"]["compute_type"])
    model = a.model or cfg["asr"]["model"]
    key = f"{model}__{dev}__{ct}__{a.glossary}"
    for clip in load_clips(cfg, a.task):
        src = audio_for(cfg, a.group, clip["id"])
        if src is None:
            print("falta audio:", clip["id"])
            continue
        for lv in a.levels:
            out = ROOT / cfg["paths"]["cache_dir"] / a.group / lv / key / f"{clip['id']}.json"
            if out.exists() and not a.force:
                continue
            info = apply_level(src, lv, ROOT / cfg["paths"]["ladder_dir"] / a.group, cfg)
            res = asr.transcribe_file(info["path"], cfg, model=model, device=dev, compute_type=ct,
                                      glossary=glossary, glossary_mode=a.glossary)
            res.update({k: v for k, v in info.items() if k.startswith("snr")})
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print(clip["id"], lv, key, f"{res['elapsed_s']}s", res["device"], "VRAM", res["vram_peak_mb"])
