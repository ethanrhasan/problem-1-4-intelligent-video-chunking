"""Boundary-accuracy evaluation on synthetic videos with known structure,
against two baselines people actually ship: fixed-length windows and
shot-cut-only segmentation."""
from __future__ import annotations

import json
import os

import numpy as np

from . import segment
from .pipeline import Params, _signals, chunk_video
from .synth import make_corpus


def prf(pred, gt, tol):
    pred, gt = sorted(pred), sorted(gt)
    used, tp = set(), 0
    for g in gt:
        best = None
        for k, p in enumerate(pred):
            if k not in used and abs(p - g) <= tol and (best is None or abs(p - g) < abs(pred[best] - g)):
                best = k
        if best is not None:
            used.add(best)
            tp += 1
    p = tp / len(pred) if pred else (1.0 if not gt else 0.0)
    r = tp / len(gt) if gt else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def fixed(duration, w):
    return list(np.arange(w, duration - 1e-6, w))


def run(out_dir: str, seed: int = 0, tol: float = 2.0) -> dict:
    data_dir = os.path.join(out_dir, "synthetic_videos")
    gts = make_corpus(data_dir, seed)
    p = Params()
    rows = []
    for gt in gts:
        res = chunk_video(gt["video_id"], gt["path"], None, p)
        scenes = [c for c in res["chunks"] if c["level"] == "scene"]
        ours = [c["start"] for c in scenes[1:]]
        sig = _signals(gt["path"], p)
        ev = segment.evidence(sig, [])
        shots = [float(t) for t in sig.times[ev["cut"] > 0.5]]
        t_tol = 3.0 if gt["genre"] == "surveillance" else tol
        row = {
            "video": gt["video_id"], "true_genre": gt["genre"], "pred_profile": res["profile"],
            "gt_boundaries": len(gt["boundaries"]), "n_chunks": len(scenes),
            "ours": prf(ours, gt["boundaries"], t_tol),
            "fixed_10s": prf(fixed(gt["duration"], 10), gt["boundaries"], t_tol),
            "fixed_30s": prf(fixed(gt["duration"], 30), gt["boundaries"], t_tol),
            "shots_only": prf(shots, gt["boundaries"], t_tol),
            "x_realtime": res["stats"]["x_realtime"],
        }
        if "events" in gt:
            # Event integrity: is each event inside ONE non-idle chunk, and how
            # much of the video was marked idle (cheap to index downstream)?
            ok = 0
            for s, e in gt["events"]:
                holders = [c for c in scenes if c["start"] <= s + t_tol and c["end"] >= e - t_tol]
                ok += int(bool(holders) and not holders[0]["idle"])
            row["events_in_single_active_chunk"] = f"{ok}/{len(gt['events'])}"
            row["idle_fraction"] = res["stats"]["idle_fraction"]
        rows.append(row)

    def mean_f(key):
        return round(float(np.mean([r[key][2] for r in rows])), 3)

    summary = {k: mean_f(k) for k in ("ours", "fixed_10s", "fixed_30s", "shots_only")}
    summary["profile_accuracy"] = round(float(np.mean([r["true_genre"] == r["pred_profile"] for r in rows])), 3)
    report = {"tolerance_s": tol, "rows": rows, "mean_boundary_f1": summary}
    with open(os.path.join(out_dir, "synthetic_eval.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def print_report(rep: dict):
    hdr = f"{'video':22s} {'true':12s} {'pred':12s} {'#gt':>4s} {'#ch':>4s} {'ours P/R/F1':>18s} {'fixed10 F1':>10s} {'fixed30 F1':>10s} {'shots F1':>9s} {'xRT':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rep["rows"]:
        o = r["ours"]
        print(f"{r['video']:22s} {r['true_genre']:12s} {r['pred_profile']:12s} {r['gt_boundaries']:4d} "
              f"{r['n_chunks']:4d} {o[0]:5.2f}/{o[1]:4.2f}/{o[2]:4.2f}     {r['fixed_10s'][2]:10.2f} "
              f"{r['fixed_30s'][2]:10.2f} {r['shots_only'][2]:9.2f} {r['x_realtime']:6.0f}")
        if "events_in_single_active_chunk" in r:
            print(f"    surveillance: events kept whole {r['events_in_single_active_chunk']}, "
                  f"idle fraction {r['idle_fraction']}")
    print("mean boundary F1:", rep["mean_boundary_f1"])
