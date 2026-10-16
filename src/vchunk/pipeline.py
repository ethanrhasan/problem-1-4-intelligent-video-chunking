"""Per-video pipeline: decode -> signals -> profile -> boundaries -> chunk records."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import decode, segment
from .profiles import PROFILES, classify
from .signals import extract
from .subtitles import parse_srt, text_between

VERSION = "1.0"


@dataclass
class Params:
    fps: float = 2.0
    width: int = 160
    height: int = 90
    profile: str = "auto"          # or a name from PROFILES
    context_pad: float = 2.0        # seconds of overlap added for retrieval context
    keyframes_per_10s: float = 1.0  # at full density
    max_keyframes: int = 8
    sections: bool = True
    fast_decode: bool = True
    threads: int = 1


def _signals(path: str, p: Params):
    audio, astats = decode.read_audio(path, hop_s=1.0 / p.fps)
    try:
        frames = decode.iter_frames(path, p.fps, p.width, p.height, p.threads, p.fast_decode)
        sig = extract(frames, p.fps, audio, 1.0 / p.fps, astats)
    except RuntimeError:
        frames = decode.iter_frames(path, p.fps, p.width, p.height, p.threads, fast=False)
        sig = extract(frames, p.fps, audio, 1.0 / p.fps, astats)
    if p.fast_decode and len(sig.times) == 0:
        frames = decode.iter_frames(path, p.fps, p.width, p.height, p.threads, fast=False)
        sig = extract(frames, p.fps, audio, 1.0 / p.fps, astats)
    return sig


def _reasons(parts: dict, i: int) -> list[str]:
    vals = {k: float(v[i]) for k, v in parts.items() if float(v[i]) > 0.15}
    return [k for k, _ in sorted(vals.items(), key=lambda kv: -kv[1])]


def chunk_video(video_id: str, path: str, srt_path: str | None, p: Params) -> dict:
    t0 = time.time()
    sig = _signals(path, p)
    t_decode = time.time() - t0
    n = len(sig.times)
    duration = n / p.fps
    cues = parse_srt(srt_path) if srt_path else []

    if p.profile == "auto":
        pname, why = classify(sig, bool(cues))
    else:
        pname, why = p.profile, {"forced": True}
    prof = PROFILES[pname]

    ev = segment.evidence(sig, cues)
    strength, parts = segment.strength_curve(ev, prof)
    density = segment.density_curve(sig, ev, cues)
    bounds = segment.select_boundaries(sig.times, strength, density, duration,
                                       prof.min_len, prof.target_len, prof.max_len)
    bounds, moved = segment.snap_to_speech(bounds, cues, prof.min_len, duration)

    edges = [0.0] + [b for b, _ in bounds] + [duration]
    bstrength = [None] + [s for _, s in bounds]
    cut_times = sig.times[ev["cut"] > 0.5]

    scenes = []
    for k in range(len(edges) - 1):
        s, e = edges[k], edges[k + 1]
        i0, i1 = int(round(s * p.fps)), max(int(round(e * p.fps)), int(round(s * p.fps)) + 1)
        i1 = min(i1, n)
        idx = np.arange(i0, i1)
        dens = float(density[i0:i1].mean()) if i1 > i0 else 0.2
        active = float(ev["_active"][i0:i1].mean()) if i1 > i0 else 0.0
        kf_n = int(np.clip(np.ceil((e - s) / 10.0 * p.keyframes_per_10s * dens), 1, p.max_keyframes))
        kf = segment.farthest_point_keyframes(sig.hist, idx, kf_n)
        bi = int(round(s * p.fps))
        rec = {
            "chunk_id": f"{video_id}:s{k:04d}",
            "video_id": video_id,
            "level": "scene",
            "index": k,
            "start": round(s, 2), "end": round(e, 2), "duration": round(e - s, 2),
            "context_start": round(max(0.0, s - p.context_pad), 2),
            "context_end": round(min(duration, e + p.context_pad), 2),
            "boundary_in": None if k == 0 else {
                "strength": round(float(bstrength[k]), 3),
                "reasons": _reasons(parts, min(bi, n - 1)),
            },
            "keyframes": [round(float(sig.times[i]), 2) for i in kf],
            "shot_cuts": [round(float(t), 2) for t in cut_times if s < t < e],
            "density": round(dens, 3),
            "idle": bool(dens < 0.35 and active < 0.05),
            "suggested_fps": round(0.1 + 1.9 * dens ** 2, 2),
        }
        if cues:
            rec["transcript"] = text_between(cues, s, e)
        scenes.append(rec)

    sections = []
    if p.sections and duration > 3 * prof.target_len and len(scenes) > 3:
        tgt = float(np.clip(duration / 6, 60, 600))
        # Section DP runs on scene boundaries only (tiny problem).
        sb = np.array(edges[1:-1])
        ss = np.array([b for b in bstrength[1:]], np.float32)
        grid = np.arange(0, duration, 1.0, dtype=np.float32)
        st = np.zeros_like(grid)
        st[np.clip(np.round(sb).astype(int), 0, len(grid) - 1)] = ss + 0.2
        sec_b = segment.select_boundaries(grid, st, np.ones_like(grid), duration,
                                          tgt / 3, tgt, tgt * 2.5, peak_radius_s=0.5)
        # Snap section edges onto actual scene edges.
        sec_edges = [0.0] + sorted({float(sb[np.argmin(np.abs(sb - b))]) for b, _ in sec_b}) + [duration]
        for k in range(len(sec_edges) - 1):
            s, e = sec_edges[k], sec_edges[k + 1]
            kids = [c["chunk_id"] for c in scenes if c["start"] >= s - 1e-3 and c["end"] <= e + 1e-3]
            sections.append({"chunk_id": f"{video_id}:p{k:03d}", "video_id": video_id,
                             "level": "section", "index": k, "start": round(s, 2),
                             "end": round(e, 2), "duration": round(e - s, 2), "children": kids})
        for c in scenes:
            for sec in sections:
                if c["chunk_id"] in sec["children"]:
                    c["parent"] = sec["chunk_id"]

    # Whole-video chunk: synopsis-type questions ("what is this video about")
    # need global context that no local chunk has.
    all_idx = np.arange(n)
    video_chunk = {
        "chunk_id": f"{video_id}:v", "video_id": video_id, "level": "video",
        "start": 0.0, "end": round(duration, 2), "duration": round(duration, 2),
        "keyframes": [round(float(sig.times[i]), 2)
                      for i in segment.farthest_point_keyframes(sig.hist, all_idx, 16)],
    }

    return {
        "video_id": video_id,
        "source": path,
        "subtitles": srt_path,
        "version": VERSION,
        "duration": round(duration, 2),
        "profile": pname,
        "profile_evidence": why,
        "params": {"fps": p.fps, "size": [p.width, p.height], **prof.as_dict()},
        "stats": {"n_scenes": len(scenes), "n_sections": len(sections),
                  "mean_scene_len": round(duration / max(len(scenes), 1), 2),
                  "boundaries_snapped_to_speech": moved,
                  "idle_fraction": round(sum(c["duration"] for c in scenes if c["idle"]) / max(duration, 1e-6), 3),
                  "decode_s": round(t_decode, 2), "total_s": round(time.time() - t0, 2),
                  "x_realtime": round(duration / max(time.time() - t0, 1e-6), 1)},
        "chunks": [video_chunk] + sections + scenes,
    }
