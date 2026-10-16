"""Corpus-level driver: discovery, parallelism, caching, manifest.

Videos are independent, so the corpus cost is the sum of per-video costs
(linear) and parallelises across processes or machines by sharding the video
list (`--shard i/n`). Outputs are content-addressed by params so re-runs skip
finished videos and a crashed job resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

from .pipeline import VERSION, Params, chunk_video

VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts", ".flv"}


def discover(inputs: list[str]) -> list[str]:
    out = []
    for x in inputs:
        if os.path.isdir(x):
            for root, _, files in os.walk(x):
                out += [os.path.join(root, f) for f in files
                        if os.path.splitext(f)[1].lower() in VIDEO_EXT]
        elif os.path.isfile(x):
            out.append(x)
    return sorted(set(out))


def find_srt(video: str, subs_dirs: list[str]) -> str | None:
    stem = os.path.splitext(os.path.basename(video))[0]
    for d in [os.path.dirname(video)] + subs_dirs:
        for cand in (os.path.join(d, stem + ".srt"), os.path.join(d, stem + ".en.srt")):
            if os.path.isfile(cand):
                return cand
    return None


def params_key(p: Params) -> str:
    return hashlib.sha1((VERSION + json.dumps(asdict(p), sort_keys=True)).encode()).hexdigest()[:10]


def _work(args):
    video, srt, out_path, p = args
    vid = os.path.splitext(os.path.basename(video))[0]
    try:
        res = chunk_video(vid, video, srt, p)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f)
        os.replace(tmp, out_path)
        return vid, res["stats"], res["profile"], None
    except Exception as e:  # one bad file must not kill a corpus run
        return vid, None, None, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"


def run(inputs, subs_dirs, out_dir, p: Params, workers: int = 1, shard: str | None = None,
        force: bool = False, limit: int | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    videos = discover(inputs)
    if shard:
        i, n = map(int, shard.split("/"))
        videos = [v for k, v in enumerate(videos) if k % n == i]
    if limit:
        videos = videos[:limit]
    key = params_key(p)
    jobs = []
    skipped = 0
    for v in videos:
        vid = os.path.splitext(os.path.basename(v))[0]
        out_path = os.path.join(out_dir, f"{vid}.chunks.json")
        if not force and os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    if json.load(f).get("_params_key") == key:
                        skipped += 1
                        continue
            except Exception:
                pass
        jobs.append((v, find_srt(v, subs_dirs), out_path, p))

    results, errors = [], []
    print(f"[vchunk] {len(videos)} videos, {skipped} cached, {len(jobs)} to process, "
          f"{workers} workers", file=sys.stderr)
    if workers <= 1:
        it = (_work(j) for j in jobs)
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        it = (f.result() for f in as_completed([ex.submit(_work, j) for j in jobs]))
    for vid, stats, prof, err in it:
        if err:
            errors.append({"video_id": vid, "error": err})
            print(f"[vchunk] FAIL {vid}: {err.splitlines()[0]}", file=sys.stderr)
        else:
            results.append({"video_id": vid, "profile": prof, **stats})
            print(f"[vchunk] {vid}: {prof:12s} {stats['n_scenes']:3d} scenes "
                  f"(mean {stats['mean_scene_len']}s) {stats['x_realtime']}x realtime",
                  file=sys.stderr)

    # Stamp params key into outputs written this run (for cache checks).
    for j in jobs:
        op = j[2]
        if os.path.exists(op):
            with open(op) as f:
                d = json.load(f)
            d["_params_key"] = key
            with open(op, "w") as f:
                json.dump(d, f)

    write_manifest(out_dir)
    summary = {"processed": len(results), "cached": skipped, "failed": len(errors),
               "errors": errors}
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump({"params": asdict(p), "params_key": key, **summary, "videos": results}, f, indent=2)
    return summary


def write_manifest(out_dir: str):
    """Flatten every chunk into one JSONL - the hand-off to the indexing component."""
    rows = 0
    with open(os.path.join(out_dir, "chunks.jsonl"), "w") as out:
        for fn in sorted(os.listdir(out_dir)):
            if not fn.endswith(".chunks.json"):
                continue
            with open(os.path.join(out_dir, fn)) as f:
                d = json.load(f)
            for c in d["chunks"]:
                c = dict(c, profile=d["profile"], source=d["source"])
                out.write(json.dumps(c) + "\n")
                rows += 1
    return rows
