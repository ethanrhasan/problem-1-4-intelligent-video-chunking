#!/usr/bin/env python3
"""vchunk - intelligent, linear-cost video chunking.

Commands
  chunk          chunk every video under the given paths -> <out>/*.chunks.json + chunks.jsonl
  download       fetch (part of) the Video-MME sample from Hugging Face
  eval-synthetic generate videos with known boundaries and score boundary accuracy
  report         summarise a chunk output directory
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vchunk import corpus, evaluate  # noqa: E402
from vchunk.pipeline import Params  # noqa: E402
from vchunk.profiles import PROFILES  # noqa: E402


def cmd_chunk(a):
    p = Params(fps=a.fps, profile=a.profile, context_pad=a.context_pad,
               sections=not a.no_sections, fast_decode=not a.no_fast_decode, threads=a.threads)
    s = corpus.run(a.inputs, a.subs or [], a.out, p, workers=a.workers, shard=a.shard,
                   force=a.force, limit=a.limit)
    print(json.dumps({k: v for k, v in s.items() if k != "errors"}))
    return 1 if s["failed"] and not s["processed"] else 0


def pick_videos(rows, durations=None, max_videos=None):
    """Choose videos, shortest buckets first, round-robin across domains so a
    small download still covers many genres."""
    order = {"short": 0, "medium": 1, "long": 2}
    by_dom: dict[str, list[str]] = {}
    seen = set()
    for r in sorted(rows, key=lambda r: (order.get(r["duration"], 3), r["video_id"])):
        if durations and r["duration"] not in durations:
            continue
        if r["video_id"] in seen:
            continue
        seen.add(r["video_id"])
        by_dom.setdefault(r.get("domain", ""), []).append(r["video_id"])
    pick = []
    while any(by_dom.values()) and (not max_videos or len(pick) < max_videos):
        for d in sorted(by_dom):
            if by_dom[d] and (not max_videos or len(pick) < max_videos):
                pick.append(by_dom[d].pop(0))
    return pick


def cmd_download(a):
    from huggingface_hub import hf_hub_download, snapshot_download

    os.makedirs(a.dest, exist_ok=True)
    try:
        qa = hf_hub_download(a.dataset, "qa/test.jsonl", repo_type="dataset", local_dir=a.dest)
    except Exception as e:
        print(f"download failed ({type(e).__name__}: {e}). Check network access to "
              f"huggingface.co, or place videos under {a.dest}/videos manually.", file=sys.stderr)
        return 1
    rows = [json.loads(l) for l in open(qa)]
    vids = pick_videos(rows, a.duration, a.max_videos)
    patterns = [f"videos/{v}.mp4" for v in vids] + [f"subtitles/{v}.srt" for v in vids]
    snapshot_download(a.dataset, repo_type="dataset", local_dir=a.dest, allow_patterns=patterns)
    print(f"downloaded {len(vids)} videos to {a.dest}")
    return 0


def cmd_eval(a):
    rep = evaluate.run(a.out, seed=a.seed, tol=a.tol)
    evaluate.print_report(rep)
    return 0


def cmd_report(a):
    rows = []
    for fn in sorted(os.listdir(a.out)):
        if fn.endswith(".chunks.json"):
            d = json.load(open(os.path.join(a.out, fn)))
            s = d["stats"]
            rows.append((d["video_id"], d["profile"], d["duration"], s["n_scenes"],
                         s["mean_scene_len"], s["idle_fraction"], s["x_realtime"]))
    print(f"{'video':14s} {'profile':12s} {'dur(s)':>8s} {'scenes':>6s} {'mean':>6s} {'idle':>5s} {'xRT':>6s}")
    for r in rows:
        print(f"{r[0]:14s} {r[1]:12s} {r[2]:8.0f} {r[3]:6d} {r[4]:6.1f} {r[5]:5.2f} {r[6]:6.0f}")
    if rows:
        tot = sum(r[2] for r in rows)
        print(f"{len(rows)} videos, {tot / 3600:.2f} h, {sum(r[3] for r in rows)} scene chunks")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vchunk", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chunk", help="chunk videos")
    c.add_argument("inputs", nargs="+", help="video files or directories")
    c.add_argument("--subs", nargs="*", help="directories with <video_id>.srt transcripts")
    c.add_argument("--out", default="out/chunks")
    c.add_argument("--profile", default="auto", choices=["auto", *PROFILES])
    c.add_argument("--fps", type=float, default=2.0, help="analysis sample rate")
    c.add_argument("--context-pad", type=float, default=2.0)
    c.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    c.add_argument("--threads", type=int, default=1, help="ffmpeg threads per video")
    c.add_argument("--shard", help="i/n: process every n-th video starting at i")
    c.add_argument("--limit", type=int)
    c.add_argument("--force", action="store_true")
    c.add_argument("--no-sections", action="store_true")
    c.add_argument("--no-fast-decode", action="store_true")
    c.set_defaults(fn=cmd_chunk)

    d = sub.add_parser("download", help="download Video-MME sample videos")
    d.add_argument("--dataset", default="aintropy-ai/video-mme-sample10")
    d.add_argument("--dest", default="data/video-mme-sample10")
    d.add_argument("--max-videos", type=int, default=6)
    d.add_argument("--duration", nargs="*", choices=["short", "medium", "long"])
    d.set_defaults(fn=cmd_download)

    e = sub.add_parser("eval-synthetic", help="boundary accuracy on generated videos")
    e.add_argument("--out", default="out/eval")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--tol", type=float, default=2.0)
    e.set_defaults(fn=cmd_eval)

    r = sub.add_parser("report", help="summarise a chunk output dir")
    r.add_argument("out", nargs="?", default="out/chunks")
    r.set_defaults(fn=cmd_report)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
