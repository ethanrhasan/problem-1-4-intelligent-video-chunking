import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vchunk import segment, synth  # noqa: E402
from vchunk.pipeline import Params, chunk_video  # noqa: E402
from vchunk.subtitles import Cue, inside_cue, parse_srt, speech_gaps, text_between  # noqa: E402


def test_dp_respects_min_max_and_hits_strong_boundaries():
    fps, dur = 2.0, 300.0
    t = np.arange(int(dur * fps)) / fps
    st = np.zeros_like(t)
    for b in (37.0, 80.0, 81.0, 150.0, 260.0):  # 80/81 too close to both be used
        st[int(b * fps)] = 1.0
    dens = np.ones_like(t)
    out = segment.select_boundaries(t, st, dens, dur, min_len=10, target_len=40, max_len=90)
    edges = [0.0] + [b for b, _ in out] + [dur]
    lens = np.diff(edges)
    assert lens.min() >= 5 - 1e-6 and lens.max() <= 90 + 1e-6
    chosen = [b for b, _ in out]
    for b in (37.0, 150.0, 260.0):
        assert any(abs(c - b) < 1e-6 for c in chosen)
    assert sum(1 for c in chosen if 79 <= c <= 82) == 1


def test_idle_span_stays_one_chunk_busy_span_splits():
    fps, dur = 2.0, 200.0
    t = np.arange(int(dur * fps)) / fps
    st = np.zeros_like(t)
    dens = np.full_like(t, 0.2)
    dens[int(100 * fps):] = 1.0  # second half busy
    out = segment.select_boundaries(t, st, dens, dur, min_len=3, target_len=30, max_len=300)
    chosen = [b for b, _ in out]
    assert not any(b < 95 for b in chosen), chosen      # idle half untouched
    assert sum(1 for b in chosen if b > 100) >= 2        # busy half split


def test_srt_and_speech_helpers(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello <i>there</i>.\n\n"
                 "2\n00:00:05,500 --> 00:00:08,000\nSecond line\n\n")
    cues = parse_srt(str(p))
    assert [c.text for c in cues] == ["Hello there.", "Second line"]
    assert inside_cue(cues, 2.0) is not None and inside_cue(cues, 4.0) is None
    gaps = speech_gaps(cues)
    assert len(gaps) == 1 and 3.0 < gaps[0][0] < 5.5
    assert text_between(cues, 0, 4) == "Hello there."


def test_snap_moves_boundary_out_of_sentence():
    cues = [Cue(0, 9.8, "a."), Cue(10.4, 20, "b.")]
    out, moved = segment.snap_to_speech([(8.0, 1.0)], cues, min_len=4, duration=30)
    assert moved == 1 and 9.8 <= out[0][0] <= 10.4


def test_keyframes_cover_distinct_states():
    feat = np.zeros((20, 64), np.float32)
    feat[:10, 0] = 1
    feat[10:, 1] = 1
    kf = segment.farthest_point_keyframes(feat, np.arange(20), 4)
    assert any(k < 10 for k in kf) and any(k >= 10 for k in kf)
    assert len(kf) == 2  # identical frames are not repeated


@pytest.fixture(scope="module")
def surveillance_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    path = str(d / "synth_surveillance.mp4")
    gt = synth.gen_surveillance(path, np.random.default_rng(0))
    return path, gt


def test_end_to_end_surveillance(surveillance_video):
    path, gt = surveillance_video
    res = chunk_video("cam", path, None, Params())
    scenes = [c for c in res["chunks"] if c["level"] == "scene"]
    assert res["profile"] == "surveillance"
    # contiguous, non-overlapping cover of the whole video
    assert scenes[0]["start"] == 0 and abs(scenes[-1]["end"] - res["duration"]) < 0.01
    for a, b in zip(scenes, scenes[1:]):
        assert a["end"] == b["start"]
    # every event lies inside a single non-idle chunk
    for s, e in gt["events"]:
        holders = [c for c in scenes if c["start"] <= s + 3 and c["end"] >= e - 3]
        assert holders and not holders[0]["idle"]
    assert res["stats"]["idle_fraction"] > 0.4
    assert any(c["level"] == "video" for c in res["chunks"])


def test_cli_chunk_writes_manifest(surveillance_video, tmp_path):
    path, _ = surveillance_video
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "main.py"), "chunk", path,
                        "--out", str(out), "--workers", "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in open(out / "chunks.jsonl")]
    assert rows and {"chunk_id", "start", "end", "level"} <= set(rows[0])
    # second run is a cache hit
    r2 = subprocess.run([sys.executable, os.path.join(ROOT, "src", "main.py"), "chunk", path,
                         "--out", str(out), "--workers", "1"], capture_output=True, text=True)
    assert "1 cached" in r2.stderr


def test_pick_videos_round_robin():
    import main
    rows = [{"video_id": f"{d}{i}", "duration": "short", "domain": d} for d in "AB" for i in range(3)]
    rows += [{"video_id": "L1", "duration": "long", "domain": "C"}]
    assert main.pick_videos(rows, ["short"], 3) == ["A0", "B0", "A1"]
    assert len(main.pick_videos(rows)) == 7
