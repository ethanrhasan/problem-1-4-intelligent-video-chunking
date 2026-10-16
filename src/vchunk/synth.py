"""Synthetic videos with known ground-truth boundaries, one per target genre.

Real corpora rarely ship with chunk-boundary labels, so we measure boundary
accuracy on generated videos whose structure we control, then sanity-check on
real footage. Each generator mimics the boundary type that matters for its
genre (hard cuts, gradual fades, slide flips, activity bursts, song gaps).
"""
from __future__ import annotations

import json
import os
import subprocess
import wave

import numpy as np

from .decode import ffmpeg_exe

W, H, FPS, SR = 320, 180, 10, 16000


def _encode(frames, audio: np.ndarray | None, path: str):
    tmp_wav = path + ".wav"
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if audio is not None:
        with wave.open(tmp_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(audio, -1, 1) * 32000).astype(np.int16).tobytes())
        cmd += ["-i", tmp_wav, "-c:a", "aac", "-b:a", "48k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
            "-g", "50", "-shortest", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)


def _speechy(dur, rng, on=True):
    """Amplitude-modulated noise bursts that look like speech to an energy detector."""
    n = int(dur * SR)
    x = rng.normal(0, 0.25, n)
    env = np.zeros(n)
    t = 0
    while on and t < n:
        L = int(rng.uniform(0.2, 0.6) * SR)
        env[t:t + L] = 1
        t += L + int(rng.uniform(0.05, 0.2) * SR)
    return x * env


def _texture(rng, scale):
    small = rng.integers(0, 255, (-(-H // scale), -(-W // scale), 3), dtype=np.uint8)
    return np.kron(small, np.ones((scale, scale, 1), np.uint8))[:H, :W]


def gen_edited(path, rng):
    """News/doc style: shots of different looks, one 2 s crossfade, speech with
    pauses at scene ends. GT = shot changes (incl. the fade midpoint)."""
    lens = [12, 25, 9, 30, 18, 14, 22]
    frames, bounds, t = [], [], 0.0
    looks = [_texture(rng, s) for s in (40, 8, 20, 60, 4, 30, 12)]
    for k, L in enumerate(lens):
        base = looks[k].astype(np.float32)
        for i in range(int(L * FPS)):
            dx = int(i * (1 + k % 3)) % W  # slow pan inside a shot
            f = np.roll(base, dx, axis=1)
            if k == 3 and i < 2 * FPS:  # crossfade from previous shot
                a = i / (2 * FPS)
                f = (1 - a) * np.roll(looks[2].astype(np.float32), dx, axis=1) + a * f
            frames.append(f.astype(np.uint8))
        t += L
        bounds.append(t)
    bounds = bounds[:-1]
    bounds[2] += 1.0  # the fade midpoint
    audio = np.concatenate([np.r_[_speechy(L - 0.8, rng), np.zeros(int(0.8 * SR))] for L in lens])
    _encode(frames, audio, path)
    return {"genre": "edited", "duration": sum(lens), "boundaries": bounds}


def gen_slides(path, rng):
    """Screen share: static slides of 'text lines', a moving cursor, continuous
    speech. GT = slide flips."""
    lens = [35, 50, 20, 60, 40]
    frames, bounds, t = [], [], 0.0
    for k, L in enumerate(lens):
        slide = np.full((H, W, 3), 245, np.uint8)
        slide[8:24, 20:20 + int(rng.integers(120, 260))] = (30, 60, 150)  # title
        for r in range(5 + k):
            y = 36 + r * 14
            if y + 6 >= H:
                break
            x = 30
            while x < W - 40:
                wl = int(rng.integers(10, 40))
                slide[y:y + 6, x:x + wl] = 40
                x += wl + 6
        for i in range(int(L * FPS)):
            f = slide.copy()
            cx = int(150 + 80 * np.sin(i / 37.0))
            cy = int(90 + 50 * np.cos(i / 53.0))
            f[cy:cy + 6, cx:cx + 4] = 0  # cursor
            frames.append(f)
        t += L
        bounds.append(t)
    audio = np.concatenate([_speechy(L, rng) for L in lens])
    _encode(frames, audio, path)
    return {"genre": "screen", "duration": sum(lens), "boundaries": bounds[:-1]}


def gen_surveillance(path, rng):
    """Fixed camera, sensor noise, two people-sized blobs crossing at known times.
    GT = activity intervals."""
    dur = 240
    bg = _texture(rng, 16).astype(np.float32) * 0.5 + 60
    events = [(40, 58), (130, 142), (200, 226)]
    frames = []
    for i in range(dur * FPS):
        t = i / FPS
        f = bg + rng.normal(0, 3, bg.shape)
        for s, e in events:
            if s <= t < e:
                x = int((t - s) / (e - s) * (W - 30))
                f[60:150, x:x + 28] = (200, 40, 40)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    _encode(frames, None, path)
    b = []
    for s, e in events:
        b += [s, e]
    return {"genre": "surveillance", "duration": dur, "boundaries": b, "events": events}


def gen_sports(path, rng):
    """Fast camera motion with frequent cuts (plays/replays). GT = cuts."""
    lens = [8, 14, 6, 11, 16, 9, 13]
    frames, bounds, t = [], [], 0.0
    for k, L in enumerate(lens):
        field = np.zeros((H, W * 3, 3), np.uint8)
        field[:] = (30 + 20 * k, 120, 40)
        stripes = np.arange(W * 3) % (40 + 10 * k) < 6
        field[:, stripes] = 230
        for _ in range(12):
            y, x = rng.integers(10, H - 20), rng.integers(0, W * 3 - 10)
            field[y:y + 14, x:x + 6] = rng.integers(0, 255, 3)
        speed = 6 + 3 * (k % 3)
        for i in range(int(L * FPS)):
            off = int((i * speed) % (W * 2))
            frames.append(field[:, off:off + W])
        t += L
        bounds.append(t)
    audio = rng.normal(0, 0.1, int(sum(lens) * SR))  # crowd noise, no pauses
    _encode(frames, audio, path)
    return {"genre": "high_motion", "duration": sum(lens), "boundaries": bounds[:-1]}


def gen_concert(path, rng):
    """Stage with slowly drifting lights, continuous music within a song,
    3 s near-silence + lighting change between songs. GT = song gaps."""
    songs = [45, 60, 40, 55]
    frames, audio, bounds, t = [], [], [], 0.0
    stage = _texture(rng, 24).astype(np.float32) * 0.3
    for k, L in enumerate(songs):
        hue = np.array([(k * 97) % 255, (k * 53 + 80) % 255, (k * 151 + 40) % 255], np.float32)
        n = int(L * FPS)
        for i in range(n):
            pulse = 0.75 + 0.25 * np.sin(i / 3.0)
            gap = i >= n - 3 * FPS
            light = hue * (0.25 if gap else pulse)
            frames.append(np.clip(stage + light, 0, 255).astype(np.uint8))
        tt = np.arange(int((L - 3) * SR)) / SR
        music = 0.3 * np.sin(2 * np.pi * (220 + 40 * k) * tt) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * tt))
        audio.append(np.r_[music, rng.normal(0, 0.003, 3 * SR)])
        t += L
        bounds.append(t - 1.5)
    _encode(frames, np.concatenate(audio), path)
    return {"genre": "performance", "duration": sum(songs), "boundaries": bounds[:-1]}


GENERATORS = {"edited": gen_edited, "screen": gen_slides, "surveillance": gen_surveillance,
              "high_motion": gen_sports, "performance": gen_concert}


def make_corpus(out_dir: str, seed: int = 0) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    gts = []
    for name, fn in GENERATORS.items():
        path = os.path.join(out_dir, f"synth_{name}.mp4")
        gt_path = path + ".gt.json"
        if os.path.exists(path) and os.path.exists(gt_path):
            with open(gt_path) as f:
                gts.append(json.load(f))
            continue
        gt = fn(path, np.random.default_rng(seed))
        gt["video_id"] = f"synth_{name}"
        gt["path"] = path
        with open(gt_path, "w") as f:
            json.dump(gt, f)
        gts.append(gt)
    return gts
