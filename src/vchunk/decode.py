"""Streaming decode: one low-resolution pass over each video, plus mono audio.

Everything downstream works on these small arrays, so cost is linear in video
duration and memory is bounded by the (tiny) per-frame feature vectors.
"""
from __future__ import annotations

import os
import subprocess
from typing import Iterator

import numpy as np


def ffmpeg_exe() -> str:
    """Prefer a system ffmpeg; fall back to the static binary from imageio-ffmpeg."""
    env = os.environ.get("VCHUNK_FFMPEG")
    if env:
        return env
    from shutil import which

    sys_ff = which("ffmpeg")
    if sys_ff:
        return sys_ff
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def iter_frames(path: str, fps: float = 2.0, width: int = 160, height: int = 90,
                threads: int = 1, fast: bool = True) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames (height, width, 3) sampled at `fps`.

    `fast` asks the decoder to skip non-reference frames. We sample at 2 fps
    anyway, so dropping B-frames loses nothing we use and roughly halves decode
    time for typical H.264 web video.
    """
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
           "-threads", str(threads)]
    if fast:
        cmd += ["-skip_frame", "noref"]
    cmd += ["-i", path, "-an", "-sn",
            "-vf", f"fps={fps},scale={width}:{height}:flags=area",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=frame_bytes * 8)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
        if proc.returncode not in (0, None) and fast and "skip_frame" in " ".join(cmd):
            # Some exotic codecs reject skip_frame; caller retries without it.
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err[-500:]}")


def read_audio(path: str, hop_s: float = 0.5, sr: int = 8000) -> tuple[np.ndarray, dict]:
    """Return (RMS per `hop_s` window, whole-file audio texture stats).

    Texture stats are computed on 64 ms frames, streamed in blocks so memory
    stays flat for multi-hour files:
      flatness       median spectral flatness of voiced frames (low = tonal music)
      micro_pause    fraction of voiced-region frames that dip near silence
                     (speech has syllable gaps; music and crowd noise don't)
    """
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
           "-i", path, "-vn", "-sn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    hop = int(sr * hop_s)
    fr = 512
    block = hop * fr * 4  # multiple of both hop and frame sizes
    rms_out, frame_rms, frame_flat = [], [], []
    carry = np.zeros(0, np.float32)
    win = np.hanning(fr).astype(np.float32)
    while True:
        buf = proc.stdout.read(block * 2)
        if not buf:
            break
        x = np.concatenate([carry, np.frombuffer(buf, np.int16).astype(np.float32) / 32768.0])
        n_full = (x.size // block) * block if len(buf) == block * 2 else x.size
        cur, carry = x[:n_full], x[n_full:]
        n = cur.size // hop
        if n:
            rms_out.append(np.sqrt((cur[: n * hop].reshape(n, hop) ** 2).mean(axis=1)))
        m = cur.size // fr
        if m:
            fx = cur[: m * fr].reshape(m, fr)
            frame_rms.append(np.sqrt((fx ** 2).mean(axis=1)))
            P = np.abs(np.fft.rfft(fx * win, axis=1)) ** 2 + 1e-12
            frame_flat.append(np.exp(np.log(P).mean(axis=1)) / P.mean(axis=1))
    proc.wait()
    if not rms_out:
        return np.zeros(0, np.float32), {"flatness": 1.0, "micro_pause": 0.0}
    rms = np.concatenate(rms_out).astype(np.float32)
    fr_rms = np.concatenate(frame_rms)
    fr_flat = np.concatenate(frame_flat)
    loud = np.percentile(fr_rms, 90) + 1e-9
    voiced = fr_rms > 0.2 * loud
    stats = {"flatness": float(np.median(fr_flat[voiced])) if voiced.any() else 1.0}
    # Micro-pauses: among frames inside active stretches (1 s windows with
    # mostly voiced frames), how many are quiet.
    k = max(1, int(sr / fr))
    if fr_rms.size > k:
        act = np.convolve(voiced.astype(float), np.ones(k) / k, mode="same") > 0.5
        quiet = fr_rms < 0.12 * loud
        stats["micro_pause"] = float((quiet & act).sum() / max(act.sum(), 1))
    else:
        stats["micro_pause"] = 0.0
    return rms, stats


def read_audio_rms(path: str, hop_s: float = 0.5, sr: int = 8000) -> np.ndarray:
    return read_audio(path, hop_s, sr)[0]
