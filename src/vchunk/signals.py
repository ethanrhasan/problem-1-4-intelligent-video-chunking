"""Cheap per-frame signals, all O(1) per sampled frame.

Each signal targets a boundary type that matters for some video genre:

  hist_diff   hard cuts (edited video, sports replays, gameplay menus)
  novelty     gradual changes: fades, lighting/scene shifts (concerts, docs)
  screen_diff content change on static, text-heavy frames (slides, screen share)
  activity    foreground motion vs. a slow background model (surveillance)
  audio       silences / energy shifts (between songs, speaker turns)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HIST_BINS = 4  # per channel -> 64-bin joint RGB histogram


@dataclass
class Signals:
    fps: float
    times: np.ndarray            # (n,) seconds
    hist: np.ndarray             # (n, 64) normalised colour histograms
    hist_diff: np.ndarray        # (n,) L1/2 distance to previous frame, [0,1]
    pix_diff: np.ndarray         # (n,) mean |gray_t - gray_{t-1}| / 255
    edge: np.ndarray             # (n,) edge density
    sat: np.ndarray              # (n,) mean saturation
    activity: np.ndarray         # (n,) foreground fraction vs background model
    mc_diff: np.ndarray          # (n,) motion-compensated profile difference (cuts under camera motion)
    audio: np.ndarray            # (n,) RMS aligned to frame times (zeros if none)
    has_audio: bool
    audio_stats: dict = field(default_factory=dict)
    novelty: np.ndarray = field(default=None)  # filled by compute_novelty


def _joint_hist(rgb: np.ndarray) -> np.ndarray:
    q = (rgb >> 6).astype(np.int32)  # 0..3 per channel
    idx = (q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]).ravel()
    h = np.bincount(idx, minlength=64).astype(np.float32)
    return h / h.sum()


def _profile_residual(a: np.ndarray, b: np.ndarray, max_shift: int) -> float:
    """Min mean |a - shift(b)| over integer shifts: ~0 for a pan, large for a cut."""
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    n = a.size
    best = np.inf
    for d in range(-max_shift, max_shift + 1):
        if d >= 0:
            r = np.abs(a[d:] - b[: n - d]).mean()
        else:
            r = np.abs(a[: n + d] - b[-d:]).mean()
        best = min(best, r)
    return float(best)


def extract(frames, fps: float, audio_rms: np.ndarray, audio_hop: float,
            audio_stats: dict | None = None) -> Signals:
    hists, hd, pdiff, edges, sats, acts, mcd = [], [], [], [], [], [], []
    prev_h = prev_g = bg = prev_px = prev_py = None
    alpha = 1.0 / (fps * 20)  # background adapts over ~20 s
    for rgb in frames:
        f = rgb.astype(np.float32)
        g = f.mean(axis=2)
        h = _joint_hist(rgb)
        mx, mn = f.max(axis=2), f.min(axis=2)
        sats.append(float(((mx - mn) / (mx + 1.0)).mean()))
        gx = np.abs(np.diff(g, axis=1)).mean()
        gy = np.abs(np.diff(g, axis=0)).mean()
        edges.append(float((gx + gy) / 255.0))
        px, py = g.mean(axis=0), g.mean(axis=1)  # column / row intensity profiles
        if prev_h is None:
            mcd.append(0.0)
            hd.append(0.0)
            pdiff.append(0.0)
            bg = g.copy()
            acts.append(0.0)
        else:
            hd.append(float(0.5 * np.abs(h - prev_h).sum()))
            mcd.append(0.5 * (_profile_residual(px, prev_px, px.size // 4) +
                              _profile_residual(py, prev_py, py.size // 4)))
            pdiff.append(float(np.abs(g - prev_g).mean() / 255.0))
            fg = np.abs(g - bg) > 25.0
            acts.append(float(fg.mean()))
            # Freeze background where foreground is present (classic selective update).
            upd = np.where(fg, alpha * 0.1, alpha)
            bg = bg * (1 - upd) + g * upd
        hists.append(h)
        prev_h, prev_g, prev_px, prev_py = h, g, px, py

    n = len(hists)
    times = np.arange(n, dtype=np.float32) / fps
    if audio_rms.size:
        idx = np.minimum((times / audio_hop).astype(int), audio_rms.size - 1)
        audio = audio_rms[idx].astype(np.float32)
    else:
        audio = np.zeros(n, np.float32)
    s = Signals(fps=fps, times=times,
                hist=np.asarray(hists, np.float32).reshape(n, 64),
                hist_diff=np.asarray(hd, np.float32), pix_diff=np.asarray(pdiff, np.float32),
                edge=np.asarray(edges, np.float32), sat=np.asarray(sats, np.float32),
                activity=np.asarray(acts, np.float32), mc_diff=np.asarray(mcd, np.float32),
                audio=audio,
                has_audio=bool(audio_rms.size and audio_rms.max() > 1e-3),
                audio_stats=audio_stats or {})
    s.novelty = window_novelty(s.hist, max(2, int(round(3 * fps))))
    return s


def window_novelty(feat: np.ndarray, w: int) -> np.ndarray:
    """Distance between mean features of [t-w, t) and [t, t+w): O(n) via cumsum.

    A boundary placed at frame t splits the video between frames t-1 and t.
    """
    n = len(feat)
    out = np.zeros(n, np.float32)
    if n < 2 * w:
        return out
    cs = np.vstack([np.zeros((1, feat.shape[1]), np.float32), np.cumsum(feat, axis=0)])
    t = np.arange(w, n - w + 1)
    left = (cs[t] - cs[t - w]) / w
    right = (cs[t + w] - cs[t]) / w
    out[t] = 0.5 * np.abs(left - right).sum(axis=1)
    return out


def robust_norm(x: np.ndarray, q: float = 99.0) -> np.ndarray:
    """Scale to [0,1] by a high percentile so one outlier doesn't flatten the rest."""
    if x.size == 0:
        return x
    hi = np.percentile(x, q)
    if hi <= 1e-9:
        return np.zeros_like(x)
    return np.clip(x / hi, 0, 1)


def local_peaks(x: np.ndarray, radius: int, min_val: float) -> np.ndarray:
    """Indices that are the maximum within +-radius and exceed min_val."""
    if x.size == 0:
        return np.zeros(0, int)
    from numpy.lib.stride_tricks import sliding_window_view

    pad = np.pad(x, radius, mode="constant", constant_values=-np.inf)
    win = sliding_window_view(pad, 2 * radius + 1).max(axis=1)
    return np.flatnonzero((x >= win) & (x > min_val))
