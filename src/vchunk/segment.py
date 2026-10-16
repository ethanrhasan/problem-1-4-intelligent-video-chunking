"""Boundary scoring and selection.

1. Per-frame boundary evidence from each signal, normalised to [0,1].
2. Weighted sum with the video's profile weights -> one boundary-strength curve.
3. Candidates = local peaks of that curve (+ a sparse fallback grid so a valid
   segmentation always exists).
4. Dynamic programming picks the boundary set that maximises
       sum(boundary strength) - sum(length cost)
   under hard min/max chunk lengths. The length cost is scaled by how much
   information the span carries, so idle stretches (empty corridor, paused
   game, silent slide) may become long cheap chunks while busy stretches are
   pushed toward the target length.

Every step is O(n) or O(n * w) with w bounded by max_len / min_spacing, so cost
stays linear in duration.
"""
from __future__ import annotations

import bisect

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .profiles import Profile
from .signals import Signals, local_peaks, robust_norm
from .subtitles import Cue, speech_gaps


def _moving(x: np.ndarray, w: int, fn) -> np.ndarray:
    if x.size == 0:
        return x
    w = max(1, min(w, x.size))
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return fn(sliding_window_view(pad, w), axis=1)


def evidence(sig: Signals, cues: list[Cue] | None) -> dict[str, np.ndarray]:
    """Per-frame boundary evidence in [0,1]; index i = boundary before frame i."""
    fps, n = sig.fps, len(sig.times)
    ev: dict[str, np.ndarray] = {}

    # Hard cuts: adaptive threshold (local median + 3 MAD) so busy footage
    # (sports) does not fire on every frame while calm footage still catches cuts.
    hd = sig.hist_diff
    w = int(20 * fps)
    med = _moving(hd, w, np.median)
    mad = _moving(np.abs(hd - med), w, np.median)
    thr = np.maximum(0.25, med + 3 * 1.4826 * mad)
    cut_h = np.clip((hd - thr) / 0.35 + 0.5, 0, 1) * (hd > thr)
    # Same palette, different shot (sports angle change, game camera swap):
    # the histogram barely moves but the motion-compensated layout does. Require
    # a spike vs. both neighbours so continuous fast motion never fires.
    mc = sig.mc_diff
    nb = np.maximum(np.r_[0, mc[:-1]], np.r_[mc[1:], 0])
    mmed = _moving(mc, w, np.median)
    spike = (mc > 0.35) & (mc > 1.8 * nb) & (mc > 2.5 * mmed)
    cut_m = np.clip((mc - nb) / 0.4, 0, 1) * spike
    ev["cut"] = np.maximum(cut_h, cut_m)

    # Gradual change: windowed histogram novelty, peaks relative to local baseline.
    nov = sig.novelty
    base = _moving(nov, int(30 * fps), np.median)
    ev["novelty"] = robust_norm(np.maximum(nov - base, 0))

    # Screen content change: a jump between two still periods on edge-rich frames
    # (slide flip, window switch in a screen share).
    still = (sig.pix_diff < 0.004).astype(np.float32)
    k = max(1, int(2 * fps))
    cs = np.concatenate([[0.0], np.cumsum(still)])
    ii = np.arange(n)
    b0 = np.maximum(ii - k, 0)
    still_before = (cs[ii] - cs[b0]) / np.maximum(ii - b0, 1)          # frames [i-k, i)
    a1 = np.minimum(ii + 1 + k, n)
    still_after = (cs[a1] - cs[np.minimum(ii + 1, n)]) / np.maximum(a1 - ii - 1, 1)  # (i, i+k]
    textish = np.clip((sig.edge - 0.03) / 0.05, 0, 1)
    ev["screen"] = (np.clip(sig.pix_diff / 0.02, 0, 1) *
                    np.minimum(still_before, still_after) * textish).astype(np.float32)

    # Activity on/off (hysteresis on smoothed foreground fraction).
    act = _moving(sig.activity, max(1, int(2 * fps)), np.mean)
    hi = max(0.015, float(np.percentile(act, 50)) * 2.0)
    lo = hi * 0.5
    state, on = False, np.zeros(n, bool)
    for i, a in enumerate(act):
        state = (a > lo) if state else (a > hi)
        on[i] = state
    trans = np.zeros(n, np.float32)
    trans[1:] = (on[1:] != on[:-1]).astype(np.float32)
    # Pad events: boundary 2 s before activity starts / after it ends.
    pad = int(2 * fps)
    t2 = np.zeros(n, np.float32)
    for i in np.flatnonzero(trans):
        j = i - pad if on[i] else i + pad
        t2[int(np.clip(j, 0, n - 1))] = 1.0
    ev["activity"] = t2
    ev["_active"] = on.astype(np.float32)

    # Audio silences: centre of each quiet run >= 0.8 s, stronger when longer.
    sil = np.zeros(n, np.float32)
    if sig.has_audio:
        a = sig.audio
        quiet = a < 0.08 * (np.percentile(a, 90) + 1e-6)
        i = 0
        while i < n:
            if quiet[i]:
                j = i
                while j < n and quiet[j]:
                    j += 1
                dur = (j - i) / fps
                if dur >= 0.8:
                    sil[(i + j) // 2] = min(1.0, dur / 3.0)
                i = j
            else:
                i += 1
    ev["silence"] = sil

    # Transcript gaps, plus a penalty map for "inside a spoken sentence".
    sp = np.zeros(n, np.float32)
    inside = np.zeros(n, np.float32)
    if cues:
        for t, s in speech_gaps(cues):
            i = int(round(t * fps))
            if 0 <= i < n:
                sp[i] = max(sp[i], s)
        for c in cues:
            i0, i1 = int(np.ceil(c.start * fps + 0.5)), int(np.floor(c.end * fps - 0.5))
            if i1 >= i0:
                inside[max(i0, 0): min(i1 + 1, n)] = 1.0
    ev["speech"] = sp
    ev["_in_speech"] = inside
    return ev


def density_curve(sig: Signals, ev: dict, cues: list[Cue] | None) -> np.ndarray:
    """How much is happening per frame, in [0.2, 1]. Drives length cost and the
    downstream sampling budget."""
    fps = sig.fps
    # Absolute motion above the video's own noise floor (sensor noise / grain must
    # not make an empty corridor look busy).
    pd = _moving(sig.pix_diff, int(2 * fps), np.mean)
    floor = float(np.percentile(sig.pix_diff, 10)) * 1.5
    motion = np.clip((pd - floor) / 0.03, 0, 1)
    activity = ev["_active"]
    speech = ev["_in_speech"]
    if not cues and sig.has_audio:
        a = sig.audio
        speech = (a > 0.15 * (np.percentile(a, 90) + 1e-6)).astype(np.float32)
    change = ((ev["cut"] > 0.5) | (ev["screen"] > 0.5) | (ev["novelty"] > 0.5)).astype(np.float32)
    change = _moving(change, int(10 * fps), np.max)
    d = np.maximum.reduce([motion, activity, speech * 0.9, change * 0.8])
    return np.clip(0.2 + 0.8 * d, 0.2, 1.0).astype(np.float32)


def strength_curve(ev: dict, p: Profile) -> tuple[np.ndarray, dict]:
    parts = {
        "cut": p.w_cut * ev["cut"],
        "novelty": p.w_novelty * ev["novelty"],
        "screen": p.w_screen * ev["screen"],
        "activity": p.w_activity * ev["activity"],
        "silence": p.w_silence * ev["silence"],
        "speech": p.w_speech * ev["speech"],
    }
    total = sum(parts.values())
    # Discourage cutting inside a spoken sentence unless the visual evidence is
    # overwhelming (a hard cut in a news package mid-sentence is still a scene change).
    total = total - 0.6 * ev["_in_speech"] * (1 - ev["cut"])
    return total.astype(np.float32), parts


def select_boundaries(times: np.ndarray, strength: np.ndarray, density: np.ndarray,
                      duration: float, min_len: float, target_len: float,
                      max_len: float, lam: float = 0.6, boundary_cost: float = 0.1,
                      peak_radius_s: float = 1.0) -> list[tuple[float, float]]:
    """DP over candidate boundaries. Returns [(time, strength)] of chosen cuts."""
    if duration <= min_len * 1.5:
        return []
    fps = 1.0 / (times[1] - times[0]) if len(times) > 1 else 1.0
    radius = max(1, int(round(peak_radius_s * fps)))
    peaks = local_peaks(strength, radius, 0.12)
    cand_t = [float(times[i]) for i in peaks]
    cand_s = [float(strength[i]) for i in peaks]
    # Fallback grid in any span with no peak for > min_len, so that a feasible
    # path always exists (long uneventful shots still get split under max_len).
    grid_step = max(min_len, 1.0)
    existing = sorted(cand_t)
    g = grid_step
    while g < duration - min_len * 0.5:
        k = bisect.bisect_left(existing, g)
        near = min([abs(existing[j] - g) for j in (k - 1, k) if 0 <= j < len(existing)] or [1e9])
        if near > grid_step * 0.5:
            cand_t.append(g)
            cand_s.append(0.0)
        g += grid_step
    order = np.argsort(cand_t)
    ct = np.array([0.0] + [cand_t[i] for i in order] + [duration])
    cs = np.array([0.0] + [cand_s[i] for i in order] + [0.0])
    m = len(ct)

    # Prefix sums of density for O(1) mean density of any span.
    dcum = np.concatenate([[0.0], np.cumsum(density)])

    def span_density(a, b):
        i0 = int(a * fps)
        i1 = max(int(b * fps), i0 + 1)
        i1 = min(i1, len(density))
        i0 = min(i0, i1 - 1)
        return (dcum[i1] - dcum[i0]) / (i1 - i0)

    best = np.full(m, -np.inf)
    back = np.full(m, -1, int)
    best[0] = 0.0
    for j in range(1, m):
        last = j == m - 1
        lo_t = ct[j] - max_len
        hi_t = ct[j] - (min_len * 0.5 if last else min_len)
        i_lo = int(np.searchsorted(ct, lo_t - 1e-6, side="left"))
        i_hi = int(np.searchsorted(ct, hi_t + 1e-6, side="right"))
        if i_hi <= i_lo:
            continue
        idx = np.arange(i_lo, min(i_hi, j))
        idx = idx[np.isfinite(best[idx])]
        if idx.size == 0:
            continue
        L = ct[j] - ct[idx]
        dens = np.array([span_density(ct[i], ct[j]) for i in idx])
        # Over-long chunks are only expensive when they're busy; short chunks are
        # always mildly penalised (they fragment context).
        over = np.maximum(L - target_len, 0) / target_len
        under = np.maximum(target_len - L, 0) / target_len
        busy = ((dens - 0.2) / 0.8) ** 2  # 0 for idle spans, 1 for fully busy
        cost = lam * (busy * over ** 2 + 0.5 * under ** 2) + (0 if last else boundary_cost)
        val = best[idx] + (0 if last else cs[j]) - cost
        k = int(np.argmax(val))
        best[j] = val[k]
        back[j] = idx[k]
    if not np.isfinite(best[-1]):
        # Infeasible under constraints (should be rare): uniform split.
        nseg = max(1, int(round(duration / target_len)))
        return [(duration * i / nseg, 0.0) for i in range(1, nseg)]
    out, j = [], back[m - 1]
    while j > 0:
        out.append((float(ct[j]), float(cs[j])))
        j = back[j]
    return sorted(out)


def snap_to_speech(bounds: list[tuple[float, float]], cues: list[Cue] | None,
                   min_len: float, duration: float, window: float = 3.0):
    """Move any boundary that still lands inside a subtitle cue to the nearest
    pause within +-window seconds, without violating half the min length."""
    if not cues:
        return bounds, 0
    from .subtitles import inside_cue

    gaps = [t for t, _ in speech_gaps(cues, min_gap=0.0)]
    times = [b for b, _ in bounds]
    moved = 0
    for k, (t, s) in enumerate(bounds):
        if inside_cue(cues, t) is None:
            continue
        prev_t = times[k - 1] if k else 0.0
        next_t = times[k + 1] if k + 1 < len(times) else duration
        options = [g for g in gaps if abs(g - t) <= window
                   and g - prev_t >= min_len * 0.5 and next_t - g >= min_len * 0.5]
        if options:
            g = min(options, key=lambda x: abs(x - t))
            bounds[k] = (g, s)
            times[k] = g
            moved += 1
    return bounds, moved


def farthest_point_keyframes(feat: np.ndarray, idx: np.ndarray, k: int) -> list[int]:
    """Pick k frames that are maximally different (covers every visual state in
    the chunk). First pick is the frame closest to the chunk's mean appearance."""
    if idx.size == 0:
        return []
    k = min(k, idx.size)
    X = feat[idx]
    mu = X.mean(axis=0)
    first = int(np.argmin(np.abs(X - mu).sum(axis=1)))
    chosen = [first]
    dmin = np.abs(X - X[first]).sum(axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dmin))
        if dmin[nxt] < 0.05:  # remaining frames look the same - stop early
            break
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.abs(X - X[nxt]).sum(axis=1))
    return sorted(int(idx[c]) for c in chosen)
