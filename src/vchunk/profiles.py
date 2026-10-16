"""Video-type profiles and a cheap, explainable classifier.

The profile only changes *weights and duration targets*; every video goes
through the same pipeline. That keeps behaviour predictable on types we never
saw (a mis-classification degrades gracefully instead of breaking chunking).

Durations (seconds) are chosen from how people query each type:
  * screen / meeting: questions are about a slide, a decision, a topic - long
    coherent spans, boundaries at slide changes and speaker pauses.
  * surveillance: questions are about events ("when did a person enter") -
    chunks wrap activity bursts; long idle stretches collapse into one chunk.
  * high_motion (sports, gameplay): plays/rounds are short; questions are about
    a specific action - short chunks.
  * performance (concert, stage): song or act level; silences and lighting
    changes separate songs.
  * edited (news, docs, vlogs, film): scene level around shot cuts + speech.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .signals import Signals


@dataclass(frozen=True)
class Profile:
    name: str
    min_len: float
    target_len: float
    max_len: float
    w_cut: float      # hard cuts
    w_novelty: float  # gradual / windowed change
    w_screen: float   # content change on static, text-like frames
    w_activity: float # foreground on/off transitions
    w_silence: float  # audio pauses
    w_speech: float   # transcript gaps (sentence ends)

    def as_dict(self):
        return asdict(self)


PROFILES = {
    "edited":      Profile("edited",       4, 20,  45, 1.0, 0.8, 0.2, 0.2, 0.4, 0.8),
    "screen":      Profile("screen",       8, 40, 120, 0.5, 0.4, 1.4, 0.0, 0.5, 1.0),
    "meeting":     Profile("meeting",     10, 45, 120, 0.6, 0.5, 0.6, 0.3, 0.8, 1.2),
    "surveillance":Profile("surveillance", 3, 30, 300, 0.3, 0.3, 0.0, 1.6, 0.2, 0.3),
    "high_motion": Profile("high_motion",  4, 15,  30, 1.0, 1.0, 0.3, 0.2, 0.3, 0.5),
    "performance": Profile("performance", 10, 45, 120, 0.5, 1.0, 0.2, 0.2, 1.0, 0.5),
}


def classify(sig: Signals, has_speech_text: bool) -> tuple[str, dict]:
    """Score each profile from whole-video statistics; return (name, scores)."""
    n = max(len(sig.times), 1)
    minutes = max(n / sig.fps / 60.0, 1e-6)
    hd = sig.hist_diff
    # Cuts = isolated spikes, not sustained change (flicker, strobe, fast pans
    # change every frame and would otherwise look like hundreds of cuts).
    nb = np.maximum(np.r_[0, hd[:-1]], np.r_[hd[1:], 0])
    spikes = (hd > 0.25) & (hd > 2.0 * nb)
    cut_rate = float(spikes.sum()) / minutes
    motion = float(np.median(sig.pix_diff))
    sustained_motion = float(np.percentile(sig.pix_diff, 25))  # camera never rests
    still_frac = float((sig.pix_diff < 0.004).mean())
    edge = float(np.median(sig.edge))
    sat = float(np.median(sig.sat))
    act_bursty = float((sig.activity > 0.02).mean())
    if sig.has_audio:
        a = sig.audio
        loud = np.percentile(a, 90) + 1e-6
        silence_frac = float((a < 0.08 * loud).mean())
    else:
        silence_frac = 1.0
    flat = float(sig.audio_stats.get("flatness", 1.0)) if sig.has_audio else 1.0
    pause = float(sig.audio_stats.get("micro_pause", 0.0)) if sig.has_audio else 0.0

    f = dict(cut_rate=round(cut_rate, 2), motion=motion, sustained_motion=sustained_motion,
             still_frac=still_frac, edge=edge, sat=sat, activity_frac=act_bursty,
             silence_frac=silence_frac, audio_flatness=flat, audio_micro_pause=pause,
             has_speech_text=has_speech_text)

    def s(x, lo, hi):  # soft ramp in [0,1]
        return float(np.clip((x - lo) / (hi - lo), 0, 1))

    music = (1 - s(flat, 0.05, 0.3)) * (1 - s(silence_frac, 0.05, 0.4))   # tonal & continuous
    speech = max(s(pause, 0.04, 0.15) * (1 - s(silence_frac, 0.5, 0.9)),
                 0.8 if has_speech_text else 0.0)
    scores = {
        # Text/slides: many identical frames, strong edges, low colour.
        "screen": s(still_frac, 0.4, 0.85) * (0.5 * s(edge, 0.04, 0.10) + 0.5 * (1 - s(sat, 0.15, 0.45))),
        # Static camera with people talking.
        "meeting": s(still_frac, 0.2, 0.7) * (1 - s(cut_rate, 0.5, 3)) * speech * 1.2,
        # Static camera, no edits, little meaningful sound.
        "surveillance": (1 - s(cut_rate, 0.2, 1.5)) * s(still_frac, 0.2, 0.7) *
                        (1 - 0.6 * speech) * (1 - 0.6 * music) * (1 - 0.5 * s(edge, 0.06, 0.12)),
        # Camera always moving (pans following play), audio not music.
        "high_motion": s(sustained_motion, 0.02, 0.05) * (1 - 0.7 * music) * (1 - 0.4 * speech),
        "performance": music * (1 - 0.5 * s(still_frac, 0.5, 0.9)) * (1 - 0.5 * speech),
        # Default for produced content; strengthened by edit rhythm + narration.
        "edited": 0.35 + 0.25 * s(cut_rate, 1, 10) + 0.2 * speech,
    }
    best = max(scores, key=scores.get)
    return best, {"features": f, "scores": {k: round(v, 3) for k, v in scores.items()}}
