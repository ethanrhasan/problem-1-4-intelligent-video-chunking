"""SRT parsing and speech-boundary helpers.

When a transcript exists (subtitles here; ASR output in production) we never
want a chunk boundary in the middle of a sentence: the downstream indexer would
get half a sentence on each side and neither chunk would answer the question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
_TAG = re.compile(r"<[^>]+>|\{[^}]+\}")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def _sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0


def parse_srt(path: str) -> list[Cue]:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        blocks = re.split(r"\n\s*\n", f.read().replace("\r\n", "\n"))
    cues = []
    for b in blocks:
        m = _TS.search(b)
        if not m:
            continue
        g = m.groups()
        text = b[m.end():].strip()
        text = _TAG.sub("", text).replace("\n", " ").strip()
        if text:
            cues.append(Cue(_sec(*g[:4]), _sec(*g[4:]), text))
    cues.sort(key=lambda c: c.start)
    return cues


def speech_gaps(cues: list[Cue], min_gap: float = 0.3) -> list[tuple[float, float]]:
    """(time, strength) at the middle of each pause between cues.

    Longer pauses and pauses after sentence-final punctuation score higher.
    """
    out = []
    for a, b in zip(cues, cues[1:]):
        gap = b.start - a.end
        sentence_end = a.text.rstrip().endswith((".", "?", "!", "。", "？", "！"))
        if gap >= min_gap or sentence_end:
            strength = min(1.0, max(gap, 0) / 2.0) * 0.7 + (0.3 if sentence_end else 0.0)
            out.append(((a.end + max(b.start, a.end)) / 2.0, strength))
    return out


def inside_cue(cues: list[Cue], t: float) -> Cue | None:
    lo, hi = 0, len(cues)
    while lo < hi:
        mid = (lo + hi) // 2
        if cues[mid].end < t:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(cues) and cues[lo].start < t < cues[lo].end:
        return cues[lo]
    return None


def text_between(cues: list[Cue], s: float, e: float) -> str:
    """Cues whose midpoint falls in [s, e) - each cue lands in exactly one chunk."""
    return " ".join(c.text for c in cues if s <= (c.start + c.end) / 2 < e)
