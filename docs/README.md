# Intelligent Video Chunking

**Goal:** split every video in a mixed corpus (meetings, screen shares, surveillance, sports, concerts, gameplay, produced video) into chunks that an indexer can describe and a retriever can return as the answer to a text question. Cost must stay linear in corpus hours.

## What makes a good chunk for retrieval

A question is answered by a *span* of video: one play, one slide, one person entering a room, one song. A chunk is good when it **contains the whole span and little else**. Two failure modes:

- **Cutting through an event** (mid-sentence, mid-play) splits the evidence. Neither half can answer the question.
- **Merging unrelated events** dilutes the embedding, so the right chunk ranks lower.

Fixed windows do both at once. Shot detection alone over-splits edited video and finds nothing in a static camera, where nearly all surveillance and meeting events live.

## Method (`src/vchunk/`)

**1. One cheap decode pass** (`decode.py`). ffmpeg decodes each video once to 160×90 RGB at 2 fps, skipping non-reference frames, plus 8 kHz mono audio. Every later step works on these small arrays, so memory per video stays bounded and cost is O(duration).

**2. Per-frame signals** (`signals.py`), each aimed at a boundary type some genre depends on:

| signal | catches | matters for |
|---|---|---|
| colour-histogram diff + adaptive spike threshold | hard cuts | produced video, replays |
| motion-compensated profile diff (row/column intensity profiles aligned by best shift) | cuts between similar shots while the camera pans | sports, gameplay |
| windowed histogram novelty (cumsum, O(n)) | fades, lighting/scene drift | concerts, documentaries |
| change between two still periods on edge-rich frames | slide flips, window switches | screen share, lectures |
| foreground vs. slow background model, with hysteresis | activity start/stop | surveillance, static meeting camera |
| audio silences; spectral flatness; micro-pause rate | gaps between songs or speakers; music vs. speech | concerts, meetings |
| subtitle/ASR cue gaps and sentence ends | never cut mid-sentence | anything with speech |

**3. Profile classification** (`profiles.py`). Whole-video statistics (cut rate, sustained motion, stillness, edge density, saturation, audio tonality, speech-like pauses) score six profiles: `screen`, `meeting`, `surveillance`, `high_motion`, `performance`, `edited`. The profile changes only **signal weights and length targets**, never the pipeline. A misclassified video degrades gracefully instead of breaking. Scores and features are saved with every output, so each decision can be explained.

**4. Boundary selection by dynamic programming** (`segment.py`). The weighted signals form one boundary-strength curve. Candidates are its local peaks, plus a sparse fallback grid that guarantees a valid solution. The DP chooses the boundary set maximising

  Σ boundary strength − Σ length cost,  subject to min_len ≤ chunk ≤ max_len.

The length cost is **scaled by information density** (motion above the noise floor, foreground activity, speech, recent change). An empty corridor or a frozen slide can become one long, cheap chunk, while busy spans are pushed toward the profile's target length. Candidates inside a spoken sentence are penalised, and survivors are snapped to the nearest pause within ±3 s. DP cost is O(n·w), where w is bounded by max_len divided by peak spacing, so it stays linear.

**5. Hierarchy and hints for the indexer.** Each video gets three levels:
- **scene** chunks: the retrieval unit.
- **section** chunks: 1–10 min, found by running the same DP over scene boundaries.
- **one whole-video chunk**: for synopsis questions like "what is this video about?", which no local chunk can answer.

Every scene also carries:
- 2 s context padding, so retrieval sees a little of what comes before and after.
- Keyframe timestamps chosen by farthest-point sampling on appearance, so each distinct visual state appears once.
- Hard-cut times inside the chunk.
- An `idle` flag and a `suggested_fps` derived from density, so the downstream extractor spends its expensive VLM/ASR budget where the information is.
- The transcript slice, when subtitles exist.

## Scalability and cost

- **Linear, and embarrassingly parallel.** Videos are independent: a process pool works inside one machine, and `--shard i/n` splits work across machines. No step compares videos to each other.
- **Throughput.** ~12× realtime per CPU core on 1080p30 H.264 and ~100× on 320p. That is ≈85 core-hours per 1,000 h of HD video, no GPU, no model calls.
- **Robust to messy corpora.** Results are cached per video and keyed by a hash of the parameters, so re-runs skip finished work and a crashed run resumes. Writes are atomic, and one corrupt file is logged and skipped rather than aborting the run.
- **Saves money downstream.** Idle chunks and `suggested_fps` cut the expensive per-chunk extraction. In the surveillance test, 67% of footage was flagged idle.
- **Optional upgrade.** A CLIP/audio-embedding check on only the ambiguous boundaries, between the strength thresholds, would add accuracy at a cost bounded by the number of candidates, not by duration. It is deliberately left out of the default path.

## Evaluation

Corpora rarely ship chunk-boundary labels, so `eval-synthetic` generates one video per genre with known structure. The structures are: shots with a crossfade; slides with a moving cursor; a fixed noisy camera with three crossing "people"; panning sports with cuts between similar shots; songs separated by quiet gaps. Boundary F1 uses ±2 s tolerance (±3 s for surveillance).

| video | profile (pred) | ours P / R / F1 | fixed 10 s | fixed 30 s | shot cuts only |
|---|---|---|---|---|---|
| edited | edited ✓ | 0.75 / 1.00 / 0.86 | 0.22 | 0.00 | 0.52 |
| screen share | screen ✓ | 1.00 / 1.00 / 1.00 | 0.00 | 0.00 | 0.00 |
| surveillance | surveillance ✓ | 1.00 / 1.00 / 1.00 | 0.34 | 0.15 | 0.00 |
| sports | high_motion ✓ | 1.00 / 1.00 / 1.00 | 0.61 | 0.25 | 0.91 |
| concert | performance ✓ | 0.60 / 1.00 / 0.75 | 0.00 | 0.00 | 0.00 |
| **mean** | 5/5 | **0.92** | 0.24 | 0.08 | 0.29 |

Surveillance: all 3 events fall inside a single active chunk, and 67% of the video is marked idle.

**Caveats, stated plainly.** The synthetic set was used to tune the classifier thresholds, so these numbers show that each mechanism works, not how well it generalises. Recall is 1.0 everywhere, and the remaining errors are over-splits. Over-splitting is the cheaper failure for retrieval, because no chunk spans two events. On real data, the end-to-end check is Task 5's benchmark: retrieval recall@k and answer accuracy on Video-MME questions, comparing these chunks against fixed windows.

## Run it

```
pip install -r requirements.txt
python src/main.py eval-synthetic --out out/eval
python src/main.py download --max-videos 6 --duration short
python src/main.py chunk data/video-mme-sample10/videos --subs data/video-mme-sample10/subtitles --out out/chunks
python src/main.py report out/chunks
```

The output, `out/chunks/chunks.jsonl`, has one row per chunk: `chunk_id, video_id, level, start, end, context_start, context_end, keyframes, shot_cuts, density, idle, suggested_fps, transcript, boundary_in{strength, reasons}, profile`.
