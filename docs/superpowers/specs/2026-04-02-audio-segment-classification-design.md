# Audio Segment Classification — Design Spec

## Overview

**concert-scribe** is a standalone CLI tool that classifies audio from video files into contiguous segments of silence, talking, music, and applause. It takes video files (typically output of SoundGraft) as input and produces a `.txt` file per clip describing the segment timeline.

## Architecture

```
video file(s) → ffmpeg extract audio → YAMNet classify (0.1s hop) → post-process → .txt output
```

### Dependencies

- Python 3.9+
- TensorFlow (CPU-only)
- tensorflow-hub (for loading YAMNet)
- ffmpeg (system tool, for audio extraction from video)

### Components

1. **CLI** (`cli.py`) — accepts input files or directory, output directory, dispatches pipeline
2. **Audio extractor** (`extract.py`) — ffmpeg to WAV (mono, 16kHz — YAMNet's expected sample rate)
3. **Classifier** (`classify.py`) — loads YAMNet via tf-hub, runs inference, maps AudioSet class scores to 4 categories
4. **Post-processor** (`postprocess.py`) — merges adjacent same-class frames, absorbs short silence, collects music sub-types
5. **Output writer** (`output.py`) — writes `.txt` segment file

## Classification Logic

### Model

YAMNet (Google), loaded via TensorFlow Hub. MobileNet-based, ~13MB, trained on AudioSet (521 classes). CPU-only inference — expected throughput roughly 1-3x realtime.

### AudioSet → Category Mapping

YAMNet produces scores for 521 AudioSet classes per frame. These are grouped into four categories:

- **Applause** — "Applause", "Clapping"
- **Talking** — "Speech", "Narration", "Conversation", "Male speech", "Female speech", and similar speech-related classes
- **Music** — "Music", "Musical instrument", "Singing", "Orchestra", "Piano", and similar music-related classes (sub-types preserved for output)
- **Silence** — default category when no other category scores above a confidence threshold

### Per-Frame Decision

1. Sum scores across all AudioSet classes belonging to each category
2. If no category exceeds a minimum confidence threshold → classify as silence
3. Otherwise → highest-scoring category wins

### Inference Resolution

- 0.48s hop between frames (YAMNet's native hop size)
- Timestamps reported at raw decimal precision (no rounding)

## Post-Processing

### Segment Merging

Adjacent frames with the same classification label are merged into contiguous segments.

### Silence Minimum Duration

Silence segments shorter than 2 seconds are absorbed into the surrounding non-silence category:
- If both sides are the same category → merge into that category
- If sides are different categories → split at midpoint

This rule applies only to silence. Short segments of talking, music, or applause are never absorbed.

### Music Sub-Type Collection

For each music segment, all AudioSet music sub-classes (e.g., "Piano", "Violin", "Singing") that contributed to the music category score in any frame within that segment are collected, deduplicated, and listed as annotations in the output. A sub-class is included if it scored above the same confidence threshold used for the per-frame category decision.

## Output Format

For each input video file, a `.txt` file with the same basename is written to the output directory. Each line describes one segment:

```
0.0-4.3: silence
4.3-15.7: talking
15.7-40.1: music (piano, orchestra, singing)
40.1-45.2: applause
45.2-47.0: silence
47.0-200.3: music (violin, orchestra)
200.3-204.5: applause
204.5-210.0: silence
```

- Music segments include parenthetical sub-type annotations
- Talking, applause, and silence segments have no sub-annotations

## CLI Interface

```
concert-scribe <input> [--output-dir DIR]
```

- `<input>` — a single video file or a directory (processes all video files found: `.mp4`, `.mov`, `.avi`, `.mkv`, `.mts`)
- `--output-dir` / `-o` — where to write `.txt` files (defaults to same directory as input)
- Progress output to stderr

Confidence threshold and silence minimum duration are hardcoded constants.

## Project Structure

```
concert-scribe/
├── AGENTS.md
├── pyproject.toml
├── Makefile
├── src/
│   └── concert_scribe/
│       ├── __init__.py
│       ├── cli.py
│       ├── extract.py
│       ├── classify.py
│       ├── postprocess.py
│       └── output.py
└── docs/
    └── superpowers/
        └── specs/
```
