# Audio Segment Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that classifies audio from video files into contiguous segments of silence, talking, music, and applause using YAMNet.

**Architecture:** Extract audio from video via ffmpeg, run YAMNet inference to get per-frame (0.48s) class scores, map 521 AudioSet classes to 4 categories, merge adjacent same-class frames into segments with silence minimum enforcement, write `.txt` output per clip.

**Tech Stack:** Python 3.9+, TensorFlow (CPU), tensorflow-hub, ffmpeg (system)

---

## File Structure

```
concert-scribe/
├── AGENTS.md
├── pyproject.toml
├── Makefile
├── src/
│   └── concert_scribe/
│       ├── __init__.py
│       ├── cli.py          # CLI entry point, arg parsing, pipeline orchestration
│       ├── extract.py      # ffmpeg audio extraction from video
│       ├── classify.py     # YAMNet loading, inference, AudioSet → 4-category mapping
│       ├── postprocess.py  # segment merging, silence absorption, music sub-types
│       └── output.py       # .txt file writer
├── tests/
│   ├── test_classify.py
│   ├── test_postprocess.py
│   └── test_output.py
└── docs/
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `src/concert_scribe/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "concert-scribe"
version = "0.1.0"
description = "Classify audio segments in concert recordings"
requires-python = ">=3.9"
dependencies = [
    "tensorflow>=2.13",
    "tensorflow-hub>=0.14",
    "numpy>=1.23",
]

[project.scripts]
concert-scribe = "concert_scribe.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create Makefile**

```makefile
.PHONY: install dev test clean

install:
	pip install .

dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
```

- [ ] **Step 3: Create src/concert_scribe/__init__.py**

```python
"""Concert Scribe — audio segment classifier for concert recordings."""
```

- [ ] **Step 4: Create empty test and source files**

Create empty files so imports don't break during incremental development:

```bash
mkdir -p src/concert_scribe tests
touch src/concert_scribe/cli.py
touch src/concert_scribe/extract.py
touch src/concert_scribe/classify.py
touch src/concert_scribe/postprocess.py
touch src/concert_scribe/output.py
touch tests/__init__.py
touch tests/test_classify.py
touch tests/test_postprocess.py
touch tests/test_output.py
```

- [ ] **Step 5: Install in dev mode and verify**

Run: `pip install -e .`
Expected: installs successfully, `concert-scribe` command registered (will fail at runtime since cli.py is empty — that's fine)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml Makefile src/ tests/
git commit -m "feat: project scaffolding"
```

---

### Task 2: Audio Extraction

**Files:**
- Create: `src/concert_scribe/extract.py`
- Create: `tests/test_extract.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extract.py
import os
import struct
import tempfile
import wave

import numpy as np
import pytest

from concert_scribe.extract import extract_audio

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".mts")


def _make_test_wav(path, duration=1.0, sample_rate=44100):
    """Create a minimal WAV file (used as a fake 'video' since ffmpeg can read it)."""
    n_samples = int(duration * sample_rate)
    samples = (np.sin(2 * np.pi * 440 * np.arange(n_samples) / sample_rate) * 32767).astype(
        np.int16
    )
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def test_extract_audio_produces_mono_16khz_wav():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test_input.wav")
        output_path = os.path.join(tmpdir, "extracted.wav")
        _make_test_wav(input_path, duration=2.0, sample_rate=44100)

        result = extract_audio(input_path, output_path)

        assert result == output_path
        assert os.path.exists(output_path)

        with wave.open(output_path, "r") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2


def test_extract_audio_nonexistent_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "out.wav")
        with pytest.raises(RuntimeError):
            extract_audio("/nonexistent/file.mp4", output_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extract.py -v`
Expected: FAIL with ImportError (extract_audio not defined)

- [ ] **Step 3: Write implementation**

```python
# src/concert_scribe/extract.py
"""Extract audio from video files using ffmpeg."""

import subprocess


def extract_audio(input_path: str, output_path: str) -> str:
    """Extract audio from a video file as mono 16kHz 16-bit WAV.

    Args:
        input_path: Path to input video file.
        output_path: Path to write the extracted WAV file.

    Returns:
        The output_path.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-y",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/concert_scribe/extract.py tests/test_extract.py
git commit -m "feat: audio extraction from video via ffmpeg"
```

---

### Task 3: AudioSet Category Mapping

**Files:**
- Create: `src/concert_scribe/classify.py`
- Create: `tests/test_classify.py`

This task builds the category mapping logic without loading the model. Task 4 adds inference.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classify.py
import numpy as np

from concert_scribe.classify import CATEGORY_CLASSES, map_scores_to_categories


def test_category_classes_cover_four_categories():
    assert "applause" in CATEGORY_CLASSES
    assert "talking" in CATEGORY_CLASSES
    assert "music" in CATEGORY_CLASSES
    assert len(CATEGORY_CLASSES) == 3  # silence is the default, not mapped


def test_map_scores_to_categories_music_wins():
    # 521 classes, all zero except one music class scores high
    scores = np.zeros((1, 521), dtype=np.float32)
    # Assume class index 0 is in a known category — we'll use the real mapping.
    # For this test, inject scores at known positions.
    class_names = [""] * 521
    class_names[137] = "Music"
    class_names[0] = "Speech"
    class_names[36] = "Applause"

    scores[0, 137] = 0.9

    categories, details = map_scores_to_categories(scores, class_names, threshold=0.1)
    assert categories[0] == "music"


def test_map_scores_to_categories_silence_when_below_threshold():
    scores = np.zeros((1, 521), dtype=np.float32)
    scores[0, :] = 0.01  # everything very low
    class_names = ["Unknown"] * 521

    categories, details = map_scores_to_categories(scores, class_names, threshold=0.1)
    assert categories[0] == "silence"


def test_map_scores_to_categories_returns_music_subtypes():
    scores = np.zeros((2, 521), dtype=np.float32)
    class_names = [""] * 521
    class_names[137] = "Music"
    class_names[294] = "Piano"
    class_names[302] = "Violin, fiddle"

    scores[0, 137] = 0.8
    scores[0, 294] = 0.5
    scores[1, 137] = 0.7
    scores[1, 302] = 0.6

    categories, details = map_scores_to_categories(scores, class_names, threshold=0.1)
    assert categories[0] == "music"
    assert categories[1] == "music"
    # details contains per-frame music sub-types above threshold
    assert "Piano" in details[0]
    assert "Violin, fiddle" in details[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_classify.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# src/concert_scribe/classify.py
"""YAMNet-based audio classification."""

from typing import Optional

import numpy as np

# Mapping of category -> set of AudioSet display_name values.
# "silence" is the default when nothing else scores above threshold.
CATEGORY_CLASSES: dict[str, set[str]] = {
    "applause": {
        "Applause",
        "Clapping",
    },
    "talking": {
        "Speech",
        "Narration, monologue",
        "Conversation",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
        "Whispering",
        "Speech synthesizer",
    },
    "music": {
        "Music",
        "Musical instrument",
        "Plucked string instrument",
        "Guitar",
        "Electric guitar",
        "Bass guitar",
        "Acoustic guitar",
        "Steel guitar, slide guitar",
        "Tapping (guitar technique)",
        "Strum",
        "Banjo",
        "Sitar",
        "Mandolin",
        "Zither",
        "Ukulele",
        "Keyboard (musical)",
        "Piano",
        "Electric piano",
        "Organ",
        "Electronic organ",
        "Hammond organ",
        "Synthesizer",
        "Sampler",
        "Harpsichord",
        "Percussion",
        "Drum kit",
        "Drum machine",
        "Drum",
        "Snare drum",
        "Rimshot",
        "Drum roll",
        "Bass drum",
        "Timpani",
        "Tabla",
        "Cymbal",
        "Hi-hat",
        "Wood block",
        "Tambourine",
        "Rattle (instrument)",
        "Maraca",
        "Gong",
        "Tubular bells",
        "Mallet percussion",
        "Marimba, xylophone",
        "Glockenspiel",
        "Vibraphone",
        "Steelpan",
        "Orchestra",
        "Brass instrument",
        "French horn",
        "Trumpet",
        "Trombone",
        "Bowed string instrument",
        "String section",
        "Violin, fiddle",
        "Pizzicato",
        "Cello",
        "Double bass",
        "Wind instrument, woodwind instrument",
        "Flute",
        "Saxophone",
        "Clarinet",
        "Harp",
        "Bell",
        "Church bell",
        "Jingle bell",
        "Tuning fork",
        "Chime",
        "Wind chime",
        "Change ringing (campanology)",
        "Harmonica",
        "Accordion",
        "Bagpipes",
        "Didgeridoo",
        "Shofar",
        "Theremin",
        "Singing",
        "Choir",
        "Yodeling",
        "Humming",
        "Singing bowl",
        "Chant",
        "Mantra",
    },
}

# Build reverse lookup: display_name -> category
_NAME_TO_CATEGORY: dict[str, str] = {}
for _cat, _names in CATEGORY_CLASSES.items():
    for _name in _names:
        _NAME_TO_CATEGORY[_name] = _cat

# Music sub-type names: everything in the music set except the generic "Music" label
_MUSIC_SUBTYPES: set[str] = CATEGORY_CLASSES["music"] - {"Music", "Musical instrument"}


def _build_category_indices(
    class_names: list[str],
) -> dict[str, list[int]]:
    """Map each category to the list of indices in the class_names list."""
    cat_indices: dict[str, list[int]] = {cat: [] for cat in CATEGORY_CLASSES}
    for i, name in enumerate(class_names):
        cat = _NAME_TO_CATEGORY.get(name)
        if cat is not None:
            cat_indices[cat].append(i)
    return cat_indices


def map_scores_to_categories(
    scores: np.ndarray,
    class_names: list[str],
    threshold: float = 0.1,
) -> tuple[list[str], list[list[str]]]:
    """Map per-frame YAMNet scores to the four segment categories.

    Args:
        scores: Array of shape (num_frames, 521) with per-class scores.
        class_names: List of 521 AudioSet display names.
        threshold: Minimum aggregate score for a category to be considered.

    Returns:
        Tuple of:
        - List of category labels per frame ("silence", "talking", "music", "applause").
        - List of music sub-type lists per frame (empty list for non-music frames).
    """
    cat_indices = _build_category_indices(class_names)

    num_frames = scores.shape[0]
    categories: list[str] = []
    music_details: list[list[str]] = []

    for i in range(num_frames):
        frame_scores = scores[i]

        # Sum scores per category
        cat_scores: dict[str, float] = {}
        for cat, indices in cat_indices.items():
            if indices:
                cat_scores[cat] = float(np.sum(frame_scores[indices]))
            else:
                cat_scores[cat] = 0.0

        # Find best category above threshold
        best_cat = "silence"
        best_score = threshold
        for cat, score in cat_scores.items():
            if score > best_score:
                best_score = score
                best_cat = cat

        categories.append(best_cat)

        # Collect music sub-types for this frame
        subtypes: list[str] = []
        if best_cat == "music":
            for idx in cat_indices["music"]:
                name = class_names[idx]
                if name in _MUSIC_SUBTYPES and frame_scores[idx] >= threshold:
                    subtypes.append(name)
        music_details.append(subtypes)

    return categories, music_details
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_classify.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/concert_scribe/classify.py tests/test_classify.py
git commit -m "feat: AudioSet category mapping logic"
```

---

### Task 4: YAMNet Model Loading and Inference

**Files:**
- Modify: `src/concert_scribe/classify.py`

This task adds the model loading and inference wrapper. No unit test for this task — it requires the actual model download. Integration testing happens in Task 7.

- [ ] **Step 1: Add model loading and inference to classify.py**

Append to `src/concert_scribe/classify.py`:

```python
import csv
import wave

import tensorflow_hub as hub
import tensorflow as tf


_model = None
_class_names: Optional[list[str]] = None

HOP_SECONDS = 0.48  # YAMNet's native hop size


def load_model():
    """Load YAMNet model and class names. Caches after first call."""
    global _model, _class_names
    if _model is not None:
        return _model, _class_names

    _model = hub.load("https://tfhub.dev/google/yamnet/1")

    class_map_path = _model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path) as f:
        _class_names = [row["display_name"] for row in csv.DictReader(f)]

    return _model, _class_names


def classify_audio(wav_path: str, threshold: float = 0.1) -> tuple[list[str], list[list[str]]]:
    """Classify a 16kHz mono WAV file into segments.

    Args:
        wav_path: Path to a 16kHz mono WAV file.
        threshold: Minimum score for a category to be considered.

    Returns:
        Tuple of (categories_per_frame, music_details_per_frame).
    """
    model, class_names = load_model()

    with wave.open(wav_path, "r") as wf:
        assert wf.getnchannels() == 1, "Expected mono audio"
        assert wf.getframerate() == 16000, "Expected 16kHz sample rate"
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    waveform = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    scores, embeddings, spectrogram = model(waveform)
    scores = scores.numpy()

    return map_scores_to_categories(scores, class_names, threshold=threshold)
```

- [ ] **Step 2: Commit**

```bash
git add src/concert_scribe/classify.py
git commit -m "feat: YAMNet model loading and inference"
```

---

### Task 5: Post-Processing

**Files:**
- Create: `src/concert_scribe/postprocess.py`
- Create: `tests/test_postprocess.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_postprocess.py
from concert_scribe.postprocess import merge_segments

HOP = 0.48


def test_merge_adjacent_same_category():
    categories = ["music", "music", "music", "silence", "silence"]
    details = [["Piano"], ["Piano", "Violin, fiddle"], ["Violin, fiddle"], [], []]

    segments = merge_segments(categories, details, hop_sec=HOP, min_silence_sec=2.0)

    assert len(segments) == 2
    assert segments[0]["category"] == "music"
    assert abs(segments[0]["start"] - 0.0) < 0.01
    assert abs(segments[0]["end"] - 1.44) < 0.01
    assert set(segments[0]["subtypes"]) == {"Piano", "Violin, fiddle"}
    assert segments[1]["category"] == "silence"


def test_short_silence_absorbed_same_neighbors():
    # music, silence (1 frame = 0.48s < 2s), music -> all music
    categories = ["music", "silence", "music"]
    details = [["Piano"], [], ["Piano"]]

    segments = merge_segments(categories, details, hop_sec=HOP, min_silence_sec=2.0)

    assert len(segments) == 1
    assert segments[0]["category"] == "music"


def test_short_silence_absorbed_different_neighbors():
    # talking, silence (1 frame), music -> silence split at midpoint
    categories = ["talking", "talking", "silence", "music", "music"]
    details = [[], [], [], ["Piano"], ["Piano"]]

    segments = merge_segments(categories, details, hop_sec=HOP, min_silence_sec=2.0)

    assert len(segments) == 2
    assert segments[0]["category"] == "talking"
    assert segments[1]["category"] == "music"


def test_long_silence_preserved():
    # silence for 5 frames = 2.4s > 2s threshold
    categories = ["music", "music"] + ["silence"] * 5 + ["talking"]
    details = [["Piano"], ["Piano"]] + [[]] * 5 + [[]]

    segments = merge_segments(categories, details, hop_sec=HOP, min_silence_sec=2.0)

    assert len(segments) == 3
    assert segments[0]["category"] == "music"
    assert segments[1]["category"] == "silence"
    assert segments[2]["category"] == "talking"


def test_silence_at_start_preserved_if_long():
    categories = ["silence"] * 5 + ["music"]
    details = [[]] * 5 + [["Piano"]]

    segments = merge_segments(categories, details, hop_sec=HOP, min_silence_sec=2.0)

    assert len(segments) == 2
    assert segments[0]["category"] == "silence"
    assert segments[1]["category"] == "music"


def test_silence_at_start_absorbed_if_short():
    categories = ["silence", "music", "music"]
    details = [[], ["Piano"], ["Piano"]]

    segments = merge_segments(categories, details, hop_sec=HOP, min_silence_sec=2.0)

    assert len(segments) == 1
    assert segments[0]["category"] == "music"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_postprocess.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# src/concert_scribe/postprocess.py
"""Post-processing: merge frames into segments, enforce silence minimum."""

from typing import TypedDict


class Segment(TypedDict):
    category: str
    start: float
    end: float
    subtypes: list[str]


def merge_segments(
    categories: list[str],
    music_details: list[list[str]],
    hop_sec: float,
    min_silence_sec: float = 2.0,
) -> list[Segment]:
    """Merge per-frame categories into contiguous segments.

    Args:
        categories: Per-frame category labels.
        music_details: Per-frame music sub-type lists.
        hop_sec: Duration of each frame hop in seconds.
        min_silence_sec: Minimum duration for silence segments.
            Shorter silence is absorbed into neighbors.

    Returns:
        List of Segment dicts with category, start, end, subtypes.
    """
    if not categories:
        return []

    # Step 1: Merge consecutive same-category frames into raw segments
    raw_segments: list[Segment] = []
    current_cat = categories[0]
    current_start = 0
    current_subtypes: set[str] = set(music_details[0])

    for i in range(1, len(categories)):
        if categories[i] != current_cat:
            raw_segments.append(
                Segment(
                    category=current_cat,
                    start=current_start * hop_sec,
                    end=i * hop_sec,
                    subtypes=sorted(current_subtypes),
                )
            )
            current_cat = categories[i]
            current_start = i
            current_subtypes = set(music_details[i])
        else:
            current_subtypes.update(music_details[i])

    raw_segments.append(
        Segment(
            category=current_cat,
            start=current_start * hop_sec,
            end=len(categories) * hop_sec,
            subtypes=sorted(current_subtypes),
        )
    )

    # Step 2: Absorb short silence segments
    if len(raw_segments) <= 1:
        return raw_segments

    result: list[Segment] = []
    for seg in raw_segments:
        if seg["category"] == "silence" and (seg["end"] - seg["start"]) < min_silence_sec:
            # Absorb into previous or next non-silence neighbor
            if result and result[-1]["category"] != "silence":
                # Extend previous segment to cover this silence
                result[-1]["end"] = seg["end"]
            # If no previous non-silence, it will be absorbed by the next segment
            # (handled below when the next segment is added)
        else:
            # If the previous segment was a short silence that couldn't be absorbed
            # backward (it was at the start), absorb it forward into this segment
            if result and result[-1]["category"] == "silence" and (result[-1]["end"] - result[-1]["start"]) < min_silence_sec:
                seg = Segment(
                    category=seg["category"],
                    start=result[-1]["start"],
                    end=seg["end"],
                    subtypes=seg["subtypes"],
                )
                result.pop()
            result.append(seg)

    # Handle trailing short silence
    if len(result) > 1 and result[-1]["category"] == "silence" and (result[-1]["end"] - result[-1]["start"]) < min_silence_sec:
        result[-2]["end"] = result[-1]["end"]
        result.pop()

    # Step 3: Re-merge adjacent segments that now share a category
    # (can happen when silence between same-category segments was absorbed)
    merged: list[Segment] = [result[0]]
    for seg in result[1:]:
        if seg["category"] == merged[-1]["category"]:
            subtypes = sorted(set(merged[-1]["subtypes"]) | set(seg["subtypes"]))
            merged[-1]["end"] = seg["end"]
            merged[-1]["subtypes"] = subtypes
        else:
            merged.append(seg)

    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_postprocess.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/concert_scribe/postprocess.py tests/test_postprocess.py
git commit -m "feat: segment merging and silence absorption"
```

---

### Task 6: Output Writer

**Files:**
- Create: `src/concert_scribe/output.py`
- Create: `tests/test_output.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_output.py
import os
import tempfile

from concert_scribe.output import write_segments
from concert_scribe.postprocess import Segment


def test_write_segments_format():
    segments = [
        Segment(category="silence", start=0.0, end=4.32, subtypes=[]),
        Segment(category="talking", start=4.32, end=15.36, subtypes=[]),
        Segment(category="music", start=15.36, end=40.32, subtypes=["Piano", "Singing"]),
        Segment(category="applause", start=40.32, end=45.12, subtypes=[]),
        Segment(category="silence", start=45.12, end=50.4, subtypes=[]),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test.txt")
        write_segments(segments, out_path)

        with open(out_path) as f:
            lines = f.read().strip().split("\n")

    assert len(lines) == 5
    assert lines[0] == "0.0-4.32: silence"
    assert lines[1] == "4.32-15.36: talking"
    assert lines[2] == "15.36-40.32: music (Piano, Singing)"
    assert lines[3] == "40.32-45.12: applause"
    assert lines[4] == "45.12-50.4: silence"


def test_write_segments_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "empty.txt")
        write_segments([], out_path)

        with open(out_path) as f:
            content = f.read()

    assert content == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_output.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# src/concert_scribe/output.py
"""Write segment classification results to text files."""

from concert_scribe.postprocess import Segment


def _format_timestamp(t: float) -> str:
    """Format a timestamp, stripping unnecessary trailing zeros."""
    s = f"{t:.2f}"
    # Strip trailing zeros after decimal, but keep at least one decimal
    s = s.rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


def write_segments(segments: list[Segment], output_path: str) -> None:
    """Write segments to a text file.

    Args:
        segments: List of classified segments.
        output_path: Path to write the output file.
    """
    with open(output_path, "w") as f:
        for seg in segments:
            start = _format_timestamp(seg["start"])
            end = _format_timestamp(seg["end"])
            line = f"{start}-{end}: {seg['category']}"
            if seg["subtypes"]:
                line += f" ({', '.join(seg['subtypes'])})"
            f.write(line + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_output.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/concert_scribe/output.py tests/test_output.py
git commit -m "feat: segment output writer"
```

---

### Task 7: CLI and Pipeline Orchestration

**Files:**
- Create: `src/concert_scribe/cli.py`

- [ ] **Step 1: Write implementation**

```python
# src/concert_scribe/cli.py
"""CLI entry point for concert-scribe."""

import argparse
import os
import sys
import tempfile

from concert_scribe.classify import HOP_SECONDS, classify_audio
from concert_scribe.extract import extract_audio
from concert_scribe.output import write_segments
from concert_scribe.postprocess import merge_segments

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mts"}

CONFIDENCE_THRESHOLD = 0.1
MIN_SILENCE_SEC = 2.0


def find_video_files(input_path: str) -> list[str]:
    """Find video files from an input path (file or directory)."""
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        files = []
        for name in sorted(os.listdir(input_path)):
            if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
                files.append(os.path.join(input_path, name))
        return files

    print(f"Error: {input_path} is not a file or directory", file=sys.stderr)
    sys.exit(1)


def process_file(video_path: str, output_dir: str) -> None:
    """Process a single video file through the full pipeline."""
    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{basename}.txt")

    print(f"Processing: {video_path}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract audio
        wav_path = os.path.join(tmpdir, "audio.wav")
        print("  Extracting audio...", file=sys.stderr)
        extract_audio(video_path, wav_path)

        # Classify
        print("  Classifying...", file=sys.stderr)
        categories, music_details = classify_audio(
            wav_path, threshold=CONFIDENCE_THRESHOLD
        )

        # Post-process
        segments = merge_segments(
            categories, music_details,
            hop_sec=HOP_SECONDS,
            min_silence_sec=MIN_SILENCE_SEC,
        )

    # Write output
    write_segments(segments, output_path)
    print(f"  Written: {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Classify audio segments in concert recordings."
    )
    parser.add_argument(
        "input",
        help="Video file or directory containing video files",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for .txt files (default: same as input)",
    )
    args = parser.parse_args()

    video_files = find_video_files(args.input)
    if not video_files:
        print("No video files found.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        if os.path.isdir(args.input):
            output_dir = args.input
        else:
            output_dir = os.path.dirname(args.input) or "."

    os.makedirs(output_dir, exist_ok=True)

    for video_path in video_files:
        process_file(video_path, output_dir)

    print(f"Done. Processed {len(video_files)} file(s).", file=sys.stderr)
```

- [ ] **Step 2: Verify CLI entry point works**

Run: `concert-scribe --help`
Expected: Shows usage with `input` and `--output-dir` arguments.

- [ ] **Step 3: Commit**

```bash
git add src/concert_scribe/cli.py
git commit -m "feat: CLI entry point and pipeline orchestration"
```

---

### Task 8: Integration Test

**Files:**
- Create: `tests/test_integration.py`

This test requires TensorFlow and YAMNet model download. It creates a synthetic audio file and runs the full pipeline.

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration test — requires TensorFlow and YAMNet model download."""

import os
import tempfile
import wave

import numpy as np
import pytest

from concert_scribe.classify import classify_audio
from concert_scribe.output import write_segments
from concert_scribe.postprocess import merge_segments
from concert_scribe.classify import HOP_SECONDS


def _make_test_wav(path: str, duration: float = 5.0):
    """Create a 16kHz mono WAV with white noise (likely classified as noise/silence)."""
    sr = 16000
    n_samples = int(duration * sr)
    # Low-amplitude white noise
    samples = (np.random.randn(n_samples) * 100).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def test_full_pipeline_produces_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "test.wav")
        out_path = os.path.join(tmpdir, "test.txt")

        _make_test_wav(wav_path, duration=5.0)

        categories, details = classify_audio(wav_path)
        assert len(categories) > 0

        segments = merge_segments(categories, details, hop_sec=HOP_SECONDS)
        assert len(segments) > 0

        write_segments(segments, out_path)
        assert os.path.exists(out_path)

        with open(out_path) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) > 0
        # Each line should match the expected format
        for line in lines:
            assert ": " in line
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/test_integration.py -v`
Expected: PASS (first run will download the YAMNet model, may take a minute)

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration test for full classification pipeline"
```

---

### Task 9: Run All Tests and Final Verification

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 2: Test CLI end-to-end with a real video file (manual)**

If a SoundGraft output video is available, run:
```bash
concert-scribe /path/to/video.mp4 -o /tmp/scribe-test/
cat /tmp/scribe-test/video.txt
```

Verify the output contains reasonable segments.

- [ ] **Step 3: Commit any fixes**

If any adjustments were needed, commit them.
