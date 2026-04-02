"""YAMNet-based audio classification."""

import csv
import wave
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

# Clean display names for AudioSet classes with clunky names
_DISPLAY_NAMES: dict[str, str] = {
    "Violin, fiddle": "Violin",
    "Marimba, xylophone": "Marimba",
    "Wind instrument, woodwind instrument": "Woodwind",
    "Steel guitar, slide guitar": "Steel guitar",
    "Male speech, man speaking": "Male speech",
    "Female speech, woman speaking": "Female speech",
    "Child speech, kid speaking": "Child speech",
    "Narration, monologue": "Narration",
    "Rattle (instrument)": "Rattle",
    "Tapping (guitar technique)": "Tapping",
    "Keyboard (musical)": "Keyboard",
    "Change ringing (campanology)": "Change ringing",
    "Plucked string instrument": "Plucked strings",
    "Bowed string instrument": "Bowed strings",
}


def display_name(name: str) -> str:
    """Return a clean display name for an AudioSet class."""
    return _DISPLAY_NAMES.get(name, name)

# AudioSet ontology: child -> parent (within our music class set)
_MUSIC_PARENT_MAP: dict[str, str] = {
    "Accordion": "Musical instrument",
    "Acoustic guitar": "Guitar",
    "Bagpipes": "Musical instrument",
    "Banjo": "Plucked string instrument",
    "Bass drum": "Drum",
    "Bass guitar": "Guitar",
    "Bell": "Musical instrument",
    "Bowed string instrument": "Musical instrument",
    "Brass instrument": "Musical instrument",
    "Cello": "Bowed string instrument",
    "Change ringing (campanology)": "Bell",
    "Chant": "Singing",
    "Chime": "Bell",
    "Choir": "Singing",
    "Church bell": "Bell",
    "Clarinet": "Wind instrument, woodwind instrument",
    "Cymbal": "Percussion",
    "Didgeridoo": "Musical instrument",
    "Double bass": "Bowed string instrument",
    "Drum": "Percussion",
    "Drum kit": "Percussion",
    "Drum machine": "Drum kit",
    "Drum roll": "Snare drum",
    "Electric guitar": "Guitar",
    "Electric piano": "Piano",
    "Electronic organ": "Organ",
    "Flute": "Wind instrument, woodwind instrument",
    "French horn": "Brass instrument",
    "Glockenspiel": "Mallet percussion",
    "Gong": "Percussion",
    "Guitar": "Plucked string instrument",
    "Hammond organ": "Organ",
    "Harmonica": "Musical instrument",
    "Harp": "Musical instrument",
    "Harpsichord": "Keyboard (musical)",
    "Hi-hat": "Cymbal",
    "Jingle bell": "Bell",
    "Keyboard (musical)": "Musical instrument",
    "Mallet percussion": "Percussion",
    "Mandolin": "Plucked string instrument",
    "Mantra": "Chant",
    "Maraca": "Rattle (instrument)",
    "Marimba, xylophone": "Mallet percussion",
    "Musical instrument": "Music",
    "Orchestra": "Musical instrument",
    "Organ": "Keyboard (musical)",
    "Percussion": "Musical instrument",
    "Piano": "Keyboard (musical)",
    "Pizzicato": "Violin, fiddle",
    "Plucked string instrument": "Musical instrument",
    "Rattle (instrument)": "Percussion",
    "Rimshot": "Snare drum",
    "Sampler": "Synthesizer",
    "Saxophone": "Wind instrument, woodwind instrument",
    "Shofar": "Musical instrument",
    "Singing bowl": "Musical instrument",
    "Sitar": "Plucked string instrument",
    "Snare drum": "Drum",
    "Steel guitar, slide guitar": "Guitar",
    "Steelpan": "Mallet percussion",
    "String section": "Bowed string instrument",
    "Strum": "Guitar",
    "Synthesizer": "Keyboard (musical)",
    "Tabla": "Drum",
    "Tambourine": "Percussion",
    "Tapping (guitar technique)": "Guitar",
    "Theremin": "Musical instrument",
    "Timpani": "Drum",
    "Trombone": "Brass instrument",
    "Trumpet": "Brass instrument",
    "Tubular bells": "Percussion",
    "Tuning fork": "Bell",
    "Ukulele": "Plucked string instrument",
    "Vibraphone": "Mallet percussion",
    "Violin, fiddle": "Bowed string instrument",
    "Wind chime": "Chime",
    "Wind instrument, woodwind instrument": "Musical instrument",
    "Wood block": "Percussion",
    "Yodeling": "Singing",
    "Zither": "Plucked string instrument",
}


def _get_ancestors(name: str) -> set[str]:
    """Get all ancestors of a class in the hierarchy."""
    ancestors: set[str] = set()
    current = name
    while current in _MUSIC_PARENT_MAP:
        parent = _MUSIC_PARENT_MAP[current]
        ancestors.add(parent)
        current = parent
    return ancestors


def _deduplicate_subtypes(subtypes: list[str]) -> list[str]:
    """Remove parent classes when a more specific child is present."""
    subtype_set = set(subtypes)
    # Collect all ancestors of all detected subtypes
    redundant: set[str] = set()
    for name in subtypes:
        redundant |= _get_ancestors(name)
    # Keep only subtypes that aren't redundant ancestors
    return [s for s in subtypes if s not in redundant]


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

        # Pick the single best music sub-type for this frame (deduplicated by hierarchy)
        subtypes: list[str] = []
        if best_cat == "music":
            raw_subtypes: list[tuple[str, float]] = []
            for idx in cat_indices["music"]:
                name = class_names[idx]
                if name in _MUSIC_SUBTYPES and frame_scores[idx] >= threshold:
                    raw_subtypes.append((name, float(frame_scores[idx])))
            # Deduplicate by hierarchy
            deduped_names = set(_deduplicate_subtypes([n for n, _ in raw_subtypes]))
            # Filter to only deduped, then pick the highest-scoring one
            deduped = [(n, s) for n, s in raw_subtypes if n in deduped_names]
            if deduped:
                best_subtype = max(deduped, key=lambda x: x[1])
                subtypes = [best_subtype[0]]
        music_details.append(subtypes)

    return categories, music_details


_model = None
_class_names: Optional[list[str]] = None

HOP_SECONDS = 0.48  # YAMNet's native hop size


def _ensure_pkg_resources():
    """Ensure pkg_resources is importable (needed by tensorflow_hub on Python 3.12+)."""
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        import sys
        import types

        # tensorflow_hub only uses pkg_resources.parse_version, so provide a minimal shim
        mod = types.ModuleType("pkg_resources")
        mod.parse_version = lambda v: tuple(int(x) for x in v.split(".") if x.isdigit())
        sys.modules["pkg_resources"] = mod


def load_model():
    """Load YAMNet model and class names. Caches after first call."""
    import os
    import sys

    global _model, _class_names
    if _model is not None:
        return _model, _class_names

    _ensure_pkg_resources()

    # Suppress C++ level logs from TF/CUDA/absl during import and model load
    # These fire at shared library load time before Python logging can catch them
    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        import tensorflow_hub as hub
        _model = hub.load("https://tfhub.dev/google/yamnet/1")
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)

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
