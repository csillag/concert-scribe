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
