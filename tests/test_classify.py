import numpy as np

from concert_scribe.classify import (
    CATEGORY_CLASSES,
    _deduplicate_subtypes,
    map_scores_to_categories,
)


def test_category_classes_cover_four_categories():
    assert "applause" in CATEGORY_CLASSES
    assert "talking" in CATEGORY_CLASSES
    assert "music" in CATEGORY_CLASSES
    assert len(CATEGORY_CLASSES) == 3  # silence is the default, not mapped


def test_map_scores_to_categories_music_wins():
    scores = np.zeros((1, 521), dtype=np.float32)
    class_names = [""] * 521
    class_names[137] = "Music"
    class_names[0] = "Speech"
    class_names[36] = "Applause"

    scores[0, 137] = 0.9

    categories, details = map_scores_to_categories(scores, class_names, threshold=0.1)
    assert categories[0] == "music"


def test_map_scores_to_categories_silence_when_below_threshold():
    scores = np.zeros((1, 521), dtype=np.float32)
    scores[0, :] = 0.01
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
    assert "Piano" in details[0]
    assert "Violin, fiddle" in details[1]


def test_deduplicate_subtypes_removes_ancestors():
    # Violin triggers Bowed string instrument, Musical instrument etc.
    subtypes = ["Violin, fiddle", "Bowed string instrument", "String section", "Orchestra"]
    result = _deduplicate_subtypes(subtypes)
    assert "Violin, fiddle" in result
    assert "Orchestra" in result
    # Bowed string instrument is ancestor of Violin
    assert "Bowed string instrument" not in result
    # String section is ancestor of... wait, String section is child of Bowed string instrument
    # String section's ancestor chain: String section -> Bowed string instrument -> Musical instrument
    # Violin's ancestor chain: Violin -> Bowed string instrument -> Musical instrument
    # So String section is NOT an ancestor of Violin. It should stay.
    assert "String section" in result


def test_deduplicate_subtypes_keeps_leaf_only():
    subtypes = ["Piano", "Keyboard (musical)"]
    result = _deduplicate_subtypes(subtypes)
    assert result == ["Piano"]
