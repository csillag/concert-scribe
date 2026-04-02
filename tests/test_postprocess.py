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
