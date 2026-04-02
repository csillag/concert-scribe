from concert_scribe.postprocess import merge_segments

HOP = 0.48
# Use 4 frames = 1.92s to exceed the 1.5s music/talking minimum
# Use 5 frames = 2.4s to exceed the 2.0s silence minimum


def test_merge_adjacent_same_category():
    # Each frame has one subtype (as classifier now outputs)
    categories = ["music"] * 6 + ["silence"] * 5
    details = [["Piano"], ["Piano"], ["Piano"], ["Piano"], ["Violin, fiddle"], ["Piano"]] + [[]] * 5

    segments = merge_segments(categories, details, hop_sec=HOP)

    assert len(segments) == 2
    assert segments[0]["category"] == "music"
    assert abs(segments[0]["start"] - 0.0) < 0.01
    assert abs(segments[0]["end"] - 2.88) < 0.01
    assert set(segments[0]["subtypes"].keys()) == {"Piano"}
    # Violin only has 0.48s (1 frame) < 1.5s minimum, so it's filtered
    assert abs(segments[0]["subtypes"]["Piano"] - 2.4) < 0.01  # 5 frames
    assert segments[1]["category"] == "silence"


def test_short_silence_absorbed_same_neighbors():
    # music (4 frames), silence (1 frame = 0.48s < 2s), music (4 frames) -> all music
    categories = ["music"] * 4 + ["silence"] + ["music"] * 4
    details = [["Piano"]] * 4 + [[]] + [["Piano"]] * 4

    segments = merge_segments(categories, details, hop_sec=HOP)

    assert len(segments) == 1
    assert segments[0]["category"] == "music"


def test_short_silence_absorbed_different_neighbors():
    # talking (4 frames), silence (1 frame), music (4 frames)
    categories = ["talking"] * 4 + ["silence"] + ["music"] * 4
    details = [[]] * 4 + [[]] + [["Piano"]] * 4

    segments = merge_segments(categories, details, hop_sec=HOP)

    assert len(segments) == 2
    assert segments[0]["category"] == "talking"
    assert segments[1]["category"] == "music"


def test_long_silence_preserved():
    # music (4 frames), silence (5 frames = 2.4s > 2s), talking (4 frames)
    categories = ["music"] * 4 + ["silence"] * 5 + ["talking"] * 4
    details = [["Piano"]] * 4 + [[]] * 5 + [[]] * 4

    segments = merge_segments(categories, details, hop_sec=HOP)

    assert len(segments) == 3
    assert segments[0]["category"] == "music"
    assert segments[1]["category"] == "silence"
    assert segments[2]["category"] == "talking"


def test_silence_at_start_preserved_if_long():
    categories = ["silence"] * 5 + ["music"] * 4
    details = [[]] * 5 + [["Piano"]] * 4

    segments = merge_segments(categories, details, hop_sec=HOP)

    assert len(segments) == 2
    assert segments[0]["category"] == "silence"
    assert segments[1]["category"] == "music"


def test_silence_at_start_absorbed_if_short():
    categories = ["silence"] + ["music"] * 4
    details = [[]] + [["Piano"]] * 4

    segments = merge_segments(categories, details, hop_sec=HOP)

    assert len(segments) == 1
    assert segments[0]["category"] == "music"


def test_short_music_absorbed_into_neighbors():
    # silence (5 frames), music (1 frame = 0.48s < 1.5s), silence (5 frames) -> all silence
    categories = ["silence"] * 5 + ["music"] + ["silence"] * 5
    details = [[]] * 5 + [["Piano"]] + [[]] * 5

    segments = merge_segments(categories, details, hop_sec=HOP)

    assert len(segments) == 1
    assert segments[0]["category"] == "silence"


def test_short_music_absorbed_into_applause():
    # applause (4 frames), music (1 frame), applause (4 frames) -> all applause
    categories = ["applause"] * 4 + ["music"] + ["applause"] * 4
    details = [[]] * 4 + [["Piano"]] + [[]] * 4

    segments = merge_segments(categories, details, hop_sec=HOP)

    assert len(segments) == 1
    assert segments[0]["category"] == "applause"


def test_short_talking_absorbed():
    # silence (5 frames), talking (1 frame = 0.48s < 1.5s), silence (5 frames) -> all silence
    categories = ["silence"] * 5 + ["talking"] + ["silence"] * 5
    details = [[]] * 11

    segments = merge_segments(categories, details, hop_sec=HOP)

    assert len(segments) == 1
    assert segments[0]["category"] == "silence"


def test_first_segment_starts_at_zero():
    categories = ["music"] * 4
    details = [["Piano"]] * 4

    segments = merge_segments(categories, details, hop_sec=HOP)

    assert segments[0]["start"] == 0.0


def test_subtype_durations_are_seconds():
    # Each frame has exactly one subtype (as classifier now outputs)
    categories = ["music"] * 10
    details = [["Violin, fiddle"]] * 6 + [["Piano"]] * 4

    segments = merge_segments(categories, details, hop_sec=HOP)

    assert len(segments) == 1
    assert abs(segments[0]["subtypes"]["Piano"] - 1.92) < 0.01  # 4 frames * 0.48
    assert abs(segments[0]["subtypes"]["Violin, fiddle"] - 2.88) < 0.01  # 6 frames * 0.48
    # Total should equal segment length
    total = sum(segments[0]["subtypes"].values())
    assert abs(total - 4.8) < 0.01  # 10 frames * 0.48


def test_short_subtypes_filtered():
    # Subtypes under 1.5s should be removed
    categories = ["music"] * 10
    details = [["Piano"]] * 8 + [["Guitar"]] * 2  # Guitar = 0.96s < 1.5s

    segments = merge_segments(categories, details, hop_sec=HOP)

    assert len(segments) == 1
    assert "Piano" in segments[0]["subtypes"]
    assert "Guitar" not in segments[0]["subtypes"]
