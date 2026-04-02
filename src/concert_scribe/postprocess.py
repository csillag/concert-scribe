"""Post-processing: merge frames into segments, enforce silence minimum."""

from collections import Counter
from typing import TypedDict


class Segment(TypedDict):
    category: str
    start: float
    end: float
    subtypes: dict[str, float]  # subtype name -> total seconds detected


def _absorb_short_segments(
    segments: list[Segment],
    category: str,
    min_duration: float,
) -> list[Segment]:
    """Absorb segments of a given category shorter than min_duration into neighbors."""
    if not segments:
        return segments

    result: list[Segment] = []
    for seg in segments:
        is_short = (
            seg["category"] == category
            and (seg["end"] - seg["start"]) < min_duration
        )
        if is_short:
            if result:
                # Absorb into previous segment
                result[-1]["end"] = seg["end"]
                # Merge subtype durations
                for k, v in seg["subtypes"].items():
                    result[-1]["subtypes"][k] = result[-1]["subtypes"].get(k, 0.0) + v
            # If no previous, will be absorbed forward (handled below)
        else:
            # If previous was a short segment that couldn't be absorbed backward
            # (at the start), absorb it forward
            if result and result[-1]["category"] == category and (result[-1]["end"] - result[-1]["start"]) < min_duration:
                seg = Segment(
                    category=seg["category"],
                    start=result[-1]["start"],
                    end=seg["end"],
                    subtypes=seg["subtypes"],
                )
                result.pop()
            result.append(seg)

    # Handle trailing short segment
    if len(result) > 1 and result[-1]["category"] == category and (result[-1]["end"] - result[-1]["start"]) < min_duration:
        result[-2]["end"] = result[-1]["end"]
        for k, v in result[-1]["subtypes"].items():
            result[-2]["subtypes"][k] = result[-2]["subtypes"].get(k, 0.0) + v
        result.pop()

    return result


def _remerge_adjacent(segments: list[Segment]) -> list[Segment]:
    """Re-merge adjacent segments that share a category."""
    if not segments:
        return segments
    merged: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        if seg["category"] == merged[-1]["category"]:
            merged[-1]["end"] = seg["end"]
            for k, v in seg["subtypes"].items():
                merged[-1]["subtypes"][k] = merged[-1]["subtypes"].get(k, 0.0) + v
        else:
            merged.append(seg)
    return merged


def merge_segments(
    categories: list[str],
    music_details: list[list[str]],
    hop_sec: float,
    min_silence_sec: float = 2.0,
    min_music_sec: float = 1.5,
    min_talking_sec: float = 1.5,
) -> list[Segment]:
    """Merge per-frame categories into contiguous segments.

    Args:
        categories: Per-frame category labels.
        music_details: Per-frame music sub-type lists.
        hop_sec: Duration of each frame hop in seconds.
        min_silence_sec: Minimum duration for silence segments.
        min_music_sec: Minimum duration for music segments.
        min_talking_sec: Minimum duration for talking segments.
            Shorter segments are absorbed into neighbors.

    Returns:
        List of Segment dicts with category, start, end, subtypes.
    """
    if not categories:
        return []

    # Step 1: Merge consecutive same-category frames into raw segments
    raw_segments: list[Segment] = []
    current_cat = categories[0]
    current_start = 0
    current_subtype_counts: Counter[str] = Counter(music_details[0])

    for i in range(1, len(categories)):
        if categories[i] != current_cat:
            # Convert frame counts to seconds
            subtype_durations = {k: v * hop_sec for k, v in current_subtype_counts.items()}
            raw_segments.append(
                Segment(
                    category=current_cat,
                    start=current_start * hop_sec,
                    end=i * hop_sec,
                    subtypes=subtype_durations,
                )
            )
            current_cat = categories[i]
            current_start = i
            current_subtype_counts = Counter(music_details[i])
        else:
            current_subtype_counts.update(music_details[i])

    subtype_durations = {k: v * hop_sec for k, v in current_subtype_counts.items()}
    raw_segments.append(
        Segment(
            category=current_cat,
            start=current_start * hop_sec,
            end=len(categories) * hop_sec,
            subtypes=subtype_durations,
        )
    )

    # Step 2: Absorb short segments (music, talking, then silence)
    result = raw_segments
    result = _absorb_short_segments(result, "music", min_music_sec)
    result = _remerge_adjacent(result)
    result = _absorb_short_segments(result, "talking", min_talking_sec)
    result = _remerge_adjacent(result)
    result = _absorb_short_segments(result, "silence", min_silence_sec)
    result = _remerge_adjacent(result)

    # Step 3: Filter short-duration subtypes within music segments
    min_subtype_sec = min_music_sec
    for seg in result:
        if seg["category"] == "music" and seg["subtypes"]:
            seg["subtypes"] = {
                k: v for k, v in seg["subtypes"].items() if v >= min_subtype_sec
            }

    # Step 4: Ensure first segment starts at 0.0
    if result and result[0]["start"] > 0.0:
        result[0]["start"] = 0.0

    # Safety: if everything got absorbed, return a single segment
    if not result and categories:
        result = [Segment(
            category="silence",
            start=0.0,
            end=len(categories) * hop_sec,
            subtypes={},
        )]

    return result
