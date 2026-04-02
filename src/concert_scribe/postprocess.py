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
    for idx, seg in enumerate(raw_segments):
        is_short_silence = (
            seg["category"] == "silence"
            and (seg["end"] - seg["start"]) < min_silence_sec
        )
        has_next = idx < len(raw_segments) - 1

        if is_short_silence and has_next:
            # Short silence with a following segment: absorb into previous non-silence
            # neighbor (extend it), or if none, absorb forward into next segment
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
