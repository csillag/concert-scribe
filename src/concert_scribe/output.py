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


def _format_duration(t: float) -> str:
    """Format a duration in seconds."""
    s = f"{t:.1f}"
    if s.endswith(".0"):
        s = s[:-2]
    return s + "s"


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
                # Sort by duration descending
                sorted_subs = sorted(seg["subtypes"].items(), key=lambda x: -x[1])
                parts = [f"{name}: {_format_duration(dur)}" for name, dur in sorted_subs]
                line += f" ({', '.join(parts)})"
            f.write(line + "\n")
