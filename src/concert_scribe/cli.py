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
