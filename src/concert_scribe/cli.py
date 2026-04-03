"""CLI entry point for concert-scribe."""

import argparse
import os
import sys
import tempfile

# Suppress TensorFlow/CUDA/absl logging before any TF imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

import logging
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import csv as csv_mod

from concert_scribe.classify import HOP_SECONDS, classify_audio, _MUSIC_SUBTYPES, display_name
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


def process_file(video_path: str, output_dir: str, verbose: bool = False, instruments_debug: bool = False, dump_scores: bool = False) -> None:
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
        (categories, music_details), raw_scores, class_names = classify_audio(
            wav_path, threshold=CONFIDENCE_THRESHOLD,
            raw_instruments=instruments_debug,
        )

        # Post-process
        segments = merge_segments(
            categories, music_details,
            hop_sec=HOP_SECONDS,
            min_silence_sec=MIN_SILENCE_SEC,
        )

    # Write output
    write_segments(segments, output_path, verbose=verbose)
    print(f"  Written: {output_path}", file=sys.stderr)

    # Dump raw scores for music frames
    if dump_scores:
        scores_path = os.path.join(output_dir, f"{basename}_scores.csv")
        # Find music subtype indices
        subtype_indices = []
        subtype_names = []
        for i, name in enumerate(class_names):
            if name in _MUSIC_SUBTYPES:
                subtype_indices.append(i)
                subtype_names.append(display_name(name))
        with open(scores_path, "w", newline="") as f:
            writer = csv_mod.writer(f)
            writer.writerow(["time", "category"] + subtype_names)
            for frame_idx in range(len(categories)):
                t = frame_idx * HOP_SECONDS
                row = [f"{t:.2f}", categories[frame_idx]]
                for si in subtype_indices:
                    row.append(f"{raw_scores[frame_idx, si]:.4f}")
                writer.writerow(row)
        print(f"  Scores: {scores_path}", file=sys.stderr)


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
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Include per-instrument durations in output",
    )
    parser.add_argument(
        "--instruments-debug",
        action="store_true",
        help="Disable instrument false-positive suppression (show all raw detections)",
    )
    parser.add_argument(
        "--dump-scores",
        action="store_true",
        help="Write per-frame instrument scores to a CSV file for analysis",
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
        process_file(video_path, output_dir, verbose=args.verbose,
                     instruments_debug=args.instruments_debug,
                     dump_scores=args.dump_scores)

    print(f"Done. Processed {len(video_files)} file(s).", file=sys.stderr)
