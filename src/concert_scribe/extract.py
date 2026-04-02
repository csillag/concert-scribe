"""Extract audio from video files using ffmpeg."""

import subprocess


def extract_audio(input_path: str, output_path: str) -> str:
    """Extract audio from a video file as mono 16kHz 16-bit WAV.

    Args:
        input_path: Path to input video file.
        output_path: Path to write the extracted WAV file.

    Returns:
        The output_path.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-y",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    return output_path
