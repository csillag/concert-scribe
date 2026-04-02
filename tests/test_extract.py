import os
import struct
import tempfile
import wave

import numpy as np
import pytest

from concert_scribe.extract import extract_audio

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".mts")


def _make_test_wav(path, duration=1.0, sample_rate=44100):
    """Create a minimal WAV file (used as a fake 'video' since ffmpeg can read it)."""
    n_samples = int(duration * sample_rate)
    samples = (np.sin(2 * np.pi * 440 * np.arange(n_samples) / sample_rate) * 32767).astype(
        np.int16
    )
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def test_extract_audio_produces_mono_16khz_wav():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test_input.wav")
        output_path = os.path.join(tmpdir, "extracted.wav")
        _make_test_wav(input_path, duration=2.0, sample_rate=44100)

        result = extract_audio(input_path, output_path)

        assert result == output_path
        assert os.path.exists(output_path)

        with wave.open(output_path, "r") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2


def test_extract_audio_nonexistent_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "out.wav")
        with pytest.raises(RuntimeError):
            extract_audio("/nonexistent/file.mp4", output_path)
