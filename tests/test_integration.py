"""Integration test — requires TensorFlow and YAMNet model download."""

import os
import tempfile
import wave

import numpy as np
import pytest

from concert_scribe.classify import classify_audio, HOP_SECONDS
from concert_scribe.output import write_segments
from concert_scribe.postprocess import merge_segments


def _make_test_wav(path: str, duration: float = 5.0):
    """Create a 16kHz mono WAV with white noise (likely classified as noise/silence)."""
    sr = 16000
    n_samples = int(duration * sr)
    # Low-amplitude white noise
    samples = (np.random.randn(n_samples) * 100).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


@pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION"),
    reason="Set RUN_INTEGRATION=1 to run (requires TensorFlow)",
)
def test_full_pipeline_produces_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "test.wav")
        out_path = os.path.join(tmpdir, "test.txt")

        _make_test_wav(wav_path, duration=5.0)

        categories, details = classify_audio(wav_path)
        assert len(categories) > 0

        segments = merge_segments(categories, details, hop_sec=HOP_SECONDS)
        assert len(segments) > 0

        write_segments(segments, out_path)
        assert os.path.exists(out_path)

        with open(out_path) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) > 0
        # Each line should match the expected format
        for line in lines:
            assert ": " in line
