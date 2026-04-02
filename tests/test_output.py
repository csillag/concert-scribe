import os
import tempfile

from concert_scribe.output import write_segments
from concert_scribe.postprocess import Segment


def test_write_segments_format():
    segments = [
        Segment(category="silence", start=0.0, end=4.32, subtypes={}),
        Segment(category="talking", start=4.32, end=15.36, subtypes={}),
        Segment(category="music", start=15.36, end=40.32, subtypes={"Piano": 20.16, "Singing": 10.08}),
        Segment(category="applause", start=40.32, end=45.12, subtypes={}),
        Segment(category="silence", start=45.12, end=50.4, subtypes={}),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test.txt")
        write_segments(segments, out_path)

        with open(out_path) as f:
            lines = f.read().strip().split("\n")

    assert len(lines) == 5
    assert lines[0] == "0.0-4.32: silence"
    assert lines[1] == "4.32-15.36: talking"
    assert lines[2] == "15.36-40.32: music (Piano: 20.2s, Singing: 10.1s)"
    assert lines[3] == "40.32-45.12: applause"
    assert lines[4] == "45.12-50.4: silence"


def test_write_segments_display_names():
    segments = [
        Segment(category="music", start=0.0, end=10.0, subtypes={"Violin, fiddle": 8.0, "Wind instrument, woodwind instrument": 2.0}),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test.txt")
        write_segments(segments, out_path)

        with open(out_path) as f:
            line = f.read().strip()

    assert "Violin:" in line
    assert "Woodwind:" in line
    assert "fiddle" not in line


def test_write_segments_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "empty.txt")
        write_segments([], out_path)

        with open(out_path) as f:
            content = f.read()

    assert content == ""
