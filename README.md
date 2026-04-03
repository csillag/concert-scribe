# Concert Scribe

![Concert Scribe](https://raw.githubusercontent.com/csillag/concert-scribe/main/images/scribe.png)

Classify audio in concert recordings into segments of **silence**, **talking**, **music**, and **applause**.

Takes video or audio files as input, extracts the audio, runs it through Google's [YAMNet](https://tfhub.dev/google/yamnet/1) model, and produces a simple text file describing the timeline.

## Example output

```
0.0-3.36: talking
3.36-33.12: silence
33.12-37.44: applause
37.44-50.4: silence
50.4-108.96: music (Cello)
108.96-118.56: silence
118.56-274.56: music (Cello, Piano)
274.56-285.6: silence
285.6-365.76: music (Cello)
365.76-377.28: applause
377.28-381.6: silence
```

With `--verbose`, instrument durations are included:

```
118.56-274.56: music (Cello: 82.6s, Piano: 15.4s)
```

## Install

```bash
pip install concert-scribe
```

Or with [pipx](https://pipx.pypa.io/):

```bash
pipx install concert-scribe
```

Requires `ffmpeg` on the system for audio extraction.

## Usage

```bash
# Single file
concert-scribe recording.mp4

# All videos in a directory
concert-scribe /path/to/videos/

# Custom output directory
concert-scribe recording.mp4 -o /path/to/output/

# Include per-instrument durations
concert-scribe recording.mp4 --verbose
```

## How it works

1. Extracts audio from video via ffmpeg (mono, 16kHz)
2. Classifies each 0.48s frame using YAMNet (521 AudioSet classes mapped to 4 categories)
3. Merges adjacent same-category frames into segments
4. Filters out short spurious segments (< 1.5s for music/talking, < 2s for silence)
5. Deduplicates music sub-types using the AudioSet hierarchy (keeps only the most specific instrument)
6. Writes a `.txt` file per input clip

## License

Apache-2.0
