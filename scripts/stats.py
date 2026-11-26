#!/usr/bin/env python3
"""Structural statistics for a transcript JSON produced by transcriber.exe.

Usage:
    python scripts/stats.py <file.json> [<file.json> ...]
    python scripts/stats.py "out/*.json"

Reports duration, segment count, word count and words-per-minute. WPM is the
useful alarm: it is stable per speaker and delivery style, so a run far outside
the expected band signals dropped content (low) or a hallucination loop (high)
without anyone listening.
"""

import glob
import json
import sys


def stats(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)

    segs = d.get("segments", [])
    words = sum(len(s.get("text", "").split()) for s in segs)
    dur = float(d.get("duration_seconds", 0)) or 0.0

    # Longest segment is another alarm: whisper segments are sentence-ish,
    # typically 2-12 s. A 90 s segment suggests a decode that ran away.
    longest = max((s.get("end", 0) - s.get("start", 0) for s in segs), default=0.0)

    return {
        "file": path.rsplit("\\", 1)[-1],
        "model": d.get("model_used", "?"),
        "dur": dur / 60.0,
        "segs": len(segs),
        "words": words,
        "wpm": (words / dur * 60.0) if dur else 0.0,
        "longest": longest,
    }


def main(argv):
    paths = []
    for a in argv[1:]:
        paths.extend(glob.glob(a) if any(c in a for c in "*?[") else [a])
    if not paths:
        print("usage: stats.py <file.json> [...]", file=sys.stderr)
        return 2

    print(f"{'file':<52} {'model':<14} {'min':>6} {'segs':>5} "
          f"{'words':>6} {'wpm':>5} {'max_s':>6}")
    print("-" * 108)
    for p in paths:
        try:
            s = stats(p)
        except Exception as e:                                    # noqa: BLE001
            print(f"{p:<52} ERROR {e}")
            continue
        print(f"{s['file']:<52.52} {s['model']:<14.14} {s['dur']:>6.1f} "
              f"{s['segs']:>5} {s['words']:>6} {s['wpm']:>5.0f} "
              f"{s['longest']:>6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
