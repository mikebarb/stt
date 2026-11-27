#!/usr/bin/env python3
"""Scan transcripts for repetition loops, reporting where they occur.

A repetition loop is whisper hallucinating over music, laughter or silence:
one phrase emitted many times, usually with a locally impossible word rate.
The discriminating measure is LOOP DENSITY - the fraction of the transcript
consumed by the most-repeated n-gram. A hymn or liturgy legitimately repeats
a short line a few times (density under 2%); a loop repeats one phrase
hundreds of times and crowds out the content (density 15-30%).

Mean token confidence does NOT detect loops: repeated text is emitted
confidently, so a looping file can score above the corpus median.

Usage:
    python scripts/loops.py <directory>
    python scripts/loops.py <directory> --all          # show every file
    python scripts/loops.py <directory> --n 8 --min-density 0.05
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

STRIP = ".,;:!?\"'“”‘’()[]—–♪"


def clock(seconds):
    s = int(max(0, seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def load_pairs(path):
    """Return [(word, start_seconds)] for every word, and the duration."""
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = data.get("segments", [])
    pairs = []
    for seg in segs:
        t = float(seg.get("start", 0.0))
        for w in seg.get("text", "").split():
            w = w.lower().strip(STRIP)
            if w:
                pairs.append((w, t))
    return pairs, float(data.get("duration_seconds", 0.0) or 0.0), len(segs)


def peak_minute_wpm(segs):
    buckets = {}
    for seg in segs:
        m = int(seg.get("start", 0) // 60)
        buckets[m] = buckets.get(m, 0) + len(seg.get("text", "").split())
    return float(max(buckets.values())) if buckets else 0.0


def analyse(path, n, min_count, min_density):
    pairs, duration, nsegs = load_pairs(path)
    total = len(pairs)
    if total < n:
        return None

    counts = Counter()
    for i in range(total - n + 1):
        counts[tuple(w for w, _ in pairs[i:i + n])] += 1
    if not counts:
        return None

    gram, count = counts.most_common(1)[0]
    positions = [i for i in range(total - n + 1)
                 if tuple(w for w, _ in pairs[i:i + n]) == gram]

    times = [pairs[p][1] for p in positions]
    density = (count * n) / total if total else 0.0

    # Densest minute, for locating the worst passage.
    per_min = Counter(int(t // 60) for t in times)
    worst_min, worst_n = per_min.most_common(1)[0]

    if count >= min_count and density >= min_density:
        klass = "LOOP"
    elif count >= min_count:
        klass = "repetition"
    else:
        klass = "ok"

    # Peak word rate needs the raw segments.
    data = json.loads(path.read_text(encoding="utf-8"))
    peak = peak_minute_wpm(data.get("segments", []))

    return {
        "file": path.name,
        "class": klass,
        "density": density,
        "count": count,
        "words": total,
        "span": f"{clock(times[0])}-{clock(times[-1])}",
        "worst": f"{clock(worst_min * 60)} ({worst_n}x)",
        "peak": peak,
        "phrase": " ".join(gram),
        "duration": duration,
        "segments": nsegs,
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--min-count", type=int, default=15)
    ap.add_argument("--min-density", type=float, default=0.05)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv[1:])

    root = Path(a.directory)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    rows = []
    for p in sorted(root.glob("*.json")):
        if p.name.endswith(".stats.json") or p.name.endswith(".stt.json"):
            continue
        try:
            r = analyse(p, a.n, a.min_count, a.min_density)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"{p.name}: {e}", file=sys.stderr)

    loops = [r for r in rows if r["class"] == "LOOP"]
    reps = [r for r in rows if r["class"] == "repetition"]

    print(f"{len(rows)} transcript(s) scanned: "
          f"{len(loops)} loop(s), {len(reps)} ordinary repetition(s)\n")

    show = rows if a.all else [r for r in rows if r["class"] != "ok"]
    if not show:
        print("no repetition loops found")
        return 0

    print(f"{'class':<11} {'density':>7} {'n':>4} {'span':<20} {'densest':<16} "
          f"{'peak':>5} {'words':>6}  file")
    print("-" * 132)
    for r in sorted(show, key=lambda r: -r["density"]):
        print(f"{r['class']:<11} {r['density']*100:>6.1f}% {r['count']:>4} "
              f"{r['span']:<20} {r['worst']:<16} {r['peak']:>5.0f} "
              f"{r['words']:>6}  {r['file'][:44]}")
        if r["class"] != "ok":
            print(f"{'':<11} \"{r['phrase'][:100]}\"")

    if loops:
        print("\nLoops need attention:")
        for r in loops:
            print(f"  {r['file']}")
            print(f"    {r['count']}x over {r['span']}, densest at {r['worst']}, "
                  f"{r['density']*100:.1f}% of the transcript")
            print(f"    listen at {r['worst'].split()[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
