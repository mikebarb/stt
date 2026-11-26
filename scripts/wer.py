#!/usr/bin/env python3
"""Compare a transcript against a reference and report Word Error Rate.

Usage:
    python scripts/wer.py reference.txt candidate.txt
"""

import argparse
import re
import sys

try:
    import jiwer
except ImportError:
    sys.exit("jiwer not installed. Run: pip install jiwer")


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)   # keep apostrophes (You'll, Jesus')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_words(path: str) -> list:
    # errors="replace" tolerates the CRLF/LF mix and any odd encoding
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    # strip timestamp prefixes like [00:01:02.340 --> 00:01:05.820]
    raw = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]", " ", raw)
    return normalise(raw).split()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    args = ap.parse_args()

    ref = " ".join(load_words(args.reference))
    hyp = " ".join(load_words(args.candidate))

    if not ref:
        print("reference transcript is empty")
        return 1

    out = jiwer.process_words(ref, hyp)

    n_ref = out.hits + out.substitutions + out.deletions
    n_hyp = out.hits + out.substitutions + out.insertions

    print("reference words : %d" % n_ref)
    print("candidate words : %d" % n_hyp)
    print("substitutions   : %d" % out.substitutions)
    print("deletions       : %d" % out.deletions)
    print("insertions      : %d" % out.insertions)
    print("hits            : %d" % out.hits)
    print("WER             : %.2f%%" % (out.wer * 100))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
