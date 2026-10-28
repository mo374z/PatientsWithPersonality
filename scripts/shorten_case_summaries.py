#!/usr/bin/env python3
"""
Condense the per-case profile digests into one sentence each for the labeling GUI.

The digests in short_desc.json are semicolon-separated problem lists, too long to
sit above a four-column comparison. This rewrites them as a single line of prose.
"""

import argparse
import json
from pathlib import Path

from google import genai

PROMPT = """Rewrite this patient case digest as ONE short sentence, for a clinician who is
about to read a transcript of the consultation.

Rules:
- One sentence, at most 18 words, no trailing full stop.
- Start with age and sex when the digest gives them, then the reason for the visit.
- Keep at most the two most salient active problems and drop everything else.
- Plain clinical prose. No semicolons, no lists, no abbreviated fragments.

Example digest: 48y Female - Annual review; Knee pain (onset March); Poor sleep; Raised blood pressure
Example sentence: 48-year-old woman attending for an annual review, reporting knee pain and raised blood pressure

Digest: {digest}
Sentence:"""


def shorten(client, model: str, digest: str) -> str:
    response = client.models.generate_content(
        model=model, contents=PROMPT.format(digest=digest)
    )
    return response.text.strip().strip('"').rstrip(".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    digests = json.loads(Path(args.source).read_text())
    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    summaries = {}
    for case, digest in digests.items():
        summaries[case] = shorten(client, args.model, digest)
        print(f"{case}: {summaries[case]}")

    out = Path(args.out)
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n")
    lengths = [len(s.split()) for s in summaries.values()]
    print(f"\nWrote {len(summaries)} summaries to {out}")
    print(
        f"  words: min {min(lengths)}, median {sorted(lengths)[len(lengths) // 2]}, max {max(lengths)}"
    )


if __name__ == "__main__":
    main()
