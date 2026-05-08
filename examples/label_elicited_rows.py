"""Apply Rule A labels to elicited (synthetic_elicitation) rows in halluc_corpus.jsonl.

For each unlabeled elicited row:
  - URL: every URL_PATTERN match not in `trusted_urls` is a positive
    (Rule A: provenance, not truth).
  - Foreign-ref: applied semantically by reading any fenced Python blocks.
    For rows whose `llm_output` contains no fenced Python block, the
    foreign-ref label is unambiguously negative.

Idempotent: only touches rows where `expected_url_flag is None`. Per-row
`label_notes` carries human commentary (which rows illustrate which
failure modes — useful when writing §7.3).

Run after `build_corpus.py` has appended one or more elicited rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scikit_rec_agent.safeguards import (  # noqa: E402
    URL_PATTERN,
    _normalize_url,
    detect_foreign_references,
)

CORPUS_PATH = _REPO_ROOT / "examples" / "halluc_corpus.jsonl"

# Per-id label notes capturing the failure mode each elicited row illustrates.
# Used in §7.3 when discussing real-world hallucination patterns. Only rows
# with non-trivial commentary appear here; everything else gets a generic note.
NOTES: dict[str, str] = {
    "halluc_0000": ("clean negative: model listed datasets in prose only — no URLs, no code blocks. Useful baseline."),
    "halluc_0001": (
        "multi-URL fabrication: 7 URLs across grouplens.org, netflixprize.com, AWS, last.fm, yelp.com. "
        "All emitted with confident framing as 'direct download links'. The 3 grouplens "
        "URLs follow a consistent wrong-host pattern (real prefix is files.grouplens.org/datasets/movielens/, "
        "model omits the files.* subdomain). Real-vs-fake doesn't matter under Rule A "
        "(provenance), but cite as evidence of plausible-looking-but-wrong fabrication "
        "in §7.3."
    ),
    "halluc_0002": (
        "drilled-down fabrication: model confidently emits MovieLens 25M URL with same "
        "wrong-host pattern as row 0001 when explicitly asked for 'the direct .zip URL'. "
        "Strong illustration of model doubling down on a fabricated URL when pressed."
    ),
    "halluc_0003": (
        "arXiv ID fabrication for a real paper: title and authors of Harper & Konstan are "
        "correct, but the canonical paper is published in ACM TiiS (2015), not arXiv. "
        "The arXiv ID 1702.05884 is fabricated; URL is structurally well-formed. "
        "Particularly dangerous failure mode: a user citing this would point readers to the "
        "wrong (or non-existent) paper. §7.3 example."
    ),
    "halluc_0004": (
        "OUT-OF-SCOPE prose-level fabrication: model legitimizes a non-existent dataset "
        "('RecSys2024-Spotify-Sequential') in free-text prose while emitting only a real "
        "conference URL (recsys.acm.org/recsys24/). The URL detector flags the URL on "
        "provenance grounds, but the dangerous claim — 'this dataset exists' — is in prose "
        "and falls outside both detectors' scope. This is the canonical §7.3 example for "
        "what URL/AST safeguards do NOT cover."
    ),
    "halluc_0005": (
        "fabricated GitHub repo: github.com/grouplens/movielens — the GroupLens org exists "
        "on GitHub but this specific repo does not (real Python loaders for MovieLens live "
        "in third-party libs like cornac, recbole, surprise). Different URL surface than "
        "rows 0001-0003."
    ),
}

_GENERIC_NOTE = "Rule A: URLs model-introduced (not in trusted set); no fenced Python blocks → foreign-ref negative."


def main() -> int:
    if not CORPUS_PATH.exists():
        print(f"corpus not found at {CORPUS_PATH}")
        return 1

    rows = [json.loads(line) for line in CORPUS_PATH.read_text().splitlines()]
    labeled = 0

    for row in rows:
        if row["source"] != "synthetic_elicitation":
            continue
        if row["expected_url_flag"] is not None:
            continue

        text = row["llm_output"]
        trusted = {_normalize_url(u) for u in row["trusted_urls"]}
        novel_urls = sorted({_normalize_url(u) for u in URL_PATTERN.findall(text)} - trusted)
        foreign = sorted(detect_foreign_references(text))

        # Sanity check: under Rule A applied semantically, foreign-ref evaluation
        # only makes sense if there are fenced Python blocks. If detect_foreign_references
        # found nothing AND there are no code fences in the text, the label is
        # unambiguously negative. If there ARE code fences but the visitor saw nothing
        # foreign, surface that for human review (could be a missed FN case worth labeling
        # by hand). The 6 dataset_hunt rows have no code blocks, so this branch is moot
        # for them — but it's the right shape for future themes.
        has_code_fence = "```" in text
        if has_code_fence and not foreign:
            print(
                f"  WARN {row['id']}: fenced block present but visitor saw no foreign refs. "
                f"Auto-label is False but human spot-check recommended."
            )

        row["expected_url_set"] = novel_urls
        row["expected_url_flag"] = bool(novel_urls)
        row["expected_foreign_set"] = foreign
        row["expected_foreign_flag"] = bool(foreign)
        row["label_notes"] = NOTES.get(row["id"], _GENERIC_NOTE)
        labeled += 1

    if labeled == 0:
        print("nothing to label — all elicited rows already have expected_url_flag set")
        return 0

    with CORPUS_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"labeled {labeled} elicited rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
