"""Fetch N Wikipedia articles, truncate each to ~N sentences, stitch with synthetic
section headers so chunker.py --wiki picks up the article boundaries as gold.

Usage:
    python stitch.py "Octopus" "French Revolution" "Photosynthesis" "Jazz" \
        --per-article 50 --out stitched.txt
"""

import argparse
import re
from pathlib import Path

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADER_RE = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$")
UA = "embedding-chunker-research/0.1 (local experiment; contact via local machine)"


def fetch_wiki(title: str) -> str:
    r = requests.get(
        WIKI_API,
        params={
            "action": "query", "format": "json", "titles": title,
            "prop": "extracts", "explaintext": "true", "redirects": "1",
        },
        headers={"User-Agent": UA, "Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    return next(iter(pages.values()))["extract"]


def first_n_sentences(text: str, n: int) -> str:
    sents: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para or HEADER_RE.match(para):
            continue
        para_norm = re.sub(r"\s+", " ", para)
        for p in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", para_norm):
            p = p.strip()
            if p:
                sents.append(p)
                if len(sents) >= n:
                    return " ".join(sents)
    return " ".join(sents)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("titles", nargs="+", help="Wikipedia article titles to stitch")
    p.add_argument("--per-article", type=int, default=60, help="max sentences per article")
    p.add_argument("--out", type=Path, default=Path("stitched.txt"))
    args = p.parse_args()

    pieces = []
    for title in args.titles:
        print(f"fetching: {title}")
        raw = fetch_wiki(title)
        body = first_n_sentences(raw, args.per_article)
        pieces.append(f"== {title} ==\n\n{body}")

    stitched = "\n\n".join(pieces)
    args.out.write_text(stitched)
    print(f"\nstitched -> {args.out}  ({len(pieces)} articles)")
    print(f"  total chars: {len(stitched)}")
    print(f"  run: .venv/bin/python chunker.py {args.out} --wiki ...")


if __name__ == "__main__":
    main()
