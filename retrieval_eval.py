"""Downstream retrieval eval: section titles as queries, chunks as documents.

Logic:
    * Parse the wiki-formatted doc into sentences + (section_title, start, end) tuples.
    * Embed all sentences (cached).
    * For each chunking method, build chunks, then embed the chunk texts.
    * For each section, use its title as a query. The set of "relevant" chunks is the
      set of chunks that contain at least one sentence from that section's range.
    * Rank chunks by cosine similarity to the query. Report hit@1, hit@3, MRR averaged
      across all queries.

A method that creates chunks well-aligned with topical structure should rank a relevant
chunk above unrelated ones for almost every query.
"""

import argparse
import re
from pathlib import Path

import numpy as np

import chunker as ch

EXCLUDE_TITLES = {
    "References", "See also", "External links", "Notes", "Citations",
    "Bibliography", "Further reading", "(intro)",
}


def parse_wiki_sections(text: str) -> tuple[list[str], list[tuple[str, int, int]]]:
    """Return (sentences, sections) where sections=[(title, sent_start, sent_end), ...]."""
    sentences: list[str] = []
    sections: list[tuple[str, int, int]] = []
    current_title, current_start = "(intro)", 0
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        m = ch.WIKI_HEADER_RE.match(para)
        if m:
            if len(sentences) > current_start:
                sections.append((current_title, current_start, len(sentences) - 1))
            current_title = m.group(1)
            current_start = len(sentences)
            continue
        para_norm = re.sub(r"\s+", " ", para)
        for p in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", para_norm):
            p = p.strip()
            if p:
                sentences.append(p)
    if len(sentences) > current_start:
        sections.append((current_title, current_start, len(sentences) - 1))
    return sentences, sections


def chunk_to_sentence_ranges(breakpoints: list[int], n_sentences: int) -> list[tuple[int, int]]:
    """Convert per-chunk end indices into (start, end) tuples (assumes non-overlapping)."""
    ranges, start = [], 0
    for bp in breakpoints:
        ranges.append((start, bp))
        start = bp + 1
    if not ranges or ranges[-1][1] < n_sentences - 1:
        ranges.append((start, n_sentences - 1))
    return ranges


def load_or_embed(input_path: Path, sentences: list[str], model: str) -> np.ndarray:
    cache = input_path.with_suffix(".embeddings.npz")
    if cache.exists():
        cached = np.load(cache, allow_pickle=True)
        if cached["sentences"].tolist() == sentences:
            return cached["embeddings"]
    print(f"embedding {len(sentences)} sentences via Ollama...")
    embs = ch.embed_batch(sentences, model=model)
    np.savez(cache, sentences=np.asarray(sentences, dtype=object), embeddings=embs, model=model)
    return embs


def run(input_path: Path, model: str = ch.DEFAULT_MODEL) -> None:
    text = input_path.read_text()
    sentences, sections = parse_wiki_sections(text)
    embeddings = load_or_embed(input_path, sentences, model)
    sentences, embeddings, dropped = ch.filter_degenerate_embeddings(sentences, embeddings)
    if dropped:
        print(f"dropped {len(dropped)} degenerate embeddings; section indices may shift")
    print(f"{len(sentences)} sentences, {len(sections)} sections in doc")

    queryable = [(t, s, e) for t, s, e in sections if t not in EXCLUDE_TITLES and e - s + 1 >= 3]
    print(f"{len(queryable)} queryable sections (min 3 sentences, excluding references/etc.)")

    queries = [t for t, _, _ in queryable]
    print("embedding queries...")
    q_embs = ch.embed_batch(queries, model=model)
    q_embs = np.nan_to_num(q_embs)
    q_embs = q_embs / np.maximum(np.linalg.norm(q_embs, axis=1, keepdims=True), 1e-9)

    methods = {
        "fixed_size(512)": lambda: ch.fixed_size_chunker(sentences, embeddings, target_tokens=512),
        "pairwise(95)":    lambda: ch.pairwise_chunker(sentences, embeddings, percentile=95.0),
        "growing_window(0.30)": lambda: ch.growing_window_chunker(sentences, embeddings, sim_threshold=0.30),
        "growing_window(0.40)": lambda: ch.growing_window_chunker(sentences, embeddings, sim_threshold=0.40),
        "max_min(0.35)":   lambda: ch.max_min_chunker(sentences, embeddings, bootstrap_threshold=0.35, window=8),
        "hybrid(d=0.40)":  lambda: ch.hybrid_chunker(sentences, embeddings, drift_threshold=0.40),
        "hybrid(d=0.30)":  lambda: ch.hybrid_chunker(sentences, embeddings, drift_threshold=0.30),
    }

    n = len(queryable)
    print(f"\n{'method':<22} {'#chunks':>8} {'hit@1':>6} {'hit@3':>6} {'MRR':>6}")
    print("-" * 56)
    for name, fn in methods.items():
        r = fn()
        ranges = chunk_to_sentence_ranges(r.breakpoints, len(sentences))
        chunk_embs = ch.embed_batch(r.chunks, model=model)
        chunk_embs = np.nan_to_num(chunk_embs)
        chunk_embs = chunk_embs / np.maximum(np.linalg.norm(chunk_embs, axis=1, keepdims=True), 1e-9)
        scores = q_embs @ chunk_embs.T          # (Q, C)
        rankings = np.argsort(-scores, axis=1)
        hit1, hit3, mrr = 0, 0, 0.0
        for q_idx, (title, sec_start, sec_end) in enumerate(queryable):
            relevant = {
                ci for ci, (cs, ce) in enumerate(ranges)
                if cs <= sec_end and ce >= sec_start          # range overlap
            }
            if not relevant:
                continue
            for rank_pos, ci in enumerate(rankings[q_idx]):
                if ci in relevant:
                    if rank_pos == 0: hit1 += 1
                    if rank_pos < 3: hit3 += 1
                    mrr += 1.0 / (rank_pos + 1)
                    break
        print(f"{name:<22} {len(r.chunks):>8} {hit1/n:>6.2f} {hit3/n:>6.2f} {mrr/n:>6.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--model", default=ch.DEFAULT_MODEL)
    args = p.parse_args()
    run(args.input, model=args.model)


if __name__ == "__main__":
    main()
