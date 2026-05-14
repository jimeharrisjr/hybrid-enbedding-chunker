"""Threshold sweep over chunking methods, scored against gold breakpoints.

Loads sentences + cached embeddings produced by chunker.py --wiki, then sweeps each
method's primary threshold and reports the F1-optimal operating point.

Usage:
    python sweep.py octopus.txt --tol 2
"""

import argparse
from pathlib import Path

import numpy as np

import chunker as ch


def sweep(input_path: Path, tol: int = 2, model: str = ch.DEFAULT_MODEL) -> None:
    text = input_path.read_text()
    sentences, gold = ch.split_sentences_with_wiki_headers(text)
    cache_path = input_path.with_suffix(".embeddings.npz")
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        embeddings = cached["embeddings"]
    else:
        print("no cache; embedding via Ollama ...")
        embeddings = ch.embed_batch(sentences, model=model)
        np.savez(cache_path, sentences=np.asarray(sentences, dtype=object),
                 embeddings=embeddings, model=model)
    sentences, embeddings, dropped = ch.filter_degenerate_embeddings(sentences, embeddings)
    gold = [g - sum(1 for d in dropped if d <= g) for g in gold if g not in dropped]
    print(f"{len(sentences)} sentences, {len(gold)} gold boundaries, dims={embeddings.shape[1]}")

    def best(name: str, runs: list[tuple[float, list[int]]]):
        rows = []
        for thr, preds in runs:
            ev = ch.evaluate_breakpoints(preds[:-1] if preds else [], gold, tolerance=tol)
            rows.append((thr, ev["n_pred"], ev["tp"], ev["precision"], ev["recall"], ev["f1"]))
        rows.sort(key=lambda r: -r[5])
        print(f"\n{name}  (best by F1, top 5):")
        print(f"  {'thr':>8} {'#pred':>6} {'tp':>4} {'P':>6} {'R':>6} {'F1':>6}")
        for thr, n_pred, tp, p, r, f1 in rows[:5]:
            print(f"  {thr:>8.3f} {n_pred:>6} {tp:>4} {p:>6.2f} {r:>6.2f} {f1:>6.2f}")

    # pairwise: sweep percentile
    pairwise_runs = []
    for pct in [70, 75, 80, 85, 88, 90, 92, 94, 95, 96, 98]:
        r = ch.pairwise_chunker(sentences, embeddings, percentile=pct)
        pairwise_runs.append((pct, r.breakpoints))
    best("pairwise (percentile)", pairwise_runs)

    # growing window: sweep sim threshold
    gw_runs = []
    for s in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        r = ch.growing_window_chunker(sentences, embeddings, sim_threshold=s)
        gw_runs.append((s, r.breakpoints))
    best("growing_window (sim_threshold)", gw_runs)

    # max-min: sweep bootstrap threshold
    mm_runs = []
    for s in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        r = ch.max_min_chunker(sentences, embeddings, bootstrap_threshold=s, window=8)
        mm_runs.append((s, r.breakpoints))
    best("max_min (bootstrap)", mm_runs)

    # fixed-size: sweep target tokens
    fs_runs = []
    for t in [256, 384, 512, 640, 768, 1024]:
        r = ch.fixed_size_chunker(sentences, embeddings, target_tokens=t, overlap_tokens=int(t * 0.2))
        fs_runs.append((t, r.breakpoints))
    best("fixed_size (target_tokens)", fs_runs)

    # hybrid: sweep drift threshold (keeping target=512, search=5, ctx=5)
    hy_runs = []
    for d in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        r = ch.hybrid_chunker(sentences, embeddings, target_tokens=512,
                              search_window=5, drift_threshold=d, context_window=5)
        hy_runs.append((d, r.breakpoints))
    best("hybrid (drift_threshold)", hy_runs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--tol", type=int, default=2)
    p.add_argument("--model", default=ch.DEFAULT_MODEL)
    args = p.parse_args()
    sweep(args.input, tol=args.tol, model=args.model)


if __name__ == "__main__":
    main()
