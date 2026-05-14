"""Visualize how three chunking strategies place boundaries on the same drift signal.

Produces a stacked-panel comparison plot for a document:
    - Top panel: fixed-size boundaries (drift-blind, evenly spaced)
    - Middle panel: growing_window boundaries (clustered at drift peaks)
    - Bottom panel: hybrid boundaries (fixed-size cadence nudged to nearest drift peak)

Each panel shows the same sliding-window cosine-drift signal so the difference
in boundary placement strategy is visible at a glance.

Usage:
    python compare_plot.py octopus.txt --out octopus_compare.png
"""

import argparse
from pathlib import Path

import numpy as np

import chunker as ch
import retrieval_eval as re_eval


def compute_drift(embeddings: np.ndarray, context_window: int = 5) -> np.ndarray:
    """Sliding-window drift: 1 - cos(mean(last K sentence embeddings), current)."""
    embs = np.nan_to_num(embeddings, nan=0.0)
    embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-9)
    n = len(embs)
    drift = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        lo = max(0, i - context_window)
        ctx = embs[lo:i].mean(axis=0)
        ctx = ctx / (np.linalg.norm(ctx) + 1e-9)
        drift[i] = 1.0 - float(np.dot(ctx, embs[i]))
    return drift


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--target-tokens", type=int, default=512)
    p.add_argument("--gw-sim", type=float, default=0.30)
    p.add_argument("--hy-drift", type=float, default=0.40)
    p.add_argument("--hy-search", type=int, default=5)
    p.add_argument("--ctx", type=int, default=5, help="context window for drift signal")
    p.add_argument("--smooth", type=int, default=5,
                   help="moving-average window for plotted drift signal (visual only)")
    p.add_argument("--gold", action="store_true",
                   help="overlay gold section boundaries (parsed from wiki headers)")
    args = p.parse_args()

    text = args.input.read_text()
    sentences, sections = re_eval.parse_wiki_sections(text)
    embeddings = re_eval.load_or_embed(args.input, sentences, ch.DEFAULT_MODEL)
    sentences, embeddings, _ = ch.filter_degenerate_embeddings(sentences, embeddings)

    drift = compute_drift(embeddings, context_window=args.ctx)
    if args.smooth > 1:
        k = args.smooth
        kernel = np.ones(k, dtype=np.float32) / k
        drift_smoothed = np.convolve(drift, kernel, mode="same")
    else:
        drift_smoothed = drift

    fixed = ch.fixed_size_chunker(sentences, embeddings, target_tokens=args.target_tokens)
    gw = ch.growing_window_chunker(sentences, embeddings, sim_threshold=args.gw_sim)
    hy = ch.hybrid_chunker(
        sentences, embeddings,
        target_tokens=args.target_tokens,
        search_window=args.hy_search,
        drift_threshold=args.hy_drift,
        context_window=args.ctx,
    )

    gold_breaks: list[int] = []
    if args.gold and sections:
        gold_breaks = sorted({end for _t, _s, end in sections[:-1]})

    panels = [
        ("fixed_size", fixed, "#555555", "--"),
        ("growing_window", gw, "#e67e22", "-"),
        ("hybrid", hy, "#c0392b", "-"),
    ]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 8.5), sharex=True)

    ymax = float(drift_smoothed.max()) * 1.10
    for ax, (name, result, color, linestyle) in zip(axes, panels):
        ax.fill_between(range(len(drift_smoothed)), 0, drift_smoothed, alpha=0.30, color="#3498db", linewidth=0)
        ax.plot(drift_smoothed, color="#2980b9", lw=1.0, alpha=0.85)
        if gold_breaks:
            for g in gold_breaks:
                ax.axvline(g, color="#27ae60", ls=":", lw=1.0, alpha=0.55)
        for bp in result.breakpoints[:-1]:
            ax.axvline(bp, color=color, ls=linestyle, lw=1.4, alpha=0.85)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(f"{name}\n({len(result.chunks)} chunks)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_lines = [
        plt.Line2D([0], [0], color="#3498db", lw=2, alpha=0.5, label="local drift signal"),
    ]
    if gold_breaks:
        legend_lines.append(plt.Line2D([0], [0], color="#27ae60", ls=":", lw=1.5,
                                       label=f"gold section boundary ({len(gold_breaks)})"))
    legend_lines.append(plt.Line2D([0], [0], color="#444", lw=1.5, label="predicted chunk boundary"))
    axes[0].legend(handles=legend_lines, loc="upper right", fontsize=9, framealpha=0.9)

    axes[-1].set_xlabel("sentence index")
    axes[0].set_title(
        f"Chunk boundary placement on {args.input.stem} "
        f"— drift signal (blue) + boundaries (colored). "
        f"Fixed: drift-blind; growing_window: drift-driven; hybrid: cadence + drift nudge.",
        fontsize=10,
    )
    fig.tight_layout()
    out = args.out or args.input.with_name(f"compare_{args.input.stem}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"plot saved -> {out}")
    print(f"chunks: fixed={len(fixed.chunks)}  growing_window={len(gw.chunks)}  hybrid={len(hy.chunks)}")


if __name__ == "__main__":
    main()
