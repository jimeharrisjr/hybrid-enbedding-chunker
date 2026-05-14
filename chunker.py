"""Compare semantic chunking strategies using EmbeddingGemma via Ollama.

Usage:
    python chunker.py                       # run on built-in heterogeneous demo text
    python chunker.py path/to/document.txt  # run on your own text file
    python chunker.py doc.txt --show-chunks # also print chunk previews

Implements four chunkers for side-by-side comparison:
    fixed_size      Naive token-count packing with overlap (baseline).
    pairwise        Kamradt-style: split at large jumps in adjacent-sentence distance.
    growing_window  Cumulative: split when next sentence diverges from running chunk mean.
    max_min         Candidate joins if its max-sim to chunk exceeds chunk's min intra-sim.
"""

import argparse
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "embeddinggemma"

DEMO_TEXT = """\
Sourdough relies on a symbiotic culture of wild yeast and lactobacilli. The starter
must be fed regularly with flour and water to keep the microbes active. A mature
starter doubles in size within four to six hours of feeding. Hydration ratios
affect both flavor development and crumb structure in the finished loaf. Bakers
often retard the dough overnight in the refrigerator to deepen the sour notes.

Neutron stars form when massive stars exhaust their nuclear fuel and collapse.
Their cores reach densities comparable to atomic nuclei. A teaspoon of neutron-star
material would weigh several billion tons on Earth. Pulsars are rapidly rotating
neutron stars whose beamed radiation sweeps past us like a cosmic lighthouse.
Some millisecond pulsars rotate hundreds of times per second.

Domestic cats descend from the African wildcat and were tamed roughly ten thousand
years ago. They retain many predatory behaviors despite millennia of cohabitation
with humans. Cats spend a large fraction of their day grooming, which helps
regulate body temperature. Their retractable claws and excellent low-light vision
are adaptations for nocturnal hunting. Most house cats sleep between twelve and
sixteen hours a day.
"""


# ---------------------------------------------------------------------------
# Sentence splitting + embedding
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Paragraph-aware sentence splitter using regex."""
    sentences = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", para)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


WIKI_HEADER_RE = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$")


def split_sentences_with_wiki_headers(text: str) -> tuple[list[str], list[int]]:
    """Like split_sentences, but treats `== Section ==` paragraphs as boundary markers.
    Returns (sentences, gold_breakpoints) where gold_breakpoints are sentence indices
    (the last index of each section as seen in the output) — i.e., a header between
    paragraphs records the boundary at len(sentences) - 1 just before its next content.
    """
    sentences: list[str] = []
    gold: list[int] = []
    saw_header = False
    for para in re.split(r"\n\s*\n", text):
        para_stripped = para.strip()
        if not para_stripped:
            continue
        if WIKI_HEADER_RE.match(para_stripped):
            if sentences:                      # header position = end of previous section
                gold.append(len(sentences) - 1)
            saw_header = True
            continue
        para_norm = re.sub(r"\s+", " ", para_stripped)
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", para_norm)
        sentences.extend(p.strip() for p in parts if p.strip())
        saw_header = False
    # dedupe + drop trailing boundary at final sentence index
    seen = set()
    gold_clean = []
    final = len(sentences) - 1
    for g in gold:
        if g in seen or g == final or g < 0:
            continue
        seen.add(g)
        gold_clean.append(g)
    return sentences, sorted(gold_clean)


def embed_batch(texts: list[str], model: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed a batch of texts via Ollama's /api/embed endpoint."""
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": model, "input": texts},
        timeout=600,
    )
    r.raise_for_status()
    return np.asarray(r.json()["embeddings"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    name: str
    chunks: list[str]
    breakpoints: list[int]                  # last sentence index of each chunk
    signal: np.ndarray | None = None        # per-step value to plot (distance/similarity)
    signal_label: str = ""
    threshold: float | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------

def fixed_size_chunker(
    sentences: list[str],
    embeddings: np.ndarray,
    target_tokens: int = 512,
    overlap_tokens: int = 100,
) -> ChunkResult:
    """Pack sentences greedily into ~target_tokens chunks; back off by overlap_tokens."""
    sizes = [max(1, len(s) // 4) for s in sentences]   # ~4 chars/token approx
    chunks, breakpoints = [], []
    i = 0
    n = len(sentences)
    while i < n:
        cum = 0
        j = i
        while j < n and (cum + sizes[j] <= target_tokens or j == i):
            cum += sizes[j]
            j += 1
        chunks.append(" ".join(sentences[i:j]))
        breakpoints.append(j - 1)
        if j >= n:
            break
        back = 0
        new_i = j
        while new_i > i + 1 and back < overlap_tokens:
            new_i -= 1
            back += sizes[new_i]
        i = new_i
    return ChunkResult(name="fixed_size", chunks=chunks, breakpoints=breakpoints)


def pairwise_chunker(
    sentences: list[str],
    embeddings: np.ndarray,
    percentile: float = 95.0,
    buffer: int = 1,
) -> ChunkResult:
    """Kamradt-style: smooth each sentence with a buffer of neighbors, split at large
    cosine-distance jumps between consecutive sentence embeddings."""
    if buffer > 0:
        smoothed = np.zeros_like(embeddings)
        for i in range(len(embeddings)):
            lo, hi = max(0, i - buffer), min(len(embeddings), i + buffer + 1)
            smoothed[i] = embeddings[lo:hi].mean(axis=0)
        embs = smoothed
    else:
        embs = embeddings
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    distances = np.array(
        [1.0 - float(embs[i] @ embs[i + 1]) for i in range(len(embs) - 1)],
        dtype=np.float32,
    )
    threshold = float(np.percentile(distances, percentile))
    breakpoints = [i for i, d in enumerate(distances) if d >= threshold]
    if not breakpoints or breakpoints[-1] != len(sentences) - 1:
        breakpoints.append(len(sentences) - 1)
    chunks, start = [], 0
    for bp in breakpoints:
        chunks.append(" ".join(sentences[start : bp + 1]))
        start = bp + 1
    return ChunkResult(
        name="pairwise",
        chunks=chunks,
        breakpoints=breakpoints,
        signal=distances,
        signal_label="cosine distance to next sentence",
        threshold=threshold,
    )


def growing_window_chunker(
    sentences: list[str],
    embeddings: np.ndarray,
    sim_threshold: float = 0.6,
    min_chunk: int = 2,
) -> ChunkResult:
    """Cumulative: maintain running mean embedding of current chunk; split when the
    next sentence's similarity to that mean drops below sim_threshold."""
    embs = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-9)
    chunks, breakpoints, drift = [], [], []
    start = 0
    chunk_emb = embs[0].copy()
    chunk_count = 1
    for i in range(1, len(embs)):
        # sim between running mean and unit-norm next sentence
        sim = float(np.dot(chunk_emb, embs[i]) / (np.linalg.norm(chunk_emb) + 1e-12))
        drift.append(1.0 - sim)
        if sim < sim_threshold and chunk_count >= min_chunk:
            chunks.append(" ".join(sentences[start:i]))
            breakpoints.append(i - 1)
            start = i
            chunk_emb = embs[i].copy()
            chunk_count = 1
        else:
            chunk_emb = (chunk_emb * chunk_count + embs[i]) / (chunk_count + 1)
            chunk_count += 1
    chunks.append(" ".join(sentences[start:]))
    breakpoints.append(len(sentences) - 1)
    return ChunkResult(
        name="growing_window",
        chunks=chunks,
        breakpoints=breakpoints,
        signal=np.asarray(drift, dtype=np.float32),
        signal_label="1 - cos(running chunk mean, next sent)",
        threshold=1.0 - sim_threshold,
        extra={"sim_threshold": sim_threshold, "min_chunk": min_chunk},
    )


def max_min_chunker(
    sentences: list[str],
    embeddings: np.ndarray,
    bootstrap_threshold: float = 0.5,
    window: int = 8,
) -> ChunkResult:
    """Max–Min with a recency window: candidate sentence joins iff its max similarity
    to the last `window` chunk members exceeds the (windowed) min pairwise similarity
    inside the chunk. Windowing keeps the floor from collapsing to ~0 on long chunks."""
    embs = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-9)
    sim = embs @ embs.T
    chunks, breakpoints, candidate_sims, baseline = [], [], [], []
    start = 0
    for i in range(1, len(embs)):
        win_start = max(start, i - window)
        chunk_idxs = list(range(win_start, i))
        cand_sim = float(sim[i, chunk_idxs].max())
        if len(chunk_idxs) == 1:
            base = bootstrap_threshold
        else:
            sub = sim[np.ix_(chunk_idxs, chunk_idxs)]
            mask = ~np.eye(len(chunk_idxs), dtype=bool)
            base = float(sub[mask].min())
        candidate_sims.append(cand_sim)
        baseline.append(base)
        if cand_sim < base:
            chunks.append(" ".join(sentences[start:i]))
            breakpoints.append(i - 1)
            start = i
    chunks.append(" ".join(sentences[start:]))
    breakpoints.append(len(sentences) - 1)
    return ChunkResult(
        name="max_min",
        chunks=chunks,
        breakpoints=breakpoints,
        signal=np.asarray(candidate_sims, dtype=np.float32),
        signal_label="max sim(candidate, recent chunk)",
        extra={"baseline": np.asarray(baseline, dtype=np.float32)},
    )


def hybrid_chunker(
    sentences: list[str],
    embeddings: np.ndarray,
    target_tokens: int = 512,
    search_window: int = 5,
    drift_threshold: float = 0.4,
    context_window: int = 5,
) -> ChunkResult:
    """Greedy fixed-size packing where each proposed chunk boundary can be nudged
    ±search_window sentences to land on a local sliding-window drift peak, if that
    peak exceeds drift_threshold. Goal: keep uniform chunk sizes (good for retrieval
    keyword coverage) while honoring real topic shifts (good for chunk coherence)."""
    embs = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-9)
    n = len(embs)

    drift = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        lo = max(0, i - context_window)
        ctx = embs[lo:i].mean(axis=0)
        ctx = ctx / (np.linalg.norm(ctx) + 1e-9)
        drift[i] = 1.0 - float(np.dot(ctx, embs[i]))

    sizes = [max(1, len(s) // 4) for s in sentences]
    chunks, breakpoints = [], []
    nudges = 0
    i = 0
    while i < n:
        cum, j = 0, i
        while j < n and (cum + sizes[j] <= target_tokens or j == i):
            cum += sizes[j]
            j += 1
        proposed_end = j - 1
        actual_end = proposed_end
        if proposed_end < n - 1:
            lo = max(i + 1, proposed_end - search_window + 1)
            hi = min(n - 1, proposed_end + search_window)
            if hi >= lo:
                local = drift[lo:hi + 1]
                best_offset = int(np.argmax(local))
                if float(local[best_offset]) >= drift_threshold:
                    candidate = max(i, lo + best_offset - 1)
                    if candidate != proposed_end:
                        nudges += 1
                    actual_end = candidate
        chunks.append(" ".join(sentences[i:actual_end + 1]))
        breakpoints.append(actual_end)
        i = actual_end + 1
    return ChunkResult(
        name="hybrid",
        chunks=chunks,
        breakpoints=breakpoints,
        signal=drift,
        signal_label=f"local drift (ctx={context_window})",
        threshold=drift_threshold,
        extra={"nudges": nudges},
    )


def filter_degenerate_embeddings(
    sentences: list[str], embeddings: np.ndarray, eps: float = 1e-6
) -> tuple[list[str], np.ndarray, list[int]]:
    """Drop sentences whose embedding has near-zero norm or contains NaN/Inf
    (Ollama can produce these on empty/whitespace/control-char inputs)."""
    finite = np.isfinite(embeddings).all(axis=1)
    embeddings_safe = np.where(finite[:, None], embeddings, 0)
    norms = np.linalg.norm(embeddings_safe, axis=1)
    keep_mask = finite & (norms > eps)
    dropped = [i for i, k in enumerate(keep_mask) if not k]
    return (
        [s for s, k in zip(sentences, keep_mask) if k],
        embeddings[keep_mask],
        dropped,
    )


def evaluate_breakpoints(
    predicted: list[int], gold: list[int], tolerance: int = 2
) -> dict:
    """Match each predicted breakpoint to the nearest gold breakpoint within
    `tolerance` sentences. Each gold can only match once. Returns precision, recall,
    F1, and the number of predictions/golds. Excludes the trailing end-of-doc index."""
    preds = sorted(set(predicted))
    golds = sorted(set(gold))
    matched_golds: set[int] = set()
    tp = 0
    for p in preds:
        best, best_d = None, tolerance + 1
        for g in golds:
            if g in matched_golds:
                continue
            d = abs(p - g)
            if d < best_d:
                best_d, best = d, g
        if best is not None and best_d <= tolerance:
            matched_golds.add(best)
            tp += 1
    n_pred, n_gold = len(preds), len(golds)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "n_pred": n_pred, "n_gold": n_gold,
        "precision": precision, "recall": recall, "f1": f1,
        "tolerance": tolerance,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(
    results: list[ChunkResult],
    output_path: Path,
    gold: list[int] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    plottable = [r for r in results if r.signal is not None]
    n = len(plottable)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, plottable):
        ax.plot(r.signal, lw=1.0, color="steelblue", label=r.signal_label)
        if r.threshold is not None:
            ax.axhline(
                r.threshold, color="orange", ls="--", lw=0.9,
                label=f"threshold={r.threshold:.3f}",
            )
        if "baseline" in r.extra:
            ax.plot(r.extra["baseline"], lw=0.8, color="orange", ls="--", label="min intra sim")
        if gold:
            for g in gold:
                ax.axvline(g, color="green", alpha=0.5, lw=1.2, ls=":")
            ax.axvline(gold[0], color="green", alpha=0.5, lw=1.2, ls=":", label="gold (section header)")
        for bp in r.breakpoints[:-1]:
            ax.axvline(bp, color="red", alpha=0.35, lw=0.7)
        ax.set_title(f"{r.name} — {len(r.chunks)} chunks")
        ax.set_ylabel("signal")
        ax.legend(loc="upper right", fontsize=8)
    xlabel = "sentence index (red = predicted boundary"
    if gold:
        xlabel += ", green dotted = gold section header"
    xlabel += ")"
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"\nplot saved -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", type=Path, nargs="?", help="input text file (omit for demo)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=Path("chunks_plot.png"))
    p.add_argument("--show-chunks", action="store_true", help="print chunk previews")
    p.add_argument("--pct", type=float, default=95.0, help="pairwise percentile threshold")
    p.add_argument("--gw-sim", type=float, default=0.6, help="growing-window similarity threshold")
    p.add_argument("--mm-bootstrap", type=float, default=0.5, help="max-min bootstrap threshold")
    p.add_argument("--mm-window", type=int, default=8, help="max-min recency window")
    p.add_argument("--hy-drift", type=float, default=0.4, help="hybrid drift threshold")
    p.add_argument("--hy-search", type=int, default=5, help="hybrid ± search window")
    p.add_argument("--hy-ctx", type=int, default=5, help="hybrid context window for drift")
    p.add_argument("--wiki", action="store_true", help="treat input as Wikipedia plain text: strip section headers and use them as gold breakpoints")
    p.add_argument("--tol", type=int, default=2, help="±sentence tolerance for breakpoint evaluation")
    args = p.parse_args()

    text = args.input.read_text() if args.input else DEMO_TEXT

    gold: list[int] = []
    if args.wiki:
        sentences, gold = split_sentences_with_wiki_headers(text)
        print(f"{len(sentences)} sentences, {len(gold)} gold section boundaries")
    else:
        sentences = split_sentences(text)
        print(f"{len(sentences)} sentences")
    if not sentences:
        raise SystemExit("no sentences found in input")

    cache_path = args.input.with_suffix(".embeddings.npz") if args.input else None
    if cache_path and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if cached["sentences"].tolist() == sentences and cached["model"] == args.model:
            print(f"loaded cached embeddings from {cache_path.name}")
            embeddings = cached["embeddings"]
        else:
            print(f"cache mismatch, re-embedding...")
            embeddings = embed_batch(sentences, model=args.model)
    else:
        print(f"embedding via Ollama ({args.model})...")
        embeddings = embed_batch(sentences, model=args.model)
    if cache_path:
        np.savez(cache_path, sentences=np.asarray(sentences, dtype=object), embeddings=embeddings, model=args.model)
    sentences, embeddings, dropped = filter_degenerate_embeddings(sentences, embeddings)
    if dropped:
        print(f"  dropped {len(dropped)} zero-norm embeddings at indices {dropped[:10]}{'...' if len(dropped) > 10 else ''}")
        # shift gold indices to account for dropped earlier indices
        gold = [g - sum(1 for d in dropped if d <= g) for g in gold if g not in dropped]
    print(f"  embeddings shape={embeddings.shape}")

    results = [
        fixed_size_chunker(sentences, embeddings),
        pairwise_chunker(sentences, embeddings, percentile=args.pct),
        growing_window_chunker(sentences, embeddings, sim_threshold=args.gw_sim),
        max_min_chunker(sentences, embeddings, bootstrap_threshold=args.mm_bootstrap, window=args.mm_window),
        hybrid_chunker(sentences, embeddings, drift_threshold=args.hy_drift,
                       search_window=args.hy_search, context_window=args.hy_ctx),
    ]

    print()
    header = f"{'method':<16} {'#chunks':>8} {'avg_chars':>10} {'min':>6} {'max':>6}"
    if gold:
        header += f"  {'P':>6} {'R':>6} {'F1':>6}  (tol=±{args.tol})"
    print(header)
    print("-" * len(header))
    for r in results:
        L = [len(c) for c in r.chunks]
        row = f"{r.name:<16} {len(r.chunks):>8} {np.mean(L):>10.0f} {min(L):>6} {max(L):>6}"
        if gold:
            preds_no_end = r.breakpoints[:-1] if r.breakpoints else []
            ev = evaluate_breakpoints(preds_no_end, gold, tolerance=args.tol)
            row += f"  {ev['precision']:>6.2f} {ev['recall']:>6.2f} {ev['f1']:>6.2f}"
        print(row)

    if args.show_chunks:
        for r in results:
            print(f"\n=== {r.name} ===")
            for i, c in enumerate(r.chunks):
                preview = c[:140].replace("\n", " ")
                print(f"  [{i}] {preview}{'...' if len(c) > 140 else ''}")

    plot_results(results, args.out, gold=gold or None)


if __name__ == "__main__":
    main()
