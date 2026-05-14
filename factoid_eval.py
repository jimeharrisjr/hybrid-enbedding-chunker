"""Factoid retrieval eval.

Generates a question bank by sampling specific sentences from the document and
asking a local LLM (qwen3 via Ollama by default) to produce one factoid question
per sentence whose answer is in that sentence. Then evaluates each chunker by
embedding the questions, scoring chunks by cosine similarity, and measuring
whether the chunk containing the answer sentence appears in the top-K results.

This is a much sharper test than section-title queries because each query has
exactly one ground-truth answer location.

Two-step workflow:
    # 1) generate questions (slow — LLM call per anchor sentence; result cached)
    python factoid_eval.py paper.txt --generate --n 25

    # 2) run eval (fast, uses cached questions + cached embeddings)
    python factoid_eval.py paper.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import requests

import chunker as ch
import retrieval_eval as re_eval

OLLAMA_URL = "http://localhost:11434"
DEFAULT_QGEN_MODEL = "qwen3:32b"

QGEN_PROMPT = """You will be shown one sentence from a longer document. Write ONE specific factoid question whose answer is captured in this sentence.

Requirements:
- The question must end with a question mark.
- Make it specific enough that someone asking this exact question would want to find this sentence (not other sentences in the document).
- It must NOT contain the answer.
- Keep it under 25 words.
- Output ONLY the question, with no preface or explanation.

Sentence:
{sentence}

Question:"""


def gen_question(sentence: str, model: str) -> str | None:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": QGEN_PROMPT.format(sentence=sentence),
            "stream": False,
            "think": False,
            "options": {"temperature": 0.3, "num_predict": 200},
        },
        timeout=600,
    )
    r.raise_for_status()
    resp = r.json().get("response", "")
    resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
    for line in resp.split("\n"):
        line = line.strip().strip('"\'').lstrip("0123456789.) ").strip()
        if line.endswith("?") and len(line) > 8:
            return line
    return None


def pick_anchors(
    sentences: list[str],
    n: int,
    *,
    min_chars: int = 90,
    max_chars: int = 400,
    skip_first: int = 5,
    skip_last: int = 5,
) -> list[int]:
    """Pick n evenly-spaced 'substantive' sentence indices."""
    keep = []
    for i, s in enumerate(sentences):
        if i < skip_first or i >= len(sentences) - skip_last:
            continue
        if not (min_chars <= len(s) <= max_chars):
            continue
        if s.count("[") + s.count("]") >= 4:        # heavy citation lines
            continue
        if re.search(r"\b(et al\.|cf\.|i\.e\.)\s*$", s):
            continue
        if sum(c.isdigit() for c in s) / max(1, len(s)) > 0.25:   # equation-y
            continue
        keep.append(i)
    if not keep:
        return []
    step = max(1, len(keep) // n)
    return keep[::step][:n]


def generate_questions_for(
    sentences: list[str], out_path: Path, n: int, model: str
) -> list[dict]:
    anchors = pick_anchors(sentences, n=n)
    if not anchors:
        sys.exit("no anchor sentences passed the filters")
    print(f"selected {len(anchors)} anchor sentences", flush=True)
    rows = []
    for j, idx in enumerate(anchors):
        sent = sentences[idx]
        print(f"\n[{j+1}/{len(anchors)}] sent[{idx}]: {sent[:90]}{'...' if len(sent) > 90 else ''}", flush=True)
        q = gen_question(sent, model=model)
        if q:
            print(f"  Q: {q}", flush=True)
            rows.append({"question": q, "answer_idx": idx, "answer_sent": sent})
        else:
            print("  (no question parsed)", flush=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved {len(rows)} questions -> {out_path}")
    return rows


def realign_answer_idx(rows: list[dict], sentences: list[str]) -> list[dict]:
    """Sentence indices in saved questions assume a fixed sentence list. If we re-run
    after re-extraction or filter changes, re-anchor by matching the saved sentence
    text. Drop questions whose anchor can't be found."""
    out = []
    pos = {s: i for i, s in enumerate(sentences)}
    for row in rows:
        if row["answer_sent"] in pos:
            row = dict(row, answer_idx=pos[row["answer_sent"]])
            out.append(row)
    return out


def eval_methods(sentences: list[str], embeddings: np.ndarray, questions: list[dict]) -> None:
    q_texts = [r["question"] for r in questions]
    q_embs = ch.embed_batch(q_texts)
    q_embs = np.nan_to_num(q_embs)
    q_embs = q_embs / np.maximum(np.linalg.norm(q_embs, axis=1, keepdims=True), 1e-9)

    methods = {
        "fixed_size(512)": lambda: ch.fixed_size_chunker(sentences, embeddings, target_tokens=512),
        "pairwise(95)": lambda: ch.pairwise_chunker(sentences, embeddings, percentile=95.0),
        "growing_window(0.30)": lambda: ch.growing_window_chunker(sentences, embeddings, sim_threshold=0.30),
        "growing_window(0.40)": lambda: ch.growing_window_chunker(sentences, embeddings, sim_threshold=0.40),
        "max_min(0.35)": lambda: ch.max_min_chunker(sentences, embeddings, bootstrap_threshold=0.35, window=8),
        "hybrid(d=0.40)": lambda: ch.hybrid_chunker(sentences, embeddings, drift_threshold=0.40),
        "hybrid(d=0.30)": lambda: ch.hybrid_chunker(sentences, embeddings, drift_threshold=0.30),
    }

    n = len(questions)
    print(f"\n{'method':<22} {'#chunks':>8} {'avg_chars':>10} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>6}")
    print("-" * 76)
    for name, fn in methods.items():
        r = fn()
        ranges = re_eval.chunk_to_sentence_ranges(r.breakpoints, len(sentences))
        chunk_embs = ch.embed_batch(r.chunks)
        chunk_embs = np.nan_to_num(chunk_embs)
        chunk_embs = chunk_embs / np.maximum(np.linalg.norm(chunk_embs, axis=1, keepdims=True), 1e-9)
        scores = q_embs @ chunk_embs.T
        rankings = np.argsort(-scores, axis=1)
        hit1 = hit3 = hit5 = 0
        mrr = 0.0
        skipped = 0
        for q_idx, q in enumerate(questions):
            answer_idx = q["answer_idx"]
            relevant = {ci for ci, (cs, ce) in enumerate(ranges) if cs <= answer_idx <= ce}
            if not relevant:
                skipped += 1
                continue
            for rp, ci in enumerate(rankings[q_idx]):
                if ci in relevant:
                    if rp == 0: hit1 += 1
                    if rp < 3: hit3 += 1
                    if rp < 5: hit5 += 1
                    mrr += 1.0 / (rp + 1)
                    break
        avg_chars = int(np.mean([len(c) for c in r.chunks]))
        eval_n = n - skipped
        print(f"{name:<22} {len(r.chunks):>8} {avg_chars:>10} "
              f"{hit1/eval_n:>6.2f} {hit3/eval_n:>6.2f} {hit5/eval_n:>6.2f} {mrr/eval_n:>6.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--questions", type=Path, default=None, help="questions.json path")
    p.add_argument("--generate", action="store_true", help="generate questions before eval")
    p.add_argument("--n", type=int, default=25, help="number of questions to generate")
    p.add_argument("--qgen-model", default=DEFAULT_QGEN_MODEL)
    args = p.parse_args()

    text = args.input.read_text()
    sentences, _ = re_eval.parse_wiki_sections(text)
    embeddings = re_eval.load_or_embed(args.input, sentences, ch.DEFAULT_MODEL)
    sentences, embeddings, _ = ch.filter_degenerate_embeddings(sentences, embeddings)
    print(f"loaded {len(sentences)} sentences ({embeddings.shape[1]}-dim embeddings)")

    q_path = args.questions or args.input.with_suffix(".questions.json")
    if args.generate or not q_path.exists():
        if not args.generate and not q_path.exists():
            print(f"no existing questions at {q_path}; use --generate to create them")
            sys.exit(1)
        questions = generate_questions_for(sentences, q_path, n=args.n, model=args.qgen_model)
    else:
        questions = json.loads(q_path.read_text())
        print(f"loaded {len(questions)} questions from {q_path}")
        questions = realign_answer_idx(questions, sentences)
        print(f"  {len(questions)} questions aligned to current sentence list")

    if not questions:
        sys.exit("no questions available for eval")
    eval_methods(sentences, embeddings, questions)


if __name__ == "__main__":
    main()
