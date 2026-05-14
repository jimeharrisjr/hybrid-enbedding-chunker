# embedding-chunker

An empirical comparison of five chunking strategies for retrieval-augmented generation (RAG), built around Google's [EmbeddingGemma](https://ollama.com/library/embeddinggemma) served locally via [Ollama](https://ollama.com). Includes a reproducible test harness, four document corpora, two evaluation methodologies, and a research-paper writeup.

**TL;DR finding:** A **hybrid chunker** that uses cumulative-embedding-drift signals to nudge fixed-size chunk boundaries beats both pure fixed-size chunking and the published pure-semantic methods (pairwise, growing-window, max–min) on factoid retrieval across coherent encyclopedia content, narrative fiction, and a math paper. It wins six of six head-to-head retrieval metrics on the documents tested, by 14–24% MRR over fixed-size, while keeping chunk sizes in a usable 1,800–2,100 character band.

The full writeup with methodology, results tables, comparison figures, and discussion is in **[REPORT.md](REPORT.md)** (also rendered to [REPORT.pdf](REPORT.pdf)).

## Quick start

Prerequisites: Python 3.10+, [Ollama](https://ollama.com) installed and running.

```bash
# 1. pull the embedding model and (optionally) the question-generation model
ollama pull embeddinggemma
ollama pull qwen3:32b               # only needed for the factoid eval

# 2. set up Python env
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. run a chunker on the included Octopus corpus
.venv/bin/python chunker.py octopus.txt --wiki \
    --pct 90 --gw-sim 0.30 --hy-drift 0.40 --tol 2 --out octopus_plot.png

# 4. visualize how the three core strategies place boundaries differently
.venv/bin/python compare_plot.py octopus.txt --gold --out compare_octopus.png

# 5. run the factoid retrieval eval (uses cached questions in octopus.questions.json)
.venv/bin/python factoid_eval.py octopus.txt
```

## Use it on your own documents

```python
from chunker import split_sentences, embed_batch, hybrid_chunker

sentences = split_sentences(open("mydoc.txt").read())
embeddings = embed_batch(sentences, model="embeddinggemma")
result = hybrid_chunker(
    sentences, embeddings,
    target_tokens=512,        # match your retrieval index chunk size
    search_window=5,          # ± sentences boundary may be nudged
    drift_threshold=0.40,     # robust default for EmbeddingGemma
    context_window=5,         # sliding context for drift signal
)
chunks = result.chunks        # ready for indexing
```

The sentence embeddings produced for chunking are reusable for retrieval indexing, so the marginal cost of any embedding-based chunker over fixed-size in a deployed RAG pipeline is zero.

## Project layout

| File | Purpose |
|---|---|
| `chunker.py` | Five chunkers: `fixed_size`, `pairwise`, `growing_window`, `max_min`, `hybrid`. CLI runs boundary detection + plotting against gold breakpoints. |
| `compare_plot.py` | Three-panel comparison figure showing how fixed_size, growing_window, and hybrid place boundaries on the same drift signal. |
| `sweep.py` | Threshold sweep over each chunker's primary parameter, scored against gold. |
| `retrieval_eval.py` | Section-title query retrieval eval. *(Weak metric — see report for caveats.)* |
| `factoid_eval.py` | LLM-generated factoid query retrieval eval — the authoritative metric. Uses qwen3 via Ollama for question generation; questions are cached. |
| `preprocess.py` | Markdown novels and arXiv PDFs → wiki-style format with section markers. PDFs use PyMuPDF + TOC anchors + position-based header/footer cropping. |
| `stitch.py` | Fetch and stitch N Wikipedia articles into a heterogeneous test corpus. |
| `REPORT.md` / `REPORT.pdf` | The full research-paper writeup. |

## Bundled corpora

| Corpus | Source | Sentences | Type |
|---|---|---|---|
| `octopus.txt` | Wikipedia "Octopus" plain-text extract | 322 | Coherent encyclopedic |
| `stitched.txt` | Octopus + French Revolution + Photosynthesis + Jazz, 50 sentences each | 193 | Heterogeneous |
| `paper.txt` | Mezić, *On Numerical Approximations of the Koopman Operator*, arXiv 2009.05883 | 547 | Vocabulary-coherent technical |

Cached sentence embeddings (`*.embeddings.npz`) and LLM-generated factoid question banks (`*.questions.json`) are included so the evals reproduce in seconds rather than minutes.

A fourth document — a private fiction work-in-progress — was used during development. Its source, embeddings, questions, and plot are gitignored; only the aggregate result numbers appear in the report.

## Reproducing the report

```bash
# Boundary detection + threshold sweep + retrieval evals on all bundled corpora
for doc in octopus stitched paper; do
    .venv/bin/python chunker.py $doc.txt --wiki \
        --pct 90 --gw-sim 0.3 --hy-drift 0.4 --tol 2 --out ${doc}_plot.png
    .venv/bin/python sweep.py $doc.txt --tol 2
    .venv/bin/python retrieval_eval.py $doc.txt
    .venv/bin/python factoid_eval.py $doc.txt          # uses cached questions
done

# Render the report PDF (requires pandoc + xelatex + R/rmarkdown)
export PATH="/opt/homebrew/Cellar/pandoc/3.9/bin:$PATH"
Rscript -e 'rmarkdown::render("REPORT.md", \
    output_format=rmarkdown::pdf_document(latex_engine="xelatex"), \
    output_file="REPORT.pdf")'
```

## Citation

If you use this code or build on the findings, please cite the report:

> Harris, J. E. *Hybrid Semantic Chunking with EmbeddingGemma: An Empirical Study.* 2026.

## Attribution of bundled data

- `octopus.txt` and the four sources used by `stitch.py` are plain-text extracts from English Wikipedia, available under the CC BY-SA 4.0 license.
- `paper.txt` is a text extraction (via PyMuPDF) from Mezić, *On Numerical Approximations of the Koopman Operator*, arXiv:2009.05883v1, included for reproducibility of the experiments. The original PDF is at <https://arxiv.org/abs/2009.05883>; consult the arXiv page for licensing.

## License

A license file has not been added yet. Until one is, default copyright applies to the code. If you'd like to release the code under an open-source license, add a `LICENSE` file (MIT and Apache 2.0 are common choices for research code) and reference it here.
