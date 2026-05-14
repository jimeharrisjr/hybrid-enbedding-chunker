"""Convert markdown and PDF-extracted text into wiki-style format the chunker
understands. Headings become `== Title ==` paragraphs so --wiki picks them up
as gold breakpoints.

Usage:
    python preprocess.py novel  "/path/to/file.md"  novel.txt
    python preprocess.py paper  "/path/to/file.pdf" paper.txt
"""

import argparse
import re
from pathlib import Path


def convert_novel(text: str) -> str:
    """Markdown novel with `# Title`, `## Title`, `N. # Title` headings and
    `\\*\\*\\*` (escaped) or `***` scene breaks. All become wiki section markers."""
    out_lines: list[str] = []
    scene_idx = 0
    for raw in text.split("\n"):
        line = raw.rstrip()
        stripped = line.strip()

        m = re.match(r"^#{1,4}\s+(.+)$", stripped)
        if m:
            title = m.group(1).strip().rstrip("#").strip()
            if title:
                out_lines.append(f"\n\n== {title} ==\n")
            continue

        m = re.match(r"^\d+\.\s+#\s+(.+)$", stripped)
        if m:
            title = m.group(1).strip()
            out_lines.append(f"\n\n== {title} ==\n")
            continue

        if re.match(r"^(\\\*){2,}\\?\*?$|^\*{3,}$", stripped):
            scene_idx += 1
            out_lines.append(f"\n\n== scene {scene_idx} ==\n")
            continue

        out_lines.append(line)

    text = "\n".join(out_lines)
    # tidy
    text = re.sub(r"\\\*", "*", text)            # unescape stray escaped asterisks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def convert_paper(pdf_path: Path) -> str:
    """arXiv PDF → wiki text. Uses PyMuPDF: crop running headers/footers by
    position, dedupe boilerplate, fix line-break hyphenation, then insert section
    markers at each TOC title's first body occurrence."""
    import fitz
    from collections import Counter

    doc = fitz.open(str(pdf_path))
    toc = doc.get_toc()

    # 1. Crop top/bottom margins (running headers, footers, page numbers).
    pages = []
    for page in doc:
        r = page.rect
        body = fitz.Rect(r.x0, r.y0 + 50, r.x1, r.y1 - 50)
        pages.append(page.get_text("text", clip=body))

    # 2. Detect boilerplate that survived the crop (e.g., paper title repeated on
    # every page within the body area).
    line_counter: Counter[str] = Counter()
    for pg in pages:
        for ln in (l.strip() for l in pg.split("\n")):
            if 4 <= len(ln) <= 80:                  # multi-char only — keep bare digits
                line_counter[ln] += 1
    threshold = max(3, int(len(pages) * 0.40))
    boilerplate = {ln for ln, c in line_counter.items() if c >= threshold}

    cleaned_pages = []
    for pg in pages:
        kept = []
        for ln in pg.split("\n"):
            if ln.strip() in boilerplate:
                continue
            kept.append(ln)
        cleaned_pages.append("\n".join(kept))
    full = "\n\n".join(cleaned_pages)

    # 3. Fix hyphenation; collapse runs of spaces but preserve newlines.
    full = re.sub(r"-\n(\w)", r"\1", full)
    full = re.sub(r"[ \t]+", " ", full)
    full = re.sub(r"\n{3,}", "\n\n", full)

    # 4. Skip TOC by jumping to the second occurrence of the first section title
    # without its number prefix (more robust to PDF quirks like "1\nIntroduction").
    if toc:
        first_title_clean = re.sub(r"^\d+(\.\d+)?\s+", "", toc[0][1].strip())
        pat = re.compile(re.escape(first_title_clean), re.IGNORECASE)
        matches = list(pat.finditer(full))
        if len(matches) >= 2:
            full = full[matches[1].start():]
        elif matches:
            full = full[matches[0].start():]

    # 5. Insert section markers at each TOC title's first occurrence in remaining
    # text. Match without the numeric prefix to tolerate "1\nIntroduction" splits.
    out_parts: list[str] = []
    cursor = 0
    for _level, title, _page in toc:
        title = title.strip()
        title_no_num = re.sub(r"^\d+(\.\d+)?\s+", "", title)
        pat = re.compile(re.escape(title_no_num), re.IGNORECASE)
        m = pat.search(full, cursor)
        if m is None:
            continue
        out_parts.append(full[cursor:m.start()].rstrip())
        out_parts.append(f"\n\n== {title} ==\n\n")
        cursor = m.end()
    out_parts.append(full[cursor:].strip())

    text = "\n".join(out_parts)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["novel", "paper"])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()

    if args.mode == "novel":
        text = args.input.read_text()
        out = convert_novel(text)
    else:
        out = convert_paper(args.input)
    args.output.write_text(out)
    headers = len(re.findall(r"^==\s+.+\s+==\s*$", out, re.MULTILINE))
    print(f"wrote {args.output}  ({len(out)} chars, {headers} section markers)")


if __name__ == "__main__":
    main()
