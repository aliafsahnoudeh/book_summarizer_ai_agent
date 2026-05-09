"""
Extract text, table of contents, visual elements, and metadata from one or
more PDFs into a structured ``.books/<book_id>/`` directory.

A *source* is either:
  - a single ``.pdf`` file  → one book, one PDF
  - a directory             → one book composed of all PDFs it contains
                              (recursive, sorted by path)

Output layout::

    .books/<book_id>/
    ├── metadata.json       # title, authors, language, total_pages,
    │                       #   source_files[], pdf_page_ranges[],
    │                       #   rtl_word_order_fix
    ├── toc.json            # flat list of {title, level, page}
    ├── pages/              # 1-based, zero-padded, concatenated across PDFs
    │   ├── 001.txt
    │   └── …
    └── visuals/
        ├── index.json
        └── page_NNN/
            └── img_NNN.(png|jpg|…)

When a source folder contains ``metadata.json`` with any of ``name``,
``authors``, ``language``, or ``rtl_word_order_fix``, those values override
the auto-detected ones.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from langdetect import LangDetectException, detect
from pypdf import PdfReader

from book_builder.normalize import normalize_persian_text
from book_builder.paths import BOOKS_DIR, book_dir, book_id_from_source


@dataclass
class BuildResult:
    """Summary of a build pass — useful for programmatic callers."""
    book_id: str
    book_dir: Path
    total_pages: int
    total_images: int
    toc_entries: int
    language: str


def _collect_pdfs(source: Path) -> list[Path]:
    """Return the list of PDFs for a source (single file or folder)."""
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(source.rglob("*.pdf"))
    raise FileNotFoundError(f"Source not found: {source}")


def _load_source_metadata(source: Path) -> dict:
    """
    Read a folder-level ``metadata.json`` if present. Recognized keys:
    ``name``, ``authors``, ``language``, ``rtl_word_order_fix``.
    """
    if source.is_dir():
        meta_file = source / "metadata.json"
        if meta_file.exists():
            try:
                return json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def _extract_outline(reader: PdfReader, page_offset: int = 0) -> list[dict]:
    """Flatten a single PDF's outline, offsetting page numbers by *page_offset*."""
    entries: list[dict] = []

    def _walk(items, level: int = 1):
        for item in items:
            if isinstance(item, list):
                _walk(item, level + 1)
            else:
                try:
                    local_page = reader.get_destination_page_number(item) + 1
                    page = local_page + page_offset
                except Exception:
                    page = None
                entries.append({
                    "title": item.title if hasattr(item, "title") else str(item),
                    "level": level,
                    "page": page,
                })

    try:
        if reader.outline:
            _walk(reader.outline)
    except Exception:
        pass
    return entries


def _extract_page_images(page, page_num: int, visuals_dir: Path) -> list[dict]:
    """Write embedded images from a single page to disk; return metadata records.

    Per-image failures (unsupported codecs like JBIG2 without ``jbig2dec``,
    malformed streams, etc.) are logged and skipped rather than aborting
    the whole page.
    """
    records: list[dict] = []
    try:
        # ``page.images`` is a lazy iterator; wrap it to tolerate decode
        # errors while iterating.
        images_iter = iter(page.images)
    except Exception:
        return records

    page_dir = visuals_dir / f"page_{page_num:03d}"
    idx = 0
    while True:
        try:
            image = next(images_iter)
        except StopIteration:
            break
        except Exception as e:
            print(f"[build_book]   (page {page_num}: skipped image — {e})")
            continue
        idx += 1
        img_name = image.name if hasattr(image, "name") else f"img_{idx:03d}"
        suffix = Path(img_name).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}:
            suffix = ".png"
        filename = f"img_{idx:03d}{suffix}"
        try:
            page_dir.mkdir(parents=True, exist_ok=True)
            (page_dir / filename).write_bytes(image.data)
        except Exception as e:
            print(f"[build_book]   (page {page_num}: failed to write {filename} — {e})")
            continue
        records.append({
            "filename": filename,
            "width": getattr(image, "width", None),
            "height": getattr(image, "height", None),
            "page": page_num,
        })
    return records


def build_book(source: str | Path, force: bool = False) -> BuildResult:
    """
    Build a processed book at ``.books/<book_id>/`` from *source*.

    Raises ``FileExistsError`` if the output directory already contains pages
    and *force* is False.
    """
    source = Path(source).resolve()
    pdfs = _collect_pdfs(source)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found at: {source}")

    book_id = book_id_from_source(source)
    out_dir = book_dir(book_id)
    pages_dir = out_dir / "pages"
    visuals_dir = out_dir / "visuals"

    if pages_dir.exists() and any(pages_dir.iterdir()) and not force:
        raise FileExistsError(
            f"{pages_dir} already contains data. Use force=True to overwrite."
        )

    # Clean stale reading state when rebuilding (used by book_summarizer)
    state_file = out_dir / "reading_state.json"
    if force and state_file.exists():
        state_file.unlink()
        print(f"[build_book] Removed stale {state_file.name}")

    source_meta = _load_source_metadata(source)
    rtl_fix = bool(source_meta.get("rtl_word_order_fix", False))

    pages_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build_book] Building '{book_id}' from {len(pdfs)} PDF(s)")

    # ── Pass 1: text extraction across all PDFs (global page numbering) ──
    print("[build_book] Extracting text…")
    global_page = 0
    pdf_page_ranges: list[dict] = []
    sample_text = ""
    readers: list[PdfReader] = []

    for pdf_path in pdfs:
        reader = PdfReader(str(pdf_path))
        readers.append(reader)
        start = global_page + 1
        for page in reader.pages:
            global_page += 1
            raw = page.extract_text() or ""
            text = normalize_persian_text(raw, fix_rtl_word_order=rtl_fix)
            (pages_dir / f"{global_page:03d}.txt").write_text(text, encoding="utf-8")
            if not sample_text and text.strip():
                sample_text = text[:500]
        pdf_page_ranges.append({
            "file": pdf_path.name,
            "start": start,
            "end": global_page,
        })

    total_pages = global_page
    print(f"[build_book]   → {total_pages} page files written")

    # ── TOC across all PDFs (offset by each PDF's start) ──
    print("[build_book] Extracting table of contents…")
    toc: list[dict] = []
    for reader, rng in zip(readers, pdf_page_ranges):
        toc.extend(_extract_outline(reader, page_offset=rng["start"] - 1))
    (out_dir / "toc.json").write_text(
        json.dumps(toc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[build_book]   → {len(toc)} TOC entries")

    # ── Visuals across all PDFs ──
    print("[build_book] Extracting visual elements…")
    all_visuals: dict[int, list[dict]] = {}
    total_images = 0
    for reader, rng in zip(readers, pdf_page_ranges):
        offset = rng["start"] - 1
        for local_page_idx, page in enumerate(reader.pages, start=1):
            global_num = local_page_idx + offset
            records = _extract_page_images(page, global_num, visuals_dir)
            if records:
                all_visuals[global_num] = records
                total_images += len(records)
    (visuals_dir / "index.json").write_text(
        json.dumps(all_visuals, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[build_book]   → {total_images} image(s) across {len(all_visuals)} page(s)")

    # ── Language: prefer source override, then auto-detect ──
    language = source_meta.get("language")
    if language:
        print(f"[build_book]   Language from source metadata: {language}")
    elif sample_text:
        try:
            language = detect(sample_text)
        except LangDetectException:
            language = "unknown"
        print(f"[build_book]   Auto-detected language: {language}")
    else:
        language = "unknown"

    # ── Title / authors: source override wins, else PDF metadata, else stem ──
    if source_meta.get("name"):
        title = source_meta["name"]
    else:
        title = None
        if pdfs:
            try:
                info = readers[0].metadata
                if info and info.title:
                    title = info.title
            except Exception:
                pass
        if not title:
            stem = source.stem if source.is_file() else source.name
            title = stem.replace("_", " ").replace("-", " ").strip()

    authors = source_meta.get("authors")
    if not authors:
        try:
            info = readers[0].metadata
            authors = [info.author] if (info and info.author) else []
        except Exception:
            authors = []

    metadata = {
        "book_id": book_id,
        "title": title,
        "authors": authors,
        "language": language,
        "total_pages": total_pages,
        "source_files": [p.name for p in pdfs],
        "pdf_page_ranges": pdf_page_ranges,
        "rtl_word_order_fix": rtl_fix,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[build_book]   → metadata.json written")
    print(f"[build_book] Done. Output: {out_dir.relative_to(BOOKS_DIR.parent)}")

    return BuildResult(
        book_id=book_id,
        book_dir=out_dir,
        total_pages=total_pages,
        total_images=total_images,
        toc_entries=len(toc),
        language=language,
    )
