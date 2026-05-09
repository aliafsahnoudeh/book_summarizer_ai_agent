# Book Builder

A reusable tool that turns one or more PDFs into a structured book directory with extracted text, table of contents, embedded images, and a per-book ChromaDB index. Handles English, Persian, Arabic, and other scripts.

Output lives at `<project_root>/.books/<book_id>/`.

> **Scope.** This tool only *builds* the artifacts and exposes a low-level ChromaDB accessor. Retrieval strategy (top-k, citation formatting, query expansion, missing-book handling) is the consumer's responsibility.

## Install

The tool ships with this repo — no extra install. Make sure dependencies are synced:

```bash
uv sync
```

## Build a single-PDF book

```bash
uv run build-book path/to/book.pdf
```

## Build a multi-PDF book

Put the PDFs in one folder and pass the folder:

```bash
uv run build-book "path/to/MyBook/"
```

All PDFs inside (recursively) become **one book**. Pages are concatenated in sorted path order — the first PDF's pages are `1..N`, the next's are `N+1..M`, etc. Name files predictably (`part1.pdf`, `part2.pdf`, …) if reading order matters.

### Optional source `metadata.json`

Drop a `metadata.json` in the folder to override the auto-detected title, authors, language, or Persian word-order fix:

```json
{
  "name": "My Book Title",
  "authors": ["Author Name"],
  "language": "fa",
  "rtl_word_order_fix": false
}
```

Use `rtl_word_order_fix: true` only for Persian PDFs where the text comes out with words in reversed reading order.

## Flags

| flag | effect |
|---|---|
| `--force` | overwrite an existing `.books/<id>/` |
| `--no-embed` | skip the ChromaDB build (extraction only) |
| `--embed-only <book_id>` | (re)build just the ChromaDB for an already-extracted book |

## Output layout

```
.books/<book_id>/
├── metadata.json       # title, authors, language, total_pages, source_files[], pdf_page_ranges[]
├── toc.json            # flat list of {title, level, page}
├── pages/001.txt …     # one file per page, 1-based, zero-padded
├── visuals/
│   ├── index.json      # {page_number: [{filename, width, height, page}, …]}
│   └── page_NNN/       # extracted images
└── chroma_db/          # per-book ChromaDB (one collection)
```

`pdf_page_ranges[]` maps a global page back to its source PDF:

```json
"pdf_page_ranges": [
  {"file": "part1.pdf", "start": 1,   "end": 120},
  {"file": "part2.pdf", "start": 121, "end": 240}
]
```

A citation like `[part2.pdf — Page 178]` from a search means 58 pages into `part2.pdf`.

## Programmatic use

```python
from book_builder import (
    build_book,
    build_vector_db,
    BookReader,
    list_books,
    get_collection,
)

# Build a book (extraction only)
result = build_book("path/to/book.pdf")           # → BuildResult(book_id=..., …)

# Build its ChromaDB index
build_vector_db(result.book_id)

# Read extracted artifacts
reader = BookReader(result.book_id)
reader.metadata()
reader.table_of_contents()
reader.content(1, 30)                              # concatenated page text
reader.search_keywords("Parthian")                 # substring search
reader.page_visuals(17)                            # image metadata for a page

# Low-level ChromaDB access (build your own retrieval on top)
col = get_collection(result.book_id)
col.query(query_texts=["fall of the empire"], n_results=5)

# Enumerate all processed books
list_books()
```

## Who uses it

- [book_summarizer](../book_summarizer/) — picks a `book_id`, reads pages + TOC, produces a multi-level summary.
