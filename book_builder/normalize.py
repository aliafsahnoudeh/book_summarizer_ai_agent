"""
Text normalization for PDF-extracted content.

The Persian pipeline addresses two common problems in PDFs:

1. **Legacy presentation forms** — old PDF generators encode Persian glyphs as
   pre-composed, positional Unicode codepoints (U+FB50–FDFF, U+FE70–FEFF)
   instead of the base characters (U+0600–06FF). NFKC normalization maps them
   back to canonical forms so the text is human-readable and searchable.

2. **Reversed word order** (opt-in via ``fix_rtl_word_order``) — some PDFs store
   right-to-left text in visual/LTR order, so ``pypdf`` emits words in reverse
   reading order. When enabled, each predominantly-Persian line has its
   whitespace-delimited tokens reversed.

Additionally:

- **Character canonicalization** — Yeh variants (ي/ى) → Farsi Yeh (ی); Arabic
  Kaf (ك) → Farsi Kaf (ک); Alef-with-Hamza variants → plain Alef (ا);
  zero-width characters (ZWNJ/ZWJ/BOM) are stripped.
- **Tatweel cleanup** — runs of ـ (U+0640) are collapsed; stand-alone tatweels
  are removed.

Apply the same pipeline at index time AND query time so normalization is
symmetric — a query for ``خشیارشا`` matches regardless of which Yeh variant
the PDF used.
"""

import re
import unicodedata


def _is_rtl_char(ch: str) -> bool:
    """Return True if *ch* is in the Persian/Arabic Unicode ranges."""
    cp = ord(ch)
    return (
        0x0600 <= cp <= 0x06FF        # Persian base characters
        or 0x0750 <= cp <= 0x077F     # Arabic Supplement
        or 0xFB50 <= cp <= 0xFDFF     # Arabic Presentation Forms-A
        or 0xFE70 <= cp <= 0xFEFF     # Arabic Presentation Forms-B
    )


def normalize_persian_text(text: str, fix_rtl_word_order: bool = False) -> str:
    """Normalize text extracted from PDFs that were typeset with older Persian tools."""
    # Step 1: NFKC — maps presentation forms back to base characters
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Persian character canonicalization
    text = text.replace("\u064a", "\u06cc")  # Arabic Yeh → Farsi Yeh
    text = text.replace("\u0649", "\u06cc")  # Alef Maqsura → Farsi Yeh
    text = text.replace("\u0643", "\u06a9")  # Arabic Kaf → Farsi Kaf
    text = text.replace("\u0623", "\u0627")  # Alef w/ Hamza Above → Alef
    text = text.replace("\u0625", "\u0627")  # Alef w/ Hamza Below → Alef
    text = text.replace("\u0671", "\u0627")  # Alef Wasla → Alef
    text = text.replace("\u200c", "")        # ZWNJ
    text = text.replace("\u200d", "")        # ZWJ
    text = text.replace("\ufeff", "")        # BOM

    # Step 3: tatweel cleanup
    text = re.sub(r"\u0640{2,}", "\u0640", text)
    text = re.sub(r"(?<=\s)\u0640+(?=\s)", "", text)
    text = re.sub(r"^\u0640+\s", "", text, flags=re.MULTILINE)

    # Step 4: optional RTL word-order reversal
    if fix_rtl_word_order:
        fixed_lines = []
        for line in text.splitlines():
            rtl_count = sum(1 for ch in line if _is_rtl_char(ch))
            alpha_count = sum(1 for ch in line if ch.isalpha())
            if alpha_count > 0 and rtl_count / alpha_count > 0.66:
                line = " ".join(reversed(line.split()))
            fixed_lines.append(line)
        text = "\n".join(fixed_lines)

    return text
