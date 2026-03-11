"""
parse_pdf.py — PDF parsing with section-aware structure detection and table extraction.

Improvements over TxDOT version:
- pdfplumber extracts tables → converted to Markdown (preserves structure)
- Detects Item numbers and section headings (Description/Materials/Construction/etc.)
- Detects Tex test method numbers
- Metadata: item, section, test_method, has_table
"""

import sys
import json
import re
import pathlib
import fitz          # PyMuPDF
import pdfplumber

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE        = pathlib.Path(__file__).parent.parent
PARSED_DIR  = BASE / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Regex patterns ─────────────────────────────────────────────────────────────

# "Item 429", "Item 420.2", "ITEM 429"
ITEM_RE = re.compile(r'\bItem\s+(\d{3,4}[\w.-]*)', re.IGNORECASE)

# "1. DESCRIPTION.", "2. MATERIALS", "Section 3 - Construction", "CONSTRUCTION."
_SECTION_WORDS = (
    r'(DESCRIPTION|MATERIALS?|EQUIPMENT|CONSTRUCTION|MEASUREMENT|PAYMENT'
    r'|LIMITATIONS?|GENERAL|SCOPE|REQUIREMENTS?|PROCEDURES?|SAMPLING|TESTING|DEFINITIONS?)'
)
SECTION_RE = re.compile(
    r'^\s*(?:\d+\.\s+|Section\s+\d+[-.\s]+)?' + _SECTION_WORDS + r'\b',
    re.IGNORECASE | re.MULTILINE,
)

# "Tex-418-A", "Tex 407", "TEX-428"
TEST_METHOD_RE = re.compile(r'\bTex[-\s](\d{3}[-\w]*)', re.IGNORECASE)

# "429.4 Construction", "429.4.A Preparation", "429.4.A.1 ..."
SUBSECTION_RE = re.compile(r'^\s*(\d{3,4}\.\d+(?:\.\w+)?)\s+([A-Z][^\n]{0,60})', re.MULTILINE)

# Cross-references to other Items: "Item 420", "Item 421"
CROSS_REF_RE = re.compile(r'\bItem\s+(\d{3,4})\b', re.IGNORECASE)

# ── Document registry ──────────────────────────────────────────────────────────

# Root PDFs: (filename, source_short, full_name)
ROOT_DOCS = [
    ("crm.pdf",            "CRM",     "Concrete Repair Manual"),
    ("spec-book-0924.pdf", "SPEC24",  "TxDOT Standard Specifications 2024"),
    ("spec-book-2014.pdf", "SPEC14",  "TxDOT Standard Specifications 2014"),
    ("mig.pdf",            "MIG",     "Material Inspection Guide"),
    ("4655.pdf",           "DMS4655", "DMS-4655 Hydraulic Cement Concrete"),
    ("6100.pdf",           "DMS6100", "DMS-6100 Epoxy and Epoxy Resin Systems"),
]

# Test Procedures subfolder
TEST_PROC_DIR = BASE / "Test Procedures - Tex Series"
TEST_PROC_DOCS = [
    ("400-0899.pdf",      "TEX-400",   "Tex-400 through 899 Test Methods"),
    ("cnn407.pdf",        "TEX-407",   "Tex-407-A Test Method"),
    ("cnn418.pdf",        "TEX-418",   "Tex-418-A Test Method"),
    ("cnn428.pdf",        "TEX-428",   "Tex-428-A Test Method"),
    ("guide_schedule.pdf","TEX-GUIDE", "Tex Test Methods Guide Schedule"),
]

# MPL subfolder — parsed simply (product approval lists)
MPL_DIR = BASE / "Material Producer List (MPL)"

# ── Helpers ────────────────────────────────────────────────────────────────────

def detect_noise_lines(fitz_doc: fitz.Document, sample_pages: int = 20) -> set:
    """Find header/footer lines that repeat across many pages."""
    from collections import Counter
    line_counts: Counter = Counter()
    total = min(len(fitz_doc), sample_pages)
    for i in range(total):
        text = fitz_doc[i].get_text("text")
        for line in text.splitlines():
            line = line.strip()
            if 2 <= len(line) <= 120:
                line_counts[line] += 1
    threshold = max(3, total // 4)
    return {line for line, count in line_counts.items() if count >= threshold}


def remove_noise(text: str, noise: set) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip() not in noise]
    return "\n".join(lines)


def extract_tables_markdown(plumber_page) -> list[str]:
    """Extract all tables from a pdfplumber page as Markdown strings."""
    md_tables = []
    try:
        tables = plumber_page.extract_tables()
    except Exception:
        return []
    for table in (tables or []):
        if not table:
            continue
        rows = []
        for i, row in enumerate(table):
            cells = [str(c or "").strip().replace("\n", " ") for c in row]
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                rows.append("|" + "|".join(["---"] * len(row)) + "|")
        if rows:
            md_tables.append("\n".join(rows))
    return md_tables


def extract_cross_refs(text: str) -> str:
    """Extract all referenced Item numbers as a comma-separated string."""
    refs = sorted(set(CROSS_REF_RE.findall(text)))
    return ",".join(refs) if refs else ""


def detect_structure(text: str):
    """Return (item, section, subsection, test_method) from page text."""
    head = text[:800]
    item = None
    section = None
    subsection = None
    test_method = None

    # Item number (e.g., "Item 429")
    m = ITEM_RE.search(head)
    if m:
        item = m.group(1)

    # Numbered subsection takes priority over generic section words
    # e.g., "429.4 Construction" → item=429, section=construction, subsection=429.4
    m = SUBSECTION_RE.search(head)
    if m:
        subsection = m.group(1)            # e.g., "429.4"
        if not item:
            item = subsection.split(".")[0]  # derive item from subsection prefix
        section_name = m.group(2).lower()
        # Map to canonical section name
        for keyword in ("description", "material", "equipment", "construction",
                        "measurement", "payment", "limitation", "general", "scope"):
            if keyword in section_name:
                section = keyword
                break
        if not section:
            section = section_name[:20]

    # Fall back to generic section words if no numbered subsection found
    if not section:
        m = SECTION_RE.search(head)
        if m:
            section = m.group(1).lower()

    m = TEST_METHOD_RE.search(head)
    if m:
        test_method = f"Tex-{m.group(1)}"

    return item, section, subsection, test_method


# ── Core parser ────────────────────────────────────────────────────────────────

def parse_pdf(path: pathlib.Path, source_short: str, source_name: str) -> list[dict]:
    """Parse a single PDF into a list of page dicts."""
    path = pathlib.Path(path)
    if not path.exists():
        print(f"  [SKIP] not found: {path.name}")
        return []

    print(f"  Parsing {path.name} ({source_short})...")
    pages_out = []

    try:
        fitz_doc    = fitz.open(str(path))
        plumber_doc = pdfplumber.open(str(path))
    except Exception as e:
        print(f"  [ERROR] cannot open {path.name}: {e}")
        return []

    noise = detect_noise_lines(fitz_doc)

    current_item       = None
    current_section    = None
    current_subsection = None
    current_test       = None

    for idx in range(len(fitz_doc)):
        raw_text   = fitz_doc[idx].get_text("text")
        clean_text = remove_noise(raw_text, noise)
        tables_md  = extract_tables_markdown(plumber_doc.pages[idx])

        # Update running structure state
        item, section, subsection, test_method = detect_structure(clean_text)
        if item:
            current_item       = item
            current_section    = None   # reset section when Item changes
            current_subsection = None
        if section:
            current_section = section
        if subsection:
            current_subsection = subsection
        if test_method:
            current_test = test_method

        # Combine text + tables
        full_text = clean_text.strip()
        if tables_md:
            full_text += "\n\n" + "\n\n".join(tables_md)

        if len(full_text.split()) < 10:
            continue

        pages_out.append({
            "source":       source_name,
            "source_short": source_short,
            "page":         idx + 1,
            "item":         current_item       or "",
            "section":      current_section    or "",
            "subsection":   current_subsection or "",
            "test_method":  current_test       or "",
            "has_table":    bool(tables_md),
            "cross_refs":   extract_cross_refs(full_text),
            "text":         full_text,
        })

    fitz_doc.close()
    plumber_doc.close()
    print(f"    -> {len(pages_out)} pages")
    return pages_out


def parse_mpl_pdf(path: pathlib.Path) -> list[dict]:
    """Simplified parser for MPL files (approved product lists)."""
    stem = path.stem.upper()
    source_short = f"MPL-{stem}"
    source_name  = f"Material Producer List: {path.stem}"
    return parse_pdf(path, source_short, source_name)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_pages = []

    # 1. Root PDFs
    print("=== Parsing root documents ===")
    for filename, short, name in ROOT_DOCS:
        path = BASE / filename
        pages = parse_pdf(path, short, name)
        all_pages.extend(pages)
        out = PARSED_DIR / f"{short}.json"
        out.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2. Test Procedure PDFs
    print("\n=== Parsing test procedures ===")
    for filename, short, name in TEST_PROC_DOCS:
        path = TEST_PROC_DIR / filename
        pages = parse_pdf(path, short, name)
        all_pages.extend(pages)
        out = PARSED_DIR / f"{short}.json"
        out.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3. MPL PDFs
    print("\n=== Parsing MPL documents ===")
    if MPL_DIR.exists():
        mpl_pages = []
        for pdf in sorted(MPL_DIR.glob("*.pdf")):
            pages = parse_mpl_pdf(pdf)
            mpl_pages.extend(pages)
            all_pages.extend(pages)
        out = PARSED_DIR / "MPL.json"
        out.write_text(json.dumps(mpl_pages, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  MPL total: {len(mpl_pages)} pages from {len(list(MPL_DIR.glob('*.pdf')))} files")

    print(f"\nTotal pages parsed: {len(all_pages)}")
    # Save combined
    (PARSED_DIR / "_all.json").write_text(
        json.dumps(all_pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved to {PARSED_DIR}/")


if __name__ == "__main__":
    main()
