"""
build_index.py — Section-aware chunking + embedding + ChromaDB indexing.

Improvements over TxDOT version:
- Section-aware chunking: never crosses Item/Section boundaries
- Metadata includes: item, section, test_method, has_table
- Deduplication before embedding
"""

import sys
import json
import pathlib
import chromadb
from openai import OpenAI
import os

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE           = pathlib.Path(__file__).parent.parent
PARSED_DIR     = BASE / "data" / "parsed"
CHROMA_DIR     = str(BASE / "db" / "chroma")
COLLECTION_NAME = "scr_core"   # structural concrete repair

MAX_WORDS     = 600
OVERLAP_WORDS = 80
BATCH_SIZE    = 100

# SPEC documents use Item-aware chunking
SPEC_SOURCES  = {"SPEC24", "SPEC14"}

# Wider window for SPEC to keep sub-sections together
SPEC_MAX_WORDS    = 800
SPEC_OVERLAP_WORDS = 150


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_item_aware(pages: list[dict]) -> list[dict]:
    """
    Item-based chunking for Standard Specifications (SPEC24, SPEC14).

    Strategy:
    - Group ALL pages that belong to the same Item number together.
    - Slide through the combined Item text with a wider window (800w / 150 overlap).
    - Each chunk carries item, section, subsection, cross_refs metadata.
    - This ensures sub-sections (429.2 Materials, 429.4 Construction) stay
      strongly associated — they appear in overlapping chunks under the same Item.
    """
    # Collect pages per item in document order
    from collections import defaultdict, OrderedDict
    item_pages: OrderedDict = OrderedDict()
    for page in pages:
        key = page.get("item") or "__no_item__"
        if key not in item_pages:
            item_pages[key] = []
        item_pages[key].append(page)

    chunks = []
    chunk_idx = 0

    for item_num, grp in item_pages.items():
        all_words: list[str] = []
        for page in grp:
            all_words.extend(page["text"].split())

        first_page  = grp[0]
        # Collect all unique cross-references across the whole Item
        all_cross = set()
        for p in grp:
            for ref in p.get("cross_refs", "").split(","):
                ref = ref.strip()
                if ref:
                    all_cross.add(ref)
        cross_refs_str = ",".join(sorted(all_cross))

        start = 0
        while start < len(all_words):
            end = min(start + SPEC_MAX_WORDS, len(all_words))
            chunk_text = " ".join(all_words[start:end])

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source":       first_page["source"],
                    "source_short": first_page["source_short"],
                    "page":         first_page["page"],
                    "item":         item_num,
                    "section":      first_page.get("section", ""),
                    "subsection":   first_page.get("subsection", ""),
                    "test_method":  first_page.get("test_method", ""),
                    "has_table":    str(any(p["has_table"] for p in grp)),
                    "cross_refs":   cross_refs_str,
                    "chunk_index":  chunk_idx,
                },
            })
            chunk_idx += 1

            if end == len(all_words):
                break
            start += SPEC_MAX_WORDS - SPEC_OVERLAP_WORDS

    return chunks


def chunk_section_aware(pages: list[dict]) -> list[dict]:
    """
    Group consecutive pages sharing the same (item, section),
    then slide within each group. Never crosses section boundaries.
    """
    # Group pages by (item, section) sequence
    groups: list[tuple[tuple, list[dict]]] = []
    prev_key = None
    current_group: list[dict] = []

    for page in pages:
        key = (page.get("item", ""), page.get("section", ""))
        if key != prev_key and current_group:
            groups.append((prev_key, current_group))
            current_group = []
        current_group.append(page)
        prev_key = key
    if current_group:
        groups.append((prev_key, current_group))

    chunks = []
    chunk_idx = 0

    for (item, section), group_pages in groups:
        # Combine all words in this section group
        all_words: list[str] = []
        for page in group_pages:
            all_words.extend(page["text"].split())

        first_page = group_pages[0]

        start = 0
        while start < len(all_words):
            end = min(start + MAX_WORDS, len(all_words))
            chunk_text = " ".join(all_words[start:end])

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source":       first_page["source"],
                    "source_short": first_page["source_short"],
                    "page":         first_page["page"],
                    "item":         item,
                    "section":      section,
                    "test_method":  first_page.get("test_method", ""),
                    "has_table":    str(any(p["has_table"] for p in group_pages)),
                    "chunk_index":  chunk_idx,
                },
            })
            chunk_idx += 1

            if end == len(all_words):
                break
            start += MAX_WORDS - OVERLAP_WORDS

    return chunks


# ── Embedding ──────────────────────────────────────────────────────────────────

def batch_embed(texts: list[str], client: OpenAI) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        embeddings.extend(item.embedding for item in resp.data)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
    return embeddings


# ── Main ───────────────────────────────────────────────────────────────────────

def load_env():
    env_path = BASE / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    load_env()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Load all parsed pages
    all_json = PARSED_DIR / "_all.json"
    if not all_json.exists():
        print("ERROR: Run parse_pdf.py first.")
        sys.exit(1)

    pages = json.loads(all_json.read_text(encoding="utf-8"))
    print(f"Loaded {len(pages)} pages")

    # Split pages by document type and apply appropriate chunking
    spec_pages  = [p for p in pages if p.get("source_short") in SPEC_SOURCES]
    other_pages = [p for p in pages if p.get("source_short") not in SPEC_SOURCES]

    print(f"  SPEC pages (item-aware chunking): {len(spec_pages)}")
    print(f"  Other pages (section-aware chunking): {len(other_pages)}")

    print("Chunking SPEC documents (item-aware)...")
    chunks = chunk_item_aware(spec_pages)
    print(f"  SPEC chunks: {len(chunks)}")

    print("Chunking other documents (section-aware)...")
    other_chunks = chunk_section_aware(other_pages)
    print(f"  Other chunks: {len(other_chunks)}")

    chunks += other_chunks
    print(f"Total chunks: {len(chunks)}")

    # Embed
    print("Embedding...")
    texts = [c["text"] for c in chunks]
    embeddings = batch_embed(texts, client)

    # Store in ChromaDB
    print("Storing in ChromaDB...")
    chroma  = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if present
    try:
        chroma.delete_collection(COLLECTION_NAME)
        print("  (deleted existing collection)")
    except Exception:
        pass

    col = chroma.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_embs   = embeddings[i:i + BATCH_SIZE]
        col.add(
            ids        = [f"chunk_{i + j}" for j in range(len(batch_chunks))],
            embeddings = batch_embs,
            documents  = [c["text"] for c in batch_chunks],
            metadatas  = [c["metadata"] for c in batch_chunks],
        )
        print(f"  Stored {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"\nDone. {col.count()} chunks indexed in '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
