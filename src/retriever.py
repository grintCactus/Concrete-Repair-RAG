"""
retriever.py — 4 retrieval strategies for A/B/C/D question types.

Type A: single-path top-7
Type B: multi-path (Haiku decomposes into sub-queries)
Type C: dual-path (procedure + definitions)
Type D: 3-phase chain — diagnose → method selection → construction procedure
"""

import re
import json
import anthropic
import chromadb
from openai import OpenAI
from glossary import expand as expand_abbr

RELEVANCE_THRESHOLD = 0.55  # cosine distance; lower = more similar

# Detect explicit Item references in user questions: "Item 429", "item 420"
ITEM_REF_RE = re.compile(r'\bItem\s+(\d{3,4})\b', re.IGNORECASE)

# ── Prompts ────────────────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = """You are a concrete repair knowledge base retrieval assistant.
Break the question into 3-4 sub-queries, each targeting a different aspect.

Available documents:
- SPEC   (TxDOT Standard Specifications): Item requirements, measurement, payment
- CRM    (Concrete Repair Manual): repair methods, selection criteria, procedures
- MIG    (Material Inspection Guide): inspection methods, acceptance criteria
- DMS4655 (DMS-4655): hydraulic cement concrete material specs
- DMS6100 (DMS-6100): epoxy and epoxy resin system specs
- TEX-*  (Test Procedures): Tex-series test methods
- MPL-*  (Material Producer Lists): approved products and manufacturers

Question: {question}
Context: {context}

Return JSON array only:
[
  {{"query": "search text", "target": "SPEC|CRM|DMS4655|any", "purpose": "why"}},
  ...
]"""

DIAGNOSE_PROMPT = """You are a structural concrete repair expert.
Analyze the user's description and extract structured damage information.

User description: {question}

Return JSON only:
{{
  "damage_type": "e.g. transverse cracking / spalling / delamination / corrosion / etc.",
  "severity": "minor / moderate / severe / unknown",
  "extent": "e.g. hairline / 0.3mm width / 30% of surface / unknown",
  "element": "e.g. bridge deck / pier / beam / abutment / unknown",
  "location": "e.g. top surface / soffit / substructure / unknown",
  "environment": "e.g. marine / inland / freeze-thaw / unknown",
  "is_active": "yes / no / unknown",
  "missing_info": ["list any critical details not provided"]
}}
"""


# ── Core helpers ───────────────────────────────────────────────────────────────

def _embed(texts: list[str], openai_client: OpenAI) -> list[list[float]]:
    resp = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in resp.data]


def _extract_item_refs(text: str) -> list[str]:
    """Return list of Item numbers explicitly mentioned in text."""
    return list(dict.fromkeys(ITEM_REF_RE.findall(text)))  # deduplicated, order-preserved


def _query_collection(
    collection: chromadb.Collection,
    embedding: list[float],
    top_k: int,
    source_filter: str | None = None,
    section_filter: str | None = None,
    item_filter: str | None = None,
) -> list[dict]:
    conditions = []
    if source_filter:
        conditions.append({"source_short": source_filter})
    if section_filter:
        conditions.append({"section": section_filter})
    if item_filter:
        conditions.append({"item": item_filter})

    if len(conditions) == 0:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}

    kwargs = {
        "query_embeddings": [embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": round(dist, 4)})
    return chunks


def check_relevance(chunks: list[dict], threshold: float = RELEVANCE_THRESHOLD) -> bool:
    """Return True if at least one chunk is similar enough to be useful."""
    if not chunks:
        return False
    return min(c["distance"] for c in chunks) < threshold


def _merge_dedup(chunk_lists: list[list[dict]]) -> list[dict]:
    seen, merged = set(), []
    for chunks in chunk_lists:
        for c in chunks:
            m = c["metadata"]
            key = f"{m['source_short']}_{m['page']}_{m.get('chunk_index', 0)}"
            if key not in seen:
                seen.add(key)
                merged.append(c)
    merged.sort(key=lambda x: x["distance"])
    return merged


# ── Type A ─────────────────────────────────────────────────────────────────────

def _retrieve_with_item_boost(
    question: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    top_k: int = 7,
    section_filter: str | None = None,
) -> list[dict]:
    """
    Semantic retrieval with Item-number boosting.
    If the question explicitly references Item numbers (e.g., 'Item 429'),
    force-include chunks from those Items via metadata filter, then merge
    with general semantic results. Item-specific chunks always rank first.
    """
    query = expand_abbr(question)
    [emb] = _embed([query], openai_client)

    item_refs = _extract_item_refs(question)

    if item_refs:
        # Phase 1: force-retrieve from each referenced Item (sorted by distance)
        item_chunks: list[dict] = []
        seen_ids: set = set()
        for item_num in item_refs:
            try:
                chunks = _query_collection(
                    collection, emb, top_k=5,
                    item_filter=item_num, section_filter=section_filter
                )
                for c in chunks:
                    m = c["metadata"]
                    key = f"{m['source_short']}_{m['page']}_{m.get('chunk_index', 0)}"
                    if key not in seen_ids:
                        seen_ids.add(key)
                        item_chunks.append(c)
            except Exception:
                pass
        item_chunks.sort(key=lambda x: x["distance"])

        # Phase 2: general semantic search — fill remaining slots
        general = _query_collection(collection, emb, top_k=top_k, section_filter=section_filter)
        for c in general:
            m = c["metadata"]
            key = f"{m['source_short']}_{m['page']}_{m.get('chunk_index', 0)}"
            if key not in seen_ids:
                seen_ids.add(key)
                item_chunks.append(c)

        # Item-specific chunks lead; general results fill the tail
        return item_chunks[:top_k + len(item_refs) * 3]

    return _query_collection(collection, emb, top_k, section_filter=section_filter)


def retrieve_type_a(
    question: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    top_k: int = 7,
) -> list[dict]:
    return _retrieve_with_item_boost(question, collection, openai_client, top_k)


# ── Type B ─────────────────────────────────────────────────────────────────────

def retrieve_type_b(
    question: str,
    context: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    claude_client: anthropic.Anthropic,
) -> list[dict]:
    resp = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(
            question=question, context=context or "None"
        )}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        sub_queries = json.loads(raw.strip())
    except json.JSONDecodeError:
        return retrieve_type_a(question, collection, openai_client)

    all_chunks = []
    for sq in sub_queries:
        query_text = expand_abbr(sq["query"])
        target = sq.get("target", "any")
        source_filter = target if target != "any" else None
        [emb] = _embed([query_text], openai_client)
        try:
            chunks = _query_collection(collection, emb, top_k=3, source_filter=source_filter)
            if not chunks:
                chunks = _query_collection(collection, emb, top_k=3)
        except Exception:
            chunks = _query_collection(collection, emb, top_k=3)
        for c in chunks:
            c["sub_query"] = sq.get("purpose", "")
        all_chunks.append(chunks)

    return _merge_dedup(all_chunks)


# ── Type C ─────────────────────────────────────────────────────────────────────

def retrieve_type_c(
    question: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
) -> list[dict]:
    q1 = expand_abbr(question)
    q2 = expand_abbr(question) + " steps procedure sequence requirements timeline"
    embs = _embed([q1, q2], openai_client)

    item_refs = _extract_item_refs(question)
    item_chunks: list[dict] = []
    for item_num in item_refs:
        try:
            c = _query_collection(collection, embs[0], top_k=4,
                                  item_filter=item_num, section_filter="construction")
            item_chunks.extend(c)
        except Exception:
            pass

    chunks_a = _query_collection(collection, embs[0], top_k=4)
    chunks_b = _query_collection(collection, embs[1], top_k=4)
    merged = _merge_dedup([item_chunks, chunks_a, chunks_b])
    return merged[:10]


# ── Type D ─────────────────────────────────────────────────────────────────────

def diagnose_damage(question: str, claude_client: anthropic.Anthropic) -> dict:
    """Phase 1: structured damage diagnosis via Haiku."""
    resp = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": DIAGNOSE_PROMPT.format(question=question)}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {
            "damage_type": "unknown", "severity": "unknown", "extent": "unknown",
            "element": "unknown", "location": "unknown", "environment": "unknown",
            "is_active": "unknown", "missing_info": [],
        }


def retrieve_type_d(
    question: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    claude_client: anthropic.Anthropic,
) -> tuple[list[dict], dict]:
    """
    3-phase retrieval for repair guidance.
    Returns (chunks, diagnosis_dict).
    """
    # Phase 1: Diagnose
    diagnosis = diagnose_damage(question, claude_client)
    damage_type = diagnosis.get("damage_type", "")
    element     = diagnosis.get("element", "")
    severity    = diagnosis.get("severity", "")

    # Phase 2: Method selection (DMS specs + CRM selection criteria)
    method_query = expand_abbr(
        f"{damage_type} {element} repair method selection criteria material requirements"
    )
    [emb2] = _embed([method_query], openai_client)
    method_chunks = _query_collection(collection, emb2, top_k=5)

    # Also try CRM specifically for method selection
    try:
        crm_chunks = _query_collection(collection, emb2, top_k=3, source_filter="CRM")
        method_chunks = _merge_dedup([method_chunks, crm_chunks])[:6]
    except Exception:
        pass

    # Phase 3: Construction procedure (prefer Construction section)
    proc_query = expand_abbr(
        f"{damage_type} {element} {severity} repair construction procedure steps surface preparation"
    )
    [emb3] = _embed([proc_query], openai_client)

    try:
        proc_chunks = _query_collection(
            collection, emb3, top_k=5, section_filter="construction"
        )
        if not proc_chunks:
            proc_chunks = _query_collection(collection, emb3, top_k=5)
    except Exception:
        proc_chunks = _query_collection(collection, emb3, top_k=5)

    all_chunks = _merge_dedup([method_chunks, proc_chunks])
    return all_chunks, diagnosis


# ── Context builder ────────────────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        label_parts = [f"[Doc {i}] {m['source']} (p.{m['page']})"]
        if m.get("item"):
            label_parts.append(f"Item {m['item']}")
        if m.get("section"):
            label_parts.append(m["section"].title())
        if m.get("test_method"):
            label_parts.append(m["test_method"])
        if c.get("sub_query"):
            label_parts.append(f"— {c['sub_query']}")
        parts.append(" | ".join(label_parts) + "\n" + c["text"])
    return "\n\n---\n\n".join(parts)
