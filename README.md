# Concrete Repair RAG

A retrieval-augmented generation (RAG) system for TxDOT structural concrete repair standards, built with OpenAI embeddings, ChromaDB, and Claude.

## Knowledge Base

| Source | Description |
|---|---|
| SPEC24 | TxDOT Standard Specifications for Construction 2024 |
| SPEC14 | TxDOT Standard Specifications for Construction 2014 |
| CRM | Concrete Repair Manual |
| MIG | Material Inspection Guide |
| DMS-4655 | DMS-4655 Hydraulic Cement Concrete |
| DMS-6100 | DMS-6100 Epoxy and Epoxy Resin Systems |
| TEX | Tex-Series Test Procedures |
| MPL | Material Producer Lists (106 approved product lists) |

## Question Types

- **Type A** — Specification lookup (e.g., "What does Item 429 require for surface preparation?")
- **Type B** — Compliance judgment (e.g., "Does 3500 psi meet the strength requirement?")
- **Type C** — Procedure / workflow (e.g., "What are the steps for crack sealing?")
- **Type D** — Repair guidance (e.g., "How do I repair major spalling on a bridge pier?") — 3-phase chain: damage diagnosis → method selection → construction procedure

## Known Limitations

Due to current PDF extraction methods, **tables and figures are not fully captured** from all documents. Complex multi-column tables (such as concrete grade tables in Item 421 and performance tables in DMS-4655) may be partially read or rendered as plain text, which can affect answers that depend on tabular threshold values.

## Anti-Hallucination Design

Several measures have been implemented to reduce fabricated answers:

- **Relevance gating**: Cosine distance threshold (0.55) — if retrieved chunks are not sufficiently similar to the question, the system declines to answer rather than generating a speculative response
- **Item-number boosting**: When a question explicitly references an Item number (e.g., "Item 429"), those chunks are force-ranked to the top of retrieval results, preventing semantically-similar but wrong Items from overriding the correct source
- **Section-aware chunking**: SPEC documents are chunked by Item boundary (never crossing Item lines), preserving sub-section associations (Materials, Construction, Payment, etc.)
- **Strict system prompt**: Claude is instructed to cite exact values from documents, flag cross-references to other Items, and explicitly state when information is not covered rather than inferring

## Stack

- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector DB**: ChromaDB (local persistent)
- **LLM**: Anthropic Claude (`claude-sonnet-4-6` for answers, `claude-haiku-4-5-20251001` for classification and diagnosis)
- **UI**: Streamlit
- **Deployment**: Railway

## Setup

```bash
py -3.12 -m venv venv
venv/Scripts/pip install -r requirements.txt
cp .env.example .env   # fill in API keys
python src/parse_pdf.py
python src/build_index.py
venv/Scripts/streamlit run app.py
```
