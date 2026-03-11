"""
app.py — Structural Concrete Repair RAG  (Streamlit UI)
Run with: venv/Scripts/streamlit.exe run app.py
"""

import os
import sys
import pathlib
import streamlit as st

SRC_DIR = pathlib.Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

import chromadb
import anthropic
from openai import OpenAI

from classifier import classify, format_clarification_request
from retriever import (
    retrieve_type_a, retrieve_type_b, retrieve_type_c, retrieve_type_d,
    build_context, check_relevance,
)

BASE            = pathlib.Path(__file__).parent
CHROMA_DIR      = str(BASE / "db" / "chroma")
COLLECTION_NAME = "scr_core"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Concrete Repair Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Login ──────────────────────────────────────────────────────────────────────
def check_login():
    if st.session_state.get("authenticated"):
        return True

    st.markdown("""
    <style>
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stAppViewContainer"] { background-color: #f9f9f8; }
    .login-box {
        max-width: 380px; margin: 120px auto 0; padding: 40px;
        background: #fff; border: 1px solid #e0ddd5;
        border-radius: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.07);
    }
    .login-title { font-size: 22px; font-weight: 700; color: #1a1a1a; margin-bottom: 4px; }
    .login-sub   { font-size: 14px; color: #888; margin-bottom: 28px; }
    </style>
    <div class="login-box">
      <div class="login-title">🏗️ Concrete Repair Assistant</div>
      <div class="login-sub">Sign in to continue</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        username  = st.text_input("Username")
        password  = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        try:
            credentials = dict(st.secrets.get("credentials", {}))
        except Exception:
            credentials = {}
        for key, val in os.environ.items():
            if key.startswith("CRED_"):
                credentials[key[5:].lower()] = val

        if username in credentials and credentials[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Incorrect username or password.")
    return False

if not check_login():
    st.stop()

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f9f9f8;
    font-family: "Söhne", "ui-sans-serif", system-ui, -apple-system, sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"]    { display: none; }
[data-testid="stDecoration"] { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1a1a1a !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #d4d4d4 !important; font-size: 14px !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; font-size: 17px !important; }
[data-testid="stSidebar"] hr { border-color: #333 !important; }
[data-testid="stSidebar"] .stButton button {
    background-color: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    color: #d4d4d4 !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    text-align: left !important;
    padding: 6px 12px !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background-color: #333 !important; border-color: #555 !important; color: #fff !important;
}
[data-testid="stSidebar"] .stDownloadButton button {
    background-color: #2a2a2a !important;
    border: 1px solid #444 !important;
    color: #aaa !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    padding: 2px 8px !important;
    min-height: 0 !important; height: 28px !important; width: 100% !important;
    margin-top: 6px;
}
[data-testid="stSidebar"] .stDownloadButton button:hover {
    background-color: #2e6b4f !important; border-color: #2e6b4f !important; color: #fff !important;
}
[data-testid="stSidebar"] .stButton:first-of-type button {
    background-color: #2e6b4f !important; border-color: #2e6b4f !important; color: #fff !important;
}
[data-testid="stSidebar"] .stButton:first-of-type button:hover {
    background-color: #245a40 !important;
}

/* Main content */
[data-testid="stMainBlockContainer"] {
    max-width: 860px; margin: 0 auto; padding: 0 24px 120px 24px;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 8px 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background-color: #f0ede6; border-radius: 18px 18px 4px 18px;
    padding: 12px 16px; display: inline-block; max-width: 85%; float: right; clear: both;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    color: #1a1a1a; font-size: 15px; line-height: 1.7;
}
[data-testid="chatAvatarIcon-user"]      { background-color: #6b6b6b !important; color: white !important; }
[data-testid="chatAvatarIcon-assistant"] { background-color: #2e6b4f !important; color: white !important; }

/* Chat input */
[data-testid="stChatInput"] {
    background-color: #ffffff; border: 1px solid #e0ddd5;
    border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.07); padding: 4px;
}
[data-testid="stChatInput"] textarea {
    font-size: 15px !important; color: #1a1a1a !important;
    background: transparent !important; border: none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #9e9e9e !important; }
[data-testid="stChatInputSubmitButton"] button { background-color: #2e6b4f !important; border-radius: 10px !important; }
[data-testid="stChatInputSubmitButton"] button:hover { background-color: #245a40 !important; }

[data-testid="stBottom"] {
    background: linear-gradient(to top, #f9f9f8 80%, transparent);
    padding: 16px 0 8px 0;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #e8e5dc !important; border-radius: 8px !important;
    background: #faf9f6 !important; margin-top: 8px !important;
}
[data-testid="stExpander"] summary { font-size: 12px !important; color: #666 !important; padding: 6px 12px !important; }

/* Badges */
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 11px; font-weight: 500; margin-top: 6px; }
.badge-a { background: #e8f0fe; color: #1a5fcc; }
.badge-b { background: #fff0e6; color: #b34400; }
.badge-c { background: #e8f5e9; color: #1b6b2a; }
.badge-d { background: #f3e8ff; color: #6b21a8; }

/* Headings */
.stMarkdown h1 { font-size: 18px !important; font-weight: 600; margin: 16px 0 8px; }
.stMarkdown h2 { font-size: 16px !important; font-weight: 600; margin: 14px 0 6px; }
.stMarkdown h3 { font-size: 14px !important; font-weight: 600; margin: 12px 0 4px; }

/* Tables */
.stMarkdown table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
.stMarkdown th { background: #f0ede6; padding: 8px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #e0ddd5; }
.stMarkdown td { padding: 7px 12px; border-bottom: 1px solid #f0ede6; }

/* Code */
.stMarkdown code { background: #f0ede6; border-radius: 4px; padding: 1px 5px; font-size: 13px; }

/* Welcome screen */
.welcome-title { text-align: center; font-size: 28px; font-weight: 700; color: #1a1a1a; margin-top: 60px; margin-bottom: 8px; }
.welcome-sub   { text-align: center; font-size: 15px; color: #666; margin-bottom: 40px; }
</style>
""", unsafe_allow_html=True)

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an extremely rigorous Texas civil engineering expert specializing in TxDOT structural concrete repair standards.

When answering, follow this strict protocol:

PRIORITY PROTOCOL:
- Item-specific provisions (e.g., Item 429 for structural concrete repair) always override general provisions (e.g., Item 420, Item 421).
- When both apply, cite the specific Item first, then note how the general Item relates.
- If a threshold or requirement appears in both a general and a specific Item, apply the stricter one.

REVERSE-CHECK LOGIC:
- After retrieving any general requirement (e.g., "4-day curing", "28-day strength"), you MUST actively check whether the reference documents contain exceptions or stricter requirements under keywords such as "Exception", "Structural", "Bridge", or "Critical".
- If a stricter exception exists, apply it and explicitly flag the override: "General rule is X, but Item 429 specifies Y for structural applications."
- Never present a general rule as final without confirming no structural/bridge exception overrides it.

ENVIRONMENTAL TRIGGERS:
- If the user mentions high temperature, hot weather, wind, humidity, or direct sunlight, you MUST retrieve and apply the "Hot Weather Concreting" provisions from the reference documents.
- If the user mentions freeze-thaw, cold weather, or low temperature, apply corresponding cold weather provisions.
- Always state the triggering condition explicitly: "Because you mentioned [condition], the following additional requirements apply..."

CROSS-REFERENCE HANDLING:
- TxDOT specifications frequently cross-reference other Items (e.g., "cure in accordance with Item 420", "as per Item 421").
- When retrieved content references another Item, explicitly flag it:
  "⚠️ Cross-reference: This provision references Item XXX — additional requirements from that Item may apply."
- If the referenced Item's content is NOT in the retrieved documents, state:
  "Item XXX is referenced but not retrieved — consult that Item directly in the specification."
- Never treat a cross-referenced requirement as satisfied unless that Item's actual content is in the retrieved documents.

CITATION RULES:
1. Answer ONLY based on the [Reference Documents] provided. Do not fabricate information.
2. Cite source for every key conclusion — format: [Source: <document>, p.<page>]
3. When citing thresholds (depths, widths, strengths, ratios), always quote the exact value from the document.
4. If the documents do not contain enough information: say "The knowledge base does not cover this topic." Do NOT guess.
5. If documents partially cover the topic, answer the covered parts and explicitly note what is missing.
6. For safety-critical repair judgments, remind the user to verify with a licensed PE.
7. Respond in the same language the user uses."""

TYPE_A_TEMPLATE = """[Reference Documents]
{context}

[User Question]
{question}

Provide a direct answer first, then supporting detail, then citations."""

TYPE_B_TEMPLATE = """[Reference Documents]
{context}

[Condition Information]
{defect_info}

[User Question]
{question}

Output format:
### Judgment
One sentence conclusion.

### Basis
Step-by-step reasoning with citations at each step.

### Recommended Actions

### Cited Provisions

### Disclaimer
Final determination must be confirmed by a licensed engineer."""

TYPE_C_TEMPLATE = """[Reference Documents]
{context}

[User Question]
{question}

Output format:
### Steps
Numbered steps with responsible party and source citation for each.

### Key Warnings

### Cited Provisions"""

TYPE_D_TEMPLATE = """[Damage Assessment]
{diagnosis_summary}

[Reference Documents — Method Selection]
{method_context}

[Reference Documents — Construction Procedure]
{procedure_context}

[User Question]
{question}

Output format:
### Damage Assessment
Brief summary of identified damage type, severity, and element.

### Recommended Repair Method
Method name and selection rationale, citing the source document and selection criteria.

### Material Requirements
Required materials, mix designs, or approved products. [Source: ...]

### Construction Procedure
Numbered steps with surface prep, application, and curing requirements. [Source: ...]

### Key Warnings
Critical quality or safety requirements.

### Cited Provisions"""


# ── Init ───────────────────────────────────────────────────────────────────────
def load_env():
    env_path = BASE / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


@st.cache_resource
def init_clients():
    load_env()
    oc  = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    cc  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    col = chroma.get_collection(COLLECTION_NAME)
    return oc, cc, col


def format_sources(chunks: list[dict]) -> list[str]:
    seen, out = set(), []
    for c in chunks:
        m = c["metadata"]
        parts = [f"{m['source_short']} p.{m['page']}"]
        if m.get("item"):
            parts.append(f"Item {m['item']}")
        if m.get("section"):
            parts.append(m["section"].title())
        k = " · ".join(parts)
        if k not in seen:
            seen.add(k)
            out.append(k)
    return sorted(out)


def generate_answer(question, q_type, chunks, extra, claude_client, history, diagnosis=None):
    context = build_context(chunks)

    if q_type == "type_b":
        prompt = TYPE_B_TEMPLATE.format(context=context, defect_info=extra or "Not provided.", question=question)
    elif q_type == "type_c":
        prompt = TYPE_C_TEMPLATE.format(context=context, question=question)
    elif q_type == "type_d":
        # Split chunks: first half = method, second half = procedure (approximate)
        mid = len(chunks) // 2
        method_ctx = build_context(chunks[:mid] or chunks)
        proc_ctx   = build_context(chunks[mid:] or chunks)
        diag_summary = ""
        if diagnosis:
            diag_summary = (
                f"Damage type: {diagnosis.get('damage_type', 'unknown')}\n"
                f"Severity: {diagnosis.get('severity', 'unknown')}\n"
                f"Extent: {diagnosis.get('extent', 'unknown')}\n"
                f"Element: {diagnosis.get('element', 'unknown')}\n"
                f"Location: {diagnosis.get('location', 'unknown')}\n"
                f"Environment: {diagnosis.get('environment', 'unknown')}\n"
                f"Active crack: {diagnosis.get('is_active', 'unknown')}"
            )
        prompt = TYPE_D_TEMPLATE.format(
            diagnosis_summary=diag_summary or "Not available.",
            method_context=method_ctx,
            procedure_context=proc_ctx,
            question=question,
        )
    else:
        prompt = TYPE_A_TEMPLATE.format(context=context, question=question)

    messages = history + [{"role": "user", "content": prompt}]
    resp = claude_client.messages.create(
        model="claude-sonnet-4-6", max_tokens=2000,
        system=SYSTEM_PROMPT, messages=messages,
    )
    return resp.content[0].text


NOT_COVERED_MSG = (
    "The knowledge base does not appear to cover this specific topic. "
    "The retrieved documents had low relevance to your question.\n\n"
    "Please consult the relevant TxDOT specification or manual directly, "
    "or try rephrasing your question with more specific concrete repair terminology."
)

# ── Session state ──────────────────────────────────────────────────────────────
for key, val in [
    ("messages", []),
    ("history", []),
    ("awaiting_clarification", False),
    ("pending", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ Concrete Repair RAG")

    username = st.session_state.get("username", "")
    col_u, col_out = st.columns([3, 1])
    with col_u:
        st.markdown(f"<span style='color:#888;font-size:13px'>👤 {username}</span>", unsafe_allow_html=True)
    with col_out:
        if st.button("⏻", help="Sign out", key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()

    try:
        openai_client, claude_client, collection = init_clients()
    except Exception as e:
        st.error(f"Index error: {e}")
        st.stop()

    st.divider()

    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history  = []
        st.session_state.awaiting_clarification = False
        st.session_state.pending  = {}
        st.rerun()

    st.divider()
    st.markdown("<span style='color:#888;font-size:12px;letter-spacing:0.8px'>KNOWLEDGE BASE</span>", unsafe_allow_html=True)

    # ── Single-file docs ──────────────────────────────────────────────────
    single_docs = [
        ("CRM",     "Concrete Repair Manual",              BASE / "crm.pdf"),
        ("SPEC",    "Standard Specifications 2024",         BASE / "spec-book-0924.pdf"),
        ("MIG",     "Material Inspection Guide",            BASE / "mig.pdf"),
        ("DMS-4655","DMS-4655 Hydraulic Cement Concrete",  BASE / "4655.pdf"),
        ("DMS-6100","DMS-6100 Epoxy Systems",              BASE / "6100.pdf"),
    ]
    for short, full, path in single_docs:
        col_label, col_btn = st.columns([3, 1])
        with col_label:
            st.markdown(
                f"<div style='font-size:13px;padding:4px 0;line-height:1.3'>"
                f"<b style='color:#ccc'>{short}</b><br>"
                f"<span style='color:#777;font-size:11px'>{full}</span></div>",
                unsafe_allow_html=True,
            )
        with col_btn:
            if path.exists():
                st.download_button("↓", data=path.read_bytes(), file_name=path.name,
                                   mime="application/pdf", key=f"dl_{short}")
            else:
                st.markdown("<span style='color:#555;font-size:11px'>—</span>", unsafe_allow_html=True)

    # ── TEX: expandable folder ────────────────────────────────────────────
    TEX_DIR = BASE / "Test Procedures - Tex Series"
    TEX_FILES = [
        ("TEX-GUIDE", "Guide Schedule",          "guide_schedule.pdf"),
        ("TEX-400",   "Tex-400 through 899",     "400-0899.pdf"),
        ("TEX-407",   "Tex-407-A",               "cnn407.pdf"),
        ("TEX-418",   "Tex-418-A",               "cnn418.pdf"),
        ("TEX-428",   "Tex-428-A",               "cnn428.pdf"),
    ]
    with st.expander("**TEX** · Tex Test Procedures"):
        for short, label, filename in TEX_FILES:
            path = TEX_DIR / filename
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"<span style='font-size:12px;color:#aaa'>{label}</span>", unsafe_allow_html=True)
            with c2:
                if path.exists():
                    st.download_button("↓", data=path.read_bytes(), file_name=filename,
                                       mime="application/pdf", key=f"dl_{short}")
                else:
                    st.markdown("<span style='color:#555;font-size:11px'>—</span>", unsafe_allow_html=True)

    # ── MPL: expandable folder ────────────────────────────────────────────
    MPL_DIR = BASE / "Material Producer List (MPL)"
    with st.expander("**MPL** · Material Producer Lists"):
        if MPL_DIR.exists():
            mpl_files = sorted(MPL_DIR.glob("*.pdf"))
            for path in mpl_files:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"<span style='font-size:12px;color:#aaa'>{path.stem}</span>", unsafe_allow_html=True)
                with c2:
                    st.download_button("↓", data=path.read_bytes(), file_name=path.name,
                                       mime="application/pdf", key=f"dl_mpl_{path.stem}")
        else:
            st.markdown("<span style='color:#555;font-size:11px'>Folder not found</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<span style='color:#888;font-size:12px;letter-spacing:0.8px'>QUESTION TYPES</span>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:14px;line-height:2.2'>
<span style='background:#e8f0fe;color:#1a5fcc;border-radius:4px;padding:2px 8px'>A</span>&nbsp; Spec lookup<br>
<span style='background:#fff0e6;color:#b34400;border-radius:4px;padding:2px 8px'>B</span>&nbsp; Compliance judgment<br>
<span style='background:#e8f5e9;color:#1b6b2a;border-radius:4px;padding:2px 8px'>C</span>&nbsp; Procedure / workflow<br>
<span style='background:#f3e8ff;color:#6b21a8;border-radius:4px;padding:2px 8px'>D</span>&nbsp; Repair guidance
</div>""", unsafe_allow_html=True)

# ── Main area ──────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown('<div class="welcome-title">Concrete Repair Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-sub">Ask about repair methods, specifications, compliance, or field procedures.</div>', unsafe_allow_html=True)

    suggestions = [
        ("Spec",     "What does DMS-4655 require for compressive strength?"),
        ("Spec",     "What is the w/c ratio limit for Item 420 concrete?"),
        ("Compliance","The measured crack width is 0.5mm — does it meet spec?"),
        ("Compliance","Is 3500 psi compressive strength acceptable for structural repair?"),
        ("Repair",   "How do I repair spalling with exposed rebar on a bridge pier?"),
        ("Repair",   "Transverse cracks on bridge deck — what repair method should I use?"),
        ("Procedure","What are the surface preparation requirements before applying epoxy?"),
        ("Procedure","What is the process for submitting a repair method for approval?"),
    ]

    col1, col2 = st.columns(2)
    for i, (label, text) in enumerate(suggestions):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(f"**{label}**\n\n{text}", key=f"sug_{i}", use_container_width=True):
                st.session_state.example_question = text
                st.rerun()

# Render chat history
BADGES = {
    "type_a": ("A", "a", "Spec Lookup"),
    "type_b": ("B", "b", "Compliance Judgment"),
    "type_c": ("C", "c", "Procedure"),
    "type_d": ("D", "d", "Repair Guidance"),
}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} sources cited"):
                for s in msg["sources"]:
                    st.markdown(f"<span style='font-size:12px;color:#666'>· {s}</span>", unsafe_allow_html=True)
        if msg.get("q_type") and msg["q_type"] in BADGES:
            letter, cls, lbl = BADGES[msg["q_type"]]
            st.markdown(f'<span class="badge badge-{cls}">Type {letter} · {lbl}</span>', unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
if "example_question" in st.session_state:
    prompt = st.session_state.pop("example_question")
else:
    prompt = st.chat_input("Ask about concrete repair specifications or methods…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # ── Clarification response ─────────────────────────────────────────
        if st.session_state.awaiting_clarification:
            pending    = st.session_state.pending
            extra_info = prompt
            q_type     = pending["q_type"]
            question_full = f"{pending['question']}\nAdditional info: {extra_info}"

            with st.spinner("Searching knowledge base…"):
                if q_type == "type_b":
                    chunks = retrieve_type_b(question_full, extra_info, collection, openai_client, claude_client)
                    diagnosis = None
                else:  # type_d
                    chunks, diagnosis = retrieve_type_d(question_full, collection, openai_client, claude_client)

            sources = format_sources(chunks)

            if not check_relevance(chunks):
                answer = NOT_COVERED_MSG
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                with st.spinner("Generating answer…"):
                    answer = generate_answer(
                        question_full, q_type, chunks, extra_info,
                        claude_client, st.session_state.history, diagnosis
                    )
                st.markdown(answer)
                with st.expander(f"📎 {len(sources)} sources cited"):
                    for s in sources:
                        st.markdown(f"<span style='font-size:12px;color:#666'>· {s}</span>", unsafe_allow_html=True)
                letter, cls, lbl = BADGES.get(q_type, ("?", "a", ""))
                st.markdown(f'<span class="badge badge-{cls}">Type {letter} · {lbl}</span>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "q_type": q_type})

            st.session_state.history += [{"role": "user", "content": question_full}, {"role": "assistant", "content": answer}]
            st.session_state.history = st.session_state.history[-6:]
            st.session_state.awaiting_clarification = False
            st.session_state.pending = {}

        # ── New question ───────────────────────────────────────────────────
        else:
            with st.spinner("Classifying question…"):
                classification = classify(prompt, claude_client)
            q_type = classification["type"]

            # Only ask for clarification if the question is genuinely vague.
            # Skip if the user already provided measurements, defect names, Item refs, or element types.
            import re as _re
            _has_numbers   = bool(_re.search(r'\d', prompt))
            _has_item_ref  = bool(_re.search(r'\bitem\s*\d+|dms[-\s]?\d+|tex[-\s]?\d+', prompt, _re.IGNORECASE))
            _has_defect    = bool(_re.search(
                r'\bspall|crack|delamination|corrosion|scaling|honeycombing|void|efflorescence|rebar|steel\b',
                prompt, _re.IGNORECASE))
            _has_element   = bool(_re.search(
                r'\bpier|deck|beam|girder|abutment|bent|cap|column|soffit|barrier|railing\b',
                prompt, _re.IGNORECASE))
            _already_detailed = _has_numbers or _has_item_ref or _has_defect or _has_element

            needs_clarification = (
                classification.get("needs_clarification")
                and classification.get("missing_info")
                and q_type in ("type_b", "type_d")
                and not _already_detailed
            )

            if needs_clarification:
                clarification_msg = format_clarification_request(classification["missing_info"], prompt)
                st.markdown(clarification_msg)
                st.session_state.messages.append({"role": "assistant", "content": clarification_msg})
                st.session_state.awaiting_clarification = True
                st.session_state.pending = {"question": prompt, "q_type": q_type}

            else:
                with st.spinner("Searching knowledge base…"):
                    diagnosis = None
                    if q_type == "type_b":
                        chunks = retrieve_type_b(prompt, "", collection, openai_client, claude_client)
                    elif q_type == "type_c":
                        chunks = retrieve_type_c(prompt, collection, openai_client)
                    elif q_type == "type_d":
                        chunks, diagnosis = retrieve_type_d(prompt, collection, openai_client, claude_client)
                    else:
                        chunks = retrieve_type_a(prompt, collection, openai_client)

                sources = format_sources(chunks)

                if not check_relevance(chunks):
                    answer = NOT_COVERED_MSG
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
                    st.session_state.history = st.session_state.history[-6:]
                else:
                    with st.spinner("Generating answer…"):
                        answer = generate_answer(
                            prompt, q_type, chunks, "", claude_client,
                            st.session_state.history, diagnosis
                        )
                    st.markdown(answer)
                    with st.expander(f"📎 {len(sources)} sources cited"):
                        for s in sources:
                            st.markdown(f"<span style='font-size:12px;color:#666'>· {s}</span>", unsafe_allow_html=True)

                    letter, cls, lbl = BADGES.get(q_type, ("?", "a", ""))
                    st.markdown(f'<span class="badge badge-{cls}">Type {letter} · {lbl}</span>', unsafe_allow_html=True)

                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "q_type": q_type})
                    st.session_state.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
                    st.session_state.history = st.session_state.history[-6:]
