"""
classifier.py — 4-type question classifier using Claude Haiku.

Type A: Regulation / spec lookup ("What does Item 429 require for...")
Type B: Compliance judgment ("Does this meet spec?", "Is this acceptable?")
Type C: Procedure / workflow ("What are the steps to...", "What is the process for...")
Type D: Repair guidance ("How do I repair...", "What method should I use to fix...")
"""

import json
import anthropic

CLASSIFY_PROMPT = """You are classifying questions for a TxDOT structural concrete repair knowledge base.

Question types:
- type_a: Looking up a regulation, specification, or definition.
  Examples: "What does DMS-4655 specify for compressive strength?", "What is the w/c ratio limit for Item 420?"
- type_b: Compliance judgment — does a specific observed condition meet a spec?
  Examples: "This crack is 0.4mm wide, does it meet tolerance?", "Is 3000 psi compressive strength acceptable for this application?"
- type_c: Procedure or workflow — steps, sequence, process.
  Examples: "What are the steps for surface preparation?", "What is the approval process for a repair method?"
- type_d: Repair guidance — what method to use and how, given a damage condition.
  Examples: "How do I repair spalling on a bridge pier?", "What repair method should I use for delamination?",
            "Transverse cracks appeared on the deck — what should I do?"

CLARIFICATION RULES — be very conservative. Only set needs_clarification=true if ALL of these are true:
1. The question type is type_b or type_d
2. The damage type or condition is completely unidentifiable (no defect name, no measurements, no element mentioned)
3. Proceeding without clarification would make retrieval impossible

Do NOT ask for clarification if:
- The user has provided measurements (depth, width, area percentage, etc.)
- The user has named a specific defect type (spall, crack, delamination, corrosion, etc.)
- The user has referenced a specific Item number or standard
- The user has identified the structural element (pier, deck, beam, abutment, etc.)
- Additional details would be helpful but the question is already answerable from the documents

When in doubt, proceed without clarification. The documents contain threshold tables that handle ranges of conditions.

Respond with JSON only:
{{
  "type": "type_a|type_b|type_c|type_d",
  "key_entities": ["list", "of", "key", "terms"],
  "needs_clarification": true|false,
  "missing_info": ["list of missing details, empty if none"]
}}

Question: {question}"""


def classify(question: str, claude_client: anthropic.Anthropic) -> dict:
    resp = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(question=question)}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"type": "type_a", "key_entities": [], "needs_clarification": False, "missing_info": []}
    return result


def format_clarification_request(missing_info: list[str], question: str) -> str:
    items = "\n".join(f"- {item}" for item in missing_info)
    return (
        f"To provide an accurate repair recommendation, I need a few more details:\n\n"
        f"{items}\n\n"
        f"Please provide as much of this information as you can."
    )
