"""Utilities to safely parse LLM output that *should* be JSON but often isn't.

Drop-in fix for:
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

Example usage
-------------
raw = corrected_vlspec.content  # or your LLM response
spec = parse_jsonish(raw)       # -> Python dict/list
"""
from __future__ import annotations

import json
import re
from typing import Any

CONTROL_WS = "\u200b\u200c\u200d\ufeff"  # zero-width, BOM, etc.
SMART_QUOTES = str.maketrans({
    "\u2018": "'", "\u2019": "'",  # single smart quotes
    "\u201C": '"', "\u201D": '"',  # double smart quotes
})


def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences (```...```) and front/back matter."""
    t = text.strip()
    if t.startswith("```"):
        # remove opening fence
        t = t.split("\n", 1)[1] if "\n" in t else ""
        # remove optional language tag at first line already handled by split
        if t.startswith("json\n"):
            t = t[5:]
        # remove closing fence
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


def extract_json_block(text: str) -> str:
    """Extract the first balanced {...} or [...] block from text.
    Falls back to the original string if braces aren't found.
    """
    s = text
    start_candidates = [(s.find("{"), "{"), (s.find("["), "[")]
    start_candidates = [(i, ch) for i, ch in start_candidates if i != -1]
    if not start_candidates:
        return s
    start, ch = min(start_candidates, key=lambda p: p[0])
    open_ch, close_ch = ("{", "}") if ch == "{" else ("[", "]")

    depth = 0
    in_str = False
    esc = False
    end = None
    for i, c in enumerate(s[start:], start):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return s[start:end] if end else s[start:]


def normalize_jsonish(text: str) -> str:
    """Light-weight normalizations to convert 'JSON-ish' into strict JSON."""
    t = text
    # Remove leading labels like "Vega-Lite Specification:" etc.
    t = re.sub(r"^[\w\- ]+:\s*(?=[\[{])", "", t.strip())
    # Strip code fences
    t = strip_code_fences(t)
    # Remove stray control whitespace and BOMs
    t = t.translate({ord(ch): None for ch in CONTROL_WS})
    # Standardize quotes
    t = t.translate(SMART_QUOTES)
    # Replace Python/SQL-ish literals
    t = re.sub(r"\bNone\b", "null", t)
    t = re.sub(r"\bNULL\b", "null", t)
    t = re.sub(r"\bTrue\b", "true", t)
    t = re.sub(r"\bFalse\b", "false", t)
    # Common unicode operators inside strings → ASCII (safe for JSON)
    t = t.replace("≤", "<=").replace("≥", ">=")
    # If text contains extra prose around JSON, try to slice the first block
    if not t.lstrip().startswith(("{", "[")):
        t = extract_json_block(t)
    return t.strip()


def parse_jsonish(text: Any) -> Any:
    """Robustly parse *text* into JSON (dict/list).

    Raises a ValueError with the first 200 chars of problematic input
    if parsing fails.
    """
    if text is None:
        raise ValueError("Input is None; LLM/content field is empty.")
    if not isinstance(text, str):
        # Some SDKs return objects; try common accessors before failing
        try:
            text = text.content  # e.g., LangChain AIMessage
        except Exception:
            pass
    if not isinstance(text, str):
        raise ValueError(f"Expected str, got {type(text).__name__}")

    t = normalize_jsonish(text)
    if not t:
        raise ValueError("Empty string after normalization; no JSON to parse.")

    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        snippet = (t[:200] + ("…" if len(t) > 200 else "")).replace("\n", "\\n")
        raise ValueError(f"Failed to parse JSON (near pos {e.pos}): {e.msg}\nSnippet: {snippet}") from e



