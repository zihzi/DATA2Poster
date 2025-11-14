from __future__ import annotations

import json
import re
from typing import Tuple, List

_JS_LINE_COMMENT = re.compile(r"//[^\n]*")
_JS_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_NAN_INF = re.compile(r"(?<![A-Za-z0-9_])(?:NaN|Infinity|-Infinity)(?![A-Za-z0-9_])")
_BAD_CONTROL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def _excerpt_around(text: str, pos: int, width: int = 160) -> Tuple[str, str]:
    half = max(width // 2, 1)
    start = max(pos - half, 0)
    end = min(pos + half, len(text))
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    slice_ = text[start:end]
    caret = " " * (pos - start) + "^"
    # Keep newlines for readability; escape tabs only
    slice_ = slice_.replace("\t", "\\t")
    return f"{prefix}{slice_}{suffix}", caret

def _likely_causes(text: str, pos: int) -> List[str]:
    neighborhood = text[max(pos - 80, 0) : min(pos + 80, len(text))]
    tips: List[str] = []
    if re.search(r",\s*[}\]]", neighborhood):
        tips.append("Trailing comma before '}' or ']'.")
    if _JS_LINE_COMMENT.search(text) or _JS_BLOCK_COMMENT.search(text):
        tips.append("JavaScript-style comments present.")
    if re.search(r"'(?:\\.|[^'\\])*'", text):
        tips.append("Single-quoted strings; JSON requires double quotes.")
    if re.search(r'\\[^"\\/bfnrtu]', neighborhood):
        tips.append("Invalid escape sequence (e.g., \\x).")
    if _NAN_INF.search(text):
        tips.append("NaN/Infinity; use null.")
    if _BAD_CONTROL.search(text):
        tips.append("Unescaped control character(s).")
    if pos >= len(text) - 1:
        tips.append("Input may be truncated.")
    return tips or ["General syntax issue near the caret."]

def _strip_bom(s: str) -> str:
    return s.lstrip("\ufeff")

def _strip_js_comments(s: str) -> str:
    # Stateful removal to avoid nuking // inside strings.
    out = []
    i = 0
    in_str = False
    esc = False
    quote = ""
    while i < len(s):
        ch = s[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            i += 1
            continue
        if ch in ("'", '"'):
            in_str, quote = True, ch
            out.append(ch); i += 1; continue
        if ch == "/" and i + 1 < len(s) and s[i + 1] == "/":
            i += 2
            while i < len(s) and s[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < len(s) and s[i + 1] == "*":
            i += 2
            while i + 1 < len(s) and not (s[i] == "*" and s[i + 1] == "/"):
                i += 1
            i = min(i + 2, len(s))
            continue
        out.append(ch); i += 1
    return "".join(out)

def _remove_trailing_commas(s: str) -> str:
    out = []
    stack = []
    i = 0
    in_str = False
    esc = False
    quote = ""
    while i < len(s):
        ch = s[i]
        out.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            i += 1
            continue
        if ch in ("'", '"'):
            in_str, quote = True, ch
            i += 1; continue
        if ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            j = len(out) - 2
            while j >= 0 and out[j].isspace():
                j -= 1
            if j >= 0 and out[j] == ",":
                del out[j]
            if stack:
                stack.pop()
        i += 1
    return "".join(out)

def _replace_nan_inf(s: str) -> str:
    return _NAN_INF.sub("null", s)

def _sanitize_best_effort(s: str) -> str:
    s2 = s.replace("\r\n", "\n").replace("\r", "\n")
    s2 = _strip_bom(s2)
    s2 = _strip_js_comments(s2)
    s2 = _replace_nan_inf(s2)
    s2 = _remove_trailing_commas(s2)
    return s2

def loads_with_diagnostics(t: str, *, try_fix: bool = True) -> object:
    """
    Strict JSON parse with clear diagnostics.
    Set try_fix=False to disable best-effort sanitization.
    """
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        if try_fix:
            repaired = _sanitize_best_effort(t)
            if repaired != t:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass  # fall through to detailed error

        excerpt, caret = _excerpt_around(t, e.pos, width=180)
        tips = "; ".join(_likely_causes(t, e.pos))
        message = (
            "Failed to parse JSON.\n"
            f"Location : line {e.lineno}, column {e.colno} (pos {e.pos})\n"
            f"Message  : {e.msg}\n"
            "Context  :\n"
            f"{excerpt}\n{caret}\n"
            f"Hint     : {tips}"
        )
        raise ValueError(message) from e
