from typing import List, Optional


class RewriteResult:
    def __init__(self, rewrites: List[str], hyde: Optional[List[str]] = None):
        self.rewrites = rewrites
        self.hyde = hyde or []


def select_topk_unique(candidates: List[str], k: int = 5) -> List[str]:
    seen = set()
    out = []
    for c in candidates:
        s = cleanup_text(c)
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= k:
            break
    return out


def cleanup_text(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    # Strip common prompt artifacts
    lower = s.lower()
    for prefix in ["paraphrase:", "paraphrase this:", "paraphrasephrase:", "paraphrase "]:
        if lower.startswith(prefix):
            s = s[len(prefix):].strip(" :-\t")
            break
    # Discard very short or non-informative strings
    if len(s) < 6:
        return ""
    # Ensure contains alphanumeric characters
    if not any(ch.isalnum() for ch in s):
        return ""
    return s

