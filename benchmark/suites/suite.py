from typing import Mapping, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase

@runtime_checkable
class IBenchmarkSuite(Protocol):
    """Sovereign benchmark suite contract: provides evaluation cases, identity, and published baselines."""
    @property
    def name(self) -> str: ...

    @property
    def manifest_hash(self) -> str: ...

    @property
    def published_baselines(self) -> Mapping[str, float]: ...

    def cases(self) -> Sequence[DecisionCase]: ...
 
def load_verified_manifest(path, expected_sha: str, url: str | None = None) -> bytes:
    import hashlib, urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        if not url: raise ValueError(f"Integrity check failed for '{path.name}': SHA256 mismatch and no source URL.")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp: path.write_bytes(resp.read())
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise ValueError(f"Integrity check failed for '{path.name}': SHA256 mismatch")
    return content

def adapt_manifest_case(d: dict, source: str) -> DecisionCase:
    import json
    state = str(d.get("state", ""))
    cat = str(d.get("category") or d.get("workflow") or "benchmark")
    split = d.get("split")

    qs_raw = d.get("questions")
    if not qs_raw:
        raise ValueError(f"Manifest record missing 'questions': {list(d.keys())}")
    qs = json.loads(qs_raw) if isinstance(qs_raw, str) else qs_raw
    gold_raw = d.get("gold")
    gold = json.loads(gold_raw) if isinstance(gold_raw, str) else (gold_raw or {})
    _, spec = next(iter(qs.items()))
    crit = spec.get("options") or spec.get("criteria")
    if isinstance(crit, dict):
        candidates = tuple(str(o) for o in crit.keys())
    elif isinstance(crit, list) and crit:
        candidates = tuple(str(c) for c in crit)
    else:
        candidates = ("false", "true")

    g_item = next(iter(gold.values())) if isinstance(gold, dict) and gold else gold
    if isinstance(g_item, dict) and "probabilities" in g_item:
        raw_probs = g_item["probabilities"]
        probs = {c: float(raw_probs.get(c if c in raw_probs else str(i), 0.0)) for i, c in enumerate(candidates)}
        if sum(probs.values()) == 0.0 and "label" in g_item:
            lbl = str(g_item["label"]).strip()
            if lbl.isdigit() and 0 <= int(lbl) < len(candidates) and lbl not in candidates:
                lbl = candidates[int(lbl)]
            probs = {c: (1.0 if c == lbl or c.lower() == lbl.lower() else 0.0) for c in candidates}
    else:
        raw_lbl = (g_item.get("label") if isinstance(g_item, dict) else g_item) if g_item else spec.get("label")
        lbl = str(raw_lbl).strip()
        if lbl.isdigit() and 0 <= int(lbl) < len(candidates) and lbl not in candidates:
            lbl = candidates[int(lbl)]
        probs = {c: (1.0 if c == lbl or c.lower() == lbl.lower() else 0.0) for c in candidates}

    if sum(probs.values()) == 0.0:
        raise ValueError(f"Target '{lbl}' not found in candidates {candidates}")

    return DecisionCase(
        state=state,
        candidates=candidates,
        probabilities=probs,
        category=cat,
        source=source,
        split=split,
    )
