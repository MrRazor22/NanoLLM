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

    def load_cases(self) -> Sequence[DecisionCase]: ...
 
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
