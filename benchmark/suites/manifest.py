import hashlib
import json
from pathlib import Path
from typing import Optional, Protocol, Sequence
import urllib.request

DEFAULT_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"

class IManifestPolicy(Protocol):
    def load(
        self,
        filename: str,
        expected_sha: str,
        url: Optional[str] = None,
    ) -> Sequence[dict]: ...

class VerifiedManifestPolicy(IManifestPolicy):
    def __init__(self, manifests_dir: Optional[Path] = None):
        self.manifests_dir = manifests_dir or DEFAULT_MANIFESTS_DIR

    def _fetch_verified_path(self, filename: str, expected_sha: str, url: Optional[str] = None) -> Path:
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        path = self.manifests_dir / filename
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            if not url:
                raise ValueError(f"Integrity check failed for '{filename}': SHA256 mismatch and no source URL.")
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request) as response:
                path.write_bytes(response.read())
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise ValueError(f"Integrity check failed for '{filename}': SHA256 mismatch")
        return path

    def load(
        self,
        filename: str,
        expected_sha: str,
        url: Optional[str] = None,
    ) -> Sequence[dict]:
        path = self._fetch_verified_path(filename, expected_sha, url)
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            return pq.read_table(path).to_pylist()

        cases: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cases.append(json.loads(line))
        return cases
