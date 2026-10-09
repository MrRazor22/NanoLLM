from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datasets import load_dataset

class Sources:
    """Consolidated curriculum and HuggingFace training sources."""

    @staticmethod
    def load_typed(repeat: int = 5) -> List[Dict[str, Any]]:
        samples = [
            {"state": "Postgres CPU at 99%", "questions": {"dept": {"type": "choice", "options": ["Billing", "Infra", "Sales", "Security"]}}, "gold": {"dept": "Infra"}},
            {"state": "Charged twice for subscription", "questions": {"dept": {"type": "choice", "options": ["Billing", "Infra", "Sales", "Security"]}}, "gold": {"dept": "Billing"}},
        ]
        return samples * repeat

    @staticmethod
    def load_glaive(limit: int = 5000, seed: int = 42) -> List[Dict[str, Any]]:
        try:
            ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
            samples = []
            for item in ds:
                if len(samples) >= limit: break
                samples.append({"state": item.get("prompt", ""), "questions": {"tool": {"type": "choice", "options": ["call", "none"]}}, "gold": {"tool": "call"}})
            return samples
        except Exception:
            return Sources.load_typed(repeat=100)

    @staticmethod
    def load_generic(dataset_name: str, split: str = "train", limit: int = 3000) -> List[Dict[str, Any]]:
        try:
            ds = load_dataset(dataset_name, split=split, streaming=True)
            out = []
            for item in ds:
                if len(out) >= limit: break
                text = str(item.get("text") or item.get("prompt") or "")
                out.append({"state": text[:500], "questions": {"intent": {"type": "choice", "options": ["general", "specific"]}}, "gold": {"intent": "general"}})
            return out
        except Exception:
            return []

__all__ = ["Sources"]
