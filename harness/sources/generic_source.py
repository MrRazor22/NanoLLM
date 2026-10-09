from typing import List, Optional, Sequence
from datasets import load_dataset
from piqo import Choice, DecisionSample
from harness.sources import ISource

class GenericSource(ISource):
    """Dynamic HuggingFace text-to-decision adapter: infers label names directly from dataset schema."""

    def __init__(self, dataset_name: str, question_name: str, instruction: str, limit: int = 3000):
        self.name = dataset_name.replace("/", "_")
        self.dataset_name = dataset_name
        self.question_name = question_name
        self.instruction = instruction
        self.limit = limit

    def load(self, limit: Optional[int] = None) -> Sequence[DecisionSample]:
        max_items = limit or self.limit
        try:
            ds = load_dataset(self.dataset_name, split="train", streaming=True)
            samples: List[DecisionSample] = []
            for item in ds:
                text = str(item.get("text") or item.get("prompt") or "")
                lbl = item.get("label_text") or item.get("label") or "general"
                label_str = str(lbl).replace("_", " ")
                samples.append(
                    DecisionSample(
                        state=text[:500],
                        questions=[
                            Choice(
                                name=self.question_name,
                                options=[label_str, "other"],
                                instruction=self.instruction,
                                target=label_str,
                            )
                        ],
                    )
                )
                if len(samples) >= max_items:
                    break
            return samples
        except Exception:
            return []

__all__ = ["GenericSource"]
