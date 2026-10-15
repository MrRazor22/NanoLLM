import json
from pathlib import Path
from typing import Optional, Sequence, Union
from data.case import DataSplits, DecisionCase, SplitTag
from data.dataset import IDataset
from data.dataset_extensions import CORPUS_DIR

class DiskStorageLayer(IDataset):
    """Sovereign composable layer (λ: IDataset -> IDataset): decorates a dataset with JSONL disk persistence."""

    def __init__(
        self,
        inner: Optional[IDataset] = None,
        corpus_dir: Optional[Union[str, Path]] = None,
    ):
        self.inner = inner
        self.corpus_dir = Path(corpus_dir) if corpus_dir is not None else CORPUS_DIR

    def prepare(self) -> DataSplits:
        if self.inner is None:
            raise ValueError("Cannot prepare dataset without an inner source dataset.")
        splits = self.inner.prepare()
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        for tag, items in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
            target = self.corpus_dir / f"{tag}.jsonl"
            lines = [json.dumps(s.to_dict(), default=str) for s in items]
            target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return splits

    def load(self, split: SplitTag = "train") -> Sequence[DecisionCase]:
        target = self.corpus_dir / f"{split}.jsonl"
        if not target.exists():
            raise FileNotFoundError(
                f"Corpus partition '{split}' not found at {target}. "
                f"Explicitly prepare the corpus first (e.g. run 'python -m data' or invoke dataset.prepare())."
            )
        return [
            DecisionCase.from_dict(json.loads(line))
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

__all__ = ["DiskStorageLayer"]
