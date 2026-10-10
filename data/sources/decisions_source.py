import json
from pathlib import Path
from typing import List, Sequence
from datasets import load_dataset
from inference.case import DecisionCase
from inference.questions import Choice
from data.sources import RAW_DIR, ADAPTED_DIR, IDataSource, SourceRecord, cached_jsonl

class DecisionsSource(IDataSource):
    """Real Typed-Decisions dataset extractor: loads multi-question typed decision benchmarks."""
    name = "typed_decisions"

    def __init__(self, repeat: int = 5):
        self.repeat = repeat

    def read(self) -> Sequence[SourceRecord]:
        adapted_path = ADAPTED_DIR / f"{self.name}.jsonl"
        if adapted_path.exists():
            records: List[SourceRecord] = []
            for line in adapted_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                qs = [
                    Choice(
                        name=q["name"],
                        options=q["options"],
                        instruction=q["instruction"],
                        target=q["target"],
                    )
                    for q in d["questions"]
                ]
                sample = DecisionCase(state=d["state"], questions=qs)
                records.append(SourceRecord(sample=sample, tag="unsplit"))
            return records * self.repeat

        rows = cached_jsonl(
            RAW_DIR / f"{self.name}.jsonl",
            lambda: list(load_dataset("LocalLLaMA/typed-decisions", "all", split="train")),
        )

        records: List[SourceRecord] = []
        cache_rows: List[dict] = []
        for row in rows:
            q_defs = row.get("questions", {})
            g_defs = row.get("gold", {})
            q_dict = json.loads(q_defs) if isinstance(q_defs, str) else q_defs
            g_dict = json.loads(g_defs) if isinstance(g_defs, str) else g_defs
            qs = []
            q_cache = []
            for qid, q_data in q_dict.items():
                if q_data.get("type") == "choice":
                    opt = q_data.get("criteria", {})
                    ins = q_data.get("instructions", "Select the best option:")
                    tgt = g_dict.get(qid, {}).get("label")
                    qs.append(Choice(name=qid, options=opt, instruction=ins, target=tgt))
                    q_cache.append({"name": qid, "options": opt, "instruction": ins, "target": tgt})
            if qs:
                sample = DecisionCase(state=row.get("state", ""), questions=qs)
                records.append(SourceRecord(sample=sample, tag="unsplit"))
                cache_rows.append({"state": sample.state, "questions": q_cache})

        if cache_rows:
            adapted_path.parent.mkdir(parents=True, exist_ok=True)
            adapted_path.write_text("\n".join(json.dumps(r) for r in cache_rows) + "\n", encoding="utf-8")

        return records * self.repeat

__all__ = ["DecisionsSource"]
