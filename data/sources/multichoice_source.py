from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
from datasets import load_dataset
import orjson

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import RAW_DIR

@dataclass(frozen=True, slots=True)
class MultiChoiceSpec:
    """Core Domain DTO: Declarative specification for multi-choice reasoning datasets."""
    dataset: str
    subset: Optional[str] = None
    state_field: Union[str, Tuple[str, ...]] = "question"
    option_fields: Optional[Tuple[str, ...]] = None
    options_field: Optional[str] = None
    fixed_options: Optional[Tuple[str, ...]] = None
    target_field: str = "label"
    one_based: bool = False
    category: str = "reasoning"
    instruction: str = "Select the best option:"
    splits: Sequence[Union[str, Tuple[str, str]]] = (("train", "train"), ("test", "test"))
    train_limit: Optional[int] = None
    test_limit: Optional[int] = 300

MULTICHOICE_REASONING_BENCHMARKS: Sequence[MultiChoiceSpec] = (
    MultiChoiceSpec("allenai/ai2_arc", subset="ARC-Challenge", target_field="answerKey", category="deductive_reasoning"),
    MultiChoiceSpec("allenai/openbookqa", subset="main", state_field="question_stem", target_field="answerKey", category="causal_reasoning", train_limit=1500),
    MultiChoiceSpec("facebook/anli", state_field=("premise", "hypothesis"), fixed_options=("Entailment", "Neutral", "Contradiction"), splits=(("train_r1", "train"), ("test_r1", "test")), category="adversarial_reasoning", train_limit=1500, test_limit=200),
    MultiChoiceSpec("allenai/art", state_field=("observation_1", "observation_2"), option_fields=("hypothesis_1", "hypothesis_2"), one_based=True, splits=(("train", "train"), ("validation", "test")), category="abductive_reasoning", train_limit=2000),
    MultiChoiceSpec("aps/super_glue", subset="copa", state_field="premise", option_fields=("choice1", "choice2"), splits=(("train", "train"), ("validation", "test")), category="causal_reasoning", train_limit=500, test_limit=200),
    MultiChoiceSpec("Rowan/hellaswag", state_field="ctx", options_field="endings", splits=(("train", "train"), ("validation", "test")), category="contextual_continuation", train_limit=2000),
    MultiChoiceSpec("GBaker/MedQA-USMLE-4-options", state_field="question", options_field="options", target_field="answer_idx", category="clinical_differential", train_limit=2000),
    MultiChoiceSpec("allenai/sciq", state_field="question", option_fields=("correct_answer", "distractor1", "distractor2", "distractor3"), target_field="correct_answer", splits=(("train", "train"), ("validation", "test")), category="scientific_reasoning", train_limit=1500, test_limit=200),
)

def _load_split(spec: MultiChoiceSpec, name: str, split: str, limit: Optional[int]) -> List[dict]:
    target = RAW_DIR / f"{name}_{split}.jsonl"
    if target.exists():
        rows = [orjson.loads(line) for line in target.read_bytes().splitlines() if line.strip()]
        return rows[:limit] if limit is not None and len(rows) > limit else rows
    args = [spec.dataset] + ([spec.subset] if spec.subset else [])
    rows = list(load_dataset(*args, split=split))
    if limit is not None:
        rows = rows[:limit]
    if rows:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\n".join(orjson.dumps(r) for r in rows) + b"\n")
    return rows

class MultiChoiceReasoningSource(IDataSource):
    """Universal declarative source adapter for multiple-choice reasoning benchmarks."""

    def __init__(self, spec: Optional[MultiChoiceSpec] = None):
        self.spec = spec or MULTICHOICE_REASONING_BENCHMARKS[0]
        self.name = f"{self.spec.dataset.replace('/', '_')}_{self.spec.subset}" if self.spec.subset else self.spec.dataset.replace("/", "_")

    def _parse(self, r: dict) -> Optional[Tuple[str, Sequence[str], str]]:
        state = " | ".join(str(r[f]).strip() for f in self.spec.state_field if f in r and r[f]) if isinstance(self.spec.state_field, tuple) else str(r.get(self.spec.state_field, "")).strip()
        if not state:
            return None
        if self.spec.fixed_options:
            opts = list(self.spec.fixed_options)
            lbl = r.get(self.spec.target_field)
            if lbl is None or not (0 <= int(lbl) < len(opts)): return None
            return state, opts, opts[int(lbl)]
        if self.spec.option_fields:
            opts = [str(r[f]).strip() for f in self.spec.option_fields if f in r and str(r[f]).strip()]
            lbl = r.get(self.spec.target_field)
            if not opts or lbl is None: return None
            tgt = str(lbl).strip() if self.spec.target_field in self.spec.option_fields else opts[int(lbl) - (1 if self.spec.one_based else 0)]
            return state, opts, tgt
        if self.spec.options_field:
            raw_opts = r.get(self.spec.options_field, {})
            opts = list(raw_opts.values()) if isinstance(raw_opts, dict) else list(raw_opts)
            raw_tgt = r.get(self.spec.target_field)
            if not opts or raw_tgt is None: return None
            tgt = raw_opts.get(raw_tgt) if isinstance(raw_opts, dict) else (opts[int(raw_tgt)] if 0 <= int(raw_tgt) < len(opts) else None)
            return (state, opts, str(tgt)) if tgt is not None else None
        choices = r.get("choices", {})
        opts = choices.get("text", [])
        labels = choices.get("label", [])
        ans = str(r.get(self.spec.target_field, "")).strip()
        if not opts or not ans: return None
        tgt = opts[labels.index(ans)] if ans in labels else (opts[int(ans) - 1] if ans.isdigit() and 0 < int(ans) <= len(opts) else None)
        return (state, opts, tgt) if tgt is not None else None

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for entry in self.spec.splits:
            split, tag = entry if isinstance(entry, (tuple, list)) else (entry, "test" if entry in ("test", "validation") else "train")
            limit = self.spec.test_limit if tag == "test" else self.spec.train_limit
            for r in _load_split(self.spec, self.name, split, limit):
                parsed = self._parse(r)
                if not parsed:
                    continue
                state, options, target = parsed
                records.append(SourceRecord(
                    sample=DecisionCase(state=state, questions=[Choice(name="choice", options=options, instruction=self.spec.instruction, target=target)], category=self.spec.category),
                    tag=tag,
                ))
        return records

__all__ = ["MultiChoiceSpec", "MULTICHOICE_REASONING_BENCHMARKS", "MultiChoiceReasoningSource"]
