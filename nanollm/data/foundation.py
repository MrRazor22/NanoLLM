import random
from typing import Any, Dict, List, Tuple
from datasets import concatenate_datasets, load_dataset
from nanollm.data.builder import split_train_val

def _state(text: str, rng: random.Random) -> Any:
    if rng.random() < 0.25:
        return {"content": text, "source": "context", "status": "active"}
    return text

class FoundationCurriculum:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _build_clinc(self, limit: int = 12000) -> List[Dict]:
        ds = load_dataset("clinc/clinc_oos", "plus", split="train")
        names = ds.features["intent"].names
        records = []
        for row in ds:
            text, label = str(row["text"]).strip(), names[int(row["intent"])]
            if not text: continue
            k = self.rng.randint(2, min(40, len(names)))
            sampled = [label] + self.rng.sample([n for n in names if n != label], k - 1)
            self.rng.shuffle(sampled)
            opts = {name: f"request related to {name.replace('_', ' ')}" for name in sampled} if self.rng.random() < 0.35 else sampled
            idx = list(opts.keys()).index(label) if isinstance(opts, dict) else opts.index(label)
            records.append({"state": _state(text, self.rng), "questions": [["intent", "choice", idx, opts, "Determine the primary intent of this input."]]})
            if len(records) >= limit: break
        return records

    def _build_hellaswag(self, limit: int = 12000) -> List[Dict]:
        ds, records = load_dataset("Rowan/hellaswag", split="train"), []
        for row in ds:
            ctx, ends = str(row["ctx"]).strip(), [str(e).strip() for e in row["endings"]]
            lbl = str(row["label"]).strip()
            if not (lbl.isdigit() and 0 <= int(lbl) < len(ends)) or not ctx: continue
            correct = ends[int(lbl)]
            self.rng.shuffle(ends)
            records.append({"state": _state(ctx, self.rng), "questions": [["next_action", "choice", ends.index(correct), ends, "Which event or action logically follows?"]]})
            if len(records) >= limit: break
        return records

    def _build_anli(self, limit: int = 12000) -> List[Dict]:
        ds = concatenate_datasets([load_dataset("facebook/anli", split="train_r1"), load_dataset("facebook/anli", split="train_r2")])
        base_opts, records = ["entailment", "neutral", "contradiction"], []
        criteria = {
            "entailment": "the claim is definitely true given the premise",
            "neutral": "the claim might be true or false given the premise",
            "contradiction": "the claim is definitely false given the premise"
        }
        for row in ds:
            prem, hyp, lbl = str(row["premise"]).strip(), str(row["hypothesis"]).strip(), int(row["label"])
            if not (prem and hyp and 0 <= lbl <= 2): continue
            correct = base_opts[lbl]
            opts = dict(criteria) if self.rng.random() < 0.4 else list(base_opts)
            keys = list(opts.keys()) if isinstance(opts, dict) else opts
            self.rng.shuffle(keys)
            opts = {k: criteria[k] for k in keys} if isinstance(opts, dict) else keys
            idx = list(opts.keys()).index(correct) if isinstance(opts, dict) else opts.index(correct)
            records.append({"state": _state(prem, self.rng), "questions": [["nli", "choice", idx, opts, f"Determine the logical relationship to: '{hyp}'"]]})
            if len(records) >= limit: break
        return records

    def _build_winogrande(self, limit: int = 10000) -> List[Dict]:
        ds, records = load_dataset("allenai/winogrande", "winogrande_xl", split="train"), []
        for row in ds:
            sent, o1, o2, ans = str(row["sentence"]).strip(), str(row["option1"]).strip(), str(row["option2"]).strip(), str(row["answer"]).strip()
            if ans not in ("1", "2") or not (sent and o1 and o2): continue
            correct, opts = (o1 if ans == "1" else o2), [o1, o2]
            self.rng.shuffle(opts)
            records.append({"state": _state(sent, self.rng), "questions": [["coref", "choice", opts.index(correct), opts, "Which candidate resolves the reference?"]]})
            if len(records) >= limit: break
        return records

    def _build_boolq(self, limit: int = 9000) -> List[Dict]:
        ds, records = load_dataset("google/boolq", split="train"), []
        opts = {"false": "no, condition does not hold", "true": "yes, condition holds"}
        for row in ds:
            p, q = str(row["passage"]).strip(), str(row["question"]).strip()
            if not (p and q): continue
            idx = 1 if row["answer"] else 0
            records.append({"state": _state(p[:1200], self.rng), "questions": [["is_true", "choice", idx, opts, f"Is this supported: '{q}'?"]]})
            if len(records) >= limit: break
        return records

    def build(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        samples = (self._build_clinc() + self._build_hellaswag() + self._build_anli() +
                   self._build_winogrande() + self._build_boolq())
        self.rng.shuffle(samples)
        return split_train_val(samples, 0.08, self.rng)
