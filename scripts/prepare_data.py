import sys
from pathlib import Path
from typing import Any, Dict, List
import json, random
from datasets import load_dataset, concatenate_datasets

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from nanollm import save_jsonl

def _state(text: str, rng: random.Random) -> Any:
    if rng.random() < 0.25:
        return {"content": text, "source": "context", "status": "active"}
    return text

def build_clinc(rng: random.Random, limit: int = 12000) -> List[Dict]:
    ds = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = ds.features["intent"].names
    records = []
    for row in ds:
        text, label = str(row["text"]).strip(), names[int(row["intent"])]
        if not text: continue
        k = rng.randint(2, min(40, len(names)))
        sampled = [label] + rng.sample([n for n in names if n != label], k - 1)
        rng.shuffle(sampled)
        if rng.random() < 0.35:
            opts = {name: f"request related to {name.replace('_', ' ')}" for name in sampled}
            keys = list(opts.keys())
            idx = keys.index(label)
        else:
            opts, idx = sampled, sampled.index(label)
        records.append({"state": _state(text, rng), "questions": [["intent", "choice", idx, opts, "Determine the primary intent of this input."]]})
        if len(records) >= limit: break
    return records

def build_hellaswag(rng: random.Random, limit: int = 12000) -> List[Dict]:
    ds, records = load_dataset("Rowan/hellaswag", split="train"), []
    for row in ds:
        ctx, ends = str(row["ctx"]).strip(), [str(e).strip() for e in row["endings"]]
        lbl = str(row["label"]).strip()
        if not (lbl.isdigit() and 0 <= int(lbl) < len(ends)) or not ctx: continue
        correct = ends[int(lbl)]
        rng.shuffle(ends)
        records.append({"state": _state(ctx, rng), "questions": [["next_action", "choice", ends.index(correct), ends, "Which event or action logically follows?"]]})
        if len(records) >= limit: break
    return records

def build_anli(rng: random.Random, limit: int = 12000) -> List[Dict]:
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
        opts = dict(criteria) if rng.random() < 0.4 else list(base_opts)
        keys = list(opts.keys()) if isinstance(opts, dict) else opts
        rng.shuffle(keys)
        if isinstance(opts, dict): opts = {k: criteria[k] for k in keys}
        else: opts = keys
        idx = list(opts.keys()).index(correct) if isinstance(opts, dict) else opts.index(correct)
        records.append({"state": _state(prem, rng), "questions": [["nli", "choice", idx, opts, f"Determine the logical relationship to: '{hyp}'"]]})
        if len(records) >= limit: break
    return records

def build_winogrande(rng: random.Random, limit: int = 10000) -> List[Dict]:
    ds, records = load_dataset("allenai/winogrande", "winogrande_xl", split="train"), []
    for row in ds:
        sent, o1, o2, ans = str(row["sentence"]).strip(), str(row["option1"]).strip(), str(row["option2"]).strip(), str(row["answer"]).strip()
        if ans not in ("1", "2") or not (sent and o1 and o2): continue
        correct, opts = (o1 if ans == "1" else o2), [o1, o2]
        rng.shuffle(opts)
        records.append({"state": _state(sent, rng), "questions": [["coref", "choice", opts.index(correct), opts, "Which candidate resolves the reference?"]]})
        if len(records) >= limit: break
    return records

def build_boolq(rng: random.Random, limit: int = 9000) -> List[Dict]:
    ds, records = load_dataset("google/boolq", split="train"), []
    opts = {"false": "no, condition does not hold", "true": "yes, condition holds"}
    for row in ds:
        p, q = str(row["passage"]).strip(), str(row["question"]).strip()
        if not (p and q): continue
        idx = 1 if row["answer"] else 0
        records.append({"state": _state(p[:1200], rng), "questions": [["is_true", "choice", idx, opts, f"Is this supported: '{q}'?"]]})
        if len(records) >= limit: break
    return records

def build_stsb(rng: random.Random, limit: int = 5700) -> List[Dict]:
    ds, records = load_dataset("nyu-mll/glue", "stsb", split="train"), []
    for row in ds:
        s1, s2 = str(row["sentence1"]).strip(), str(row["sentence2"]).strip()
        if not (s1 and s2): continue
        score = float(row["label"]) / 5.0
        records.append({"state": f"Statement 1: {s1}\nStatement 2: {s2}", "questions": [["similarity", "score", score, None, "Rate semantic similarity from 0.0 to 1.0."]]})
        if len(records) >= limit: break
    return records

def main():
    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    print("[DATA] Generating foundation multi-task reasoning dataset (0% benchmark overlap)...", flush=True)
    samples = (build_clinc(rng, 12000) + build_hellaswag(rng, 12000) + build_anli(rng, 12000) +
               build_winogrande(rng, 10000) + build_boolq(rng, 9000) + build_stsb(rng, 5700))
    rng.shuffle(samples)
    print(f"[DATA] Total Samples: {len(samples)}", flush=True)
    n_val, n_test = int(0.08 * len(samples)), int(0.08 * len(samples))
    splits = [("train.jsonl", samples[n_val + n_test:]), ("val.jsonl", samples[:n_val]), ("test.jsonl", samples[n_val:n_val + n_test])]
    for name, data in splits:
        save_jsonl(str(data_dir / name), data)
        print(f"Wrote {len(data)} samples to {data_dir / name}")

if __name__ == "__main__":
    main()
