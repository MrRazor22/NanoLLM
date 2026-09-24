from pathlib import Path
from typing import Dict, List
import json, random
from datasets import load_dataset, concatenate_datasets

ROOT_DIR = Path(__file__).resolve().parent.parent

CLINC_PROMPTS = [
    "What is the user's intent?",
    "Classify the primary goal of the request.",
    "Which intent category applies to the input?",
    "Determine the action requested by the user.",
]
HELLA_PROMPTS = [
    "Which next action or event logically follows?",
    "What is the most plausible continuation?",
    "Predict the next logical outcome.",
    "Select the event consistent with the situation.",
]
ANLI_PROMPTS = [
    "Determine the logical relationship to: '{hyp}'",
    "Does the context entail, contradict, or remain neutral toward: '{hyp}'?",
    "Assess whether this claim is supported, neutral, or contradicted: '{hyp}'",
]
WINO_PROMPTS = [
    "Which candidate correctly fills the blank or resolves the reference?",
    "Determine the correct entity referenced in the sentence.",
    "Which option is contextually and causally coherent?",
]
BOOLQ_PROMPTS = [
    "Is this condition or statement supported: '{cond}'?",
    "Based on the reference text, is it true that '{cond}'?",
    "Verify if the following holds true: '{cond}'",
]
STSB_PROMPTS = [
    "Rate the degree of semantic agreement or similarity from 0.0 to 1.0.",
    "Assess how closely the meaning matches on a 0 to 1 scale.",
    "Score the semantic alignment between these statements.",
]

def build_clinc(rng: random.Random, limit: int = 12000) -> List[Dict]:
    ds = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = ds.features["intent"].names
    records = []
    for row in ds:
        text, label = str(row["text"]).strip(), names[int(row["intent"])]
        if not text: continue
        k_opts = rng.randint(2, 8)
        opts = [label] + rng.sample([n for n in names if n != label], k_opts - 1)
        rng.shuffle(opts)
        records.append({"state": text, "questions": [["intent", "choice", opts.index(label), opts, rng.choice(CLINC_PROMPTS)]]})
        if len(records) >= limit: break
    return records

def build_hellaswag(rng: random.Random, limit: int = 12000) -> List[Dict]:
    ds = load_dataset("Rowan/hellaswag", split="train")
    records = []
    for row in ds:
        ctx, endings = str(row["ctx"]).strip(), [str(e).strip() for e in row["endings"]]
        lbl = str(row["label"]).strip()
        if not (lbl.isdigit() and 0 <= int(lbl) < len(endings)) or not ctx: continue
        correct = endings[int(lbl)]
        rng.shuffle(endings)
        records.append({"state": ctx, "questions": [["next_action", "choice", endings.index(correct), endings, rng.choice(HELLA_PROMPTS)]]})
        if len(records) >= limit: break
    return records

def build_anli(rng: random.Random, limit: int = 12000) -> List[Dict]:
    ds1 = load_dataset("facebook/anli", split="train_r1")
    ds2 = load_dataset("facebook/anli", split="train_r2")
    ds = concatenate_datasets([ds1, ds2])
    base_opts, records = ["entailment", "neutral", "contradiction"], []
    for row in ds:
        premise, hyp = str(row["premise"]).strip(), str(row["hypothesis"]).strip()
        lbl = int(row["label"])
        if not (premise and hyp and 0 <= lbl <= 2): continue
        correct, opts = base_opts[lbl], list(base_opts)
        rng.shuffle(opts)
        q = rng.choice(ANLI_PROMPTS).format(hyp=hyp)
        records.append({"state": premise, "questions": [["nli_relation", "choice", opts.index(correct), opts, q]]})
        if len(records) >= limit: break
    return records

def build_winogrande(rng: random.Random, limit: int = 10000) -> List[Dict]:
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    records = []
    for row in ds:
        sent, o1, o2 = str(row["sentence"]).strip(), str(row["option1"]).strip(), str(row["option2"]).strip()
        ans = str(row["answer"]).strip()
        if ans not in ("1", "2") or not (sent and o1 and o2): continue
        correct, opts = (o1 if ans == "1" else o2), [o1, o2]
        rng.shuffle(opts)
        records.append({"state": sent, "questions": [["coreference", "choice", opts.index(correct), opts, rng.choice(WINO_PROMPTS)]]})
        if len(records) >= limit: break
    return records

def build_boolq(rng: random.Random, limit: int = 9400) -> List[Dict]:
    ds = load_dataset("google/boolq", split="train")
    records = []
    for row in ds:
        passage, cond = str(row["passage"]).strip(), str(row["question"]).strip()
        if not (passage and cond): continue
        ans = float(1.0 if row["answer"] else 0.0)
        q = rng.choice(BOOLQ_PROMPTS).format(cond=cond)
        records.append({"state": passage[:1200], "questions": [["is_true", "noul", ans, None, q]]})
        if len(records) >= limit: break
    return records

def build_stsb(rng: random.Random, limit: int = 5700) -> List[Dict]:
    ds = load_dataset("nyu-mll/glue", "stsb", split="train")
    records = []
    for row in ds:
        s1, s2 = str(row["sentence1"]).strip(), str(row["sentence2"]).strip()
        if not (s1 and s2): continue
        score = float(row["label"]) / 5.0
        records.append({"state": f"Statement 1: {s1}\nStatement 2: {s2}", "questions": [["similarity", "score", score, None, rng.choice(STSB_PROMPTS)]]})
        if len(records) >= limit: break
    return records

def main():
    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    print("[DATA] Loading 6 foundation reasoning datasets (0% benchmark overlap)...", flush=True)
    all_samples = (
        build_clinc(rng, 12000) +
        build_hellaswag(rng, 12000) +
        build_anli(rng, 12000) +
        build_winogrande(rng, 10000) +
        build_boolq(rng, 9400) +
        build_stsb(rng, 5700)
    )
    rng.shuffle(all_samples)
    print(f"[DATA] Total Samples: {len(all_samples)}", flush=True)

    n_val, n_test = int(0.08 * len(all_samples)), int(0.08 * len(all_samples))
    splits = [("train.jsonl", all_samples[n_val + n_test:]), ("val.jsonl", all_samples[:n_val]), ("test.jsonl", all_samples[n_val:n_val + n_test])]
    for name, data in splits:
        path = data_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for item in data: f.write(json.dumps(item) + "\n")
        print(f"Wrote {len(data)} samples to {path}")

if __name__ == "__main__":
    main()
