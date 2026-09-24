from pathlib import Path
from typing import Any, Dict, List
import json, random
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent

def build_typed_decisions() -> List[Dict]:
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    records = []
    for row in ds:
        q_dict, g_dict = json.loads(row["questions"]), json.loads(row["gold"])
        qs = []
        for name, spec in q_dict.items():
            t, ins, crit = spec.get("type"), spec.get("instructions"), spec.get("criteria", {})
            gold = g_dict.get(name, {})
            if t == "choice":
                opts = crit if isinstance(crit, dict) else list(crit)
                opt_keys = list(opts.keys()) if isinstance(opts, dict) else opts
                lbl = gold.get("label")
                if lbl in opt_keys:
                    qs.append([name, "choice", opt_keys.index(lbl), opts, ins])
            elif t == "noul":
                crit_dict = crit if isinstance(crit, dict) and crit else {"false": "no, condition does not hold", "true": "yes, condition holds"}
                target_bool = gold.get("label") == "true" or gold.get("noul", 0) >= 0.5
                qs.append([name, "choice", 1 if target_bool else 0, crit_dict, ins])
            elif t == "score":
                opts = {str(i): c for i, c in enumerate(crit)} if isinstance(crit, list) else crit
                lbl = str(gold.get("label"))
                if lbl in opts:
                    qs.append([name, "choice", list(opts.keys()).index(lbl), opts, ins])
        if qs: records.append({"state": row["state"], "questions": qs})
    return records

def build_choice_set(path: str, sub: Any, split: str, tcol: str, lcol: str, qname: str, instr: str, rng: random.Random, limit: int, opts_dict: Any = None) -> List[Dict]:
    ds = load_dataset(path, sub, split=split) if sub else load_dataset(path, split=split)
    labels = opts_dict if opts_dict else sorted(list(set(ds[lcol])))
    opt_keys = list(labels.keys()) if isinstance(labels, dict) else labels
    records = []
    for row in ds:
        text, raw = str(row[tcol]).strip(), row[lcol]
        if not text: continue
        lbl_str = opt_keys[raw] if isinstance(raw, int) and isinstance(labels, dict) else (labels[raw] if isinstance(raw, int) else str(raw))
        if lbl_str not in opt_keys: continue
        records.append({"state": text, "questions": [[qname, "choice", opt_keys.index(lbl_str), labels, instr]]})
        if len(records) >= limit: break
    return records

def main():
    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    print("[ADAPT-DATA] Loading multi-task agent adaptation datasets...", flush=True)
    raw_td = build_typed_decisions()
    td = raw_td * 3
    print(f"  Loaded {len(raw_td)} raw typed-decisions (upsampled 3x to {len(td)})", flush=True)

    massive = build_choice_set("mteb/amazon_massive_intent", "en", "train", "text", "label_text", "action", "What is the user's intent in this utterance?", rng, 8000)
    print(f"  Loaded {len(massive)} MASSIVE intent samples", flush=True)

    ag_opts = {
        "World": "international news, politics, conflicts",
        "Sports": "sports, games, athletes",
        "Business": "companies, markets, economy",
        "Sci/Tech": "science, technology, software, space"
    }
    ag_news = build_choice_set("fancyzhx/ag_news", None, "train", "text", "label", "label", "What is the topic of the article?", rng, 6000, opts_dict=ag_opts)
    print(f"  Loaded {len(ag_news)} AG News samples", flush=True)

    # Replay buffer from foundation train set to prevent catastrophic forgetting
    foundation_path = data_dir / "train.jsonl"
    replay = []
    if foundation_path.exists():
        with open(foundation_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            rng.shuffle(all_lines)
            for line in all_lines[:6000]:
                replay.append(json.loads(line))
    print(f"  Loaded {len(replay)} foundation reasoning replay samples", flush=True)

    all_samples = td + massive + ag_news + replay
    rng.shuffle(all_samples)
    print(f"[ADAPT-DATA] Total Adaptation Samples: {len(all_samples)}", flush=True)

    n_val = int(0.08 * len(all_samples))
    splits = [("train_adapt.jsonl", all_samples[n_val:]), ("val_adapt.jsonl", all_samples[:n_val])]
    for name, data in splits:
        path = data_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for item in data: f.write(json.dumps(item) + "\n")
        print(f"  Wrote {len(data)} samples to {path}")

if __name__ == "__main__":
    main()
