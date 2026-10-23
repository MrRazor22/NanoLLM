import logging
import time
from collections import Counter
import typer

from data.dataset import Dataset

app = typer.Typer(add_completion=False, help="Piqo Dataset Corpus Builder CLI")

@app.command()
def main() -> None:
    logging.basicConfig(format="[%(name)s] %(message)s")
    logging.getLogger("data").setLevel(logging.INFO)
    t0 = time.perf_counter()

    cases = Dataset().prepare()
    train_cases = [c for c in cases if c.meta.split == "train"]
    val_cases = [c for c in cases if c.meta.split == "val"]
    test_cases = [c for c in cases if c.meta.split == "test"]

    print(f"\n[data] Built corpus in {time.perf_counter()-t0:.2f}s:")
    print(f"  train: {len(train_cases):,} | val: {len(val_cases):,} | test: {len(test_cases):,}")

    tr_by_cat = Counter(c.meta.category for c in train_cases)
    va_by_cat = Counter(c.meta.category for c in val_cases)
    te_by_cat = Counter(c.meta.category for c in test_cases)
    all_cats = sorted(set(tr_by_cat) | set(va_by_cat) | set(te_by_cat))

    print("\n--- Corpus Category Composition ---", flush=True)
    for cat in all_cats:
        tr = tr_by_cat.get(cat, 0)
        va = va_by_cat.get(cat, 0)
        te = te_by_cat.get(cat, 0)
        print(f"  {cat:<28} total: {tr + va + te:>5,} | train: {tr:>5,} | val: {va:>4,} | test: {te:>4,}", flush=True)
    print(f"Total: {len(train_cases):,} train, {len(val_cases):,} val, {len(test_cases):,} test ({len(all_cats)} categories)\n", flush=True)

if __name__ == "__main__":
    app()
