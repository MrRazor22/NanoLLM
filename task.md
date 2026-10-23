# Category-First Chunked Pipeline Plan

## 1. Domain Modeling (`data/case.py`)
- [x] Fail-fast `CaseMeta`: `category`, `source`, `candidate_type` (`choice`, `noul`, `score`) required.
- [x] Clean `DecisionCase`: normalized string fields in `__init__`, zero reflection.

## 2. Balancer Architecture (`data/policies/balancer.py`)
- [ ] Refactor `ICaseBalancer`:
  - Signature: `balance(cases: Sequence[DecisionCase], limit: Optional[int] = None) -> Sequence[DecisionCase]`.
  - Single category focus: receives a sequence of cases belonging to a category.
  - Balances equally across the 5 intra-category dimensions:
    `(source, candidate_type, xxhash(instruction), len(candidates), target_index, choice_signature)`.
  - Zero category grouping inside the balancer since all cases in the call share the same category.

## 3. Eliminate Split Policy
- [ ] Delete `data/policies/split.py`.
- [ ] Remove `ISplitPolicy` / `RatioSplitPolicy` exports from `piqo.py`.

## 4. Orchestration (`data/dataset.py`)
- [ ] Constructor configuration:
  - `quota_per_category: int = 1500`
  - `val_ratio: float = 0.10`
  - `chunk_factor: int = 2`
  - Injected `balancer_policy: ICaseBalancer` and `leakage_policy: ILeakagePolicy`.
- [ ] `Dataset.prepare(source=None)`:
  - Separate `test` holdouts: `holdouts = [c for c in cases if c.meta and c.meta.split == "test"]`.
  - Group holdouts by category for localized fast decontamination.
  - Group training pool by category.
  - For each category:
    - While clean quota is unfilled and category candidates remain:
      - Slice chunk of size `(quota - collected) * chunk_factor`.
      - Balance chunk via `balancer.balance(chunk)`.
      - Decontaminate balanced chunk via `leakage.decontaminate(balanced_chunk, cat_holdouts)`.
      - Collect clean cases.
    - Final pass: `category_clean = balancer.balance(collected_clean, limit=quota_per_category)`.
    - Carve validation split: `val_count = max(1, int(len(category_clean) * val_ratio))`.
    - Tag `split="val"` on validation subset and `split="train"` on remainder.
  - Save `test`, `train`, `val` via `storage_policy`.

## 5. Verification
- [ ] Unit tests for multi-category chunked balancing, decontamination, and val split.
