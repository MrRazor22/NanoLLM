# Data Balancer & Decision Case Alignment Task Plan

## 1. Domain Modeling (`data/case.py`)
- [x] Fail-fast `CaseMeta`: `category`, `source`, `option_type` (`choice`, `noul`, `score`) required without default fallbacks.
- [x] Eliminate `DataSplits` wrapper class: all boundaries operate on `Sequence[DecisionCase]` directly.
- [x] Clean serialization: `to_dict` and `from_dict` include `instruction` and `option_type` without backward-compat shims.

## 2. Case Balancer Policy (`data/split/policies/balancer.py`)
- [x] `ICaseBalancer` protocol with `balance(cases: Sequence[DecisionCase], quota: int, rng: random.Random) -> Sequence[DecisionCase]`.
- [x] `RoundRobinCaseBalancer` with 7-dimension compound slot key:
  1. `meta.category`
  2. `meta.source`
  3. `meta.option_type`
  4. `xxhash.xxh64_intdigest(instruction)` (64-bit template signature)
  5. `len(candidates)`
  6. `target_index()`
  7. Per-source choice signature (`c.choice()` if `source_choice_counts[(source, choice)] >= 2` else `""`)
- [x] Remove obsolete `simhash16` bit-loop in favor of `xxhash`.

## 3. Split Pipeline & Storage Integration
- [x] `BudgetSplitPolicy.partition` returns `Sequence[DecisionCase]`.
- [x] `IStoragePolicy` and `DiskStoragePolicy` save and load `Sequence[DecisionCase]`.
- [x] `Dataset.prepare` and `Dataset.load(split=...)` return `Sequence[DecisionCase]`.
- [x] Updated all 11 dataset sources to pass `option_type="choice"` explicitly to `CaseMeta`.

## 4. Documentation & Verification
- [x] Updated `AGENT.md` guidelines.
- [x] Ran fast verification on all components.
