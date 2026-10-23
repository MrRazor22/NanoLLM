# Data Balancer & Decision Case Alignment Task Plan

## 1. Domain Modeling (`data/case.py`)
- [x] Fail-fast `CaseMeta`: `category`, `source`, `candidate_type` (`choice`, `noul`, `score`) required without default fallbacks.
- [x] Normalized slots constructor: `DecisionCase` normalizes `state`, `instruction`, `candidates`, and `probabilities` directly in `__init__` without reflection.
- [x] Eliminate `DataSplits` wrapper class: all boundaries operate on `Sequence[DecisionCase]` directly.
- [x] Clean serialization: `to_dict` and `from_dict` include `instruction` and `candidate_type` without backward-compat shims.

## 2. Case Balancer Policy (`data/policies/balancer.py`)
- [x] `ICaseBalancer` protocol with `balance(cases: Sequence[DecisionCase], quota: int, rng: random.Random, holdouts: Sequence[DecisionCase] = ()) -> Sequence[DecisionCase]`.
- [x] `RoundRobinCaseBalancer` with 7-dimension compound slot key:
  1. `meta.category`
  2. `meta.source`
  3. `meta.candidate_type`
  4. `xxhash.xxh64_intdigest(instruction)` (64-bit template signature)
  5. `len(candidates)`
  6. `target_index()`
  7. Per-source choice signature (`c.choice()` if `source_choice_counts[(source, choice)] >= 2` else `""`)
- [x] On-demand per-bucket decontamination: pops replacement candidates from the same slot on leakage without unbalancing.
- [x] Relocate `balancer.py` and `leakage.py` to `data/policies/` for domain symmetry.

## 3. Dataset Orchestration & Elimination of `split.py`
- [x] Eliminate `data/split/split.py` completely.
- [x] `Dataset.prepare` orchestrates `test`, `val`, `train` directly via injected `balancer`.
- [x] Update `piqo.py` root exports.
- [x] Updated all 11 dataset sources to pass `candidate_type="choice"` explicitly to `CaseMeta`.

## 4. Documentation & Verification
- [x] Updated `AGENT.md` guidelines.
- [x] Fast verification test passed.
