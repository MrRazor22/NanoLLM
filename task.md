# Data Domain Sibling Policy Architecture Task Plan

## 1. Domain Modeling (`data/case.py`)
- [x] Fail-fast `CaseMeta`: `category`, `source`, `candidate_type` (`choice`, `noul`, `score`) required without default fallbacks.
- [x] Normalized slots constructor: `DecisionCase` normalizes `state`, `instruction`, `candidates`, and `probabilities` directly in `__init__` without reflection.
- [x] Clean serialization: `to_dict` and `from_dict` include `instruction` and `candidate_type` without backward-compat shims.

## 2. Flat Sibling Policies (`data/policies/`)
- [x] `ISplitPolicy` (`data/policies/split.py`): `RatioSplitPolicy` pure partitioning into `val` and `train`, retaining author benchmark `test` splits.
- [x] `ILeakagePolicy` (`data/policies/leakage.py`): `JaccardLeakagePolicy` decontaminates `train`/`val` against benchmark holdouts.
- [x] `ICaseBalancer` (`data/policies/balancer.py`): `RoundRobinCaseBalancer` pure distribution across 7 compound dimensions up to `limit`.
- [x] `IStoragePolicy` (`data/storage.py`): `DiskStoragePolicy` persists `train.jsonl` and `val.jsonl`.

## 3. Core Primitive Orchestration (`data/dataset.py`)
- [x] `Dataset.prepare(source=None)` executes a 5-line pure sequential pipeline:
  `cases -> split -> decontaminate -> balance -> save`.
- [x] Zero nesting: none of the sibling policies own or couple to each other.
- [x] Aligned `piqo.py` root exports.

## 4. Verification
- [x] Run fast unit verification across all 5 components.
