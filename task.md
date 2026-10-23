# Data Balancer & Decision Case Alignment Task Plan

## 1. Domain Modeling (`data/case.py`)
- [x] Align `CaseMeta`: `category`, `source`, `option_type` (`choice`, `noul`, `score`), `split`.
- [x] Invariant stripping: `DecisionCase` guarantees `state`, `instruction`, `candidates`, and `probabilities` keys are stripped at initialization.
- [x] Clean serialization: `to_dict` and `from_dict` include `instruction` and `option_type`. Zero backward-compatibility shims.

## 2. Case Balancer Policy (`data/split/policies/balancer.py`)
- [x] Define `ICaseBalancer` protocol with `balance(cases: Sequence[DecisionCase], quota: int, rng: random.Random) -> Sequence[DecisionCase]`.
- [x] Implement `RoundRobinCaseBalancer` with 7-dimension compound slot key:
  1. `meta.category`
  2. `meta.source`
  3. `meta.option_type`
  4. `simhash16(instruction)`
  5. `len(candidates)`
  6. `target_index()`
  7. Per-source choice signature (`c.choice()` if `source_choice_counts[(source, choice)] >= 2` else `""`)
- [x] Remove obsolete `data/split/policies/stratification.py`.

## 3. Split Pipeline Integration (`data/split/split.py`)
- [x] Update `BudgetSplitPolicy` to depend on `ICaseBalancer` and default to `RoundRobinCaseBalancer`.
- [x] Align `piqo.py` root exports.

## 4. Documentation & Guidelines (`AGENT.md`)
- [x] Document data balancing architecture, 7-dimension slot invariants, and `option_type` domain symmetry with inference options.

## 5. Verification
- [x] Run fast import/syntax check and test runs on dataset samples.
