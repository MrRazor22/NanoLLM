# Task: Refactor Census Policy and CategoryBucket Architecture

## Objective
Clean up the census and bucket architecture to strictly adhere to Primitive-First Design (PFD):
1. **`CategoryBucket`**: Represents the discrete strata coordinates of a category (`sources`, `candidate_types`, `instructions`, `candidate_counts`, `target_indices`, `candidates`).
2. **`DatasetCensus`**: Represents the aggregate dataset profile (`categories: tuple[CategoryBucket, ...]`, `holdouts: tuple[frozenset[int], ...]`). Implements immutable `.union(other: Self) -> Self` and `.get_bucket(category: str) -> Optional[CategoryBucket]`.
3. **`ICensusPolicy`**: Pure contract `create(cases: Sequence[DecisionCase]) -> DatasetCensus`. Takes plain cases without streaming/generator baggage.
4. **File Placement**: In `data/policies/census.py` (policy folder).
5. **Caller Updates**: Update `data/policies/balancer.py`, `data/policies/leakage.py`, `data/dataset.py`, and `piqo.py`.

---

## Action Items
- [x] 1. Create and update `task.md` with the full plan.
- [x] 2. Implement `data/policies/census.py` with `CategoryBucket`, `DatasetCensus`, `ICensusPolicy`, and `CaseCensusPolicy`.
- [x] 3. Update `data/policies/balancer.py` to consume `CategoryBucket`.
- [x] 4. Update `data/policies/leakage.py` to accept `holdouts: Sequence[frozenset[int]]`.
- [x] 5. Update `data/dataset.py` to accumulate `DatasetCensus` via `census = census.union(census_policy.create(chunk))` and Pass 2 balance/decontaminate.
- [x] 6. Update `piqo.py` exports.
- [x] 7. Fast targeted verification (imports, syntax, and contract alignment).
