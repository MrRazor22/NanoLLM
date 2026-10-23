# DATA CURATION PIPELINE SPECIFICATION (IMMUTABLE INVARIANT)

## 1. Core Principle: `category` is King
- `category` is the fundamental semantic boundary across all sources, tasks, and modalities.
- All decisions, quotas, balancing, and decontamination happen **strictly per category**.

## 2. Ingestion & Test Isolation
- Ingest consolidated `Sequence[DecisionCase]` from `source`.
- Immediately isolate `test` split cases into `test_by_category`. Test cases are author benchmark holdouts; they must never enter training or validation pools.
- Group remaining candidate cases into `train_by_category`.

## 3. The 6-Dimension Bucket Definition
To eliminate dominance and positional bias across diverse datasets, a **Bucket** represents the discrete structural profile of a decision case:
1. `source`: where data comes from (e.g. `sciq`, `anli`, `medqa`)
2. `candidate_type`: decision geometry (`choice`, `noul`, `score`)
3. `instruction`: instruction text / template
4. `option`: target option text (the answer text)
5. `num_candidates`: number of options (`len(candidates)`, e.g. 2, 4)
6. `target_index`: target index of ground-truth answer (positional bias mitigation)

## 4. Scalable Curation Architecture
- Scaling is an uncompromising architectural invariant.
- `Dataset` prepares the active buckets for the category.
- Streaming/chunking reads through category candidates:
  - Decontaminates against `test_by_category` (Jaccard leakage policy).
  - Routes into active buckets up to the fair category quota.
  - Prevents common early cases from starving rare/late combinations (zero early-stopping bias).
- Balanced round-robin interleaving across active buckets.
- Slices final balanced cases into `train` and `val` splits.
- Persists all splits (`test`, `train`, `val`) to disk storage.
