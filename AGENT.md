# Piqo Agent Guidelines

## 1. Architectural Doctrine: Primitive-First Design (PFD)
- **Always Apply PFD:** All architecture, boundary definitions, refactorings, and reviews in this repository strictly follow Primitive-First Design (PFD).
- **Authoritative Reference:** Always read and adhere to `~/.gemini/config/skills/pfd/SKILL.md` (invocable via `/primitive-first-design`). Never use partial heuristics or speculative wrapper abstractions.
- **The 5 Irreducible Sovereign Domains:**
  1. **Model:** [`model/`](file:///d:/CodeBase/Piqo/model) — [`IModel`](file:///d:/CodeBase/Piqo/model/torch_model.py), [`TorchModel`](file:///d:/CodeBase/Piqo/model/torch_model.py), [`ILoss`](file:///d:/CodeBase/Piqo/model/policies/calibrated_loss.py), [`CalibratedLoss`](file:///d:/CodeBase/Piqo/model/policies/calibrated_loss.py)
  2. **Inference:** [`inference/`](file:///d:/CodeBase/Piqo/inference) — [`IDecisionEngine`](file:///d:/CodeBase/Piqo/inference/decision_engine.py), [`DecisionEngine`](file:///d:/CodeBase/Piqo/inference/decision_engine.py)
  3. **Training:** [`training/`](file:///d:/CodeBase/Piqo/training) — [`ITrainer`](file:///d:/CodeBase/Piqo/training/epoch_trainer.py), [`EpochTrainer`](file:///d:/CodeBase/Piqo/training/epoch_trainer.py)
  4. **Data:** [`data/`](file:///d:/CodeBase/Piqo/data) — [`IDataset`](file:///d:/CodeBase/Piqo/data/dataset.py), [`Dataset`](file:///d:/CodeBase/Piqo/data/dataset.py)
  5. **Benchmark:** [`benchmark/`](file:///d:/CodeBase/Piqo/benchmark) — [`IScorecard`](file:///d:/CodeBase/Piqo/benchmark/scorecard.py), [`Scorecard`](file:///d:/CodeBase/Piqo/benchmark/scorecard.py)
- **Zero Meta-Buckets & Zero Nesting:** Root commands the sovereign domains directly as siblings with 1:1 disk topology.

## 2. Operational Workflows & Commands
- **Interactive Terminal Playground:**
  ```bash
  python -m inference --play
  ```
- **Run One-Shot Inference:**
  ```bash
  python -m inference "Database CPU reached 99% and connection pool is exhausted"
  python -m inference "Look up encryption policy" --preset routing
  ```
- **Build / Adapt Training Curriculum (Zero Leakage):**
  ```bash
  python -m data
  ```
- **Run Training (1-Epoch Adaptation):**
  ```bash
  python -m training --epochs 1 --lr 1.5e-5
  ```
- **Run Evaluation Benchmarks:**
  ```bash
  python -m benchmark
  ```
