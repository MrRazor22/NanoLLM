# Piqo Agent Guidelines

## 1. Architectural Doctrine: Primitive-First Design (PFD)
- **Always Apply PFD:** All architecture, boundary definitions, refactorings, and reviews in this repository strictly follow Primitive-First Design (PFD).
- **Authoritative Reference:** Always read and adhere to `~/.gemini/config/skills/pfd/SKILL.md` (invocable via `/primitive-first-design`). Never use partial heuristics, legacy ATA terminology, or speculative wrapper abstractions.
- **The 4 Irreducible Sovereign Domains:**
  1. **Inference:** [`inference/`](file:///d:/CodeBase/Piqo/inference) — [`IDecisionEngine`](file:///d:/CodeBase/Piqo/inference/decision_engine.py), [`DecisionEngine`](file:///d:/CodeBase/Piqo/inference/decision_engine.py)
  2. **Training:** [`training/`](file:///d:/CodeBase/Piqo/training) — [`ITrainer`](file:///d:/CodeBase/Piqo/training/epoch_trainer.py), [`EpochTrainer`](file:///d:/CodeBase/Piqo/training/epoch_trainer.py)
  3. **Data:** [`data/`](file:///d:/CodeBase/Piqo/data) — [`IDataset`](file:///d:/CodeBase/Piqo/data/dataset.py), [`Dataset`](file:///d:/CodeBase/Piqo/data/dataset.py), [`ILeakagePolicy`](file:///d:/CodeBase/Piqo/data/policies/leakage.py)
  4. **Evaluation:** [`evaluation/`](file:///d:/CodeBase/Piqo/evaluation) — [`IEvaluator`](file:///d:/CodeBase/Piqo/evaluation/scorecard.py), [`Scorecard`](file:///d:/CodeBase/Piqo/evaluation/scorecard.py)
- **Zero Meta-Buckets & Zero Nesting:** Root commands the 4 domains directly as siblings. Zero nested `piqo/` folder, and zero `tests/` meta-bucket.

## 2. Operational Workflows & Commands
- **Interactive Terminal Playground:**
  ```bash
  python cli.py --play
  ```
- **Run One-Shot Inference:**
  ```bash
  python cli.py "Database CPU reached 99% and connection pool is exhausted"
  python cli.py "Look up encryption policy" --preset routing
  ```
- **Build / Adapt Training Curriculum (Zero Leakage):**
  ```bash
  python -m data
  ```
- **Run Training (1-Epoch Adaptation):**
  ```bash
  python train.py --epochs 1 --lr 2e-5
  # Or directly via domain module:
  python -m training --epochs 1 --lr 2e-5
  ```
- **Run Evaluation Benchmarks:**
  ```bash
  python -m evaluation
  python -m evaluation --suite agentic
  python -m evaluation --suite typed_decisions
  python -m evaluation --suite abstention
  ```
