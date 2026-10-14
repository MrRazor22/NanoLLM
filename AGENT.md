# Piqo Agent Guidelines

## 1. Architectural Doctrine: Primitive-First Design (PFD)
- **Always Apply PFD:** All architecture, boundary definitions, refactorings, and reviews in this repository strictly follow Primitive-First Design (PFD).
- **Authoritative Reference:** Always read and adhere to `~/.gemini/config/skills/pfd/SKILL.md` (invocable via `/primitive-first-design`). Never use partial heuristics or speculative wrapper abstractions.
- **The 5 Irreducible Sovereign Domains:**
  1. **Model:** [`model/`](file:///d:/CodeBase/Piqo/model) — [`IModel`](file:///d:/CodeBase/Piqo/model/torch_model.py), [`TorchModel`](file:///d:/CodeBase/Piqo/model/torch_model.py), [`ILoss`](file:///d:/CodeBase/Piqo/model/policies/calibrated_loss.py), [`CalibratedLoss`](file:///d:/CodeBase/Piqo/model/policies/calibrated_loss.py)
  2. **Inference:** [`inference/`](file:///d:/CodeBase/Piqo/inference) — [`IDecisionEngine`](file:///d:/CodeBase/Piqo/inference/decision_engine.py), [`DecisionEngine`](file:///d:/CodeBase/Piqo/inference/decision_engine.py)
  3. **Training:** [`training/`](file:///d:/CodeBase/Piqo/training) — [`ITrainer`](file:///d:/CodeBase/Piqo/training/epoch_trainer.py), [`EpochTrainer`](file:///d:/CodeBase/Piqo/training/epoch_trainer.py), [`ISchedulePolicy`](file:///d:/CodeBase/Piqo/training/policies/schedule_policy.py), [`CosineSchedulePolicy`](file:///d:/CodeBase/Piqo/training/policies/schedule_policy.py)
  4. **Data:** [`data/`](file:///d:/CodeBase/Piqo/data) — [`IDataset`](file:///d:/CodeBase/Piqo/data/dataset.py), [`Dataset`](file:///d:/CodeBase/Piqo/data/dataset.py)
  5. **Benchmark:** [`benchmark/`](file:///d:/CodeBase/Piqo/benchmark) — [`IScorecard`](file:///d:/CodeBase/Piqo/benchmark/scorecard.py), [`Scorecard`](file:///d:/CodeBase/Piqo/benchmark/scorecard.py)
- **Zero Meta-Buckets & Zero Nesting:** Root commands the sovereign domains directly as siblings with 1:1 disk topology.

## 2. Operational Workflows & Commands (Zero-Flag Self-Wiring Defaults)
- **Interactive Terminal Playground:**
  ```bash
  python -m inference --play
  python -m benchmark --play
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
- **Run Training (Self-Wiring Defaults):**
  ```bash
  python -m training                  # Continuous adaptation (defaults: 1 epoch, lr=1e-5, on latest checkpoint)
  python -m training --from-scratch   # Pristine run from base ModernBERT (defaults: 2 epochs, lr=3e-5, cosine schedule)
  ```
- **Run Evaluation Benchmarks:**
  ```bash
  python -m benchmark
  python -m benchmark --play
  ```
- **Self-Wiring Defaults Invariant:** All command-line interfaces and primitives must resolve sensible operational defaults out of the box without requiring manual flag specifications. Never require callers to pass arguments that the command can deterministically resolve.

## 3. Training & Data Doctrine: Pure Reasoning Channeling (Zero Domain-Specific Fine-Tuning)
- **Zero Domain Memorization:** Never fine-tune the model to memorize domain-specific trivia, target facts, or narrow technical jargon. The pretrained foundation backbone inherently possesses broad world knowledge.
- **Knowledge Channeling Over Knowledge Injection:** Adaptation exists strictly to channel the backbone's latent representations into structured decision slots, calibrating confidence and aligning question constraints with candidate choices.
- **Pure Reasoning Curricula:** The training corpus is strictly composed of domain-agnostic reasoning, reading comprehension, differential elimination, and counterfactual constraint resolution. Domain capabilities emerge naturally from structured deduction, not memorized answers.

## 4. Strict Operational Gating: Zero Autonomous Execution
- **Zero Unprompted Data Builds, Training, or Tests:** NEVER run dataset preparation (`python -m data`, `dataset.prepare()`), model training (`python -m training`), benchmark evaluation (`python -m benchmark`), or tests automatically or in the background without an unmistakable, explicit user command.
- **Explicit Command Barrier:** Focus strictly on sound architecture, contract design, and code edits. Always delegate execution to the user by providing the exact command line to run.


