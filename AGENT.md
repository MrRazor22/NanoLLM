# Piqo Agent Guidelines

## 1. Architectural Doctrine: Primitive-First Design (PFD)
- **Always Apply PFD:** All architecture, boundary definitions, refactorings, and reviews in this repository strictly follow Primitive-First Design (PFD).
- **Authoritative Reference:** Always read and adhere to `~/.gemini/config/skills/pfd/SKILL.md` (invocable via `/primitive-first-design`). Never use partial heuristics or speculative wrapper abstractions.
- **The 5 Irreducible Sovereign Domains:**
  1. **Model:** [`model/`](file:///d:/CodeBase/Piqo/model) — [`IModel`](file:///d:/CodeBase/Piqo/model/torch_model.py), [`TorchModel`](file:///d:/CodeBase/Piqo/model/torch_model.py), [`ILoss`](file:///d:/CodeBase/Piqo/model/policies/calibrated_loss.py), [`CalibratedLoss`](file:///d:/CodeBase/Piqo/model/policies/calibrated_loss.py)
  2. **Inference:** [`inference/`](file:///d:/CodeBase/Piqo/inference) — [`IDecisionEngine`](file:///d:/CodeBase/Piqo/inference/decision_engine.py), [`DecisionEngine`](file:///d:/CodeBase/Piqo/inference/decision_engine.py)
  3. **Training:** [`training/`](file:///d:/CodeBase/Piqo/training) — [`ITrainer`](file:///d:/CodeBase/Piqo/training/epoch_trainer.py), [`EpochTrainer`](file:///d:/CodeBase/Piqo/training/epoch_trainer.py), [`ISchedule`](file:///d:/CodeBase/Piqo/training/policies/schedule.py), [`CosineSchedule`](file:///d:/CodeBase/Piqo/training/policies/schedule.py)
  4. **Data:** [`data/`](file:///d:/CodeBase/Piqo/data) — [`IDataset`](file:///d:/CodeBase/Piqo/data/dataset.py), [`Dataset`](file:///d:/CodeBase/Piqo/data/dataset.py)
  5. **Benchmark:** [`benchmark/`](file:///d:/CodeBase/Piqo/benchmark) — [`IBenchmark`](file:///d:/CodeBase/Piqo/benchmark/benchmark.py), [`Benchmark`](file:///d:/CodeBase/Piqo/benchmark/benchmark.py), [`IScorerPolicy`](file:///d:/CodeBase/Piqo/benchmark/policies/scorer.py), [`ILedgerPolicy`](file:///d:/CodeBase/Piqo/benchmark/policies/ledger.py), [`IReporterPolicy`](file:///d:/CodeBase/Piqo/benchmark/policies/reporter.py)
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
- **Pristine Reset / Clean Slate Workflow:**
  ```powershell
  # Wipe stale corpus splits, weights, and scores (preserves data/raw/ cache)
  Get-ChildItem data\corpus, model\weights -File -Recurse | Remove-Item -Force
  if (Test-Path benchmark\scores.json) { Remove-Item -Force benchmark\scores.json }
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

## 4. Operational Gating: Fast Verification vs. Heavy Compute
- **Heavy Compute Gated:** NEVER run heavy dataset builds (`python -m data`), full training (`python -m training`), or long benchmark evaluation (`python -m benchmark`) automatically without an explicit user command; delegate them with ready-to-run snippets.
- **Fast, Lightweight Verification Permitted:** If a test or verification check is fast and small (e.g. quick unit tests, import checks, smoke validation), run it directly and swiftly to confirm correctness without stalling dialogue.

## 5. Competitor Reference Architectures (External)
- Competitor open-source codebases are isolated externally at `D:\CodeBase\systemonemodels/` (outside Piqo's git tree to maintain repository cleanliness and zero IP/dependency pollution):
  - `D:\CodeBase\systemonemodels\gliner` (GLiNER / Fastino)
  - `D:\CodeBase\systemonemodels\laya` (Laya router)
  - `D:\CodeBase\systemonemodels\openJev-verdict-2.0` (Verdict 2.0 / openJev)
- Consult these external source trees when verifying competitor API protocols, contract parity, or benchmark input/output compatibility.


