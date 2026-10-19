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
- **Zero Billboard Comments & Decorative Noise:** NEVER write ASCII banner lines, decorative dividers, or billboard section headers (`# --------...`, `// ======...`). Comments must be rare, subtle, and strictly concise explanations of non-obvious logic.
- **System One Unified Candidate Decision Doctrine:**
  - **Single Universal Model Primitive:** Model boundary evaluates only $\text{Context} \to \text{Candidates} \to \mathbf{p} \in \Delta^{K-1}$. The neural backbone has zero awareness of question types.
  - **Pure Data Primitive:** `DecisionCase` in `data/case.py` contains strictly `(state, candidates, probabilities, category, source)`. Zero option or answer types exist in the data layer. `case_choice()` and `case_confidence()` are pure derived projections.
  - **Fail-Fast Invariant (Zero Silent Fallbacks):** Never write defensive fallback branches, magical `0` defaults, or sneaky swallowing of missing candidates. If a derived choice is not present in `candidates`, throw immediately (`ValueError`). If a state or category is missing, abort. Defensive fallbacks corrupt training data and mask upstream pipeline bugs.
  - **High-Level Ergonomic Layer:** `ChoiceOption`, `NoulOption`, and `ScoreOption` live strictly at the user-facing inference boundary (`inference/options.py`).
    - `ChoiceOption`: Unordered option selection $\to$ argmax and probability distribution.
    - `NoulOption`: Binary statement evaluation $\to$ candidates `["false: does not hold", "true: holds"]` $\to$ scalar probability $P(\text{true})$.
    - `ScoreOption`: Discrete rubric criteria $\to$ candidates derived from semantic levels $\to$ continuous expected level $\sum_{i=0}^{K-1} i \cdot p_i$.
  - **Consolidated Corpus:** Training and evaluation splits live in a single consolidated schema inside `data/corpus/` (`train.jsonl`, `val.jsonl`, `test.jsonl`). No type-fragmented subdirectories. Raw downloads remain strictly isolated in `data/raw/`.

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

## 6. Sober, High-Signal UI (Zero Decorative Noise & Zero Idle Moments)
- **Calculated Clarity Over Aesthetics:** Terminal and CLI outputs exist solely to convey essential operational signal (live progress, errors, verifiable metrics) with maximum readability.
- **Strictly Zero Idle/Blank Terminal Moments:** Long-running pipeline operations (ingesting sources, partitioning splits, auditing leakage, downloading datasets) must NEVER run silently in a black box. The UI must emit concise, live, real-time status updates per phase and per source so the operator always knows exactly what step is executing.
- **Zero Decorative UI Bloat:** Strictly zero gratuitous colors, decorative banner graphics, multi-table repetition, or animated gimmicks. UI is an operational instrument, not a toy showcase.
- **Sufficient & Non-Overloaded Signal:** Present just enough vital information to verify operations without under-informing the operator or flooding the screen with unreadable walls of numbers.
- **Minimal Code Footprint:** Output formatting code must be lean and direct. Never add dozens of lines of UI ceremony or display bloat.


