# NanoLLM

NanoLLM is a high-performance micro-decision engine designed for ultra-low latency semantic routing, dynamic tool selection, incident triage, and safety guardrails. Powered by a ModernBERT-base backbone with calibrated multi-task decision heads, it operates at ~32–41 ms P50 latency on CUDA, outperforming Laya and Jev across complex decision workloads. Built following ATA design principles (single primitive per boundary, forward pipe composition, and zero toy mocks).

---

## Codebase Topology

```text
d:/CodeBase/NanoLLM/
├── cli.py                     # User-facing inference CLI
├── pipeline.py                # ATA Universal Pipeline Helper (PipelineComposable)
├── harness/                   # External training orchestration boundary (outside nanollm)
│   ├── __main__.py            # python -m harness training entrypoint
│   └── dataset/               # Training dataset boundary
│       ├── training_dataset.py# TrainingDataset primitive (P)
│       ├── curriculum.py      # Dataset curriculum recipe & builder
│       ├── __main__.py        # python -m harness.dataset entrypoint
│       ├── sources/           # Dataset source adapters (Glaive, Generic, Typed)
│       ├── collator.py        # MultiQuestionCollator policy (π)
│       └── data/
│           ├── raw/           # Raw unadapted corpora & source drops (*.jsonl)
│           ├── adapted/       # 11 individual adapted source files (*.jsonl)
│           └── splits/        # Production multi-task splits (train.jsonl, val.jsonl)
├── nanollm/                   # Core production library
│   ├── model/
│   │   ├── nano_model.py      # NanoModel primitive (P), ModelConfig
│   │   └── checkpoints/       # checkpoint_champion_v4.pt (production weights)
│   ├── inference/
│   │   ├── decision_engine.py # DecisionEngine primitive (P)
│   │   ├── assembler.py       # SlotAssembler policy (π)
│   │   ├── profiling_layer.py # ProfilingLayer (λ)
│   │   └── schema.py          # Choice, Noul, Score, DecisionResult
│   └── training/              # Core neural training engine
│       ├── epoch_trainer.py   # EpochTrainer primitive (P)
│       ├── loss.py            # CalibratedLoss policy (π)
│       ├── checkpointing_layer.py # CheckpointingLayer (λ)
│       └── metrics_layer.py   # MetricsLayer (λ)
└── benchmark/                 # Independent verification boundary (outside nanollm)
    ├── report/                # Reporting boundary: evaluation orchestration & presentation
    │   ├── scorecard.py       # Scorecard primitive (P) & IScorecard contract
    │   ├── renderer.py        # ConsoleScorecardRenderer policy (π) & IReportRenderer contract
    │   └── baselines.py       # JsonBaselineProvider policy (π) & IBaselineProvider contract
    ├── baselines/             # Verified baseline numbers & competitor cache
    ├── __main__.py            # python -m benchmark entrypoint
    └── dataset/               # Benchmark dataset boundary (fully decoupled)
        ├── benchmark_dataset.py # BenchmarkDataset primitive (P) & IDataSource contract
        ├── sources/           # Suite source extraction adapters (Agentic, Laya, etc.)
        └── data/
            ├── raw/           # Raw unadapted evaluation slices (slice_*.jsonl)
            └── adapted/       # Standardized evaluation tracks (agentic.json, abstention.json, laya.json)
```

---

## Operational Entrypoints

* **Run Inference:**
  ```bash
  python cli.py "Database CPU reached 99% and connection pool is exhausted"
  ```
* **Build / Adapt Training Dataset (from raw & sources):**
  ```bash
  python -m harness.dataset
  ```
* **Run Training (1-Epoch Adaptation):**
  ```bash
  python train.py --epochs 1 --lr 2e-5
  # or: python -m harness --epochs 1 --lr 2e-5
  ```
* **Run Benchmark (Full 2,400-case Laya / Jev Head-to-Head):**
  ```bash
  python -m benchmark
  ```
* **Run Benchmark (400-case Publication-Grade Agentic Suite vs Live Laya):**
  ```bash
  python -m benchmark --agentic
  ```
* **Run Benchmark (2,000-decision Typed Decisions Showdown vs Verdict 2.0 & Laya):**
  ```bash
  python -m benchmark --typed-decisions
  ```
* **Run Benchmark (Abstention & Safety Suite vs Verdict 2.0):**
  ```bash
  python -m benchmark --abstention
  ```

---

## Key Learnings & Proving Ground Facts

1. **Production Checkpoint (`checkpoint_champion_v4.pt`):**
   * Initialized from adaptation run, trained on balanced multi-domain curriculum (`train_adapt.jsonl` with bounded state sequence assembly, balanced English queue distribution, and calibrated loss).
   * Drove validation loss down to **0.2498**, establishing decisive head-to-head dominance over Laya SOTA across both the 2,400 multi-suite benchmark and custom agentic workloads while maintaining ~32 ms P50 CUDA latency.

2. **Head-to-Head Verification vs Laya & Jev:**
   * **77-Way Choice Dominance (`banking77`):** NanoLLM heavily outperforms Laya on high-cardinality semantic routing.
   * **Emotion & Spam:** NanoLLM decisively outperforms Laya on nuanced emotion classification and email spam detection.
   * **Custom Agentic Suite:** Outperforms live Laya across tool selection and negative constraints.
   * **Latency:** NanoLLM operates at ~31–33 ms P50 on CUDA, running faster than Laya SOTA and ~7x faster than Jev.

3. **Decoupled Verification Boundary:**
   * `benchmark/` lives strictly outside `nanollm/`. Evaluation data, ground-truth suites, and reporter tools never pollute production library code.

4. **Forward Pipe Composition:**
   * Replaced inside-out constructor nesting with forward composition across all capabilities:
     `engine | ProfilingLayer`, `trainer | CheckpointingLayer`, `evaluator | ProfilingEvaluatorLayer`.

5. **Strict Evidence Grounding:**
   * Unit tests in `tests/` verify mechanics only; no toy mocks dumping fake tables. All performance scorecards are produced by live evaluations on real datasets.

6. **Data Richness & Anti-Contamination Invariant:**
   * Any dataset prepared or curated must be genuinely rich, semantically deep, and structurally complete—never poor, shallow, or degraded toy data.
   * Domain adaptation performance is strictly bounded by data quality: high-signal, rich multi-domain distributions drive real capability, while evaluation splits remain strictly isolated to prevent test leakage.

7. **Domain & Schema Parity Invariant:**
   * Before launching any adaptation or fine-tuning run, always systematically validate that all evaluation domains, task types, and candidate action spaces (e.g., verifying 4-choice contrastive options vs binary pairs) are fully represented and aligned in the training mixture. Training on a reduced or mismatched action space causes severe evaluation divergence.



