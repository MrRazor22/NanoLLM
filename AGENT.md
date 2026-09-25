# NanoLLM

NanoLLM is a high-performance micro-decision engine designed for ultra-low latency semantic routing, dynamic tool selection, incident triage, and safety guardrails. Powered by a ModernBERT-base backbone with calibrated multi-task decision heads, it operates at ~32–41 ms P50 latency on CUDA, outperforming Laya and Jev across complex decision workloads. Built following ATA design principles (single primitive per boundary, forward pipe composition, and zero toy mocks).

---

## Codebase Topology

```text
d:/CodeBase/NanoLLM/
├── cli.py                     # User-facing inference CLI
├── train.py                   # External training runner
├── nanollm/                   # Core production library
│   ├── model/                 # NanoModel primitive, ModelConfig, neural checkpoints
│   │   └── checkpoints/       # checkpoint_champion_v2.pt (production weights)
│   ├── inference/             # DecisionEngine primitive
│   │   ├── policies/          # SlotAssembler, DecisionResolver, SubwordTokenizer
│   │   ├── layers/            # ProfilingLayer, HierarchicalLayer
│   │   └── schema.py          # Choice, Noul, Score, DecisionResult
│   └── training/              # EpochTrainer primitive
│       ├── policies/          # CalibratedLoss, MultiQuestionCollator, AdaptationCurriculum
│       ├── checkpointing_layer.py
│       └── data/              # train_adapt.jsonl, val_adapt.jsonl
└── benchmark/                 # Independent verification boundary (outside nanollm)
    ├── evaluator.py           # ModelEvaluator primitive
    ├── profiling_layer.py     # ProfilingEvaluatorLayer
    ├── reporter.py            # Head-to-head scorecard printer (Delta vs Best)
    ├── __main__.py            # python -m benchmark runner
    └── data/                  # laya_benchmark.json (2,400 cases), benchmark.json (120 cases)
```

---

## Operational Entrypoints

* **Run Inference:**
  ```bash
  python cli.py "Database CPU reached 99% and connection pool is exhausted"
  ```
* **Run Training (1-Epoch Adaptation):**
  ```bash
  python train.py --epochs 1 --lr 2e-5
  ```
* **Run Benchmark (Full 2,400-case Laya / Jev Head-to-Head):**
  ```bash
  python -m benchmark
  ```
* **Run Benchmark (120-case Custom Agentic Suite vs Live Laya):**
  ```bash
  python -m benchmark --agentic
  ```

---

## Key Learnings & Proving Ground Facts

1. **Production Checkpoint (`checkpoint_champion_v2.pt`):**
   * Initialized from v1 champion, trained for **1 epoch** on `train_adapt.jsonl` (combining 5,000 Glaive dynamic function-calling samples, 3x oversampled typed-decisions, and foundation replay).
   * Drove validation loss down to **0.348**, boosting Agent Tool Routing by **+16.7%** without regressing latency.

2. **Head-to-Head Verification vs Laya & Jev:**
   * **77-Way Choice Dominance (`banking77`):** NanoLLM scores **64.0%** vs Laya's **42.5%** (**+21.5% lead**).
   * **Emotion:** NanoLLM scores **52.0%**, beating Jev's published **48.0%** (**+4.0%**).
   * **Custom Agentic Suite (120 cases / 180 questions):** NanoLLM scores **70.6%** vs Laya's **60.0%** (**+10.6% overall**), winning Tool Routing (76.7% vs 70.0%), Triage (66.7% vs 53.3%), and Negative Constraints (93.3% vs 70.0%).
   * **Latency:** NanoLLM runs at **~35 ms P50** on CUDA (~6x faster than Jev's 246 ms, parity with Laya's 33 ms).

3. **Decoupled Verification Boundary:**
   * `benchmark/` lives strictly outside `nanollm/`. Evaluation data, ground-truth suites, and reporter tools never pollute production library code.

4. **Forward Pipe Composition:**
   * Replaced inside-out constructor nesting with forward composition across all capabilities:
     `engine | ProfilingLayer`, `trainer | CheckpointingLayer`, `evaluator | ProfilingEvaluatorLayer`.

5. **Strict Evidence Grounding:**
   * Unit tests in `tests/` verify mechanics only; no toy mocks dumping fake tables. All performance scorecards are produced by live evaluations on real datasets.
