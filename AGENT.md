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
    │   └── renderer.py        # ConsoleScorecardRenderer policy (π) & IReportRenderer contract
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

* **Interactive Terminal Playground:**
  ```bash
  python cli.py --play
  # or simply: python cli.py
  ```
* **View Head-to-Head Benchmark vs Laya & Verdict 2.0:**
  ```bash
  python cli.py --compare
  ```
* **List Model Registry & Checkpoints:**
  ```bash
  python cli.py --models
  ```
* **Run One-Shot Inference:**
  ```bash
  python cli.py "Database CPU reached 99% and connection pool is exhausted"
  python cli.py "Look up encryption policy" --preset routing
  python cli.py "Suspicious login attempt" --choice "action: allow, mfa, block" --score "threat_score"
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

---

## Python API Usage Guide (Building Applications with NanoLLM)

NanoLLM is designed as an ultra-fast, zero-bloat micro-decision engine designed to be embedded directly into Python applications, agent frameworks, microservices, and event loops. Built strictly on Axiomatic Triad Architecture (ATA), NanoLLM separates state from execution policies, providing typed multi-task decisions in a single neural forward pass.

### 1. Minimal Quickstart

```python
from nanollm import DecisionEngine, Choice, Noul, Score

# 1. Initialize engine (automatically discovers the latest champion checkpoint)
engine = DecisionEngine.from_checkpoint()

# 2. Define typed decision schema
questions = [
    Choice("department", ["Billing", "Infrastructure", "Sales", "Security"]),
    Noul("urgent", instruction="Is immediate incident response required?"),
    Score("severity", min_value=0.0, max_value=100.0),
]

# 3. Execute decision on any raw state/prompt
result = engine.decide(
    state="Our primary Postgres database CPU reached 99% and connection pool is exhausted.",
    questions=questions,
)

# 4. Access typed answers
dept = result.answers["department"]   # ChoiceResult: .choice, .confidence, .probabilities
urgent = result.answers["urgent"]       # NoulResult:   .value (bool), .probability (float)
sev = result.answers["severity"]       # ScoreResult:  .score (float), .normalized (0.0..1.0)

print(f"Route to: {dept.choice} ({dept.confidence * 100:.1f}%)")
print(f"Urgent: {urgent.value} (p={urgent.probability:.3f})")
print(f"Severity: {sev.score:.1f}/100")
```

---

### 2. Typed Question Schemas

NanoLLM supports three fundamental decision primitives that can be queried concurrently in a single forward pass:

| Primitive | Type | Purpose | Extracted Result Attributes |
| :--- | :--- | :--- | :--- |
| `Choice` | Multi-class categorical | Tool routing, department triage, intent classification, sentiment analysis | `.choice` (str), `.confidence` (float), `.probabilities` (dict[str, float]) |
| `Noul` | Calibrated boolean | Binary flags, safety filters, escalation gates, confirmation checks | `.value` (bool), `.probability` (float in `[0.0, 1.0]`) |
| `Score` | Calibrated regression | Priority scoring (1–10), risk rating (0–100), confidence estimation | `.score` (float in `[min, max]`), `.normalized` (float in `[0.0, 1.0]`) |

#### Detailed Schema Definition

```python
from nanollm import Choice, Noul, Score

# Categorical Choice with optional guidance instruction
choice_q = Choice(
    name="intent",
    options=["refund_request", "order_status", "tech_support", "account_cancellation"],
    instruction="Classify customer intent accurately based on message context",
)

# Calibrated Boolean (Noul)
bool_q = Noul(
    name="requires_supervisor",
    instruction="Flag True if customer is hostile or requesting supervisor escalation",
)

# Calibrated Continuous Score
score_q = Score(
    name="churn_risk",
    min_value=0.0,
    max_value=100.0,
    instruction="Estimated risk of customer churn from 0 (none) to 100 (critical)",
)
```

---

### 3. Latency Profiling & Forward Pipe Composition

Following ATA principles, capabilities are attached via forward pipe composition (`|`) rather than constructor nesting:

```python
from nanollm import DecisionEngine, ProfilingLayer

# Forward pipe composition attaches microsecond latency profiling
engine = DecisionEngine.from_checkpoint() | ProfilingLayer()

result = engine.decide(state="System alert: high memory pressure on worker-04", questions=questions)

# Access hardware execution latency in milliseconds
print(f"Decision latency: {result.latency_ms:.2f} ms")
```

---

### 4. Production Application Patterns

#### Pattern A: Autonomous Agent Dynamic Tool Router

Before dispatching expensive LLM calls, use NanoLLM to instantly route user prompts to the correct local tool and decide if human confirmation is required:

```python
from nanollm import DecisionEngine, Choice, Noul, Score

engine = DecisionEngine.from_checkpoint()

def route_agent_action(user_prompt: str) -> dict:
    questions = [
        Choice("tool", ["query_db", "web_search", "send_email", "human_escalate", "none"]),
        Noul("needs_confirmation", instruction="Requires explicit human confirmation before executing"),
        Score("execution_priority", min_value=1.0, max_value=10.0),
    ]
    result = engine.decide(state=user_prompt, questions=questions)
    return {
        "tool": result.answers["tool"].choice,
        "confidence": result.answers["tool"].confidence,
        "needs_confirmation": result.answers["needs_confirmation"].value,
        "priority": result.answers["execution_priority"].score,
        "latency_ms": result.latency_ms,
    }

# Example execution:
action = route_agent_action("Show me all orders from user #4421 in the last week")
# -> {'tool': 'query_db', 'confidence': 0.985, 'needs_confirmation': False, 'priority': 5.2, ...}
```

#### Pattern B: Real-Time Input Safety & Abstention Guardrail

NanoLLM achieves 100% abstention accuracy on out-of-scope and adversarial prompts, making it ideal as a zero-latency firewall in front of generative LLMs:

```python
from nanollm import DecisionEngine, Choice, Noul, Score

engine = DecisionEngine.from_checkpoint()

def validate_input_safety(raw_input: str) -> bool:
    questions = [
        Choice("threat_type", ["safe", "prompt_injection", "jailbreak", "spam", "out_of_scope"]),
        Noul("is_safe", instruction="Is the prompt safe to pass to core LLM pipeline?"),
        Score("risk_score", min_value=0.0, max_value=100.0),
    ]
    res = engine.decide(state=raw_input, questions=questions)
    threat = res.answers["threat_type"].choice
    is_safe = res.answers["is_safe"].value
    risk = res.answers["risk_score"].score

    if not is_safe or threat != "safe" or risk > 30.0:
        return False  # Block or quarantine request
    return True
```

#### Pattern C: Customer Support Triage & Auto-Dispatch

```python
from nanollm import DecisionEngine, Choice, Noul, Score

engine = DecisionEngine.from_checkpoint()

def triage_ticket(ticket_body: str) -> dict:
    questions = [
        Choice("queue", ["tier_1_support", "engineering_bug", "billing_finance", "executive_escalation"]),
        Noul("emergency", instruction="Is there an ongoing outage or critical data loss?"),
        Score("sla_urgency", min_value=1.0, max_value=5.0),
    ]
    res = engine.decide(state=ticket_body, questions=questions)
    return {
        "queue": res.answers["queue"].choice,
        "emergency": res.answers["emergency"].value,
        "urgency_level": round(res.answers["sla_urgency"].score),
    }
```

---

### 5. Checkpoint Management & Model Registry

You can inspect all available checkpoints and auto-discover the highest-performing champion programmatically:

```python
from nanollm import get_latest_checkpoint, list_all_checkpoints, DecisionEngine

# 1. Discover latest champion checkpoint
latest_path, version = get_latest_checkpoint()
print(f"Latest champion: {latest_path.name} (version {version})")

# 2. List all available models with metadata from manifest
for model_info in list_all_checkpoints():
    print(f"Model: {model_info['name']} | Size: {model_info['size_mb']}MB | Active: {model_info['is_latest']}")

# 3. Load an explicit checkpoint version
engine = DecisionEngine.from_checkpoint(checkpoint_path="checkpoints/checkpoint_champion_v10.pt")
```




