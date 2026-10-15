# Piqo

Ultra-fast, calibrated micro-decision engine designed with **Primitive-First Design (PFD)**. Evaluates high-cardinality multi-task questions (classification, boolean decisions, calibrated scores) in single-pass sub-40ms inference.

---

## End-to-End System Map

```mermaid
flowchart TD
    classDef client fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1;
    classDef input fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#92400e;
    classDef engine fill:#ecfdf5,stroke:#059669,stroke-width:2px,color:#065f46;
    classDef output fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#15803d;
    classDef trainer fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8;
    classDef curriculum fill:#fff1f2,stroke:#e11d48,stroke-width:2px,color:#9f1239;

    subgraph Client ["1. Client & Invocation Surfaces"]
        CLI["CLI: python cli.py"]
        SDK["Python SDK: from piqo import ..."]
    end

    subgraph Inputs ["2. Input Bundle"]
        CTX["Context State: 'Postgres CPU reached 99%, pool exhausted'"]
        QS["Questions DTO: Choice(...) + Noul(...) + Score(...)"]
    end

    subgraph InferencePrimitive ["3. Inference Primitive (piqo/engine)"]
        direction TB
        DE["DecisionEngine (Contract: IDecisionEngine)"]

        subgraph InfLayers ["Outer Decorators (layers/)"]
            PROF["ProfilingLayer: Per-stage latency & timing traces"]
        end

        subgraph InfPolicies ["Irreducible Policies (policies/)"]
            RES["DecisionResolver (IResolver): Calibrated softmax, sigmoid & min-max scale"]
        end
    end

    subgraph ModelPrimitive ["4. Model Primitive (model/)"]
        direction TB
        MODEL["TorchModel (Contract: IModel): Backbone + Tokenizer + Prediction Layout"]
        LOSS["CalibratedLoss (ILoss): Multi-task loss registry (CE, BCE, MSE)"]
    end

    subgraph Outputs ["5. Instant Calibrated Result (sub-40ms)"]
        RES_OUT["DecisionResult: department='Billing' (0.82), urgent=True (0.96), severity=97.3"]
    end

    subgraph TrainingPrimitive ["6. Training Primitive (training/)"]
        direction TB
        ET["EpochTrainer (Contract: ITrainer)"]

        subgraph TrainLayers ["Outer Decorators (layers/)"]
            METRICS["MetricsLayer: Live loss, throughput & step telemetry"]
            CKPT["CheckpointingLayer: Auto-evaluates & saves champion weights"]
        end
    end

    subgraph DataEngine ["7. Data Domain (data/)"]
        SOURCES["12+ Task Sources: Banking, Security, Glaive Tools, Intent, Guardrails"]
        TDS["Dataset: Self-wiring budget split, n-gram leakage, and cache policies"]
    end

    %% Inference Flow
    Client --> Inputs
    Inputs --> DE
    DE --> PROF
    PROF --> MODEL
    MODEL --> RES
    RES --> RES_OUT

    %% Training Flow
    SOURCES --> TDS
    TDS --> ET
    ET --> MODEL
    ET --> METRICS
    ET --> CKPT
    CKPT -. "Loads champion weights (.safetensors)" .-> MODEL

    class CLI,SDK client;
    class CTX,QS input;
    class DE,PROF,RES engine;
    class MODEL,LOSS model;
    class RES_OUT output;
    class ET,METRICS,CKPT trainer;
    class SOURCES,TDS data;
```

---

## Architectural Topology & Responsibilities

| Domain | Boundary Contract | Primary Implementation | Role & PFD Discipline |
| :--- | :--- | :--- | :--- |
| **Inference (`inference/`)** | `IDecisionEngine` | `DecisionEngine` | Orchestrates single-pass micro-decisions; strictly decoupled from frameworks. |
| **Inference Policies** | `IResolver` | `DecisionResolver` | Calibrated softmax, sigmoid & min-max scale resolution. |
| **Inference Layers** | Decorator ($\lambda: P \to P$) | `ProfilingLayer` | Non-invasive instrumentation for sub-millisecond stage tracking. |
| **Model (`model/`)** | `IModel` | `TorchModel` | Encapsulates weights, tokenizer layout, and gradient step executions. |
| **Model Policies** | `ILoss` | `CalibratedLoss` | Dynamic loss composition tailored to multi-task question heads. |
| **Training (`training/`)** | `ITrainer` | `EpochTrainer` | Drives multi-epoch model convergence and validation loops. |
| **Training Layers** | Decorator ($\lambda: P \to P$) | `MetricsLayer`, `CheckpointingLayer` | Pluggable telemetry monitoring and champion checkpoint persistence. |
| **Data (`data/`)** | `IDataset` | `Dataset` | Clean ingestion, budget splitting, 13-gram leakage policy, and disk caching. |
| **Benchmark (`benchmark/`)** | `IScorecard` | `Scorecard` | Evaluates engine accuracy & latency against competitive baselines. |

---

## Quickstart

### 1. Run Inference
```python
from piqo import DecisionEngine, Choice, Noul, Score, ProfilingLayer

# Initialize engine decorated with profiling layer
engine = DecisionEngine.from_checkpoint() | ProfilingLayer()

# Run single-pass multi-question evaluation
result = engine.decide(
    state="Postgres CPU reached 99% and connection pool is exhausted.",
    questions=[
        Choice("department", ["Billing", "Infrastructure", "Security"]),
        Noul("urgent", instruction="Is immediate incident response required?"),
        Score("severity", min_value=0.0, max_value=100.0),
    ]
)

print(f"Latency: {result.latency_ms:.2f} ms")
for name, ans in result.answers.items():
    print(f"  {name}: {ans}")
```

### 2. Run Training
```python
from piqo import Dataset, EpochTrainer, MetricsLayer, CheckpointingLayer, TorchModel

# Self-wiring dataset loader
dataset = Dataset()
train_cases = dataset.load("train")
val_cases = dataset.load("val")

# Pipeline composition with decorators
trainer = (
    EpochTrainer(model=TorchModel())
    | MetricsLayer()
    | CheckpointingLayer(output_path="checkpoints/checkpoint_champion.safetensors")
)

# Stream training telemetry
for stats in trainer.fit(train_cases, val_cases=val_cases, epochs=1):
    print(f"Epoch {stats.epoch} | Loss: {stats.loss:.4f} | Val: {stats.val_loss:.4f}")
```

### 3. Interactive CLI
```bash
# Incident triage preset
python -m inference "Postgres CPU at 99%" --preset triage

# Build dataset corpus
python -m data --train-quota 2000 --val-quota 500

# Train model
python -m training --epochs 1 --batch-size 16

# Run benchmark evaluation
python -m benchmark
```

---

## Architectural Principles (Primitive-First Design)

1. **Zero Substrate Leakage in Contracts:** Interfaces (`IDecisionEngine`, `ITrainer`, `IModel`, `IDataset`) exchange pure domain DTOs. Zero token IDs or PyTorch tensors leak across primitive boundaries.
2. **Fractal Directory Topology:** Disk folders match the architectural tree 1:1. The root hosts only sovereign entrypoints; internal strategies live in `policies/`, and decorators live in `layers/`.
3. **The 150-Line Cognitive Limit:** Every source file in the repository fits within human working memory ($\le 150$ lines).
4. **Self-Wiring Primitives:** Every constructor provides production defaults; callers never need to import internal default policies.
5. **Zero Backward-Compatibility Bloat:** Lean, zero-loss functional compression.

