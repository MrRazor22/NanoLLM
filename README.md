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
    classDef harness fill:#fff1f2,stroke:#e11d48,stroke-width:2px,color:#9f1239;

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
            ENC["SlotEncoder (IEncoder): Packs state + questions into slot layout"]
            SCORER["TorchScorer (IScorer): Single-pass forward pass (ModernBERT)"]
            RES["DecisionResolver (IResolver): Calibrated softmax, sigmoid & min-max scale"]
        end
    end

    subgraph Outputs ["4. Instant Calibrated Result (sub-40ms)"]
        RES_OUT["DecisionResult: department='Infra' (0.97), urgent=True (0.99), severity=94.2"]
    end

    subgraph TrainingPrimitive ["5. Optimization Primitive (piqo/trainer)"]
        direction TB
        ET["EpochTrainer (Contract: ITrainer)"]

        subgraph TrainLayers ["Outer Decorators (layers/)"]
            METRICS["MetricsLayer: Live loss, throughput & step telemetry"]
            CKPT["CheckpointingLayer: Auto-evaluates & saves champion weights"]
        end

        subgraph TrainPolicies ["Irreducible Policies (policies/)"]
            LOSS["CalibratedLoss (ILoss): Multi-task loss registry (CE, BCE, MSE)"]
        end
    end

    subgraph DataEngine ["6. Data Harness (harness/)"]
        SOURCES["12+ Task Sources: Banking, Security, Glaive Tools, Intent, Guardrails"]
        TDS["TrainingDataset: Streamer & slot-aligned batch generator"]
    end

    %% Inference Flow
    Client --> Inputs
    Inputs --> DE
    DE --> PROF
    PROF --> ENC
    ENC --> SCORER
    SCORER --> RES
    RES --> RES_OUT

    %% Training Flow
    SOURCES --> TDS
    TDS --> ET
    ET --> LOSS
    ET --> METRICS
    ET --> CKPT
    CKPT -. "Loads champion weights (.pt)" .-> SCORER

    class CLI,SDK client;
    class CTX,QS input;
    class DE,PROF,ENC,SCORER,RES engine;
    class RES_OUT output;
    class ET,LOSS,METRICS,CKPT trainer;
    class SOURCES,TDS harness;
```

---

## Architectural Topology & Responsibilities

| Subsystem | Boundary Contract | Implementation | Role & PFD Discipline |
| :--- | :--- | :--- | :--- |
| **Inference** | `IDecisionEngine` | `DecisionEngine` | Orchestrates single-pass micro-decisions; strictly decoupled from frameworks. |
| **Inference Policies** | `IEncoder`, `IScorer`, `IResolver` | `SlotEncoder`, `TorchScorer`, `DecisionResolver` | Discrete sub-primitives handling slot serialization, transformer scoring, and calibration. |
| **Inference Layers** | Decorator ($\lambda: P \to P$) | `ProfilingLayer` | Non-invasive instrumentation for sub-millisecond stage tracking. |
| **Optimization** | `ITrainer` | `EpochTrainer` | Drives multi-epoch model convergence and validation loops. |
| **Trainer Policies** | `ILoss`, `LossRegistry` | `CalibratedLoss` | Dynamic loss composition tailored to multi-task question heads. |
| **Trainer Layers** | Decorator ($\lambda: P \to P$) | `MetricsLayer`, `CheckpointingLayer` | Pluggable telemetry monitoring and champion checkpoint persistence. |
| **Data Harness** | Pure Python Iterators | `TrainingDataset`, Sources | Standardizes 12+ real-world heterogeneous datasets into slot layouts. |

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
from piqo import EpochTrainer, MetricsLayer, CheckpointingLayer
from harness.dataset import TrainingDataset

# Pipeline composition
trainer = (
    EpochTrainer.from_backbone("answerdotai/ModernBERT-base")
    | MetricsLayer()
    | CheckpointingLayer(output_path="checkpoints/checkpoint_champion.pt")
)

train_loader, val_loader = TrainingDataset.loaders("harness/dataset/data/splits/train.jsonl")

# Stream training statistics
for stats in trainer.fit(train_loader, val_batches=val_loader, epochs=3):
    pass
```

### 3. Interactive CLI
```bash
# Terminal playground
python cli.py --play

# Incident triage preset
python cli.py "Postgres CPU at 99%" --preset triage

# View available checkpoints
python cli.py --models
```

---

## Architectural Principles (Primitive-First Design)

1. **Zero Framework Types in Contracts:** Interfaces (`IDecisionEngine`, `ITrainer`, `ILayoutEncoder`, etc.) exchange pure Python DTOs. Zero PyTorch tensors leak across boundary contracts.
2. **Fractal Directory Topology:** Disk folders match the architectural tree 1:1. The root hosts only sovereign entrypoints; internal strategies live in `policies/`, and decorators live in `layers/`.
3. **The 150-Line Cognitive Limit:** Every source file in the repository fits within human working memory ($\le 142$ lines).
4. **Zero Backward-Compatibility Bloat:** High-density, zero-loss functional compression (~1,750 lines end-to-end).
