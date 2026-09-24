# NanoLLM Design Philosophy: Axiomatic Derivation Architecture (ADA)

## Core Principle & Soul

> **"Correctly identifying the fundamental primitive of a system naturally minimizes its architecture, because unnecessary abstractions become redundant when the primitive itself correctly represents the underlying problem; once the primitive is correct, composition, injectable policies, and generic layers provide extensibility without requiring the core primitive to continuously grow new abstractions."**

Less code is a natural side effect of correct design, not a goal unto itself. Bloat comes from developer laziness—introducing convenience wrappers, helper overloads, and speculative abstraction layers instead of thinking deeply to fix latent flaws in the bedrock primitive.

---

## The Operational Boundaries of NanoLLM

NanoLLM is decomposed strictly by operational capabilities into irreducible **Axioms (Primitives)**, with swappable internal strategies (**Derived Policies**), endomorphic boundary decorators (**Derived Layers**), and shared foundational substrates:

```text
┌─────────────────────────────────────────────────────────────┐
│                 External Consumers & Runners                │
│                 (examples/, benchmarks, CLI)                │
└──────────────────────────────┬──────────────────────────────┘
                               │ orchestrates
┌──────────────────────────────▼──────────────────────────────┐
│                    nanollm/ (Autonomous Domains)            │
│                                                             │
│  nanollm/inference/               nanollm/training/         │
│  ├── IDecisionEngine (Axiom)      ├── ITrainer (Axiom)      │
│  ├── schema.py                    ├── checkpointing_        │
│  ├── data/                        │   layer.py              │
│  │   ├── benchmark.json           ├── data/                 │
│  │   └── baseline.json            │   ├── train_adapt.jsonl │
│  ├── layers/                      │   └── val_adapt.jsonl   │
│  │   ├── profiling.py             └── policies/             │
│  │   └── hierarchical.py              ├── loss.py           │
│  └── policies/                        ├── dataset.py        │
│      ├── assembler.py                 └── curriculum.py     │
│      ├── resolver.py                                        │
│      └── tokenizer.py                                       │
│                                                             │
│  nanollm/model/ (Autonomous Shared Foundation Substrate)    │
│  ├── NanoModel (Substrate Primitive)                        │
│  └── checkpoints/                                           │
│      └── checkpoint_champion_v2.pt                          │
└─────────────────────────────────────────────────────────────┘
```

### 1. Operational Boundaries (1 Boundary = 1 Primitive)
Each boundary in `nanollm/` is fully autonomous and owns its functional code, policies, layers, and operational assets:
- **`nanollm/inference/` (Semantic Decision Execution):** 
  - **Axiom ($P$):** `IDecisionEngine` (`DecisionEngine`)
  - **Schema:** `schema.py` (`Choice`, `Noul`, `Score`, `DecisionResult`)
  - **Assets:** `data/benchmark.json` (180 golden evaluation questions), `data/baseline.json` (verified score history)
  - **Derived Policies ($\mathcal{P}(P)$):** `ISlotAssembler` (`SlotAssembler`), `IResolver` (`DecisionResolver`), `ITokenizer` (`SubwordTokenizer`)
  - **Derived Layers ($\text{End}(P)$):** `ProfilingLayer`, `HierarchicalLayer`
- **`nanollm/training/` (Parameter Optimization):** 
  - **Axiom ($P$):** `ITrainer` (`EpochTrainer`)
  - **Derived Layer ($\text{End}(P)$):** `CheckpointingLayer`
  - **Assets:** `data/train_adapt.jsonl`, `data/val_adapt.jsonl`
  - **Derived Policies ($\mathcal{P}(P)$):** `CalibratedLoss`, `MultiQuestionCollator`, `AdaptationCurriculum`, `FoundationCurriculum`
- **`nanollm/model/` (Autonomous Shared Foundation Substrate):**
  - **Substrate Primitive:** `NanoModel` (`model.py`)
  - **Assets:** `checkpoints/checkpoint_champion_v2.pt` (neural weights)
  - Consumed cleanly across inference and training as an injected substrate dependency.

### 2. Verification & Benchmarking Are Operational Consumers
Testing and evaluation are **operational consumers**, not an artificial architectural boundary. Running a benchmark against ground truth is simply exercising the execution primitive (`DecisionEngine.decide()`) over test data. Inventing fake primitives (`IEvaluator`) or fake layers just to wrap a test loop is Abstraction Theater.

### 3. Universal ADA Code Taxonomy
Every line of code in NanoLLM strictly belongs to one of four categories:
1. **The Axiom & Derivations (Behavior):**
   - **Axiom ($P$):** Irreducible contract defining *what* the boundary does.
   - **Policy ($\pi$):** Swappable strategy defining *how* an internal step executes (domain nouns, no `*Policy` suffix).
   - **Layer ($\lambda$):** Endomorphic decorator ($\lambda_P: P \to P$) decorating the primitive externally (MUST carry `*Layer` suffix).
2. **DTOs / Schema (State):** Pure, immutable domain schemas (`Choice`, `DecisionResult`, `DecisionSample`).
3. **Pure Functions / Extension Methods (Stateless Transforms):** Zero side-effect transforms (`save_jsonl`, `load_jsonl`, `choice_question`).
4. **Composition Roots / Runners (External Wiring):** External scripts and runners in `examples/` orchestrating boundaries.

### 4. Interfaces Are the System; Implementations Are Transient
- The core primitive interfaces (`IDecisionEngine`, `ITrainer`) define the fundamental domain boundaries.
- The injected policy interfaces (`ISlotAssembler`, `IResolver`, `ITokenizer`) define swappable strategy points.
- Concrete classes coordinate through razor-sharp contracts, preventing coupling and accidental bloat.

---

## Retrospective: Mistakes Made & Evolutionary Breakthroughs

1. **Procedural Script Eradication:** Deleted all 6 ad-hoc scripts in `scripts/` (saving 573 lines of procedural rot), replacing them with unified domain primitives and lean drivers.
2. **Eliminated "Inside vs Outside" Fallacy:** Replaced messy procedural CLI loops, table formatters, and closures with clean boundary capabilities (`ModelEvaluator.from_engine`, `print_benchmark_table`).
3. **The Clutter-Threshold Rule:** Cleanly flattened lean 2-file boundaries (`evaluation/`) while subordinating cluttered boundaries (`engine/policies/`, `training/policies/`), achieving zero 1-file subdirectories.
4. **Folder-Namespace 1:1 Isomorphism:** Every directory maps 1:1 to an explicit logical namespace with `__init__.py`.
