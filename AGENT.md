# NanoLLM Design Philosophy

## Core Principle

Less code is a side effect of correct logic. Not a goal unto itself—bloat comes from wrong abstractions.

---

## The Boundary-First Axiomatic Triad Architecture

Every operational domain in NanoLLM is modeled as an autonomous, cohesive **Boundary Triad**:

```text
┌─────────────────────────────────────────────────────────────┐
│                 External Consumers / Drivers                │
│                 (drivers/cli.py, examples/)                 │
└──────────────────────────────┬──────────────────────────────┘
                               │ orchestrates
┌──────────────────────────────▼──────────────────────────────┐
│                    Operational Boundaries                   │
│                                                             │
│  nanollm/engine/          nanollm/training/    nanollm/eval/│
│  ├── IDecisionEngine      ├── ITrainer         ├── IEvaluator
│  ├── schema.py            ├── checkpointing_   ├── profiling_
│  ├── layers/              │   layer.py         │   layer.py  
│  │   ├── profiling.py     └── policies/        └── reporter.py
│  │   └── hierarchical.py      ├── loss.py                   │
│  └── policies/                ├── dataset.py                │
│      ├── substrate.py         ├── curriculum.py             │
│      ├── assembler.py         ├── foundation.py             │
│      ├── resolver.py          ├── taxonomies.py             │
│      └── tokenizer.py         └── builder.py                │
└─────────────────────────────────────────────────────────────┘
```

### 1. Operational Boundaries (1 Boundary = 1 Primitive)
Each subsystem owns its complete Triad:
- **`nanollm/engine/` (Inference):** 
  - Primitive: `IDecisionEngine` (`DecisionEngine`)
  - Schema: `schema.py` (`Choice`, `Noul`, `Score`, `DecisionResult`)
  - Policies ($\ge 4$ items $\implies$ `policies/`): `ISubstrate` (`DecisionSubstrate`, `NanoModel`), `ISlotAssembler` (`SlotAssembler`), `IResolver` (`DecisionResolver`), `ITokenizer` (`SubwordTokenizer`)
  - Layers ($\ge 2$ items $\implies$ `layers/`): `ProfilingLayer`, `HierarchicalLayer`
- **`nanollm/training/` (Optimization):** 
  - Primitive: `ITrainer` (`EpochTrainer`)
  - Layer (Lean $\implies$ Flat): `CheckpointingLayer`
  - Policies ($\ge 5$ items $\implies$ `policies/`): `CalibratedLoss`, `MultiQuestionCollator`, `AdaptationCurriculum`, `FoundationCurriculum`, taxonomies, builder
- **`nanollm/evaluation/` (Verification):** 
  - Primitive: `IEvaluator` (`ModelEvaluator`)
  - Layer (Lean $\implies$ Flat): `ProfilingEvaluatorLayer`
  - Utilities: `reporter.py` (`print_benchmark_table`)

### 2. Universal ATA Taxonomy: Behavior, State, and Utilities
Every piece of code in NanoLLM strictly belongs to one of four canonical types:
1. **The Triad (Behavior):**
   - **Primitive ($P$):** Irreducible contract defining *what* the boundary does.
   - **Policy ($\pi$):** Swappable strategy defining *how* an internal step executes (domain nouns, no `*Policy` suffix).
   - **Layer ($\lambda$):** Endomorphic decorator ($\lambda_P: P \to P$) decorating the primitive externally (MUST carry `*Layer` suffix).
2. **DTOs / Schema (State):** Pure, immutable domain schemas (`Choice`, `DecisionResult`, `DecisionSample`).
3. **Pure Functions / Extension Methods (Stateless Utilities):** Zero side-effect transforms (`save_jsonl`, `load_jsonl`, `choice_question`).
4. **Composition Root / Driver (Zero-Logic Wiring):** Declarative wiring entrypoint in `drivers/` ($\le 50-80$ lines).

### 3. Consumers & Drivers (`drivers/cli.py`, `examples/`)
- Standalone execution drivers reside in `drivers/`, keeping repository root pristine.
- Zero business or presentation logic: the driver purely parses CLI args, wires primitives, decorates them with layers, and invokes them.

### 4. Interfaces Are the System; Implementations Are Transient
- The entire architecture is anchored strictly on razor-sharp interfaces (`typing.Protocol`).
- The core primitive interfaces (`IDecisionEngine`, `ITrainer`, `IEvaluator`) define the fundamental domain boundaries.
- The injected policy interfaces (`ISubstrate`, `ISlotAssembler`, `IResolver`, `ITokenizer`) define the swappable strategy points.
- Concrete classes are interchangeable implementation details; components coordinate only through interface contracts.

---

## Retrospective: Mistakes Made & Evolutionary Breakthroughs

1. **Procedural Script Eradication:** Deleted all 6 ad-hoc scripts in `scripts/` (saving 573 lines of procedural rot), replacing them with unified domain primitives and lean drivers.
2. **Eliminated "Inside vs Outside" Fallacy:** Replaced messy procedural CLI loops, table formatters, and closures with clean boundary capabilities (`ModelEvaluator.from_engine`, `print_benchmark_table`).
3. **The Clutter-Threshold Rule:** Cleanly flattened lean 2-file boundaries (`evaluation/`) while subordinating cluttered boundaries (`engine/policies/`, `training/policies/`), achieving zero 1-file subdirectories.
4. **Folder-Namespace 1:1 Isomorphism:** Every directory maps 1:1 to an explicit logical namespace with `__init__.py`.
