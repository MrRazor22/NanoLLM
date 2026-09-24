# NanoLLM Design Philosophy

## Core Principle

Less code is a side effect of correct logic. Not a goal unto itself—bloat comes from wrong abstractions.

---

## The Axiomatic Triad Architecture

Every component in NanoLLM strictly belongs to one of four architectural tiers:

```text
┌─────────────────────────────────────────────────────────────┐
│                    Consumers / Drivers                      │
│            (scripts/train.py, scripts/benchmark.py)         │
└──────────────────────────────┬──────────────────────────────┘
                               │ orchestrates
┌──────────────────────────────▼──────────────────────────────┐
│                  Outer Composable Layers                    │
│            (nanollm/layers/profiling.py - λ_F: F -> F)      │
└──────────────────────────────┬──────────────────────────────┘
                               │ wraps
┌──────────────────────────────▼──────────────────────────────┐
│                    The Core Primitives                      │
│     IDecisionEngine (Domain)  │  ISubstrate (Neural Tensor) │
└──────────────────────────────▲──────────────────────────────┘
                               │ injects
┌──────────────────────────────┴──────────────────────────────┐
│                     Injected Policies                       │
│     SlotAssembler  │  DecisionResolver  │  CalibratedLoss    │
└─────────────────────────────────────────────────────────────┘
```

### 1. The Core Primitives (`nanollm/core/`)
- Irreducible, orthogonal contracts defining *what* the system does.
- `IDecisionEngine`: The single domain entry point (`decide(state, questions) -> DecisionResult`).
- `ISubstrate`: The hardware tensor compute primitive (`forward(input_ids, mask) -> scores`).
- Kept razor-thin (<80 lines). Zero intermediate glue.

### 2. Injected Policies (`nanollm/policies/`)
- Pure, swappable strategies configured into the primitive at initialization.
- `ISlotAssembler`: Formats sequence layout and maps token slot coordinates. Single source of truth for both training and inference.
- `IResolver`: Maps raw scalar logits at slot coordinates into typed, calibrated decision results (`Choice`, `Noul`, `Score`).
- `ITokenizer`: Text-to-token encoding and decoding (`SubwordTokenizer`, `ByteTokenizer`).
- `CalibratedLoss`: Multi-task objective function combining Cross-Entropy, BCE, and Brier score penalty.

### 3. Composable Layers (`nanollm/layers/`)
- Endomorphic wrappers ($\lambda_F: F \to F$) that decorate the primitive from the outside without contract mutation.
- `ProfilingLayer`: Decorates `IDecisionEngine` with CUDA-synchronized latency timing.
- New capabilities (caching, audit logging, telemetry) must always be added as outer layers, never monkey-patched into core primitives.

### 4. Consumers & Orchestrators (`scripts/`, `examples/`)
- Standalone execution drivers (`train.py`, `benchmark.py`, `evaluate.py`, `basic_decision.py`).
- Strictly isolated from the `nanollm` library package to ensure zero dependency bloat and eliminate root clutter.

### 5. Interfaces Are the System; Implementations Are Transient
- The entire architecture is anchored strictly on razor-sharp interfaces (`typing.Protocol`).
- The core primitive interfaces (`IDecisionEngine`, `ISubstrate`) define the fundamental domain boundary.
- The injected policy interfaces (`ISlotAssembler`, `IResolver`, `ITokenizer`) define the swappable strategy points.
- Concrete classes are interchangeable implementation details; the engine coordinates only through interface contracts.

---

## Retrospective: Mistakes Made & Evolutionary Breakthroughs

### 1. The Bi-Encoder Fallacy (Mistake 1)
- **What went wrong:** We initially built a dual-tower bi-encoder—encoding state in one forward pass and option candidates in a second pass, computing cosine similarity between pooled vectors.
- **The Failure:** 
  1. *Terrible Latency:* Two full forward passes took ~50ms on GPU, destroying the <2ms goal.
  2. *Catastrophic Accuracy Drop:* Options could not attend to the state context during representation formation, leading to severe underfitting (22% accuracy).
  3. *Inability to Calibrate:* Cosine similarity cannot be naturally calibrated for true epistemic probabilities.
- **The Fix (Single-Pass Cross-Attention):** Moved to single-sequence cross-attention (`state [SEP] question: [MASK] opt1 [MASK] opt2 ...`). All tokens attend to each other in one pass. Latency dropped to **1.1ms** (45x faster) with zero-shot accuracy jumping to near 100%.

### 2. Finding the Right Primitive (Mistake 2)
- **What went wrong:** We initially created speculative abstractions—separate QuestionCollators, distinct Head layers for Choice vs Noul vs Score, and multi-stage pipeline classes.
- **The Breakthrough:** There is only ONE neural operation: *extracting scalar logits at target slot token positions*. Choice is just softmax across slot logits; Noul is sigmoid on one slot logit; Score is scaled sigmoid on one slot logit. The primitive is simply `ISubstrate`, and the decoding strategy is an injected `DecisionResolver`.

### 3. Layout Train/Serve Skew (Mistake 3)
- **What went wrong:** Sequence formatting was previously duplicated between training collators and inference prediction loops, risking subtle coordinate misalignment.
- **The Fix:** `SlotAssembler` was promoted to an injected policy. It is the single source of truth for prompt layout and token coordinate indexing across both training batches and real-time inference.

### 4. The Flat File Trap (Mistake 4)
- **What went wrong:** Dumping all files flat in `nanollm/` obscured architectural boundaries and blurred the distinction between primitives, policies, and outer layers.
- **The Fix:** Grouped strictly by role (`core/`, `policies/`, `layers/`, `data/`, `scripts/`, `examples/`). Every file is now under 80 lines and immediately reveals its exact responsibility.

---

## Guidelines

### No Needless Convenience
If the concept is clear, the code is short. Stupid logic needs convenience wrappers. We don't add helper methods just to save a few keystrokes—we add them when they express a clear, reusable concept.

### Match or Exceed the Competition
Express Jev's System 1 architecture in 1/10th the code through sharper design. Feature parity with frontier decision models, zero line-count bloat.

### No Design Smell
Reducing code that introduces smell is worse than the bloat it replaced. If a simplification makes the code harder to understand or maintain, it's not a simplification—it's a regression.

### Data Over Behavior
Schemas, questions, states, and decisions are pure immutable data structures (dataclasses), not behavioral objects with lifecycle hooks. Keep the system simple, composable, and serialization-ready.

### Interface-First for Behaviors
Define small, focused interfaces for any class exhibiting behavior (encoders, heads, calibrators). Always depend on abstractions rather than concrete classes.

### Strict Size & Complexity Limits
- Max 150 lines per file: If a file exceeds 150 lines, it is doing too much and has a design smell.
- Max 4–5 methods per class: If a class has more than 4–5 methods, it violates Single Responsibility and must be decomposed.

### Self-Documenting Code & No Explanatory Comments
If code requires comments to explain what it does, it is a design smell. Code must be self-documenting through clear naming and clean structure.

### Generic & Multimodal by Design
Keep root representations and decision contracts generic. Never attach single-modality assumptions to root pipeline interfaces.

### Production Calibrated Decisions
All probabilities must be epistemically calibrated (calibrated cross-entropy / Brier score loss) so output probabilities reliably reflect true certainty.

### Zero Speculative Tweaking & Compute-Respecting Discipline
- Speculative ML tweaking, unverified trial-and-error architecture modifications, and ungrounded "hunches" are STRICTLY FORBIDDEN.
- GPU compute and user time are constrained, serious resources; NEVER waste compute or training cycles on speculative layers or unproven changes.
- Every architectural change must be justified by prior empirical or mathematical necessity, verified against minimal primitives, and aligned with the user before burning a single GPU second.
- If an approach hits saturation or regression, immediately report the empirical facts and revert to the last verified baseline rather than chasing speculative patches.
