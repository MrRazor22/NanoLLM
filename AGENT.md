# NanoLLM Design Philosophy

## Core Principle

Less code is a side effect of correct logic. Not a goal unto itself—bloat comes from wrong abstractions.

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
