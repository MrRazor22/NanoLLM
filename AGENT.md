# NanoLLM Agent Guidelines

## 1. Architectural Doctrine: Primitive-First Design (PFD)
- **Always Apply PFD:** All architecture, boundary definitions, refactorings, and reviews in this repository strictly follow Primitive-First Design (PFD).
- **Authoritative Reference:** Always read and adhere to `~/.gemini/config/skills/pfd/SKILL.md` (invocable via `/primitive-first-design`). Never use partial heuristics, legacy ATA terminology, or speculative wrapper abstractions.
- **Ignore Legacy Codebase Labels:** Do not accept past labels or conventional coding patterns as architectural truth. Primitives are discovered from domain boundaries and consumer demand, not from legacy comments or file names.

## 2. Operational Workflows & Commands
- **Interactive Terminal Playground:**
  ```bash
  python cli.py --play
  ```
- **Run One-Shot Inference:**
  ```bash
  python cli.py "Database CPU reached 99% and connection pool is exhausted"
  python cli.py "Look up encryption policy" --preset routing
  ```
- **Build / Adapt Training Dataset:**
  ```bash
  python -m harness.dataset
  ```
- **Run Training (1-Epoch Adaptation):**
  ```bash
  python train.py --epochs 1 --lr 2e-5
  ```
- **Run Benchmarks:**
  ```bash
  python -m benchmark
  python -m benchmark --agentic
  python -m benchmark --typed-decisions
  python -m benchmark --abstention
  ```
