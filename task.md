# System One 1-to-1 Alignment Task Plan

## 1. Documentation & Competitor Assets
- [x] Download authoritative TypeSafe AI System One specification docs (`docs/spec/systemone/`)
- [x] Move `d:\CodeBase\systemonemodels` into `Piqo/systemonemodels/`
- [x] Add `systemonemodels/` to `.gitignore`
- [x] Add authoritative System One reference note to `AGENT.md`

## 2. Decision Case & Layout (Instruction Support)
- [x] Update `DecisionCase` in `data/case.py` to support `instruction: str = ""`
- [x] Update `MaskedLayout` in `model/layout.py` to cleanly encode `instruction` when present

## 3. Inference Layer (System One 1-to-1 Spec)
- [x] Update `inference/options.py`:
  - `ChoiceOption(options: Sequence[str] | Mapping[str, str], instruction: str = "")`
  - `NoulOption(instruction: str, criteria: Mapping[str, str] | None = None)`
  - `ScoreOption(criteria: Sequence[str], instruction: str = "")` (No fake hardcoded defaults; criteria required)
  - `NoulAnswer(noul: float, decision: bool, confidence: float)`
  - `ChoiceAnswer(choice: str, confidence: float, probabilities: Mapping[str, float])`
  - `ScoreAnswer(score: float, confidence: float, probabilities: Sequence[float])`
  - `option_to_candidates(option: IOption) -> Sequence[str]`
  - `scores_to_answer(option: IOption, scores: Sequence[float]) -> IAnswer`
- [x] Update `inference/decision_engine.py` to pass `state`, `instruction`, and `candidates` to `DecisionCase`
- [x] Update `inference/profiling_layer.py` for signature consistency

## 4. Competitor Adapters & Callers
- [x] Align competitor adapters (`gliner_decide.py`, `jev.py`, `kev.py`, `laya.py`, `verdict.py`)
- [x] Align CLI (`inference/__main__.py`) and benchmark scorer (`benchmark/policies/scorer.py`)
- [x] Verify exports in `piqo.py`

## 5. Verification & Clean Commit
- [x] Run fast import/syntax check
- [x] Commit changes cleanly (`1681879`)
