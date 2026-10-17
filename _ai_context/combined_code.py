// ============================================================================
// FILE: benchmark/__main__.py (116 code lines, 128 total)
// ============================================================================

import os, warnings
from pathlib import Path
from typing import Mapping, Optional, Sequence
import typer

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from benchmark.benchmark import Benchmark
from benchmark.registry import AVAILABLE_ENGINES, BENCHMARK_SUITES
from benchmark.suites.suite import IBenchmarkSuite
from inference.decision_engine import IDecisionEngine
from inference.profiling_layer import ProfilingLayer
from inference.questions import Choice

app = typer.Typer(add_completion=False, help="Piqo Sovereign Benchmark CLI")

def resolve_suites(suite_arg: str) -> Sequence[IBenchmarkSuite]:
    """Resolves CLI suite argument into sovereign IBenchmarkSuite primitives."""
    arg = suite_arg.strip().lower()
    if arg in ("all", "all_suites", "official"):
        return [cls() for cls in BENCHMARK_SUITES.values()]
    queries = [s.strip().lower() for s in suite_arg.split(",") if s.strip()]
    matched = [cls for key, cls in BENCHMARK_SUITES.items() if any(q in key for q in queries)]
    if not matched:
        raise typer.BadParameter(f"No benchmark suites matched '{suite_arg}'. Available: {list(BENCHMARK_SUITES.keys())}")
    return [cls() for cls in dict.fromkeys(matched)]

def resolve_engines(
    model_arg: str,
    device: Optional[str] = None,
    checkpoint: Optional[str] = None,
) -> Mapping[str, IDecisionEngine]:
    """Resolves CLI model argument into instantiated IDecisionEngine primitives."""
    arg = model_arg.strip().lower()
    if arg == "all":
        keys = list(AVAILABLE_ENGINES.keys())
    else:
        requested = [k.strip() for k in arg.split(",") if k.strip()]
        invalid = [k for k in requested if k not in AVAILABLE_ENGINES]
        if invalid:
            raise typer.BadParameter(f"Unrecognized engine(s): {invalid}. Available: {list(AVAILABLE_ENGINES.keys())}")
        keys = requested
    engines: dict[str, IDecisionEngine] = {}
    for k in keys:
        cls = AVAILABLE_ENGINES[k]
        kwargs = {"device": device}
        if k == "piqo" and checkpoint is not None:
            kwargs["checkpoint"] = checkpoint
        try:
            engines[k.title()] = cls(**kwargs)
        except Exception as e:
            print(f"[{k.title()}] Skipping: could not initialize ({e})")
    if not engines:
        raise typer.BadParameter(f"No engines could be initialized for '{model_arg}'.")
    return engines

def run_playground(engines: Sequence[tuple[str, IDecisionEngine]]) -> None:
    names = ", ".join(n for n, _ in engines)
    print(f"\nPiqo Benchmark Playground [{names}]. Enter query (or 'q' to quit):")
    while True:
        try:
            ctx = input("\nContext (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if ctx.lower() in ("q", "quit", "exit") or not ctx:
            if not ctx: continue
            break
        try:
            raw_choices = input("Choices (comma-separated): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        opts = [o.strip() for o in raw_choices.split(",") if o.strip()]
        if not opts:
            continue
        qs = [Choice("decision", opts)]
        print(f"\nState: {ctx}\n" + "-" * 55)
        for m_name, eng in engines:
            try:
                res = eng.decide(ctx, qs)
                ans = res.answers.get("decision")
                val = getattr(ans, "choice", getattr(ans, "value", str(ans)))
                conf = getattr(ans, "confidence", 1.0)
                print(f"[{m_name:<10}] Choice: {val:<20} | Conf: {conf*100:.1f}% | Lat: {res.latency_ms:.1f}ms")
            except Exception as e:
                print(f"[{m_name:<10}] Error: {e}")
        print("-" * 55)

@app.command()
def main(
    suite: str = typer.Option("all", "--suite", "-s", help="Benchmark suite(s): all, laya, jev, kev (comma-separated)"),
    model: str = typer.Option("all", "--model", "-m", help="Model(s) to evaluate: all, piqo, gliner, jev, kev, laya, verdict (comma-separated)"),
    report: bool = typer.Option(False, "--report", "-r", help="Show leaderboard report without running models"),
    limit: Optional[int] = typer.Option(None, help="Item limit per suite"),
    device: Optional[str] = typer.Option(None, help="Target compute device (cpu, cuda, mps)"),
    checkpoint: Optional[str] = typer.Option(None, help="Model checkpoint path for Piqo"),
    skip_cached: bool = typer.Option(False, "--skip-cached", help="Skip models that already have scores in ledger"),
    play: bool = typer.Option(False, "--play", "-p", help="Playground mode"),
) -> None:
    bench = Benchmark()
    active_suites = resolve_suites(suite)

    if report:
        bench.report(suites=active_suites)
        return

    active_engines = resolve_engines(model, device=device, checkpoint=checkpoint)

    if play:
        live_engines = []
        for name, eng in active_engines.items():
            try:
                profiled = eng | ProfilingLayer()
                profiled.decide("warmup", [Choice("decision", ["warm"])])
                live_engines.append((name, profiled))
            except Exception as e:
                print(f"[{name}] Skipping playground: {e}")
        if not live_engines:
            print("No active engines available for playground.")
            return
        run_playground(live_engines)
        return

    bench.run(active_suites, active_engines, limit=limit, skip_cached=skip_cached)

if __name__ == "__main__":
    app()


// ============================================================================
// FILE: benchmark/benchmark.py (76 code lines, 90 total)
// ============================================================================

from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence, Union, runtime_checkable
from benchmark.policies.ledger import DiskLedgerPolicy, ILedgerPolicy
from benchmark.policies.reporter import ConsoleReporterPolicy, IReporterPolicy
from benchmark.policies.scorer import IScorerPolicy, ScorerPolicy
from benchmark.suites.suite import IBenchmarkSuite
from inference.decision_engine import IDecisionEngine

@runtime_checkable
class IBenchmark(Protocol):
    """Core Primitive Contract: Sovereign operational boundary for benchmark evaluation and reporting."""

    def run(
        self,
        suites: Sequence[IBenchmarkSuite],
        engines: Mapping[str, IDecisionEngine],
        limit: Optional[int] = None,
        skip_cached: bool = False,
    ) -> None: ...

    def report(self, suites: Optional[Sequence[IBenchmarkSuite]] = None) -> None: ...

class Benchmark(IBenchmark):
    """Core Primitive Implementation: Sovereign coordinator of benchmark suites, engines, and policies."""

    def __init__(
        self,
        scorer: Optional[IScorerPolicy] = None,
        ledger: Optional[ILedgerPolicy] = None,
        reporter: Optional[IReporterPolicy] = None,
    ):
        self.ledger = ledger or DiskLedgerPolicy()
        self.scorer = scorer or ScorerPolicy()
        self.reporter = reporter or ConsoleReporterPolicy()

    def run(
        self,
        suites: Sequence[IBenchmarkSuite],
        engines: Mapping[str, IDecisionEngine],
        limit: Optional[int] = None,
        skip_cached: bool = False,
    ) -> None:
        active_names: list[str] = []
        for suite in suites:
            try:
                items = list(suite.load_cases())
            except FileNotFoundError as e:
                print(f"\n[UNAVAILABLE] Suite '{suite.name.upper()}': {e}")
                continue

            active_names.append(suite.name)
            if limit:
                items = items[:limit]

            print(f"\n{'=' * 65}\nEvaluating Suite: {suite.name.upper()} ({len(items)} items)")
            if suite.published_baselines:
                print("Published Baselines:")
                for b_name, b_score in suite.published_baselines.items():
                    print(f"  {b_name:<24}: {b_score * 100:.1f}%")
            print(f"{'=' * 65}")

            s_hash = getattr(suite, "manifest_hash", None)
            for label, engine in engines.items():
                if skip_cached and self.ledger.get(label, suite=suite.name, expected_count=len(items), expected_hash=s_hash) is not None:
                    print(f"[{label}] Cached run found for '{suite.name}'. Skipping.")
                    continue
                try:
                    report = self.scorer.score(
                        engine,
                        items,
                        model_name=label,
                        suite=suite.name,
                    )
                    self.ledger.put(report, manifest_hash=s_hash)
                except Exception as e:
                    print(f"Skipping run for '{label}': {e}")

            self.reporter.suite(self.ledger.all(suite.name), suite.name)

        if len(active_names) > 1:
            self.reporter.summary(self.ledger.all(), suites=active_names)

    def report(self, suites: Optional[Sequence[IBenchmarkSuite]] = None) -> None:
        suite_names = list(dict.fromkeys(s.name for s in suites)) if suites else list(dict.fromkeys(s for s, d in self.ledger.all().items() if isinstance(d, dict)))
        for s_name in suite_names:
            self.reporter.suite(self.ledger.all(s_name), s_name)

        if len(suite_names) > 1:
            self.reporter.summary(self.ledger.all(), suites=suite_names)



// ============================================================================
// FILE: benchmark/competitors/gliner_decide.py (65 code lines, 71 total)
// ============================================================================

from typing import Optional, Sequence
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from inference.questions import (
    Choice, ChoiceResult, DecisionResult, DecisionState,
    Noul, NoulResult, Question, Score, ScoreResult,
)
from model.device import resolve_device

class GlinerDecideEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps Fastino's GLiNER2.5-Decide."""

    def __init__(
        self,
        model_id: str = "fastino/GLiNER2.5-Decide",
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = resolve_device(device)
        self._model = None

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        if self._model is None:
            try:
                try:
                    from gliner2 import AutoExtractor
                    m = AutoExtractor.from_pretrained(self.model_id)
                except (ImportError, TypeError):
                    from gliner import GLiNER
                    m = GLiNER.from_pretrained(self.model_id)
                if hasattr(m, "to"): m.to(self.device)
                self._model = m
            except Exception as e:
                raise ImportError(f"Cannot load GLiNER2.5-Decide ({self.model_id}): {e}. To run live: pip install gliner2")

        model = self._model
        answers = {}
        for q in questions:
            prompt_text = f"Question: {q.instruction}\n\nContext:\n{state}" if q.instruction else str(state)

            if isinstance(q, Choice):
                opt_map = {desc or k: k for k, desc in q.options.items()} if isinstance(q.options, dict) else {opt: opt for opt in q.options}
                if hasattr(model, "classify_text"):
                    raw = model.classify_text(prompt_text, {q.name: list(opt_map.keys())}).get(q.name, "")
                    if raw in opt_map:
                        k = opt_map[raw]
                        answers[q.name] = ChoiceResult(choice=str(k), confidence=1.0, probabilities={v: (1.0 if v == k else 0.0) for v in opt_map.values()})
                else:
                    preds = model.classify(prompt_text, list(opt_map.keys())) if hasattr(model, "classify") else model.predict_entities(prompt_text, list(opt_map.keys()))
                    scores = {opt_map[p["label"]]: float(p.get("score", 0.0)) for p in (preds or []) if p.get("label") in opt_map}
                    if scores and preds and preds[0].get("label") in opt_map:
                        total = sum(scores.values()) or 1.0
                        norm_p = {k: v / total for k, v in scores.items()}
                        top_k = opt_map[preds[0]["label"]]
                        answers[q.name] = ChoiceResult(choice=str(top_k), confidence=norm_p.get(top_k, 1.0), probabilities=norm_p)
            elif isinstance(q, Noul):
                if hasattr(model, "classify_text"):
                    ans_label = model.classify_text(prompt_text, {q.name: ["Yes", "No"]}).get(q.name, "")
                    if ans_label in ("Yes", "No"):
                        answers[q.name] = NoulResult(is_positive=(ans_label == "Yes"), probability=1.0 if ans_label == "Yes" else 0.0)
                else:
                    preds = model.classify(prompt_text, ["Yes", "No"]) if hasattr(model, "classify") else model.predict_entities(prompt_text, ["Yes", "No"])
                    if preds and preds[0].get("label") in ("Yes", "No"):
                        is_pos = (preds[0]["label"] == "Yes")
                        score = float(preds[0].get("score", 0.5))
                        answers[q.name] = NoulResult(is_positive=is_pos, probability=score if is_pos else 1.0 - score)
            elif isinstance(q, Score):
                raise NotImplementedError(f"GLiNER does not support continuous regression questions ({type(q).__name__})")
        return DecisionResult(answers=answers)

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)


// ============================================================================
// FILE: benchmark/competitors/jev.py (81 code lines, 90 total)
// ============================================================================

import os
from typing import Optional, Sequence
import httpx
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from inference.questions import (
    Choice, ChoiceResult, DecisionResult, DecisionState,
    Noul, NoulResult, Question, Score, ScoreResult, score_to_norm,
)

class JevEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps TypeSafe AI's hosted Jev API (/v1/systemone)."""

    def __init__(
        self,
        endpoint: str = "",
        api_key: str = "",
        model: str = "jev-1.13.0",
        timeout: float = 30.0,
        device: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("JEV_API_KEY", "")
        self.endpoint = endpoint or os.environ.get("JEV_ENDPOINT", "https://api.typesafe.ai/v1/systemone")
        self.timeout = timeout

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        if not self.api_key and "api.typesafe.ai" in self.endpoint:
            raise ValueError("TypeSafe Jev API key not set. Set JEV_API_KEY environment variable to evaluate hosted Jev.")

        q_dict = {}
        for q in questions:
            if isinstance(q, Choice):
                q_dict[q.name] = {
                    "type": "choice",
                    "instructions": q.instruction or "",
                    "criteria": q.options if isinstance(q.options, dict) else {o: o for o in q.options},
                }
            elif isinstance(q, Noul):
                q_dict[q.name] = {
                    "type": "noul",
                    "instructions": q.instruction or "",
                    "criteria": {"false": "No", "true": "Yes"},
                }
            elif isinstance(q, Score):
                q_dict[q.name] = {
                    "type": "score",
                    "instructions": q.instruction or "",
                    "min": q.min_value,
                    "max": q.max_value,
                }

        payload = {
            "model": self.model,
            "state": state,
            "questions": q_dict,
        }

        resp = httpx.post(
            self.endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        answers = {}
        results = data.get("results", {})
        for q in questions:
            if q.name not in results:
                continue
            res = results[q.name]
            if isinstance(q, Choice):
                choice = res.get("choice", "")
                probs = res.get("probabilities", {})
                conf = res.get("confidence", probs.get(choice, 1.0))
                answers[q.name] = ChoiceResult(choice=str(choice), confidence=float(conf), probabilities=probs)
            elif isinstance(q, Noul):
                is_pos = res.get("decision", False)
                p = res.get("probability", 1.0 if is_pos else 0.0)
                answers[q.name] = NoulResult(is_positive=bool(is_pos), probability=float(p))
            elif isinstance(q, Score) and "score" in res:
                val = float(res["score"])
                answers[q.name] = ScoreResult(score=val, normalized=score_to_norm(q, val))

        latency = float(data.get("latency_ms", 0.0))
        return DecisionResult(answers=answers, latency_ms=latency)

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)


// ============================================================================
// FILE: benchmark/competitors/kev.py (85 code lines, 94 total)
// ============================================================================

from typing import Optional, Sequence
import httpx
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from inference.questions import (
    Choice, ChoiceResult, DecisionResult, DecisionState,
    Noul, NoulResult, Question, Score, ScoreResult, score_to_norm,
)

class KevEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps Jared Palmer's Kev decision server (/v1/systemone)."""

    def __init__(
        self,
        endpoint: str = "",
        timeout: float = 30.0,
        device: Optional[str] = None,
        model: str = "kev-4b",
        api_key: str = "lm-studio",
    ):
        import os
        self.model = model
        self.api_key = api_key or os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
        self.endpoint = endpoint or os.environ.get("KEV_ENDPOINT", "")
        self.timeout = timeout
        self.candidate_endpoints = [self.endpoint] if self.endpoint else [
            "http://127.0.0.1:1234/v1/systemone",
            "http://127.0.0.1:8008/v1/systemone",
        ]

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        q_dict = {}
        for q in questions:
            if isinstance(q, Choice):
                q_dict[q.name] = {
                    "type": "choice",
                    "instructions": q.instruction or "",
                    "criteria": q.options if isinstance(q.options, dict) else {o: o for o in q.options},
                }
            elif isinstance(q, Noul):
                q_dict[q.name] = {"type": "noul", "instructions": q.instruction or ""}
            elif isinstance(q, Score):
                q_dict[q.name] = {
                    "type": "score",
                    "instructions": q.instruction or "",
                    "criteria": [str(q.min_value), str(q.max_value)],
                }

        payload = {"model": self.model, "state": str(state), "questions": q_dict}
        raw, last_err = None, None
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        for ep in self.candidate_endpoints:
            try:
                resp = httpx.post(ep, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                raw = resp.json()
                self.candidate_endpoints = [ep]
                break
            except Exception as e:
                last_err = e
                continue

        if raw is None:
            raise ConnectionError(
                f"Cannot connect to Kev decision server (tried {self.candidate_endpoints}): {last_err}. "
                "Ensure LM Studio local server (port 1234) or 'llama serve' (port 8008) is running."
            )

        answers = {}
        choices = raw.get("choices", raw.get("answers", {}))
        nouls = raw.get("nouls", {})
        scores = raw.get("scores", {})

        for q in questions:
            if isinstance(q, Choice) and q.name in choices:
                c_data = choices[q.name]
                choice_str = str(c_data.get("choice", ""))
                conf = float(c_data.get("confidence", 1.0))
                answers[q.name] = ChoiceResult(choice=choice_str, confidence=conf, probabilities=c_data.get("probabilities", {choice_str: conf}))
            elif isinstance(q, Noul) and (q.name in nouls or q.name in choices):
                n_data = nouls.get(q.name, choices.get(q.name, {}))
                val = n_data.get("noul", n_data.get("decision"))
                if val is not None:
                    p = float(n_data.get("probability", 1.0 if val else 0.0))
                    answers[q.name] = NoulResult(is_positive=bool(val), probability=p)
            elif isinstance(q, Score) and (q.name in scores or q.name in choices):
                s_data = scores.get(q.name, choices.get(q.name, {}))
                if "score" in s_data:
                    score_val = float(s_data["score"])
                    answers[q.name] = ScoreResult(score=score_val, normalized=score_to_norm(q, score_val))

        return DecisionResult(answers=answers)

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)


// ============================================================================
// FILE: benchmark/competitors/laya.py (55 code lines, 62 total)
// ============================================================================

from typing import Optional, Sequence
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from inference.questions import (
    Choice, ChoiceResult, DecisionResult, DecisionState,
    Noul, NoulResult, Question, Score, ScoreResult, score_to_norm,
)
from model.device import resolve_device

class LayaEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps Convai Innovations' Laya decision router."""

    def __init__(
        self,
        model_variant: str = "convaiinnovations/laya",
        device: Optional[str] = None,
    ):
        self.model_variant = model_variant
        self.device = resolve_device(device)
        self._agent = None

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        if self._agent is None:
            try:
                from laya import Agent
                self._agent = Agent(self.model_variant, device=self.device)
            except ImportError:
                raise ImportError("Laya is not installed. To run live: pip install laya")

        q_dict = {}
        for q in questions:
            instr = q.instruction.strip() if q.instruction and q.instruction.strip() else f"Select the best option for {q.name}."
            if isinstance(q, Choice):
                q_dict[q.name] = {
                    "type": "choice",
                    "instructions": instr,
                    "criteria": q.options if isinstance(q.options, dict) else {o: o for o in q.options},
                }
            elif isinstance(q, Noul):
                q_dict[q.name] = {"type": "noul", "instructions": instr}
            elif isinstance(q, Score):
                q_dict[q.name] = {"type": "score", "instructions": instr, "criteria": [str(q.min_value), str(q.max_value)]}

        raw = self._agent.system_one(str(state), q_dict)
        raw_answers = raw.get("answers", {})
        answers = {}
        for q in questions:
            if q.name not in raw_answers:
                continue
            ans_raw = raw_answers[q.name]
            if isinstance(q, Choice):
                answers[q.name] = ChoiceResult(choice=str(ans_raw.get("choice", "")), confidence=float(ans_raw.get("confidence", 1.0)), probabilities=ans_raw.get("probabilities", {}))
            elif isinstance(q, Noul):
                p = float(ans_raw.get("noul", ans_raw.get("probability", 0.0)))
                answers[q.name] = NoulResult(is_positive=p >= 0.5, probability=p)
            elif isinstance(q, Score) and "score" in ans_raw:
                s = float(ans_raw["score"])
                answers[q.name] = ScoreResult(score=s, normalized=score_to_norm(q, s))

        return DecisionResult(answers=answers)

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)


// ============================================================================
// FILE: benchmark/competitors/verdict.py (55 code lines, 60 total)
// ============================================================================

from typing import Optional, Sequence
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer
from inference.questions import (
    Choice, ChoiceResult, DecisionResult, DecisionState,
    Noul, NoulResult, Question, Score, ScoreResult,
)
from model.device import resolve_device

class VerdictEngine(IDecisionEngine):
    """Sovereign competitor adapter: wraps Verdict 2.0 (heman10x/rlcd-modernbert-151m)."""

    def __init__(
        self,
        model_id: str = "heman10x/rlcd-modernbert-151m",
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = resolve_device(device)
        self._pipeline = None

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        if self._pipeline is None:
            try:
                from gliclass import GLiClassModel, ZeroShotClassificationPipeline
                from transformers import AutoTokenizer
                model = GLiClassModel.from_pretrained(self.model_id)
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._pipeline = ZeroShotClassificationPipeline(
                    model, tokenizer, classification_type="single_label", device=self.device, progress_bar=False
                )
            except Exception as e:
                raise ImportError(f"Cannot load Verdict ({self.model_id}): {e}. To run live: pip install gliclass transformers")

        pipe = self._pipeline
        answers = {}
        for q in questions:
            prompt_text = f"Question: {q.instruction}\n\nContext:\n{state}" if q.instruction else str(state)
            if isinstance(q, Choice):
                opt_map = {f"It is {desc or k}": k for k, desc in q.options.items()} if isinstance(q.options, dict) else {f"It is {o}": o for o in q.options}
                res = pipe(prompt_text, list(opt_map.keys()))
                items = res[0] if (res and isinstance(res, list) and isinstance(res[0], list)) else (res or [])
                cand_scores = {opt_map[it["label"]]: float(it.get("score", 0.0)) for it in items if it.get("label") in opt_map}
                if not cand_scores: continue
                total = sum(cand_scores.values()) or 1.0
                norm_p = {k: v / total for k, v in cand_scores.items()}
                top_k = opt_map.get(items[0].get("label", ""))
                answers[q.name] = ChoiceResult(choice=str(top_k), confidence=norm_p.get(top_k, 1.0), probabilities=norm_p)
            elif isinstance(q, Noul):
                res = pipe(prompt_text, ["It is true", "It is false"])
                items = res[0] if (res and isinstance(res, list) and isinstance(res[0], list)) else (res or [])
                s_map = {it["label"]: float(it.get("score", 0.0)) for it in items if "label" in it}
                if not s_map: continue
                prob = s_map.get("It is true", 0.0) / ((s_map.get("It is true", 0.0) + s_map.get("It is false", 0.0)) or 1.0)
                answers[q.name] = NoulResult(is_positive=prob >= 0.5, probability=prob)
            elif isinstance(q, Score):
                raise NotImplementedError(f"Verdict does not support continuous regression questions ({type(q).__name__})")
        return DecisionResult(answers=answers)

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)


// ============================================================================
// FILE: benchmark/policies/ledger.py (49 code lines, 59 total)
// ============================================================================

import orjson
from pathlib import Path
from typing import Mapping, Optional, Protocol, Union, runtime_checkable
from benchmark.policies.scorer import BenchmarkReport

DEFAULT_SCORES_FILE = Path(__file__).resolve().parent.parent / "scores.json"

SuiteEntry = Mapping[str, Union[float, int, str]]
SuiteScores = Mapping[str, SuiteEntry]
LedgerScores = Mapping[str, SuiteScores]

@runtime_checkable
class ILedgerPolicy(Protocol):
    """Sovereign opinion sub-primitive: manages storage, retrieval, and caching of benchmark reports."""
    def get(self, model: str, suite: str = "default", expected_count: Optional[int] = None, expected_hash: Optional[str] = None) -> Optional[Mapping[str, float]]: ...
    def put(self, report: BenchmarkReport, manifest_hash: Optional[str] = None) -> None: ...
    def all(self, suite: Optional[str] = None) -> Mapping[str, dict]: ...

class DiskLedgerPolicy(ILedgerPolicy):
    """Local JSON filesystem ledger policy with per-suite caching."""

    def __init__(self, scores_path: Optional[Union[str, Path]] = None):
        self.scores_path = Path(scores_path or DEFAULT_SCORES_FILE)

    def get(self, model: str, suite: str = "default", expected_count: Optional[int] = None, expected_hash: Optional[str] = None) -> Optional[Mapping[str, float]]:
        suite_data = self.all(suite)
        if expected_count is not None and suite_data.get("count") != expected_count:
            return None
        if expected_hash is not None and suite_data.get("manifest_hash") != expected_hash:
            return None
        scores = {t: entry[model] for t, entry in suite_data.items() if isinstance(entry, dict) and model in entry}
        return scores if scores else None

    def put(self, report: BenchmarkReport, manifest_hash: Optional[str] = None) -> None:
        data = self.all()
        suite_data = data.setdefault(report.suite or "default", {})
        suite_data["count"] = report.count
        if manifest_hash is not None:
            suite_data["manifest_hash"] = manifest_hash

        for cat, acc in report.slices.items():
            entry = suite_data.setdefault(cat, {})
            entry["count"], entry[report.model] = report.counts.get(cat, 0), round(acc, 4)
        overall = suite_data.setdefault("OVERALL", {})
        overall["count"], overall[report.model] = report.count, round(report.overall, 4)
        lat = suite_data.setdefault("LATENCY_P50_MS", {})
        lat["count"], lat[report.model] = "-", round(report.p50_ms, 2)
        if report.brier_score > 0:
            brier_entry = suite_data.setdefault("BRIER_SCORE", {})
            brier_entry["count"], brier_entry[report.model] = "-", round(report.brier_score, 4)
        if report.regression_mse > 0:
            reg_entry = suite_data.setdefault("REGRESSION_MSE", {})
            reg_entry["count"], reg_entry[report.model] = "-", round(report.regression_mse, 4)

        self.scores_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def all(self, suite: Optional[str] = None) -> Mapping[str, dict]:
        data = orjson.loads(self.scores_path.read_bytes()) if self.scores_path.exists() and self.scores_path.stat().st_size > 0 else {}
        return data.get(suite, {}) if suite is not None else data


// ============================================================================
// FILE: benchmark/policies/reporter.py (77 code lines, 91 total)
// ============================================================================

from typing import Iterable, List, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable
from rich import box
from rich.console import Console
from rich.table import Table

SuiteData = Mapping[str, Mapping[str, Union[float, int, str]]]
LedgerData = Mapping[str, SuiteData]

@runtime_checkable
class IReporterPolicy(Protocol):
    """Sovereign opinion sub-primitive: presents benchmark suite and summary leaderboards."""
    def suite(self, data: SuiteData, suite_name: str) -> None: ...
    def summary(self, ledger: LedgerData, suites: Optional[Sequence[str]] = None) -> None: ...

def order_models(models: Iterable[str]) -> List[str]:
    """Pure domain transform: sorts model identifiers with Piqo pinned first."""
    return sorted(models, key=lambda m: (m != "Piqo", m))

class ConsoleReporterPolicy(IReporterPolicy):
    """Rich console presentation policy for formatted benchmark tables and playground."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def suite(self, data: SuiteData, suite_name: str) -> None:
        if not data:
            return
        tracks_data = {k: v for k, v in data.items() if isinstance(v, dict) and k != "count"}
        models = order_models({m for entry in tracks_data.values() for m in entry if m != "count"})
        if not models:
            return

        summary_keys = ("OVERALL", "BRIER_SCORE", "REGRESSION_MSE", "LATENCY_P50_MS")
        tracks = [t for t in sorted(tracks_data) if t not in summary_keys] + [t for t in summary_keys if t in tracks_data]

        table = Table(
            title=f"Benchmark Suite: {suite_name.upper()}",
            box=box.HORIZONTALS,
            header_style="bold magenta",
            title_style="bold white",
        )
        table.add_column("Track / Metric", style="cyan", no_wrap=True)
        table.add_column("Count", justify="right", style="dim")
        for m in models:
            table.add_column(m, justify="right", style="bold green" if m == "Piqo" else "white")

        for track in tracks:
            entry = tracks_data.get(track, {})
            cnt = str(entry.get("count", "-"))
            if track == "LATENCY_P50_MS":
                cols = [f"{entry[m]:.2f} ms" if m in entry else "-" for m in models]
            elif track in ("BRIER_SCORE", "REGRESSION_MSE"):
                cols = [f"{entry[m]:.4f}" if m in entry else "-" for m in models]
            else:
                cols = [f"{entry[m]*100:.1f}%" if m in entry else "-" for m in models]
            is_summary = track in summary_keys
            table.add_row(f"[bold]{track}[/bold]" if is_summary else track, cnt, *cols, end_section=is_summary)

        self.console.print("\n", table, "\n")

    def summary(self, ledger: LedgerData, suites: Optional[Sequence[str]] = None) -> None:
        if not ledger:
            return
        active_suites = [s for s in suites if s in ledger] if suites else [s for s, d in ledger.items() if isinstance(d, dict)]
        if not active_suites:
            return
        all_models = {
            m for s_name in active_suites if isinstance(s_data := ledger.get(s_name), dict)
            for entry in s_data.values() if isinstance(entry, dict)
            for m in entry if m != "count"
        }
        if not all_models:
            return

        sorted_models = order_models(all_models)
        table = Table(
            title="EXECUTIVE BENCHMARK SUMMARY (OVERALL ACCURACY)",
            box=box.DOUBLE_EDGE,
            header_style="bold cyan",
            title_style="bold yellow",
        )
        table.add_column("Benchmark Suite", style="bold white")
        for m in sorted_models:
            table.add_column(m, justify="right", style="bold green" if m == "Piqo" else "white")

        for s_name in sorted(active_suites):
            overall = ledger[s_name].get("OVERALL", {})
            row = [f"{score*100:.1f}%" if isinstance(score := overall.get(m), (int, float)) else "-" for m in sorted_models]
            table.add_row(s_name, *row)

        self.console.print("\n", table, "\n")


// ============================================================================
// FILE: benchmark/policies/scorer.py (102 code lines, 112 total)
// ============================================================================

from dataclasses import dataclass, field
import statistics
import time
from typing import Dict, Mapping, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase
from inference.decision_engine import IDecisionEngine
from inference.questions import Choice, Noul, Score, choice_keys, matches_answer, score_to_norm

@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Pure domain DTO: verified evaluation report for a decision engine on a benchmark suite."""
    model: str
    overall: float
    count: int
    p50_ms: float = 0.0
    brier_score: float = 0.0
    regression_mse: float = 0.0
    slices: Mapping[str, float] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)
    suite: str = "default"
    omissions: int = 0

@runtime_checkable
class IScorerPolicy(Protocol):
    """Opinion sub-primitive: scores decisions against ground truth to produce a report."""
    def score(
        self,
        engine: IDecisionEngine,
        cases: Sequence[DecisionCase],
        model_name: str = "Piqo",
        suite: str = "default",
    ) -> BenchmarkReport: ...

class ScorerPolicy(IScorerPolicy):
    """Concrete evaluation policy: runs live engine inference, tracks metrics, and computes scores."""

    def __init__(self, show_progress: bool = True, log_interval_sec: float = 3.0):
        self.show_progress = show_progress
        self.log_interval_sec = log_interval_sec

    def score(
        self,
        engine: IDecisionEngine,
        cases: Sequence[DecisionCase],
        model_name: str = "Piqo",
        suite: str = "default",
    ) -> BenchmarkReport:
        cat_correct: Dict[str, int] = {}
        cat_total: Dict[str, int] = {}
        latencies: list[float] = []
        hits: list[float] = []
        brier_diffs: list[float] = []
        regression_diffs: list[float] = []
        omissions = 0
        total_items = len(cases)
        start_time = time.perf_counter()
        last_log = start_time

        for idx, item in enumerate(cases, 1):
            cat = item.category or "general"
            t_call = time.perf_counter()
            res = engine.decide(item.state, item.questions)
            dt_ms = res.latency_ms if res.latency_ms > 0 else (time.perf_counter() - t_call) * 1000.0
            latencies.append(dt_ms)

            for q in item.questions:
                if q.target is not None:
                    cat_total[cat] = cat_total.get(cat, 0) + 1
                    if q.name in res.answers:
                        ans = res.answers[q.name]
                        is_hit = matches_answer(q, ans)
                        hits.append(1.0 if is_hit else 0.0)
                        if is_hit:
                            cat_correct[cat] = cat_correct.get(cat, 0) + 1

                        if isinstance(q, Choice):
                            probs = getattr(ans, "probabilities", None)
                            keys = choice_keys(q)
                            b_val = sum((float(probs.get(k, 0.0)) - (1.0 if str(k) == str(q.target) else 0.0)) ** 2 for k in keys) if (probs and len(probs) >= len(keys)) else 1.0
                            brier_diffs.append(b_val)
                        elif isinstance(q, Noul):
                            prob = getattr(ans, "probability", None)
                            tgt_b = 1.0 if (q.target is True or str(q.target).lower() in ("true", "1", "yes")) else 0.0
                            brier_diffs.append((float(prob) - tgt_b) ** 2 if prob is not None else 1.0)
                        elif isinstance(q, Score):
                            norm = getattr(ans, "normalized", None)
                            regression_diffs.append((float(norm) - score_to_norm(q, float(q.target))) ** 2 if norm is not None else 1.0)
                    else:
                        hits.append(0.0)
                        omissions += 1
                        (regression_diffs if isinstance(q, Score) else brier_diffs).append(1.0)

            now = time.perf_counter()
            if self.show_progress and (now - last_log >= self.log_interval_sec or idx == total_items):
                p50 = float(statistics.median(latencies)) if latencies else 0.0
                acc = (sum(hits) / max(1, len(hits))) * 100.0
                speed = idx / max(0.001, now - start_time)
                print(f"[{model_name}] {idx}/{total_items} ({idx*100.0/total_items:.1f}%) | Acc: {acc:.1f}% | P50: {p50:.1f}ms | {speed:.1f} it/s", flush=True)
                last_log = now

        return BenchmarkReport(
            model=model_name,
            overall=sum(hits) / max(1, len(hits)),
            count=len(hits),
            p50_ms=float(statistics.median(latencies)) if latencies else 0.0,
            brier_score=float(sum(brier_diffs) / max(1, len(brier_diffs))) if brier_diffs else 0.0,
            regression_mse=float(sum(regression_diffs) / max(1, len(regression_diffs))) if regression_diffs else 0.0,
            slices={c: cat_correct.get(c, 0) / cat_total[c] for c in cat_total},
            counts=cat_total,
            suite=suite,
            omissions=omissions,
        )


// ============================================================================
// FILE: benchmark/registry.py (25 code lines, 28 total)
// ============================================================================

"""Benchmark registry for available decision engines and published suites (PFD)."""

from typing import Type
from benchmark.competitors.gliner_decide import GlinerDecideEngine
from benchmark.competitors.jev import JevEngine
from benchmark.competitors.kev import KevEngine
from benchmark.competitors.laya import LayaEngine
from benchmark.competitors.verdict import VerdictEngine
from benchmark.suites.jevbench import JevBenchSuite
from benchmark.suites.kev_transfer import KevTransferSuite
from benchmark.suites.laya_typed import LayaTypedDecisionsSuite
from benchmark.suites.suite import IBenchmarkSuite
from inference.decision_engine import DecisionEngine, IDecisionEngine

AVAILABLE_ENGINES: dict[str, Type[IDecisionEngine]] = {
    "piqo": DecisionEngine,
    "gliner": GlinerDecideEngine,
    "jev": JevEngine,
    "kev": KevEngine,
    "laya": LayaEngine,
    "verdict": VerdictEngine,
}

BENCHMARK_SUITES: dict[str, Type[IBenchmarkSuite]] = {
    "laya_typed_decisions": LayaTypedDecisionsSuite,
    "jevbench_231": JevBenchSuite,
    "kev_transfer": KevTransferSuite,
}


// ============================================================================
// FILE: benchmark/suites/jevbench.py (25 code lines, 31 total)
// ============================================================================

from pathlib import Path
from typing import Mapping, Sequence
from data.case import DecisionCase, case_from_dict
from benchmark.suites.suite import IBenchmarkSuite, load_verified_manifest

BENCHMARK_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"

class JevBenchSuite(IBenchmarkSuite):
    """TypeSafe AI / JevBench official 231 public evaluation tasks (Easy 48, Standard 72, Hard 111)."""

    @property
    def name(self) -> str:
        return "jevbench_231"

    @property
    def manifest_hash(self) -> str:
        return "d2677f4d6f1e2063657183a36f4dbd92c0383136c050084701e2ee3d6ccc49a6"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "Jev-1.13": 0.845,
            "SemIf (Qwen3.5)": 0.732,
            "Verdict v1.4": 0.646,
            "Laya (Zero-Shot)": 0.584,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        import json
        content = load_verified_manifest(BENCHMARK_MANIFESTS_DIR / "jevbench_231.jsonl", "d2677f4d6f1e2063657183a36f4dbd92c0383136c050084701e2ee3d6ccc49a6")
        return [case_from_dict(json.loads(line)) for line in content.decode("utf-8").splitlines() if line.strip()]


// ============================================================================
// FILE: benchmark/suites/kev_transfer.py (25 code lines, 31 total)
// ============================================================================

from pathlib import Path
from typing import Mapping, Sequence
from data.case import DecisionCase, case_from_dict
from benchmark.suites.suite import IBenchmarkSuite, load_verified_manifest

BENCHMARK_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"

class KevTransferSuite(IBenchmarkSuite):
    """Jared Palmer's official Kev out-of-domain transfer evaluation suite (SciQ, ANLI, Emotion, ARC)."""

    @property
    def name(self) -> str:
        return "kev_transfer"

    @property
    def manifest_hash(self) -> str:
        return "c30a91274f9b483aac9e4f02ada5dea953b3f1a2829456e0bc07806c4e73b517"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "Kev-4B": 0.720,
            "Verdict-151M": 0.428,
            "GLiNER2.5-Decide": 0.404,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        import json
        url = "https://raw.githubusercontent.com/jaredpalmer/kev/main/evals/v4/transfer-v4/test.jsonl"
        content = load_verified_manifest(BENCHMARK_MANIFESTS_DIR / "kev_transfer_v4.jsonl", "c30a91274f9b483aac9e4f02ada5dea953b3f1a2829456e0bc07806c4e73b517", url)
        return [case_from_dict(json.loads(line)) for line in content.decode("utf-8").splitlines() if line.strip()]


// ============================================================================
// FILE: benchmark/suites/laya_typed.py (27 code lines, 33 total)
// ============================================================================

from pathlib import Path
from typing import Mapping, Sequence
from data.case import DecisionCase, case_from_dict
from benchmark.suites.suite import IBenchmarkSuite, load_verified_manifest

BENCHMARK_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"

class LayaTypedDecisionsSuite(IBenchmarkSuite):
    """Official LocalLLaMA/typed-decisions 400-case test benchmark (Laya, Jev, Verdict)."""

    @property
    def name(self) -> str:
        return "laya_typed_decisions"

    @property
    def manifest_hash(self) -> str:
        return "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "Laya (Fine-Tuned)": 0.767,
            "Jev 1.13.0": 0.727,
            "Verdict-151M": 0.261,
            "Laya (Zero-Shot)": 0.362,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        import pyarrow.parquet as pq
        path = BENCHMARK_MANIFESTS_DIR / "typed_decisions_test.parquet"
        url = "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/all/test-00000-of-00001.parquet"
        load_verified_manifest(path, "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c", url)
        return [case_from_dict(r) for r in pq.read_table(path).to_pylist()]


// ============================================================================
// FILE: benchmark/suites/suite.py (23 code lines, 28 total)
// ============================================================================

from typing import Mapping, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase

@runtime_checkable
class IBenchmarkSuite(Protocol):
    """Sovereign benchmark suite contract: provides evaluation cases, identity, and published baselines."""
    @property
    def name(self) -> str: ...

    @property
    def manifest_hash(self) -> str: ...

    @property
    def published_baselines(self) -> Mapping[str, float]: ...

    def load_cases(self) -> Sequence[DecisionCase]: ...
 
def load_verified_manifest(path, expected_sha: str, url: str | None = None) -> bytes:
    import hashlib, urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        if not url: raise ValueError(f"Integrity check failed for '{path.name}': SHA256 mismatch and no source URL.")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp: path.write_bytes(resp.read())
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise ValueError(f"Integrity check failed for '{path.name}': SHA256 mismatch")
    return content


// ============================================================================
// FILE: data/__main__.py (71 code lines, 81 total)
// ============================================================================

from collections import Counter
from typing import Optional
from rich import box
from rich.console import Console
from rich.table import Table
import typer

from data.dataset import Dataset
from data.curation.budget_split import BudgetSplitPolicy
from data.curation.ngram_leakage import NGramLeakagePolicy

app = typer.Typer(add_completion=False, help="Piqo Dataset Corpus Builder CLI")

@app.command()
def main(
    seed: int = typer.Option(42, help="Random seed"),
    train_quota: int = typer.Option(40000, help="Train quota"),
    train_quota_per_category: Optional[int] = typer.Option(1500, help="Train quota per category"),
    val_quota: int = typer.Option(2500, help="Val quota"),
    test_quota: Optional[int] = typer.Option(None, help="Test quota"),
    n_gram: int = typer.Option(10, help="N-gram leakage window"),
) -> None:
    console = Console()
    dataset = Dataset(
        split_policy=BudgetSplitPolicy(
            train_quota=train_quota,
            train_quota_per_category=train_quota_per_category,
            val_quota=val_quota,
            test_quota=test_quota,
            seed=seed,
        ),
        leakage_policy=NGramLeakagePolicy(n_gram=n_gram),
    )
    console.print(f"\n[bold cyan]Piqo Data Pipeline[/bold cyan]\n")

    src_table = Table(title="Registered Sources", box=box.HORIZONTALS, title_style="bold magenta")
    src_table.add_column("Source Name", style="cyan")
    src_table.add_column("Category", style="yellow")
    src_table.add_column("Read Count", justify="right", style="green")

    source_counts = []
    total_src = len(dataset.sources)
    for i, src in enumerate(dataset.sources, 1):
        console.print(f"[{i:02d}/{total_src:02d}] Ingesting {src.name}...", end="\r")
        recs = src.read()
        cat = recs[0].sample.category if recs else "unknown"
        source_counts.append((src.name, cat, len(recs)))
        src_table.add_row(src.name, cat, f"{len(recs):,}")

    console.print(src_table)
    console.print(f"Total Ingested Records: [bold green]{sum(c[2] for c in source_counts):,}[/bold green]\n")

    console.print("[dim]Partitioning splits & auditing zero-leakage invariant...[/dim]\n")
    splits = dataset.prepare()

    summary_table = Table(title="Corpus Summary & Leakage Scorecard", box=box.HORIZONTALS, title_style="bold white")
    summary_table.add_column("Metric / Split", style="cyan")
    summary_table.add_column("Cases", justify="right", style="bold green")
    summary_table.add_column("Invariant", justify="right", style="dim")
    summary_table.add_row("Train Split", f"{len(splits.train):,}", "-")
    summary_table.add_row("Validation Split", f"{len(splits.val):,}", "-")
    summary_table.add_row("Test Split", f"{len(splits.test):,}", "-")
    summary_table.add_row("Pruned Overlaps", f"{dataset.leak_audit.get('total_unique', 0):,}", "-")
    summary_table.add_row("Test Leaks", f"{dataset.leak_audit.get('test_leaks', 0)}", "[bold green]0[/bold green]")
    summary_table.add_row("Val Leaks", f"{dataset.leak_audit.get('val_leaks', 0)}", "[bold green]0[/bold green]")
    console.print(summary_table)

    dist_table = Table(title="Train Category Distribution", box=box.HORIZONTALS, title_style="bold white")
    dist_table.add_column("Category", style="yellow")
    dist_table.add_column("Cases", justify="right", style="green")
    dist_table.add_column("Share", justify="right", style="cyan")
    train_counts = Counter(c.category or "general" for c in splits.train)
    total_train = len(splits.train)
    for cat, cnt in train_counts.most_common():
        pct = (cnt / max(1, total_train)) * 100
        dist_table.add_row(cat, f"{cnt:,}", f"{pct:.1f}%")
    console.print(dist_table)
    console.print("")

if __name__ == "__main__":
    app()


// ============================================================================
// FILE: data/case.py (62 code lines, 72 total)
// ============================================================================

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence
from inference.questions import Choice, DecisionState, Noul, Question, Score

SplitTag = Literal["train", "val", "test"]

@dataclass(frozen=True, slots=True)
class DecisionCase:
    """Core Domain DTO: Represents a situated decision problem with associated questions."""
    state: DecisionState
    questions: Sequence[Question]
    category: str = "general"

def case_gold(case: DecisionCase) -> Mapping[str, str | int | float | bool]:
    """Pure domain transform: extracts target labels from case questions."""
    return {q.name: q.target for q in case.questions if q.target is not None}

def case_to_dict(case: DecisionCase) -> dict:
    """Pure domain transform: serializes DecisionCase DTO to dictionary with exact polymorphic types."""
    q_dict = {
        q.name: ({"type": "choice", "options": q.options, "instructions": q.instruction} if isinstance(q, Choice)
                 else ({"type": "score", "min_value": q.min_value, "max_value": q.max_value, "instructions": q.instruction} if isinstance(q, Score)
                       else {"type": "noul", "instructions": q.instruction}))
        for q in case.questions
    }
    gold = {q.name: q.target for q in case.questions if q.target is not None}
    return {"state": case.state, "category": case.category, "questions": q_dict, "gold": gold}

def case_from_dict(d: dict) -> DecisionCase:
    """Pure domain transform: deserializes dictionary to DecisionCase preserving exact Question variants."""
    import json
    state = d.get("state", "")
    cat = d.get("category") or d.get("workflow", "general")
    raw_qs, raw_gold = d.get("questions", {}), d.get("gold", {}) or {}
    qs = json.loads(raw_qs) if isinstance(raw_qs, str) else (raw_qs or {})
    gold = json.loads(raw_gold) if isinstance(raw_gold, str) else (raw_gold or {})

    specs: list[Question] = []
    if isinstance(qs, dict):
        for name, spec in qs.items():
            ins = spec.get("instructions") or spec.get("instruction")
            q_type = spec.get("type")
            g_item = gold.get(name) if isinstance(gold, dict) else None
            tgt = g_item.get("label") if isinstance(g_item, dict) else (g_item if g_item is not None else spec.get("label", spec.get("target")))

            if q_type == "score" or (q_type is None and "min_value" in spec):
                min_v = float(spec.get("min_value", 0.0))
                max_v = float(spec.get("max_value", 100.0))
                specs.append(Score(name=name, min_value=min_v, max_value=max_v, instruction=ins, target=float(tgt) if tgt is not None else None))
            elif q_type == "noul" or (q_type is None and not (spec.get("options") or spec.get("criteria"))):
                specs.append(Noul(name=name, instruction=ins, target=tgt))
            else:
                crit = spec.get("options") or spec.get("criteria")
                opts = crit if isinstance(crit, dict) else {o: o for o in (crit or [])}
                if not opts:
                    raise ValueError(f"Choice question '{name}' has empty options in serialized case.")
                specs.append(Choice(name=name, options=opts, instruction=ins, target=tgt))

    return DecisionCase(state=state, questions=specs, category=cat)

@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Core Domain DTO: Ingested decision case tagged with origin partition."""
    sample: DecisionCase
    tag: SplitTag = "train"

@dataclass(frozen=True, slots=True)
class DataSplits:
    """Core Domain DTO: Consolidated partitioned dataset splits."""
    train: Sequence[DecisionCase]
    val: Sequence[DecisionCase]
    test: Sequence[DecisionCase]


// ============================================================================
// FILE: data/curation/budget_split.py (85 code lines, 101 total)
// ============================================================================

import random
from typing import Optional, Protocol, Sequence, runtime_checkable
from data.case import DataSplits, DecisionCase, SourceRecord, case_gold

@runtime_checkable
class ISplitPolicy(Protocol):
    """Sovereign policy contract: partitions source records into train, val, and test splits."""
    def partition(self, records: Sequence[SourceRecord]) -> DataSplits: ...

class BudgetSplitPolicy(ISplitPolicy):
    """Partitions source records by preserving explicit holdout tags, stratifying holdouts, and balancing pools."""
    def __init__(
        self,
        train_quota: int = 40000,
        train_quota_per_category: Optional[int] = 1500,
        val_quota: int = 2500,
        test_quota: Optional[int] = None,
        test_quota_per_category: Optional[int] = 400,
        balanced_test: bool = True,
        seed: int = 42,
    ):
        self.train_quota = train_quota
        self.train_quota_per_category = train_quota_per_category
        self.val_quota = val_quota
        self.test_quota = test_quota
        self.test_quota_per_category = test_quota_per_category
        self.balanced_test = balanced_test
        self.seed = seed

    def partition(self, records: Sequence[SourceRecord]) -> DataSplits:
        rng = random.Random(self.seed)
        splits: dict[str, list[DecisionCase]] = {"train": [], "val": [], "test": []}
        for r in records:
            if r.tag not in splits:
                raise ValueError(f"Invalid split tag '{r.tag}'. Must be 'train', 'val', or 'test'.")
            splits[r.tag].append(r.sample)
        train_recs, val_recs, test_recs = splits["train"], splits["val"], splits["test"]

        if self.test_quota_per_category is not None and test_recs:
            by_category: dict[str, list[DecisionCase]] = {}
            for c in test_recs:
                by_category.setdefault(c.category or "general", []).append(c)

            balanced_test_recs: list[DecisionCase] = []
            for cat, cat_items in by_category.items():
                if len(cat_items) <= self.test_quota_per_category:
                    balanced_test_recs.extend(cat_items)
                    continue

                if not self.balanced_test:
                    rng.shuffle(cat_items)
                    balanced_test_recs.extend(cat_items[: self.test_quota_per_category])
                    continue

                by_class: dict[str, list[DecisionCase]] = {}
                for item in cat_items:
                    gold = case_gold(item)
                    target_val = str(next(iter(gold.values()))) if gold else "default"
                    by_class.setdefault(target_val, []).append(item)

                for group in by_class.values():
                    rng.shuffle(group)

                target_per_class = max(1, self.test_quota_per_category // max(1, len(by_class)))
                per_class = min(target_per_class, min(len(g) for g in by_class.values())) if len(by_class) == 2 else target_per_class
                selected = [item for group in by_class.values() for item in group[:per_class]]
                if len(by_class) != 2 and len(selected) < self.test_quota_per_category:
                    remaining = [item for group in by_class.values() for item in group[target_per_class:]]
                    rng.shuffle(remaining)
                    selected.extend(remaining[: self.test_quota_per_category - len(selected)])
                balanced_test_recs.extend(selected)

            test_recs = balanced_test_recs

        if self.train_quota_per_category is not None and train_recs:
            by_train_cat: dict[str, list[DecisionCase]] = {}
            for c in train_recs:
                by_train_cat.setdefault(c.category or "general", []).append(c)

            balanced_train: list[DecisionCase] = []
            for cat_items in by_train_cat.values():
                rng.shuffle(cat_items)
                balanced_train.extend(cat_items[: self.train_quota_per_category])
            train_recs = balanced_train

        needed_val = max(0, self.val_quota - len(val_recs))
        if needed_val > 0 and train_recs:
            take = min(needed_val, max(1, len(train_recs) // 5)) if len(train_recs) <= needed_val else needed_val
            rng.shuffle(train_recs)
            val_recs.extend(train_recs[:take])
            train_recs = train_recs[take:]

        if len(train_recs) > self.train_quota:
            rng.shuffle(train_recs)
            train_recs = train_recs[:self.train_quota]

        if self.test_quota and len(test_recs) > self.test_quota:
            rng.shuffle(test_recs)
            test_recs = test_recs[:self.test_quota]

        return DataSplits(train=train_recs, val=val_recs, test=test_recs)


// ============================================================================
// FILE: data/curation/ngram_leakage.py (103 code lines, 121 total)
// ============================================================================

import re
from typing import Protocol, Sequence, Set, Tuple, Union, runtime_checkable
from data.case import DecisionCase, case_gold
from inference.questions import DecisionState

@runtime_checkable
class ILeakagePolicy(Protocol):
    """Sovereign policy contract: audits candidate training cases against holdout evaluation suites."""
    def find_leaks(
        self,
        candidates: Sequence[DecisionCase],
        holdouts: Sequence[DecisionCase],
    ) -> Sequence[DecisionCase]: ...

def tokenize_state(st: Union[DecisionState, str, dict]) -> Tuple[str, ...]:
    """Pure domain transform: decomposes state text/mapping into lowercase lexical tokens."""
    txt = " ".join(f"{k} {v}" for k, v in st.items()) if isinstance(st, dict) else str(st or "")
    return tuple(re.findall(r"\w+", txt.lower()))

class NGramLeakagePolicy(ILeakagePolicy):
    """Input-state and entity-level leakage policy: exact prompt duplicates, n-gram shingling, token Jaccard similarity, and isolated entity holdouts."""

    def __init__(
        self,
        n_gram: int = 10,
        min_short_tokens: int = 4,
        jaccard_threshold: float = 0.70,
        disjoint_categories: Sequence[str] = ("function_calling",),
        exempt_categories: Sequence[str] = ("typed_decisions",),
    ):
        self.n_gram = n_gram
        self.min_short_tokens = min_short_tokens
        self.jaccard_threshold = jaccard_threshold
        self.disjoint_categories = tuple(disjoint_categories)
        self.exempt_categories = tuple(exempt_categories)

    def find_leaks(
        self,
        candidates: Sequence[DecisionCase],
        holdouts: Sequence[DecisionCase],
    ) -> Sequence[DecisionCase]:
        if not candidates or not holdouts:
            return []

        exact_holdouts: Set[Tuple[str, ...]] = set()
        holdout_shingles: Set[Tuple[str, ...]] = set()
        holdout_texts: list[str] = []
        short_holdout_by_first: dict[str, list[str]] = {}
        holdout_targets_by_cat: dict[str, Set[str]] = {}
        holdouts_by_cat: dict[str, list[Tuple[Set[str], int]]] = {}

        disjoint_set = set(self.disjoint_categories)
        exempt_set = set(self.exempt_categories)

        for h in holdouts:
            toks = tokenize_state(h.state)
            cat = h.category or "general"
            if toks:
                exact_holdouts.add(toks)
                if len(toks) >= self.n_gram:
                    holdout_shingles.update(tuple(toks[i : i + self.n_gram]) for i in range(len(toks) - self.n_gram + 1))
                elif len(toks) >= self.min_short_tokens:
                    short_holdout_by_first.setdefault(toks[0], []).append(f" {' '.join(toks)} ")
                holdout_texts.append(" ".join(toks))

                tok_set = set(toks)
                holdouts_by_cat.setdefault(cat, []).append((tok_set, len(tok_set)))

            if cat in disjoint_set and (h_gold := case_gold(h)):
                for tgt in h_gold.values():
                    holdout_targets_by_cat.setdefault(cat, set()).add(str(tgt))

        holdout_blob = (" " + " \n ".join(holdout_texts) + " ") if holdout_texts else ""

        leaked: list[DecisionCase] = []
        for c in candidates:
            cat = c.category or "general"
            if cat in exempt_set:
                continue

            if cat in disjoint_set and (c_gold := case_gold(c)) and cat in holdout_targets_by_cat:
                if any(str(tgt) in holdout_targets_by_cat[cat] for tgt in c_gold.values()):
                    leaked.append(c)
                    continue

            toks = tokenize_state(c.state)
            if not toks:
                continue

            if toks in exact_holdouts:
                leaked.append(c)
                continue

            cand_str = f" {' '.join(toks)} "
            if len(toks) >= self.n_gram:
                cand_shingles = {tuple(toks[i : i + self.n_gram]) for i in range(len(toks) - self.n_gram + 1)}
                if cand_shingles & holdout_shingles:
                    leaked.append(c)
                    continue
                cand_set = set(toks)
                if any(p in cand_str for tok in cand_set if tok in short_holdout_by_first for p in short_holdout_by_first[tok]):
                    leaked.append(c)
                    continue
            elif len(toks) >= self.min_short_tokens and cand_str in holdout_blob:
                leaked.append(c)
                continue

            if len(toks) < 30 and cat in holdouts_by_cat:
                cand_set = set(toks)
                cand_len = len(cand_set)
                found_fuzzy = False
                for h_set, h_len in holdouts_by_cat[cat]:
                    common = len(cand_set & h_set)
                    if common / (cand_len + h_len - common) >= self.jaccard_threshold:
                        leaked.append(c)
                        found_fuzzy = True
                        break
                if found_fuzzy:
                    continue

        return leaked


// ============================================================================
// FILE: data/dataset.py (71 code lines, 85 total)
// ============================================================================

from typing import List, Optional, Protocol, Sequence, runtime_checkable

from data.case import DataSplits, DecisionCase, SourceRecord, SplitTag
from data.curation.budget_split import BudgetSplitPolicy, ISplitPolicy
from data.curation.ngram_leakage import ILeakagePolicy, NGramLeakagePolicy
from data.registry import AVAILABLE_SOURCES
from data.sources.source import IDataSource
from data.storage_policy import DiskStoragePolicy, IStoragePolicy

@runtime_checkable
class IDataset(Protocol):
    """Sovereign primitive contract: manages dataset source preparation, leakage audit, and split access."""
    def prepare(self) -> DataSplits: ...
    def load(self, split: SplitTag = "train") -> Sequence[DecisionCase]: ...

class Dataset(IDataset):
    """Sovereign Core Primitive: manages source registration, split preparation, zero leakage, and in-memory access."""

    def __init__(
        self,
        split_policy: Optional[ISplitPolicy] = None,
        leakage_policy: Optional[ILeakagePolicy] = None,
        storage_policy: Optional[IStoragePolicy] = None,
        sources: Optional[Sequence[IDataSource]] = None,
    ):
        self.split_policy = split_policy or BudgetSplitPolicy()
        self.leakage_policy = leakage_policy or NGramLeakagePolicy()
        self.storage = storage_policy or DiskStoragePolicy()
        self.sources: List[IDataSource] = list(sources) if sources is not None else [src() for src in AVAILABLE_SOURCES.values()]
        self.leak_audit: dict[str, int] = {}
        self._splits: Optional[DataSplits] = None

    def prepare(self) -> DataSplits:
        """Extracts all registered sources, partitions splits, audits for leakage, and holds splits in memory."""
        if not self.sources:
            raise ValueError("No data sources registered for dataset preparation.")

        records: List[SourceRecord] = []
        for src in self.sources:
            records.extend(src.read())

        splits = self.split_policy.partition(records)

        # Enforce zero-leakage invariant: prune candidate training and validation samples overlapping holdouts
        test_leaks = list(self.leakage_policy.find_leaks(splits.train, splits.test))
        val_leaks = list(self.leakage_policy.find_leaks(splits.train, splits.val))
        val_test_leaks = list(self.leakage_policy.find_leaks(splits.val, splits.test))

        unique_val_test = {id(c): c for c in val_test_leaks}
        unique_train_leaks = {id(c): c for c in (test_leaks + val_leaks)}
        if unique_val_test or unique_train_leaks:
            splits = DataSplits(
                train=[c for c in splits.train if id(c) not in unique_train_leaks],
                val=[c for c in splits.val if id(c) not in unique_val_test],
                test=splits.test,
            )

        from collections import Counter
        all_unique_leaks = {**unique_train_leaks, **unique_val_test}
        self.leak_audit = {
            "total_unique": len(all_unique_leaks),
            "test_leaks": len(test_leaks),
            "val_leaks": len(val_leaks),
            "val_test_leaks": len(val_test_leaks),
            "by_category": dict(Counter(c.category or "unknown" for c in all_unique_leaks.values())),
        }

        if self.leakage_policy.find_leaks(splits.train, splits.test):
            raise ValueError("Contamination invariant violated: train split leaks test holdout.")
        if self.leakage_policy.find_leaks(splits.train, splits.val):
            raise ValueError("Contamination invariant violated: train split leaks validation.")
        if self.leakage_policy.find_leaks(splits.val, splits.test):
            raise ValueError("Contamination invariant violated: val split leaks test holdout.")

        self._splits = splits
        self.storage.save(splits)
        return splits

    def load(self, split: SplitTag = "train") -> Sequence[DecisionCase]:
        """Loads domain decision cases from in-memory splits or persistent storage policy."""
        if self._splits is not None:
            return getattr(self._splits, split)
        return self.storage.load(split)

__all__ = ["IDataset", "Dataset"]


// ============================================================================
// FILE: data/registry.py (27 code lines, 30 total)
// ============================================================================

"""Data source registry for available decision and reasoning sources (PFD)."""

from data.sources.source import IDataSource
from data.sources.anli_source import ANLISource
from data.sources.art_source import ARTSource
from data.sources.copa_source import COPASource
from data.sources.decisions_source import DecisionsSource
from data.sources.glaive_source import GlaiveSource
from data.sources.hellaswag_source import HellaSwagSource
from data.sources.huggingface_source import HuggingFaceSource, STANDARD_BENCHMARKS
from data.sources.medqa_source import MedQASource
from data.sources.multichoice_source import MultiChoiceReasoningSource, MULTICHOICE_REASONING_BENCHMARKS
from data.sources.sciq_source import SciQSource
from data.sources.reclor_source import ReClorSource

AVAILABLE_SOURCES = {
    "glaive": GlaiveSource,
    "decisions": DecisionsSource,
    "anli": ANLISource,
    "hellaswag": HellaSwagSource,
    "medqa": MedQASource,
    "art": ARTSource,
    "copa": COPASource,
    "sciq": SciQSource,
    "reclor": ReClorSource,
    **{s.category: (lambda spec=s: HuggingFaceSource(spec)) for s in STANDARD_BENCHMARKS},
    **{s.category: (lambda spec=s: MultiChoiceReasoningSource(spec)) for s in MULTICHOICE_REASONING_BENCHMARKS},
}

__all__ = ["AVAILABLE_SOURCES"]


// ============================================================================
// FILE: data/sources/anli_source.py (47 code lines, 55 total)
// ============================================================================

from typing import List, Sequence

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import load_raw_jsonl

NLI_OPTIONS = ["Entailment", "Neutral", "Contradiction"]
NLI_INSTRUCTION = "What is the logical relationship between the context and the statement?"

class ANLISource(IDataSource):
    """Adversarial NLI (ANLI) dataset source: extracts human-crafted counterfactuals and refutations."""
    name = "facebook_anli"

    def __init__(
        self,
        rounds: Sequence[str] = ("r1", "r2"),
        train_limit_per_round: int = 1500,
        test_limit_per_round: int = 200,
    ):
        self.rounds = list(rounds)
        self.train_limit = train_limit_per_round
        self.test_limit = test_limit_per_round

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for r_name in self.rounds:
            for split_type, tag in [("train", "train"), ("test", "test")]:
                limit = self.test_limit if split_type == "test" else self.train_limit
                hf_split = f"{split_type}_{r_name}"
                rows = load_raw_jsonl(f"anli_{hf_split}", "facebook/anli", split=hf_split, limit=limit)
                for r in rows:
                    premise = str(r.get("premise", "")).strip()
                    hypothesis = str(r.get("hypothesis", "")).strip()
                    label = r.get("label")

                    if not premise or not hypothesis or label not in (0, 1, 2):
                        continue

                    state = f"Context: {premise}\n\nStatement: {hypothesis}"
                    case = DecisionCase(
                        state=state,
                        questions=[
                            Choice(
                                name="relation",
                                options=list(NLI_OPTIONS),
                                instruction=NLI_INSTRUCTION,
                                target=NLI_OPTIONS[int(label)],
                            )
                        ],
                        category="adversarial_reasoning",
                    )
                    records.append(SourceRecord(sample=case, tag=tag))

        return records


// ============================================================================
// FILE: data/sources/art_source.py (50 code lines, 62 total)
// ============================================================================

from typing import List, Sequence

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import load_raw_jsonl

ART_INSTRUCTION = "Which hypothesis best explains what happened between the observations?"

class ARTSource(IDataSource):
    """Abductive Reasoning in Natural Language (aNLI / ART) Source: deduces latent unstated hypotheses."""
    name = "allenai_art"

    def __init__(self, train_limit: int = 2000, test_limit: int = 300):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name, tag in [("train", "train"), ("validation", "test")]:
            limit = self.test_limit if split_name == "validation" else self.train_limit
            rows = load_raw_jsonl(f"art_{split_name}", "allenai/art", split=split_name, limit=limit)
            for r in rows:
                obs1 = str(r.get("observation_1", "")).strip()
                obs2 = str(r.get("observation_2", "")).strip()
                hyp1 = str(r.get("hypothesis_1", "")).strip()
                hyp2 = str(r.get("hypothesis_2", "")).strip()
                label_raw = r.get("label")

                if not obs1 or not obs2 or not hyp1 or not hyp2 or label_raw is None:
                    continue

                try:
                    raw_idx = int(label_raw)
                    target_idx = (raw_idx - 1) if raw_idx in (1, 2) else raw_idx
                except ValueError:
                    continue

                options = [hyp1, hyp2]
                if target_idx < 0 or target_idx >= len(options):
                    continue

                state = f"Observation 1: {obs1}\nObservation 2: {obs2}"
                target_str = options[target_idx]

                case = DecisionCase(
                    state=state,
                    questions=[
                        Choice(
                            name="explanation",
                            options=options,
                            instruction=ART_INSTRUCTION,
                            target=target_str,
                        )
                    ],
                    category="abductive_reasoning",
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records

__all__ = ["ARTSource", "ART_INSTRUCTION"]


// ============================================================================
// FILE: data/sources/copa_source.py (48 code lines, 59 total)
// ============================================================================

from typing import List, Sequence

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import load_raw_jsonl

class COPASource(IDataSource):
    """Choice of Plausible Alternatives (COPA) Source: causal directionality and root cause deduction."""
    name = "superglue_copa"

    def __init__(self, train_limit: int = 500, test_limit: int = 200):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name, tag in [("train", "train"), ("validation", "test")]:
            limit = self.test_limit if split_name in ("validation", "test") else self.train_limit
            rows = load_raw_jsonl(f"copa_{split_name}", "aps/super_glue", split=split_name, subset="copa", limit=limit)
            for r in rows:
                premise = str(r.get("premise", "")).strip()
                choice1 = str(r.get("choice1", "")).strip()
                choice2 = str(r.get("choice2", "")).strip()
                q_type = str(r.get("question", "cause")).strip().lower()
                label_raw = r.get("label")

                if not premise or not choice1 or not choice2 or label_raw is None:
                    continue

                try:
                    target_idx = int(label_raw)
                except ValueError:
                    continue

                options = [choice1, choice2]
                if target_idx < 0 or target_idx >= len(options):
                    continue

                instruction = "What was the cause of the premise?" if q_type == "cause" else "What was the effect of the premise?"
                target_str = options[target_idx]

                case = DecisionCase(
                    state=premise,
                    questions=[
                        Choice(
                            name="causal_link",
                            options=options,
                            instruction=instruction,
                            target=target_str,
                        )
                    ],
                    category="causal_reasoning",
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records

__all__ = ["COPASource"]


// ============================================================================
// FILE: data/sources/decisions_source.py (16 code lines, 19 total)
// ============================================================================

from typing import List, Sequence
from data.case import SourceRecord, case_from_dict
from data.sources.source import IDataSource
from data.sources.huggingface_source import load_raw_jsonl

class DecisionsSource(IDataSource):
    """Real Typed-Decisions dataset extractor: loads multi-question typed decision benchmarks."""
    name = "typed_decisions"

    def __init__(self, repeat: int = 5):
        self.repeat = repeat

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name, tag in [("train", "train"), ("test", "test")]:
            rows = load_raw_jsonl(f"{self.name}_{split_name}", "LocalLLaMA/typed-decisions", split=split_name, subset="all")
            split_records = [SourceRecord(sample=case_from_dict(r), tag=tag) for r in rows]
            records.extend(split_records * self.repeat if tag == "train" else split_records)
        return records


// ============================================================================
// FILE: data/sources/glaive_source.py (54 code lines, 61 total)
// ============================================================================

import random
import re
from typing import List, Sequence
from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import load_raw_jsonl

class GlaiveSource(IDataSource):
    """Real Glaive function-calling extractor: parses tool calls and yields tagged source records."""
    name = "glaive_tools"

    def __init__(self, seed: int = 42, test_ratio: float = 0.1):
        self.seed = seed
        self.test_ratio = test_ratio

    def read(self) -> Sequence[SourceRecord]:
        rng = random.Random(self.seed)
        rows = load_raw_jsonl(self.name, "glaiveai/glaive-function-calling-v2", split="train")

        raw_pairs, tool_registry = [], {}
        for row in rows:
            system, chat = row.get("system", ""), row.get("chat", "")
            fn_match = re.search(r'<functioncall>\s*\{\s*"name":\s*"([^"]+)"', chat)
            user_match = re.search(r'USER:\s*(.*?)(?=\n\s*(?:ASSISTANT:|<\|endoftext\|>|$))', chat, re.DOTALL)
            if not (fn_match and user_match):
                continue
            fn_name, user_query = fn_match.group(1), user_match.group(1).strip()
            tools = dict(re.findall(r'\{\s*"name":\s*"([^"]+)",\s*"description":\s*"([^"]+)"', system))
            if fn_name not in tools:
                continue
            tool_registry.update(tools)
            raw_pairs.append((user_query, fn_name, tools))

        # Partition tools into strictly disjoint train and test subsets (Zero-Shot Tool Holdout)
        all_tools = sorted(list(tool_registry.keys()))
        rng.shuffle(all_tools)
        test_tool_count = max(1, int(len(all_tools) * self.test_ratio)) if all_tools else 0
        test_tool_set = set(all_tools[:test_tool_count])
        train_tool_set = set(all_tools[test_tool_count:])

        records: List[SourceRecord] = []
        for query, fn, sample_tools in raw_pairs:
            is_test = fn in test_tool_set
            pool = test_tool_set if is_test else train_tool_set
            candidates = {k: v for k, v in sample_tools.items() if k in pool}
            candidates[fn] = tool_registry[fn]
            if len(candidates) < 5 and len(pool) >= 5:
                distractor_names = [n for n in pool if n != fn and n not in candidates]
                sample_k = min(len(distractor_names), 5 - len(candidates))
                if sample_k > 0:
                    for d in rng.sample(distractor_names, sample_k):
                        candidates[d] = tool_registry[d]

            case = DecisionCase(
                state=query,
                questions=[Choice(name="tool", options=candidates, instruction="Select the tool to invoke:", target=fn)],
                category="function_calling",
            )
            records.append(SourceRecord(sample=case, tag="test" if is_test else "train"))
        return records


// ============================================================================
// FILE: data/sources/hellaswag_source.py (46 code lines, 58 total)
// ============================================================================

from typing import List, Sequence

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import load_raw_jsonl

HELLASWAG_INSTRUCTION = "Which ending logically completes the scenario?"

class HellaSwagSource(IDataSource):
    """Adversarially Filtered Grounding Source: extracts HellaSwag continuation decisions."""
    name = "rowan_hellaswag"

    def __init__(self, train_limit: int = 2000, test_limit: int = 300):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name, tag in [("train", "train"), ("validation", "test")]:
            limit = self.test_limit if split_name == "validation" else self.train_limit
            rows = load_raw_jsonl(f"hellaswag_{split_name}", "Rowan/hellaswag", split=split_name, limit=limit)
            for r in rows:
                state = str(r.get("ctx", "")).strip()
                endings = r.get("endings", [])
                label_raw = r.get("label")

                if not state or not endings or label_raw is None:
                    continue

                try:
                    target_idx = int(label_raw)
                except ValueError:
                    continue

                if target_idx < 0 or target_idx >= len(endings):
                    continue

                cleaned_endings = [e.lstrip(", ").strip() for e in endings]
                target_str = cleaned_endings[target_idx]

                case = DecisionCase(
                    state=state,
                    questions=[
                        Choice(
                            name="ending",
                            options=cleaned_endings,
                            instruction=HELLASWAG_INSTRUCTION,
                            target=target_str,
                        )
                    ],
                    category="adversarial_grounding",
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records

__all__ = ["HellaSwagSource", "HELLASWAG_INSTRUCTION"]


// ============================================================================
// FILE: data/sources/huggingface_source.py (106 code lines, 120 total)
// ============================================================================

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional, Sequence
from datasets import load_dataset

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"

def load_raw_jsonl(
    target_name: str,
    dataset: str,
    split: str,
    subset: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    """Loads dataset rows from local disk cache, fetching and persisting from HuggingFace if absent."""
    target = RAW_DIR / f"{target_name}.jsonl"
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
            return rows[:limit] if limit is not None else rows

    args = [dataset]
    if subset:
        args.append(subset)
    rows = list(load_dataset(*args, split=split))
    if limit is not None:
        rows = rows[:limit]
    if rows:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
    return rows

@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Core Domain DTO: Declarative specification for benchmark decision tasks."""
    dataset: str
    question: str = "decision"
    instruction: str = "Select the best option:"
    category: str = "general"
    text_field: str = "text"
    label_field: str = "label"
    label_names: Sequence[str] | None = None
    subset: str | None = None
    splits: Sequence[str] = ("train", "test")
    train_limit: int | None = None
    test_limit: int | None = 400

STANDARD_BENCHMARKS: Sequence[BenchmarkSpec] = (
    BenchmarkSpec("mteb/banking77", "intent", "What is the primary customer request?", "banking77", label_field="label_text"),
    BenchmarkSpec("fancyzhx/ag_news", "topic", "What is the topic of article?", "ag_news", label_names=("World", "Sports", "Business", "Sci/Tech"), train_limit=1000),
    BenchmarkSpec("dair-ai/emotion", "emotion", "Which emotion is most strongly expressed?", "emotion", label_names=("sadness", "joy", "love", "anger", "fear", "surprise"), train_limit=1000),
    BenchmarkSpec("Tobi-Bueck/customer-support-tickets", "queue", "Which support queue should handle this ticket?", "customer_support", text_field="body", label_field="queue", splits=("train",)),
    BenchmarkSpec("zefang-liu/phishing-email-dataset", "is_phishing", "Is this email a phishing or scam attempt?", "phishing_email", text_field="Email Text", label_field="Email Type", label_names=("Safe Email", "Phishing Email"), splits=("train",), train_limit=1000),
    BenchmarkSpec("SetFit/enron_spam", "is_spam", "Is this email unsolicited spam or bulk marketing?", "enron_spam", label_names=("ham", "spam"), label_field="label_text", train_limit=1000),
    BenchmarkSpec("lmsys/toxic-chat", "jailbreak", "Does prompt try to bypass system safety instructions?", "toxic_chat", label_names=("benign", "jailbreak"), text_field="user_input", label_field="jailbreaking", subset="toxicchat0124", train_limit=1000),
    BenchmarkSpec("SetFit/amazon_massive_intent_en-US", "intent", "What is the user intent in this utterance?", "amazon_massive", label_field="label_text", train_limit=2000),
)

class HuggingFaceSource(IDataSource):
    """Declarative benchmark dataset adapter: unifies train and test partitions under canonical shared candidate options."""

    def __init__(self, spec: Optional[BenchmarkSpec] = None):
        self.spec = spec or STANDARD_BENCHMARKS[0]
        self.name = self.spec.dataset.replace("/", "_")
        if self.spec.subset:
            self.name = f"{self.name}_{self.spec.subset}"

    def read(self) -> Sequence[SourceRecord]:
        split_rows = {
            s: load_raw_jsonl(
                f"{self.name}_{s}",
                self.spec.dataset,
                split=s,
                subset=self.spec.subset,
                limit=(self.spec.test_limit if s == "test" else self.spec.train_limit),
            )
            for s in self.spec.splits
        }

        if self.spec.label_names:
            all_options = list(self.spec.label_names)
        else:
            all_rows = [r for rows in split_rows.values() for r in rows]
            labels = {
                str(r[self.spec.label_field]).replace("_", " ").strip()
                for r in all_rows
                if self.spec.label_field in r and r[self.spec.label_field] is not None
            }
            all_options = sorted(list(labels))

        records: List[SourceRecord] = []
        for split_name, rows in split_rows.items():
            tag = "test" if split_name == "test" else "train"
            for r in rows:
                text = str(r.get(self.spec.text_field) or r.get("prompt") or r.get("body") or "").strip()
                if not text or self.spec.label_field not in r or r[self.spec.label_field] is None:
                    continue
                raw_val = r[self.spec.label_field]
                if self.spec.label_names and isinstance(raw_val, int) and 0 <= raw_val < len(self.spec.label_names):
                    label_str = self.spec.label_names[raw_val]
                else:
                    label_str = str(raw_val).replace("_", " ").strip()

                options = all_options
                if label_str not in options:
                    options = sorted(list(set(options + [label_str])))

                case = DecisionCase(
                    state=text[:2000],
                    questions=[Choice(name=self.spec.question, options=options, instruction=self.spec.instruction, target=label_str)],
                    category=self.spec.category,
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records


// ============================================================================
// FILE: data/sources/medqa_source.py (42 code lines, 52 total)
// ============================================================================

from typing import List, Sequence

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import load_raw_jsonl

MEDQA_INSTRUCTION = "What is the most likely diagnosis, mechanism, or correct next step?"

class MedQASource(IDataSource):
    """Differential Elimination Source: extracts multi-variable clinical elimination decisions."""
    name = "gbaker_medqa"

    def __init__(self, train_limit: int = 2000, test_limit: int = 300):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name in ("train", "test"):
            limit = self.test_limit if split_name == "test" else self.train_limit
            rows = load_raw_jsonl(f"medqa_{split_name}", "GBaker/MedQA-USMLE-4-options", split=split_name, limit=limit)
            tag = "test" if split_name == "test" else "train"
            for r in rows:
                state = str(r.get("question", "")).strip()
                opts_dict = r.get("options", {})
                answer_str = str(r.get("answer", "")).strip()

                if not state or not opts_dict or not answer_str:
                    continue

                options_list = list(opts_dict.values())
                if len(options_list) < 2 or answer_str not in options_list:
                    continue

                case = DecisionCase(
                    state=state,
                    questions=[
                        Choice(
                            name="diagnosis",
                            options=options_list,
                            instruction=MEDQA_INSTRUCTION,
                            target=answer_str,
                        )
                    ],
                    category="differential_elimination",
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records

__all__ = ["MedQASource", "MEDQA_INSTRUCTION"]


// ============================================================================
// FILE: data/sources/multichoice_source.py (102 code lines, 113 total)
// ============================================================================

from dataclasses import dataclass
from typing import List, Optional, Sequence
from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import load_raw_jsonl

@dataclass(frozen=True, slots=True)
class MultiChoiceSpec:
    """Core Domain DTO: Declarative specification for multi-choice reasoning datasets."""
    dataset: str
    subset: Optional[str] = None
    question_field: str = "question"
    choices_field: str = "choices"
    answer_field: str = "answerKey"
    category: str = "reasoning"
    instruction: str = "Select the best conclusion:"
    splits: Sequence[str] = ("train", "test")
    train_limit: Optional[int] = None
    test_limit: Optional[int] = 300

MULTICHOICE_REASONING_BENCHMARKS: Sequence[MultiChoiceSpec] = (
    MultiChoiceSpec(
        dataset="allenai/ai2_arc",
        subset="ARC-Challenge",
        question_field="question",
        category="deductive_reasoning",
        instruction="Which option best answers the question?",
    ),
    MultiChoiceSpec(
        dataset="allenai/openbookqa",
        subset="main",
        question_field="question_stem",
        category="causal_reasoning",
        instruction="Which conclusion logically follows?",
        train_limit=1500,
    ),
    MultiChoiceSpec(
        dataset="tau/commonsense_qa",
        question_field="question",
        category="commonsense_reasoning",
        instruction="Which option best answers the question?",
        train_limit=2000,
    ),
    MultiChoiceSpec(
        dataset="allenai/qasc",
        question_field="question",
        category="multihop_reasoning",
        instruction="Which option best completes the reasoning chain?",
        train_limit=2000,
    ),
)

class MultiChoiceReasoningSource(IDataSource):
    """Universal source adapter for multi-choice reasoning benchmarks (ARC, OpenBookQA)."""

    def __init__(self, spec: Optional[MultiChoiceSpec] = None):
        self.spec = spec or MULTICHOICE_REASONING_BENCHMARKS[0]
        self.name = self.spec.dataset.replace("/", "_")
        if self.spec.subset:
            self.name = f"{self.name}_{self.spec.subset}"

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name in self.spec.splits:
            limit = self.spec.test_limit if split_name == "test" else self.spec.train_limit
            rows = load_raw_jsonl(
                f"{self.name}_{split_name}",
                self.spec.dataset,
                split=split_name,
                subset=self.spec.subset,
                limit=limit,
            )
            tag = "test" if split_name == "test" else "train"
            for r in rows:
                state = r.get(self.spec.question_field, "").strip()
                choices_raw = r.get(self.spec.choices_field, {})
                ans_key = str(r.get(self.spec.answer_field, "")).strip()

                if not state or not choices_raw or not ans_key:
                    continue

                texts = choices_raw.get("text", [])
                labels = choices_raw.get("label", [])
                if not texts or len(texts) < 2:
                    continue

                target_idx: Optional[int] = None
                if ans_key in labels:
                    target_idx = labels.index(ans_key)
                elif ans_key.isdigit():
                    num_idx = int(ans_key) - 1
                    if 0 <= num_idx < len(texts):
                        target_idx = num_idx

                if target_idx is None or target_idx >= len(texts):
                    continue

                case = DecisionCase(
                    state=state,
                    questions=[
                        Choice(
                            name="choice",
                            options=texts,
                            instruction=self.spec.instruction,
                            target=texts[target_idx],
                        )
                    ],
                    category=self.spec.category,
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records


// ============================================================================
// FILE: data/sources/reclor_source.py (45 code lines, 55 total)
// ============================================================================

from typing import List, Sequence

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import load_raw_jsonl

class ReClorSource(IDataSource):
    """ReClor Logical Reasoning Source: LSAT multi-constraint argument analysis and fallacy elimination."""
    name = "reclor"

    def __init__(self, train_limit: int = 2000, test_limit: int = 400):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name, tag in [("train", "train"), ("validation", "test")]:
            limit = self.test_limit if tag == "test" else self.train_limit
            rows = load_raw_jsonl(f"reclor_{split_name}", "metaeval/reclor", split=split_name, limit=limit)
            for r in rows:
                context = str(r.get("context", "")).strip()
                question = str(r.get("question", "")).strip()
                answers = r.get("answers", [])
                label = r.get("label")

                if not context or not answers or len(answers) < 2 or label is None:
                    continue

                try:
                    idx = int(label)
                except (ValueError, TypeError):
                    continue

                if idx < 0 or idx >= len(answers):
                    continue

                state = f"Context: {context}\nQuestion: {question}" if question else context
                case = DecisionCase(
                    state=state,
                    questions=[
                        Choice(
                            name="logical_deduction",
                            options=answers,
                            instruction=question or "Which conclusion logically follows?",
                            target=answers[idx],
                        )
                    ],
                    category="logical_reasoning",
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records

__all__ = ["ReClorSource"]


// ============================================================================
// FILE: data/sources/sciq_source.py (43 code lines, 53 total)
// ============================================================================

from typing import List, Sequence

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import load_raw_jsonl

SCIQ_INSTRUCTION = "Which option correctly answers the question?"

class SciQSource(IDataSource):
    """Scientific Mechanism & Distractor Elimination Source: scientific reasoning with negative distractors."""
    name = "allenai_sciq"

    def __init__(self, train_limit: int = 1500, test_limit: int = 200):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name, tag in [("train", "train"), ("validation", "test")]:
            limit = self.test_limit if split_name in ("validation", "test") else self.train_limit
            rows = load_raw_jsonl(f"sciq_{split_name}", "allenai/sciq", split=split_name, limit=limit)
            for r in rows:
                question = str(r.get("question", "")).strip()
                correct = str(r.get("correct_answer", "")).strip()
                d1 = str(r.get("distractor1", "")).strip()
                d2 = str(r.get("distractor2", "")).strip()
                d3 = str(r.get("distractor3", "")).strip()

                if not question or not correct or not d1 or not d2 or not d3:
                    continue

                options = [correct, d1, d2, d3]
                if len(set(options)) < 4:
                    continue

                case = DecisionCase(
                    state=question,
                    questions=[
                        Choice(
                            name="answer",
                            options=options,
                            instruction=SCIQ_INSTRUCTION,
                            target=correct,
                        )
                    ],
                    category="scientific_reasoning",
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records

__all__ = ["SciQSource", "SCIQ_INSTRUCTION"]


// ============================================================================
// FILE: data/sources/source.py (7 code lines, 8 total)
// ============================================================================

from typing import Protocol, Sequence, runtime_checkable
from data.case import SourceRecord

@runtime_checkable
class IDataSource(Protocol):
    """Sovereign policy contract: extracts domain records from external data origins."""
    name: str
    def read(self) -> Sequence[SourceRecord]: ...


// ============================================================================
// FILE: data/storage_policy.py (29 code lines, 36 total)
// ============================================================================

import orjson
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union, runtime_checkable
from data.case import DataSplits, DecisionCase, SplitTag, case_from_dict, case_to_dict

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

@runtime_checkable
class IStoragePolicy(Protocol):
    """Sub-Primitive Policy: Manages persistence and retrieval of partitioned dataset splits."""
    def save(self, splits: DataSplits) -> None: ...
    def load(self, split: SplitTag = "train") -> Sequence[DecisionCase]: ...

class DiskStoragePolicy(IStoragePolicy):
    """Default disk persistence policy for JSONL corpus storage."""

    def __init__(self, corpus_dir: Optional[Union[str, Path]] = None):
        self.corpus_dir = Path(corpus_dir) if corpus_dir is not None else DEFAULT_CORPUS_DIR

    def save(self, splits: DataSplits) -> None:
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        for tag, items in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
            target = self.corpus_dir / f"{tag}.jsonl"
            payload = b"\n".join(orjson.dumps(case_to_dict(s), default=str) for s in items)
            target.write_bytes(payload + (b"\n" if payload else b""))

    def load(self, split: SplitTag = "train") -> Sequence[DecisionCase]:
        target = self.corpus_dir / f"{split}.jsonl"
        if not target.exists():
            raise FileNotFoundError(
                f"Corpus partition '{split}' not found at {target}. "
                f"Explicitly prepare the corpus first (run 'python -m data' or invoke dataset.prepare())."
            )
        return [case_from_dict(orjson.loads(line)) for line in target.read_bytes().splitlines() if line.strip()]

__all__ = ["IStoragePolicy", "DiskStoragePolicy"]


// ============================================================================
// FILE: inference/__main__.py (79 code lines, 89 total)
// ============================================================================

import os, warnings
from typing import List, Optional
from rich import box
from rich.console import Console
from rich.table import Table
import typer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from inference.decision_engine import DecisionEngine, IDecisionEngine
from inference.profiling_layer import ProfilingLayer
from inference.questions import Choice, ChoiceResult, DecisionResult, Noul, NoulResult, Question, Score, ScoreResult
from model.weights import list_weights

console = Console()
app = typer.Typer(add_completion=False, help="Piqo Micro-Decision Engine CLI")

def parse_cli_questions(choice: list[str], noul: list[str], score: list[str]) -> list[Question]:
    qs: list[Question] = []
    for c in choice:
        name, _, opts = c.partition(":")
        opt_list = [o.strip() for o in opts.split(",") if o.strip()]
        if not name.strip() or not opt_list: raise typer.BadParameter(f"Invalid --choice '{c}'. Format: 'name:opt1,opt2'")
        qs.append(Choice(name.strip(), opt_list))
    for n in noul:
        name, sep, ins = n.partition(":")
        if not name.strip(): raise typer.BadParameter(f"Invalid --noul '{n}'. Format: 'name' or 'name:instruction'")
        qs.append(Noul(name.strip(), instruction=ins.strip() if sep and ins.strip() else None))
    for s in score:
        name, sep, ins = s.partition(":")
        if not name.strip(): raise typer.BadParameter(f"Invalid --score '{s}'. Format: 'name' or 'name:instruction'")
        qs.append(Score(name.strip(), instruction=ins.strip() if sep and ins.strip() else None))
    return qs

def render(res: DecisionResult, state: str) -> None:
    console.print(f"\n[bold]Context:[/] {state}")
    t = Table(title=f"Micro-Decision (Latency: {res.latency_ms:.2f} ms)", box=box.ROUNDED)
    t.add_column("Question", style="bold cyan")
    t.add_column("Decision", style="bold green")
    t.add_column("Confidence", justify="right")
    t.add_column("Details / Distribution", style="dim")
    for k, v in res.answers.items():
        match v:
            case ChoiceResult(choice=c, confidence=p, probabilities=dist):
                t.add_row(k, c, f"{p*100:.1f}%", ", ".join(f"{o}: {x*100:.1f}%" for o, x in dist.items()))
            case NoulResult(is_positive=pos, probability=p):
                t.add_row(k, "[bold green]YES[/]" if pos else "[bold red]NO[/]", f"{p*100:.1f}%", f"p={p:.4f}")
            case ScoreResult(score=s, normalized=norm):
                t.add_row(k, f"{s:.1f}", f"{norm:.3f}", "range 0-100")
    console.print(t)

@app.command()
def main(
    state: Optional[str] = typer.Argument(None, help="Context/state to evaluate"),
    play: bool = typer.Option(False, "--play", "-p", help="Playground REPL"),
    weights: bool = typer.Option(False, "--weights", "-w", help="List saved weights"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Weights checkpoint path/id"),
    choice: Optional[List[str]] = typer.Option(None, "--choice", "-c", help="Choice slot ('name:opt1,opt2')"),
    noul: Optional[List[str]] = typer.Option(None, "--noul", "-n", help="Noul slot ('name:instruction')"),
    score: Optional[List[str]] = typer.Option(None, "--score", "-s", help="Score slot ('name:instruction')"),
    device: Optional[str] = typer.Option(None, "--device", help="Compute device override"),
) -> None:
    if weights:
        t = Table(title="Saved Weights", box=box.ROUNDED)
        for col in ("Identity", "Size", "Val Loss", "Saved At"): t.add_column(col)
        for c in list_weights(): t.add_row(c.name, f"{c.size_mb:.1f} MB", f"{c.meta.val_loss:.4f}" if c.meta.val_loss else "-", c.meta.saved_at or "-")
        console.print(t); return

    try:
        engine: IDecisionEngine = DecisionEngine(checkpoint=model, device=device) | ProfilingLayer()
    except FileNotFoundError as e:
        console.print(f"[bold red]Model Error:[/] {e}")
        available = [w.name for w in list_weights()]
        if available:
            console.print(f"Available weights in cache: {available}")
        raise typer.Exit(code=1)

    if play or state is None:
        console.print(f"[bold cyan]Piqo Playground[/] [{model or 'piqo'}]. Enter context (or 'q' to quit):")
        while (ctx := console.input("\n[bold]Context[/]: ").strip()) and ctx.lower() not in ("q", "quit", "exit"):
            opts = [o.strip() for o in console.input("[bold]Choices[/]: ").split(",") if o.strip()]
            if opts: render(engine.decide(ctx, [Choice("decision", opts)]), ctx)
        return

    render(engine.decide(state, parse_cli_questions(choice or [], noul or [], score or []) or [Choice("decision", ["Yes", "No"])]), state)

if __name__ == "__main__":
    app()


// ============================================================================
// FILE: inference/decision_engine.py (36 code lines, 46 total)
// ============================================================================

from pathlib import Path
from typing import Optional, Protocol, Sequence, Union, runtime_checkable

from inference.questions import DecisionResult, DecisionState, Question
from inference.decision_resolver import DecisionResolver, IResolver
from model.torch_model import IModel

@runtime_checkable
class IDecisionEngineLayer(Protocol):
    """Layer Decorator Contract (lambda: P -> P): Instruments or wraps decision engine execution."""
    def __call__(self, inner: "IDecisionEngine") -> "IDecisionEngine": ...

@runtime_checkable
class IDecisionEngine(Protocol):
    """Core Primitive: Sovereign operational boundary for domain decision resolution."""
    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult: ...
    def __or__(self, layer: IDecisionEngineLayer) -> "IDecisionEngine": ...

class DecisionEngine(IDecisionEngine):
    """Core Primitive Implementation: Coordinates neural prediction and decision resolution."""

    def __init__(
        self,
        model: Optional["IModel"] = None,
        device: Optional[str] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        resolver: Optional[IResolver] = None,
    ):
        if model is None:
            from model.torch_model import TorchModel
            from model.weights import has_weights
            model = TorchModel(device=device)
            target_ckpt = checkpoint or ("piqo" if has_weights("piqo") else None)
            if target_ckpt is not None:
                model.load(target_ckpt)
        self.model, self.resolver = model, resolver or DecisionResolver()

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        pred = self.model.predict(state, questions)
        return DecisionResult(answers=self.resolver.resolve(questions, pred.slots, pred.scores))

    def __or__(self, layer: IDecisionEngineLayer) -> "IDecisionEngine":
        return layer(self)





// ============================================================================
// FILE: inference/decision_resolver.py (43 code lines, 46 total)
// ============================================================================

from typing import Dict, Mapping, Protocol, Sequence, runtime_checkable
import scipy.special as sp
from inference.questions import (
    Answer, Choice, ChoiceResult, Noul, NoulResult, Question, Score, ScoreResult,
    score_from_norm,
)

@runtime_checkable
class IResolver(Protocol):
    """Injected Opinion Sub-Primitive: Decodes raw float scores into typed domain Answers."""
    def resolve(
        self,
        questions: Sequence[Question],
        slots: Sequence[Sequence[int]],
        scores: Sequence[float],
    ) -> Mapping[str, Answer]:
        ...

class DecisionResolver(IResolver):
    """Numerically stable resolver: decodes raw float scores using industry-standard primitives."""

    def resolve(
        self,
        questions: Sequence[Question],
        slots: Sequence[Sequence[int]],
        scores: Sequence[float],
    ) -> Mapping[str, Answer]:
        answers: Dict[str, Answer] = {}
        for q, slot_indices in zip(questions, slots):
            match q:
                case Choice(name=name, options=opts):
                    keys = list(opts.keys()) if isinstance(opts, dict) else list(opts)
                    probs = sp.softmax([scores[i] for i in slot_indices])
                    best = int(probs.argmax())
                    answers[name] = ChoiceResult(
                        choice=keys[best],
                        confidence=float(probs[best]),
                        probabilities={k: float(p) for k, p in zip(keys, probs)},
                    )
                case Noul(name=name):
                    prob = float(sp.expit(scores[slot_indices[0]]))
                    answers[name] = NoulResult(is_positive=prob >= 0.5, probability=prob)
                case Score(name=name):
                    norm = float(sp.expit(scores[slot_indices[0]]))
                    answers[name] = ScoreResult(score=score_from_norm(q, norm), normalized=norm)
        return answers


// ============================================================================
// FILE: inference/profiling_layer.py (29 code lines, 34 total)
// ============================================================================

import time
from typing import Optional, Sequence
from inference.questions import DecisionResult, DecisionState, Question
from inference.decision_engine import IDecisionEngine, IDecisionEngineLayer

def synchronize_device() -> None:
    """Domain Transform: Synchronizes CUDA compute target if active."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, AttributeError):
        pass

class ProfilingLayer(IDecisionEngine):
    """Layer Decorator (lambda: P -> P): Measures wall-clock decision latency."""
    def __init__(self, inner: Optional[IDecisionEngine] = None):
        self.inner = inner

    def __call__(self, inner: IDecisionEngine) -> "ProfilingLayer":
        self.inner = inner
        return self

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        if self.inner is None: raise RuntimeError("ProfilingLayer is not attached to an inner engine.")
        synchronize_device()
        t0 = time.perf_counter()
        result = self.inner.decide(state, questions)
        synchronize_device()
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def __or__(self, layer: IDecisionEngineLayer) -> IDecisionEngine:
        return layer(self)


// ============================================================================
// FILE: inference/questions.py (117 code lines, 136 total)
// ============================================================================

from dataclasses import dataclass
import json
from typing import Mapping, Protocol, Sequence, runtime_checkable

DecisionState = str | Mapping[str, str]

@runtime_checkable
class IAnswer(Protocol):
    """Polymorphic Data Stream Contract: Shared capability for typed answer values."""
    @property
    def value(self) -> str | bool | float: ...

AnswerInput = IAnswer | str | int | float | bool | None

@dataclass(frozen=True, slots=True)
class Choice:
    name: str
    options: Sequence[str] | Mapping[str, str]
    instruction: str | None = None
    target: str | int | None = None

@dataclass(frozen=True, slots=True)
class Noul:
    name: str
    instruction: str | None = None
    target: bool | float | int | str | None = None

@dataclass(frozen=True, slots=True)
class Score:
    name: str
    min_value: float = 0.0
    max_value: float = 100.0
    instruction: str | None = None
    target: float | None = None

Question = Choice | Noul | Score

@dataclass(frozen=True, slots=True)
class ChoiceResult:
    choice: str
    confidence: float
    probabilities: Mapping[str, float]
    @property
    def value(self) -> str: return self.choice

@dataclass(frozen=True, slots=True)
class NoulResult:
    is_positive: bool
    probability: float
    @property
    def value(self) -> bool: return self.is_positive

@dataclass(frozen=True, slots=True)
class ScoreResult:
    score: float
    normalized: float
    @property
    def value(self) -> float: return self.score

Answer = ChoiceResult | NoulResult | ScoreResult

@dataclass(slots=True)
class DecisionResult:
    answers: Mapping[str, Answer]
    latency_ms: float = 0.0

def choice_keys(q: Choice) -> Sequence[str]:
    """Pure domain transform: extracts option keys from sequence or mapping."""
    return tuple(q.options.keys()) if isinstance(q.options, dict) else tuple(q.options)

def choice_target_index(q: Choice) -> int:
    """Pure domain transform: resolves integer target index from option keys."""
    if q.target is None:
        raise ValueError(f"Choice question '{q.name}' has no target defined.")
    keys = choice_keys(q)
    if isinstance(q.target, int):
        if 0 <= q.target < len(keys): return q.target
        raise ValueError(f"Target index {q.target} is out of bounds for options of length {len(keys)} in '{q.name}'.")
    tgt = str(q.target).strip()
    if tgt in keys: return keys.index(tgt)
    lower_keys = [k.lower() for k in keys]
    if tgt.lower() in lower_keys: return lower_keys.index(tgt.lower())
    if tgt.isdigit() and 0 <= int(tgt) < len(keys): return int(tgt)
    raise ValueError(f"Target '{q.target}' is not in options {keys} for question '{q.name}'.")

def shuffle_choice(q: Choice) -> Choice:
    """Pure domain transform: permutes choice candidate order preserving target alignment."""
    import random
    keys = list(choice_keys(q))
    indices = list(range(len(keys)))
    random.shuffle(indices)
    target = choice_target_index(q) if q.target is not None else None
    new_target = indices.index(target) if (target is not None and 0 <= target < len(indices)) else None
    opts = {keys[i]: q.options[keys[i]] for i in indices} if isinstance(q.options, dict) else [keys[i] for i in indices]
    return Choice(name=q.name, options=opts, instruction=q.instruction, target=new_target)

def score_to_norm(q: Score, value: float) -> float:
    """Pure domain transform: projects continuous score value to bounded [0, 1] interval."""
    span = q.max_value - q.min_value
    return max(0.0, min(1.0, (float(value) - q.min_value) / span)) if span > 0 else 0.0

def score_from_norm(q: Score, norm: float) -> float:
    """Pure domain transform: projects bounded [0, 1] activation back to continuous score domain."""
    return q.min_value + max(0.0, min(1.0, float(norm))) * (q.max_value - q.min_value)

def matches_answer(q: Question, answer: AnswerInput, tolerance: float = 0.05) -> bool:
    """Pure domain transform: evaluates polymorphic answer match against question target."""
    if answer is None or q.target is None: return False
    match q:
        case Choice():
            val, tgt = str(getattr(answer, "value", answer)).strip(), str(q.target).strip()
            if val == tgt or val.lower() == tgt.lower(): return True
            keys = choice_keys(q)
            if tgt.isdigit() and 0 <= int(tgt) < len(keys):
                return val == str(keys[int(tgt)]) or val.lower() == str(keys[int(tgt)]).lower()
            return val in keys and tgt in keys and keys.index(val) == keys.index(tgt)
        case Noul():
            val = getattr(answer, "value", answer)
            val_bool = val.strip().lower() in ("true", "1", "yes") if isinstance(val, str) else bool(val)
            tgt_bool = q.target.strip().lower() in ("true", "1", "yes") if isinstance(q.target, str) else bool(q.target)
            return val_bool == tgt_bool
        case Score():
            val = float(getattr(answer, "value", answer))
            return abs(score_to_norm(q, val) - score_to_norm(q, float(q.target))) <= tolerance
    return False

def format_state(state: DecisionState) -> str:
    """Colocated pure transform: formats state string or dictionary for token encoding."""
    if isinstance(state, dict): return " | ".join(f"{k}: {v}" for k, v in state.items())
    if isinstance(state, str) and state.startswith("{") and state.endswith("}"):
        try:
            if isinstance(parsed := json.loads(state), dict):
                return " | ".join(f"{k}: {v}" for k, v in parsed.items())
        except json.JSONDecodeError:
            pass
    return str(state)


// ============================================================================
// FILE: model/device.py (26 code lines, 31 total)
// ============================================================================

import os
from typing import Optional
import torch

def resolve_device(preferred: Optional[str] = None) -> str:
    """Domain Operation: Resolves optimal compute target device name without substrate leakage."""
    if preferred and preferred.strip():
        return preferred.strip().lower()

    env = os.environ.get("PIQO_DEVICE")
    if env and env.strip():
        return env.strip().lower()

    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except (ImportError, AttributeError):
        pass

    return "cpu"

def resolve_vram_gb() -> Optional[float]:
    """Domain Operation: Inspects GPU reserved memory in GB without substrate leakage."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / (1024**3)
    except (ImportError, AttributeError):
        pass
    return None


// ============================================================================
// FILE: model/policies/calibrated_loss.py (62 code lines, 70 total)
// ============================================================================

from typing import Protocol, Sequence, runtime_checkable
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.token_layout import SampleTargets

@runtime_checkable
class ILoss(Protocol):
    """Injected Opinion Sub-Primitive: Evaluates multi-task loss over predictions and targets."""
    def compute(self, predictions: torch.Tensor, targets: Sequence[SampleTargets]) -> torch.Tensor: ...

class CalibratedLoss(nn.Module, ILoss):
    """Brier-calibrated multi-task loss policy: vectorizes cross-entropy, Brier, and MSE losses."""

    def __init__(self, brier_weight: float = 0.5):
        super().__init__()
        self.brier_weight = brier_weight

    def compute(self, predictions: torch.Tensor, targets: Sequence[SampleTargets]) -> torch.Tensor:
        dev = predictions.device
        c_logits, c_tgts = [], []
        b_idx, b_pos, b_tgts = [], [], []
        s_idx, s_pos, s_tgts = [], [], []

        for b, sample in enumerate(targets):
            for item in sample.slots:
                t, pos, val = item.target_type, item.indices, item.value
                if t == "choice":
                    c_logits.append(predictions[b, pos])
                    c_tgts.append(int(val))
                elif t == "noul":
                    b_idx.append(b); b_pos.append(pos[0]); b_tgts.append(float(val))
                elif t == "score":
                    s_idx.append(b); s_pos.append(pos[0]); s_tgts.append(float(val))

        losses, counts = [], []
        if c_logits:
            max_c = max(l.size(0) for l in c_logits)
            pad = torch.full((len(c_logits), max_c), -1e4, device=dev)
            for i, l in enumerate(c_logits):
                pad[i, :l.size(0)] = l
            tgt_tensor = torch.as_tensor(c_tgts, dtype=torch.long, device=dev)
            ce_loss = F.cross_entropy(pad, tgt_tensor, reduction="mean")
            if self.brier_weight > 0:
                probs = F.softmax(pad, dim=-1)
                one_hot = F.one_hot(tgt_tensor, num_classes=max_c).float()
                brier_loss = torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))
                losses.append((1.0 - self.brier_weight) * ce_loss + self.brier_weight * brier_loss)
            else:
                losses.append(ce_loss)
            counts.append(len(c_logits))
        if b_idx:
            pred = predictions[torch.as_tensor(b_idx, device=dev), torch.as_tensor(b_pos, device=dev)]
            tgt = torch.as_tensor(b_tgts, dtype=torch.float, device=dev)
            prob = torch.sigmoid(pred)
            bce_loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="mean")
            brier_bce = torch.mean((prob - tgt) ** 2)
            losses.append((1.0 - self.brier_weight) * bce_loss + self.brier_weight * brier_bce)
            counts.append(len(b_idx))
        if s_idx:
            pred = predictions[torch.as_tensor(s_idx, device=dev), torch.as_tensor(s_pos, device=dev)]
            tgt = torch.as_tensor(s_tgts, dtype=torch.float, device=dev)
            losses.append(F.mse_loss(torch.sigmoid(pred), tgt))
            counts.append(len(s_idx))

        if not losses:
            return torch.tensor(0.0, device=dev)
        weights = torch.as_tensor(counts, dtype=torch.float, device=dev)
        return torch.sum(torch.stack(losses) * weights) / torch.sum(weights)


// ============================================================================
// FILE: model/policies/head.py (20 code lines, 25 total)
// ============================================================================

from typing import Protocol, runtime_checkable
import torch
import torch.nn as nn

@runtime_checkable
class IHead(Protocol):
    """Opinion Sub-Primitive: Projects hidden token representations to scalar decision scores."""
    def score(self, representations: torch.Tensor) -> torch.Tensor: ...

class MLPHead(nn.Module, IHead):
    """Default decision head: opinion sub-primitive composing LayerNorm + Dropout + Linear + GELU + Linear."""

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def score(self, representations: torch.Tensor) -> torch.Tensor:
        return self._net(representations).squeeze(-1)



// ============================================================================
// FILE: model/policies/layout.py (87 code lines, 103 total)
// ============================================================================

from typing import Optional, Protocol, Sequence, runtime_checkable
import torch

from data.case import DecisionCase
from inference.questions import (
    Choice, DecisionState, Question, Score, choice_target_index, format_state,
    score_to_norm, shuffle_choice,
)
from model.token_layout import Batch, SampleTargets, TargetSlot, TokenLayout, TokenSlot

@runtime_checkable
class ITokenizer(Protocol):
    """Sub-Primitive Protocol: Tokenizes text into integer token ids."""
    cls_token_id: int
    sep_token_id: int
    mask_token_id: int
    pad_token_id: int
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

@runtime_checkable
class ILayout(Protocol):
    """Opinion Sub-Primitive: Formats and tokenizes decision states and cases into model layouts."""
    def encode(self, state: DecisionState, questions: Sequence[Question]) -> TokenLayout: ...
    def collate(self, cases: Sequence[DecisionCase], shuffle: bool = False) -> Batch: ...

class MaskedLayout(ILayout):
    """Standard masked token layout policy encoding questions and candidate slots."""

    def __init__(self, tokenizer: Optional[ITokenizer] = None, max_state_tokens: int = 8192):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.tokenizer = tokenizer
        self.max_state_tokens = max_state_tokens

    def encode(self, state: DecisionState, questions: Sequence[Question]) -> TokenLayout:
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        mask_id = self.tokenizer.mask_token_id

        ids = [cls_id]
        slots = []
        for q in questions:
            ids.append(sep_id)
            q_type = "choice" if isinstance(q, Choice) else ("score" if isinstance(q, Score) else "noul")
            prompt = f" {q_type} question: {q.instruction}" if q.instruction else f" {q_type} question for {q.name}:"
            ids.extend(self.tokenizer.encode(prompt, add_special_tokens=False))

            if isinstance(q, Choice):
                pos = []
                opt_items = q.options.items() if isinstance(q.options, dict) else [(o, None) for o in q.options]
                for k, v in opt_items:
                    ids.append(mask_id)
                    pos.append(len(ids) - 1)
                    opt_text = f" {k}: {v}" if v else f" {k}"
                    ids.extend(self.tokenizer.encode(opt_text, add_special_tokens=False))
                slots.append(TokenSlot(pos))
            else:
                ids.append(mask_id)
                slots.append(TokenSlot([len(ids) - 1]))

        ids.append(sep_id)
        state_tokens = self.tokenizer.encode(format_state(state), add_special_tokens=False)
        if len(state_tokens) > self.max_state_tokens:
            state_tokens = state_tokens[:self.max_state_tokens]
        ids.extend(state_tokens)
        ids.append(sep_id)

        return TokenLayout(input_ids=ids, mask=[1.0] * len(ids), slots=slots)

    def collate(self, cases: Sequence[DecisionCase], shuffle: bool = False) -> Batch:
        pad_id = self.tokenizer.pad_token_id
        padded_layouts: list[TokenLayout] = []
        all_targets: list[SampleTargets] = []

        for c in cases:
            questions = [shuffle_choice(q) if shuffle and isinstance(q, Choice) else q for q in c.questions]
            layout = self.encode(c.state, questions)
            padded_layouts.append(layout)

            slots = []
            for q, pos in zip(questions, layout.slots):
                if isinstance(q, Choice):
                    slots.append(TargetSlot("choice", pos, choice_target_index(q)))
                elif isinstance(q, Score):
                    target_val = score_to_norm(q, q.target) if q.target is not None else 0.0
                    slots.append(TargetSlot("score", pos, target_val))
                else:
                    tgt = q.target
                    is_pos = (tgt is True or str(tgt).strip().lower() in ("true", "1", "yes")) if tgt is not None else False
                    slots.append(TargetSlot("noul", pos, 1.0 if is_pos else 0.0))
            all_targets.append(SampleTargets(slots))

        max_len = max((len(l.input_ids) for l in padded_layouts), default=0)
        padded_ids = torch.full((len(padded_layouts), max_len), pad_id, dtype=torch.long)
        masks = torch.zeros((len(padded_layouts), max_len), dtype=torch.float)

        for i, layout in enumerate(padded_layouts):
            sz = len(layout.input_ids)
            padded_ids[i, :sz] = torch.as_tensor(layout.input_ids, dtype=torch.long)
            masks[i, :sz] = 1.0

        return Batch(input_ids=padded_ids, mask=masks, targets=all_targets)


// ============================================================================
// FILE: model/policies/optimizer.py (63 code lines, 72 total)
// ============================================================================

from typing import Iterable, Optional, Protocol, Union, runtime_checkable
import torch

@runtime_checkable
class IGradientNode(Protocol):
    """Structural protocol for computational loss graph nodes supporting differentiation."""
    def backward(self) -> None: ...

@runtime_checkable
class IOptimizerPolicy(Protocol):
    """Sub-Primitive Policy: Coordinates parameter update strategy, gradient scaling, and accumulation."""
    def step(self, loss: IGradientNode, lr: Optional[float] = None) -> None: ...
    def flush(self) -> None: ...
    def zero_grad(self) -> None: ...

class AdamWOptimizer(IOptimizerPolicy):
    """Default optimizer policy: AdamW with AMP GradScaler and gradient accumulation."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-5,
        accum_steps: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.accum_steps = max(1, accum_steps)
        dev = torch.device(device) if isinstance(device, str) else device
        is_cuda = dev.type == "cuda" if dev is not None else torch.cuda.is_available()
        self.optimizer = torch.optim.AdamW(params, lr=lr, fused=is_cuda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)
        self.opt_count = 0

    def step(self, loss: IGradientNode, lr: Optional[float] = None) -> None:
        if lr is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = lr

        loss_node = (loss / self.accum_steps) if isinstance(loss, torch.Tensor) else loss
        if self.scaler.is_enabled() and isinstance(loss_node, torch.Tensor):
            scaled = self.scaler.scale(loss_node)
            scaled.backward()
        else:
            loss_node.backward()

        self.opt_count += 1
        if self.opt_count % self.accum_steps == 0:
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def flush(self) -> None:
        """Flushes any pending accumulated gradients before epoch or iteration boundaries."""
        remainder = self.opt_count % self.accum_steps
        if remainder != 0:
            scale_factor = self.accum_steps / remainder
            for p in self.optimizer.param_groups:
                for param in p["params"]:
                    if param.grad is not None:
                        param.grad.mul_(scale_factor)
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.opt_count = 0

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)


// ============================================================================
// FILE: model/policies/weight_cache.py (46 code lines, 52 total)
// ============================================================================

import orjson
from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable
from model.weights import DEFAULT_WEIGHTS_DIR, WeightMeta, Weights, weights_from_bytes

@runtime_checkable
class IWeightCache(Protocol):
    """Internal opinion sub-primitive contract: local persistence for neural weights."""
    def get(self, target: Union[str, Path]) -> Optional[Weights]: ...
    def put(self, target: Union[str, Path], weights: Weights) -> Path: ...

class DiskWeightCache(IWeightCache):
    """Local filesystem weight cache implementation."""

    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_WEIGHTS_DIR

    def get(self, target: Union[str, Path]) -> Optional[Weights]:
        p = Path(target)
        w_file = p if p.is_file() else (self.root_dir / p.name if (self.root_dir / p.name).is_file() else (self.root_dir / p.stem).with_suffix(".safetensors"))
        if not w_file.is_file():
            return None
        meta_file = w_file.with_suffix(".json")
        meta_dict = orjson.loads(meta_file.read_bytes()) if meta_file.exists() else {}
        meta = WeightMeta(
            epoch=meta_dict.get("epoch", 1),
            train_loss=meta_dict.get("train_loss"),
            val_loss=meta_dict.get("val_loss"),
            saved_at=meta_dict.get("saved_at"),
        )
        return weights_from_bytes(
            payload=w_file.read_bytes(),
            name=meta_dict.get("name", w_file.stem),
            meta=meta,
        )

    def put(self, target: Union[str, Path], weights: Weights) -> Path:
        p = Path(target)
        w_file = (p if p.is_absolute() or p.parent != Path(".") else self.root_dir / p.stem).with_suffix(".safetensors")
        w_file.parent.mkdir(parents=True, exist_ok=True)
        if weights.payload:
            w_file.write_bytes(weights.payload)

        meta = {
            "name": weights.name or w_file.stem,
            "epoch": weights.meta.epoch,
            "train_loss": weights.meta.train_loss,
            "val_loss": weights.meta.val_loss,
            "saved_at": weights.meta.saved_at,
        }
        w_file.with_suffix(".json").write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
        return w_file


// ============================================================================
// FILE: model/token_layout.py (40 code lines, 49 total)
// ============================================================================

from dataclasses import dataclass
from typing import Sequence
import torch

@dataclass(frozen=True, slots=True)
class TokenSlot:
    """Explicit domain DTO for a question's candidate token slot positions."""
    indices: Sequence[int]

    def __iter__(self):
        return iter(self.indices)

    def __getitem__(self, idx: int) -> int:
        return self.indices[idx]

    def __len__(self) -> int:
        return len(self.indices)

@dataclass(frozen=True, slots=True)
class TokenLayout:
    """Explicit domain DTO for tokenized input sequence and question slot positions."""
    input_ids: Sequence[int]
    mask: Sequence[float]
    slots: Sequence[TokenSlot]

@dataclass(frozen=True, slots=True)
class PredictionLayout:
    """Explicit domain DTO for model output scores and corresponding question slot positions."""
    scores: Sequence[float]
    slots: Sequence[TokenSlot]

@dataclass(frozen=True, slots=True)
class TargetSlot:
    """Explicit domain DTO for a question's masked slot positions and target value."""
    target_type: str
    indices: Sequence[int]
    value: int | float

@dataclass(frozen=True, slots=True)
class SampleTargets:
    """Explicit domain DTO for all target slots in a single sample."""
    slots: Sequence[TargetSlot]

@dataclass(frozen=True, slots=True)
class Batch:
    """Explicit domain DTO for a batch of encoded tokens and targets."""
    input_ids: torch.Tensor | Sequence[Sequence[int]]
    mask: torch.Tensor | Sequence[Sequence[float]]
    targets: Sequence[SampleTargets]


// ============================================================================
// FILE: model/torch_model.py (119 code lines, 133 total)
// ============================================================================

from pathlib import Path
import time
from typing import Optional, Protocol, Sequence, Union, runtime_checkable
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, logging

logging.set_verbosity_error()

from data.case import DecisionCase
from inference.questions import DecisionState, Question
from model.device import resolve_device
from model.policies.calibrated_loss import CalibratedLoss, ILoss
from model.policies.head import IHead, MLPHead
from model.policies.layout import ILayout, MaskedLayout
from model.policies.optimizer import AdamWOptimizer, IOptimizerPolicy
from model.policies.weight_cache import DiskWeightCache, IWeightCache
from model.token_layout import PredictionLayout
from model.weights import WeightMeta, Weights

@runtime_checkable
class IModel(Protocol):
    """Core Primitive: Neural computation boundary for inference scoring and loss computation."""
    def predict(self, state: DecisionState, questions: Sequence[Question]) -> PredictionLayout: ...
    def step(self, cases: Sequence[DecisionCase], backward: bool = True, lr: Optional[float] = None) -> float: ...
    def flush(self) -> None: ...
    def save(self, target: Union[str, Path] = "piqo", meta: Optional[WeightMeta] = None) -> Path: ...
    def load(self, source: Optional[Union[Weights, str, Path]] = None, strict: bool = True) -> None: ...

class TorchModel(IModel):
    """Core Primitive Implementation: ModernBERT backbone with modular decision head, layout, and optimizer."""

    def __init__(
        self,
        backbone_name: str = "answerdotai/ModernBERT-base",
        backbone: Optional[nn.Module] = None,
        head: Optional[IHead] = None,
        layout: Optional[ILayout] = None,
        loss_policy: Optional[ILoss] = None,
        optimizer_policy: Optional[IOptimizerPolicy] = None,
        weight_cache: Optional[IWeightCache] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        weights: Optional[Weights] = None,
        lr: float = 2e-5,
        accum_steps: int = 1,
        max_state_tokens: int = 2048,
        device: Optional[Union[str, torch.device]] = None,
        compile: bool = False,
    ):
        target_dev = device if isinstance(device, torch.device) else resolve_device(device)
        self.device = torch.device(target_dev) if isinstance(target_dev, str) else target_dev
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model_kwargs = {"attn_implementation": "sdpa"} if self.device.type == "cuda" else {}
        self.backbone = (backbone or AutoModel.from_pretrained(backbone_name, **model_kwargs)).to(self.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(backbone_name)
        hidden_dim = getattr(self.backbone.config, "hidden_size", 768) if hasattr(self.backbone, "config") else 768

        self.head = (head or MLPHead(hidden_dim=hidden_dim)).to(self.device)
        self.layout = layout or MaskedLayout(tokenizer=self.tokenizer, max_state_tokens=max_state_tokens)
        self.loss_policy = loss_policy or CalibratedLoss()
        self.weight_cache = weight_cache or DiskWeightCache()

        params = list(self.backbone.parameters()) + list(self.head.parameters())
        self.optimizer_policy = optimizer_policy or AdamWOptimizer(params, lr=lr, accum_steps=accum_steps, device=self.device)

        if weights is not None:
            self.load(weights)
        if compile and hasattr(torch, "compile"):
            self.backbone = torch.compile(self.backbone)

    def predict(self, state: DecisionState, questions: Sequence[Question]) -> PredictionLayout:
        layout = self.layout.encode(state, questions)
        self.backbone.eval()
        self.head.eval()
        ids = torch.as_tensor(layout.input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(layout.mask, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            scores = self.head.score(hidden).squeeze(0).cpu().tolist()
        return PredictionLayout(scores=scores, slots=layout.slots)

    def step(self, cases: Sequence[DecisionCase], backward: bool = True, lr: Optional[float] = None) -> float:
        batch = self.layout.collate(cases, shuffle=backward)
        self.backbone.train(backward)
        self.head.train(backward)
        ids = batch.input_ids.to(self.device, non_blocking=True)
        mask = batch.mask.to(self.device, non_blocking=True)
        with torch.set_grad_enabled(backward), torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            preds = self.head.score(hidden)
            loss = self.loss_policy.compute(preds, batch.targets)
        if backward and self.optimizer_policy:
            self.optimizer_policy.step(loss, lr=lr)
        return loss.item()

    def flush(self) -> None:
        if self.optimizer_policy:
            self.optimizer_policy.flush()

    def save(self, target: Union[str, Path] = "piqo", meta: Optional[WeightMeta] = None) -> Path:
        from safetensors.torch import save
        tensors = {f"backbone.{k}": v.contiguous() for k, v in self.backbone.state_dict().items()}
        tensors.update({f"head.{k}": v.contiguous() for k, v in self.head.state_dict().items()})
        raw_bytes = save(tensors)
        metadata = meta or WeightMeta(saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        if not metadata.saved_at:
            metadata = WeightMeta(epoch=metadata.epoch, train_loss=metadata.train_loss, val_loss=metadata.val_loss, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        return self.weight_cache.put(target, Weights(payload=raw_bytes, name=Path(target).stem, meta=metadata))

    def load(self, source: Optional[Union[Weights, str, Path]] = None, strict: bool = True) -> None:
        target = "piqo" if source is None else source
        weights = target if isinstance(target, Weights) else self.weight_cache.get(target)
        if not weights:
            raise FileNotFoundError(f"Weights not found: {target}")
        if not weights.payload:
            return
        from safetensors.torch import load
        sd = load(weights.payload)
        bb_sd = {k.removeprefix("backbone."): v for k, v in sd.items() if k.startswith("backbone.")}
        h_sd = {k.removeprefix("head."): v for k, v in sd.items() if k.startswith("head.")}
        if bb_sd:
            res_bb = self.backbone.load_state_dict(bb_sd, strict=strict)
            if not strict and (res_bb.missing_keys or res_bb.unexpected_keys):
                import sys
                sys.stderr.write(f"\n[CHECKPOINT WARNING] '{target}' backbone missing: {res_bb.missing_keys}, unexpected: {res_bb.unexpected_keys}\n")
        if h_sd:
            res_h = self.head.load_state_dict(h_sd, strict=strict)
            if not strict and (res_h.missing_keys or res_h.unexpected_keys):
                import sys
                sys.stderr.write(f"\n[CHECKPOINT WARNING] '{target}' head missing: {res_h.missing_keys}, unexpected: {res_h.unexpected_keys}\n")


// ============================================================================
// FILE: model/weights.py (49 code lines, 55 total)
// ============================================================================

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional, Sequence

DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

@dataclass(frozen=True, slots=True)
class WeightMeta:
    """Domain DTO: Training checkpoint telemetry and provenance metadata."""
    epoch: int = 1
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    saved_at: Optional[str] = None

@dataclass(frozen=True, slots=True)
class Weights:
    """Core Domain DTO: Substrate-neutral parameter payload and metadata."""
    payload: bytes = b""
    name: str = "piqo"
    meta: WeightMeta = field(default_factory=WeightMeta)
    size_mb: float = 0.0

def weights_from_bytes(payload: bytes, name: str = "piqo", meta: Optional[WeightMeta] = None) -> Weights:
    """Pure Transform: constructs Weights DTO from raw byte payload and metadata."""
    size_mb = round(len(payload) / (1024 * 1024), 1) if payload else 0.0
    return Weights(payload=payload, name=name, meta=meta or WeightMeta(), size_mb=size_mb)

def list_weights(root: Optional[Path] = None) -> Sequence[Weights]:
    """Domain Operation: Discovers saved weights and their metadata from weights directory."""
    weights_dir = root or DEFAULT_WEIGHTS_DIR
    if not weights_dir.exists():
        return []
    files = {f.stem: f for f in weights_dir.glob("*.safetensors")}
    results = []
    for stem, w_file in files.items():
        meta_file = w_file.with_suffix(".json")
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        results.append(
            Weights(
                name=meta.get("name", stem),
                size_mb=round(w_file.stat().st_size / (1024 * 1024), 1),
                meta=WeightMeta(
                    epoch=meta.get("epoch", 1),
                    train_loss=meta.get("train_loss"),
                    val_loss=meta.get("val_loss"),
                    saved_at=meta.get("saved_at"),
                ),
            )
        )
    return sorted(results, key=lambda x: x.meta.saved_at or "", reverse=True)

def has_weights(name: str = "piqo", root: Optional[Path] = None) -> bool:
    """Domain Operation: Verifies whether a weight checkpoint is available in the cache."""
    return any(w.name == name for w in list_weights(root))


// ============================================================================
// FILE: piqo.py (92 code lines, 94 total)
// ============================================================================

"""Piqo: Sovereign Composition Root (PFD)."""

from model.torch_model import IModel, TorchModel
from model.policies.calibrated_loss import ILoss, CalibratedLoss
from model.policies.head import IHead, MLPHead
from model.policies.optimizer import IOptimizerPolicy, AdamWOptimizer
from model.policies.layout import ILayout, MaskedLayout
from model.policies.weight_cache import (
    DEFAULT_WEIGHTS_DIR, DiskWeightCache, IWeightCache,
)
from model.device import resolve_device, resolve_vram_gb
from model.weights import Weights, WeightMeta, list_weights, has_weights
from model.token_layout import TokenLayout, PredictionLayout, TokenSlot, Batch, SampleTargets, TargetSlot
from inference.decision_engine import IDecisionEngine, DecisionEngine
from inference.decision_resolver import IResolver, DecisionResolver
from inference.profiling_layer import ProfilingLayer
from inference.questions import (
    DecisionState, Choice, Noul, Score, Question,
    IAnswer, ChoiceResult, NoulResult, ScoreResult, Answer, DecisionResult,
    format_state, choice_keys, choice_target_index, shuffle_choice,
    score_to_norm, score_from_norm, matches_answer,
)
from training.epoch_trainer import ITrainer, EpochTrainer
from training.policies.schedule import ISchedule, CosineSchedule, ConstantSchedule
from training.telemetry.checkpointing_layer import CheckpointingLayer
from training.telemetry.metrics_layer import MetricsLayer
from training.stats import StepStats
from data.dataset import IDataset, Dataset
from data.case import DataSplits, SourceRecord, DecisionCase, case_to_dict, case_from_dict, case_gold
from data.storage_policy import IStoragePolicy, DiskStoragePolicy
from data.curation.budget_split import ISplitPolicy, BudgetSplitPolicy
from data.curation.ngram_leakage import ILeakagePolicy, NGramLeakagePolicy
from data.sources.source import IDataSource
from data.registry import AVAILABLE_SOURCES
from data.sources.glaive_source import GlaiveSource
from data.sources.decisions_source import DecisionsSource
from data.sources.huggingface_source import HuggingFaceSource, BenchmarkSpec, RAW_DIR
from benchmark.policies.scorer import BenchmarkReport, IScorerPolicy, ScorerPolicy
from benchmark.policies.ledger import ILedgerPolicy, DiskLedgerPolicy
from benchmark.policies.reporter import IReporterPolicy, ConsoleReporterPolicy
from benchmark.benchmark import IBenchmark, Benchmark
from benchmark.registry import AVAILABLE_ENGINES, BENCHMARK_SUITES
from benchmark.competitors.gliner_decide import GlinerDecideEngine
from benchmark.competitors.jev import JevEngine
from benchmark.competitors.kev import KevEngine
from benchmark.competitors.laya import LayaEngine
from benchmark.competitors.verdict import VerdictEngine

__all__ = [
    # Core Primitives
    "IModel", "TorchModel",
    "IDecisionEngine", "DecisionEngine",
    "ITrainer", "EpochTrainer",
    "IDataset", "Dataset",
    "IBenchmark", "Benchmark", "GlinerDecideEngine", "JevEngine", "KevEngine", "LayaEngine", "VerdictEngine",
    # Policies & Injected Sub-Primitives
    "IHead", "MLPHead",
    "IOptimizerPolicy", "AdamWOptimizer",
    "ILoss", "CalibratedLoss",
    "ILayout", "MaskedLayout",
    "IWeightCache", "DiskWeightCache", "DEFAULT_WEIGHTS_DIR",
    "ISchedule", "CosineSchedule", "ConstantSchedule",
    "IResolver", "DecisionResolver",
    "IStoragePolicy", "DiskStoragePolicy",
    "ISplitPolicy", "BudgetSplitPolicy", "ILeakagePolicy", "NGramLeakagePolicy",
    "ILedgerPolicy", "DiskLedgerPolicy",
    "IScorerPolicy", "ScorerPolicy",
    "IReporterPolicy", "ConsoleReporterPolicy",
    "AVAILABLE_ENGINES", "BENCHMARK_SUITES", "AVAILABLE_SOURCES",
    "IDataSource", "GlaiveSource", "DecisionsSource", "HuggingFaceSource", "BenchmarkSpec",
    # Layers (Lambda: P -> P)
    "ProfilingLayer", "CheckpointingLayer", "MetricsLayer",
    # Domain DTOs & Models
    "Weights",
    "DecisionState", "Choice", "Noul", "Score", "Question",
    "IAnswer", "ChoiceResult", "NoulResult", "ScoreResult", "Answer", "DecisionResult",
    "TokenLayout", "PredictionLayout", "DecisionCase", "DataSplits", "SourceRecord",
    "Batch", "StepStats", "TokenSlot", "TargetSlot", "SampleTargets",
    "BenchmarkReport",
    # Transforms & Paths
    "format_state",
    "choice_keys",
    "choice_target_index",
    "shuffle_choice",
    "score_to_norm",
    "score_from_norm",
    "matches_answer",
    "case_to_dict",
    "case_from_dict",
    "case_gold",
    "list_weights",
    "DEFAULT_WEIGHTS_DIR", "RAW_DIR",
    "resolve_device",
]


// ============================================================================
// FILE: training/__main__.py (93 code lines, 110 total)
// ============================================================================

import random
import time
from typing import Optional
from rich import box
from rich.console import Console
from rich.table import Table
import torch
import transformers
import typer

transformers.logging.set_verbosity_error()

from data.dataset import Dataset
from model.torch_model import TorchModel
from model.weights import list_weights
from training.epoch_trainer import EpochTrainer, ITrainer
from training.policies.schedule import ConstantSchedule, CosineSchedule
from training.telemetry.checkpointing_layer import CheckpointingLayer
from training.telemetry.metrics_layer import MetricsLayer

DEFAULT_BACKBONE = "answerdotai/ModernBERT-base"

app = typer.Typer(add_completion=False, help="Piqo Model Trainer CLI")

def make_card(title: str, rows: list[tuple[str, str]], style: str = "bold magenta") -> Table:
    table = Table(title=title, box=box.HORIZONTALS, title_style=style)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold green" if style == "bold white" else "green")
    for k, v in rows:
        table.add_row(k, v)
    return table

@app.command()
def main(
    backbone: str = typer.Option(DEFAULT_BACKBONE, help="HuggingFace backbone model"),
    name: str = typer.Option("piqo", "--name", "-n", help="Target weights name/prefix (default: 'piqo')"),
    init_weights: Optional[str] = typer.Option(None, "--init", help="Initial weights to resume from"),
    from_scratch: bool = typer.Option(False, "--from-scratch", help="Train from raw base backbone"),
    epochs: Optional[int] = typer.Option(None, help="Number of training epochs (default: 2 for --from-scratch, 1 for adaptation)"),
    batch_size: int = typer.Option(16, help="Batch size"),
    lr: Optional[float] = typer.Option(None, help="Learning rate (default: 3e-5 for --from-scratch, 1e-5 for adaptation)"),
    accum_steps: int = typer.Option(2, help="Gradient accumulation steps"),
    seed: int = typer.Option(42, help="Random seed"),
    compile: bool = typer.Option(False, "--compile", help="Compile model with torch.compile"),
    schedule: str = typer.Option("cosine", help="Learning rate schedule policy ('cosine', 'constant')"),
    warmup_ratio: float = typer.Option(0.05, help="Warmup ratio for cosine schedule"),
    eval_test: bool = typer.Option(False, "--eval-test", help="Evaluate internal holdout test split after training"),
) -> None:
    console = Console()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    has_piqo = any(w.name == "piqo" for w in list_weights())
    init_ckpt = None if from_scratch or (init_weights and init_weights.lower() in ("none", "")) else (init_weights or ("piqo" if has_piqo else None))
    out_target = name

    target_epochs = epochs if epochs is not None else (2 if from_scratch else 1)
    target_lr = lr if lr is not None else (3e-5 if from_scratch else 1e-5)

    model = TorchModel(backbone_name=backbone, accum_steps=accum_steps, compile=compile)
    if init_ckpt:
        model.load(init_ckpt)

    schedule_policy = ConstantSchedule() if schedule.lower() == "constant" else CosineSchedule(warmup_ratio=warmup_ratio)
    trainer: ITrainer = (
        EpochTrainer(model=model, batch_size=batch_size, base_lr=target_lr, schedule=schedule_policy)
        | MetricsLayer(log_interval_sec=5.0)
        | CheckpointingLayer(output_path=out_target)
    )

    dataset = Dataset()
    train_cases = dataset.load("train")
    val_cases = dataset.load("val")

    init_desc = f"Checkpoint ({init_ckpt})" if init_ckpt else "Scratch (Raw Backbone + Xavier Head)"
    console.print(f"\n[bold cyan]Piqo Model Trainer[/bold cyan] | Target: [bold green]model/weights/{out_target}[/bold green]\n")
    console.print(make_card("Training Configuration", [
        ("Backbone", backbone),
        ("Initialization", init_desc),
        ("Epochs", str(target_epochs)),
        ("Batch Size (Effective)", f"{batch_size} (Effective: {batch_size * accum_steps})"),
        ("Learning Rate", f"{target_lr:.2e} ({schedule})"),
        ("Dataset Splits", f"{len(train_cases):,} train, {len(val_cases):,} val cases"),
    ]))
    console.print("")

    stats = list(trainer.fit(train_cases, epochs=target_epochs, val_cases=val_cases))

    res = [("Status", "Completed Successfully")]
    if stats:
        res.append(("Final Train Loss", f"{stats[-1].loss:.4f}"))
        if stats[-1].val_loss is not None:
            res.append(("Final Val Loss", f"{stats[-1].val_loss:.4f}"))

    if eval_test:
        test_cases = dataset.load("test")
        console.print(f"\n[dim]Evaluating internal holdout test split ({len(test_cases):,} cases)...[/dim]")
        t_eval = time.perf_counter()
        test_loss = trainer.evaluate(test_cases)
        ms = (time.perf_counter() - t_eval) * 1000.0 / max(1, len(test_cases))
        res.append(("Test Holdout Loss", f"{test_loss:.4f} ({ms:.2f}ms/case)"))

    res.append(("Output Weights", f"model/weights/{out_target}"))
    console.print(make_card("Training Results", res, style="bold white"))
    console.print("")

if __name__ == "__main__":
    app()


// ============================================================================
// FILE: training/epoch_trainer.py (83 code lines, 98 total)
// ============================================================================

import time
from typing import Iterator, Optional, Protocol, Sequence, runtime_checkable

from data.case import DecisionCase
from model.torch_model import IModel, TorchModel
from training.policies.batch import BucketedBatch, IBatch, SequentialBatch
from training.policies.schedule import CosineSchedule, ISchedule
from training.stats import StepStats

@runtime_checkable
class ITrainerLayer(Protocol):
    """Layer Decorator Contract (lambda: P -> P): Instruments or wraps trainer execution."""
    def __call__(self, inner: "ITrainer") -> "ITrainer": ...

@runtime_checkable
class ITrainer(Protocol):
    """Core Primitive: Operational boundary for model optimization and evaluation."""
    def fit(
        self,
        cases: Sequence[DecisionCase],
        epochs: int = 1,
        val_cases: Optional[Sequence[DecisionCase]] = None,
    ) -> Iterator[StepStats]: ...

    def evaluate(self, cases: Sequence[DecisionCase]) -> float: ...
    def __or__(self, layer: ITrainerLayer) -> "ITrainer": ...

class EpochTrainer(ITrainer):
    """Core Primitive Implementation: Supervised training loop over domain decision cases."""

    def __init__(
        self,
        model: Optional[IModel] = None,
        batch_size: int = 16,
        base_lr: float = 2e-5,
        schedule: Optional[ISchedule] = None,
        batching: Optional[IBatch] = None,
        eval_batching: Optional[IBatch] = None,
        warmup_ratio: float = 0.05,
    ):
        self.model = model or TorchModel()
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.schedule = schedule or CosineSchedule(warmup_ratio=warmup_ratio)
        self.batching = batching or BucketedBatch()
        self.eval_batching = eval_batching or SequentialBatch()

    def evaluate(self, cases: Sequence[DecisionCase]) -> float:
        """Evaluates model performance over candidate cases."""
        batches = list(self.eval_batching.batch(cases, batch_size=self.batch_size))
        total = len(batches)
        total_loss = sum(self.model.step(b, backward=False) for b in batches)
        return total_loss / max(1, total)

    def fit(
        self,
        cases: Sequence[DecisionCase],
        epochs: int = 1,
        val_cases: Optional[Sequence[DecisionCase]] = None,
    ) -> Iterator[StepStats]:
        items = list(cases)
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size if items else 0
        total_steps = epochs * total_batches

        for ep in range(1, epochs + 1):
            total_loss = 0.0
            batches = list(self.batching.batch(items, batch_size=self.batch_size))

            for n, batch in enumerate(batches):
                step = (ep - 1) * total_batches + (n + 1)
                cur_lr = self.schedule.get_lr(step, total_steps, self.base_lr)

                t0 = time.perf_counter()
                loss = self.model.step(batch, backward=True, lr=cur_lr)
                dt = (time.perf_counter() - t0) * 1000.0
                total_loss += loss

                if n + 1 == total_batches:
                    self.model.flush()

                is_val_cadence = val_cases and (n + 1 == total_batches or n + 1 == (total_batches // 2))
                val_loss = self.evaluate(val_cases) if is_val_cadence else None

                yield StepStats(
                    epoch=ep,
                    total_epochs=epochs,
                    step=step,
                    total_steps=total_steps,
                    loss=total_loss / (n + 1),
                    elapsed_ms=dt,
                    lr=cur_lr,
                    val_loss=val_loss,
                    batch=n + 1,
                    total_batches=total_batches,
                )

    def __or__(self, layer: ITrainerLayer) -> "ITrainer":
        return layer(self)


// ============================================================================
// FILE: training/policies/batch.py (29 code lines, 36 total)
// ============================================================================

import itertools
import random
from typing import Iterator, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase

@runtime_checkable
class IBatch(Protocol):
    """Policy contract for partitioning and ordering domain cases into batches."""
    def batch(self, cases: Sequence[DecisionCase], batch_size: int = 16) -> Iterator[Sequence[DecisionCase]]: ...

def estimate_case_length(case: DecisionCase) -> int:
    """Pure Transform: Estimates token footprint of a case for bucketed batching."""
    state_len = len(case.state) // 4 if isinstance(case.state, str) else 30
    opt_len = sum(len(q.options) * 5 if hasattr(q, "options") and q.options else 10 for q in case.questions)
    return state_len + opt_len

class BucketedBatch(IBatch):
    """Batches cases sorted by length heuristic to minimize sequence padding waste."""

    def __init__(self, shuffle: bool = True):
        self.shuffle = shuffle

    def batch(self, cases: Sequence[DecisionCase], batch_size: int = 16) -> Iterator[Sequence[DecisionCase]]:
        if not cases:
            return
        sorted_cases = sorted(cases, key=estimate_case_length)
        chunks = list(itertools.batched(sorted_cases, batch_size))
        if self.shuffle:
            random.shuffle(chunks)
        yield from chunks

class SequentialBatch(IBatch):
    """Deterministic linear chunking without reordering, ideal for evaluation."""

    def batch(self, cases: Sequence[DecisionCase], batch_size: int = 16) -> Iterator[Sequence[DecisionCase]]:
        yield from itertools.batched(cases, batch_size)


// ============================================================================
// FILE: training/policies/schedule.py (27 code lines, 33 total)
// ============================================================================

import math
from typing import Protocol, runtime_checkable

@runtime_checkable
class ISchedule(Protocol):
    """Policy Contract: determines parameter learning rate per step given total step budget."""
    def get_lr(self, step: int, total_steps: int, base_lr: float) -> float: ...

class CosineSchedule(ISchedule):
    """Linear warmup followed by cosine annealing decay."""

    def __init__(self, warmup_ratio: float = 0.05, min_lr_ratio: float = 0.1):
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio

    def get_lr(self, step: int, total_steps: int, base_lr: float) -> float:
        if total_steps <= 0:
            return base_lr
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        if step < warmup_steps:
            return base_lr * (float(step) / float(warmup_steps))
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 0:
            return base_lr
        progress = min(max(float(step - warmup_steps) / float(decay_steps), 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay)

class ConstantSchedule(ISchedule):
    """Preserves base learning rate across all steps unconditionally."""

    def get_lr(self, step: int, total_steps: int, base_lr: float) -> float:
        return base_lr


// ============================================================================
// FILE: training/stats.py (22 code lines, 24 total)
// ============================================================================

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class StepStats:
    """Immutable domain DTO for step-level optimization metrics."""
    epoch: int
    total_epochs: int
    step: int
    total_steps: int
    loss: float
    elapsed_ms: float
    lr: float = 0.0
    val_loss: float | None = None
    batch: int = 0
    total_batches: int = 0

@dataclass(frozen=True, slots=True)
class EvalStats:
    """Immutable domain DTO for evaluation and holdout progress telemetry."""
    step: int
    total_steps: int
    loss: float
    elapsed_sec: float
    vram_gb: float = 0.0


// ============================================================================
// FILE: training/telemetry/checkpointing_layer.py (53 code lines, 61 total)
// ============================================================================

from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

from data.case import DecisionCase
from training.epoch_trainer import ITrainer, ITrainerLayer
from training.stats import StepStats

class CheckpointingLayer(ITrainer):
    """Layer Decorator (lambda: P -> P): Sovereign saving telemetry at epoch-end."""

    def __init__(
        self,
        inner: Optional[ITrainer] = None,
        output_path: Optional[Union[str, Path]] = None,
    ):
        self.inner = inner
        self.output_path = Path(output_path) if output_path else None
        self.best_val_loss: float = float("inf")

    @property
    def model(self):
        return getattr(self.inner, "model", None)

    def __call__(self, inner: ITrainer) -> "CheckpointingLayer":
        self.inner = inner
        return self

    def fit(
        self,
        cases: Sequence[DecisionCase],
        epochs: int = 1,
        val_cases: Optional[Sequence[DecisionCase]] = None,
    ) -> Iterator[StepStats]:
        if not self.inner:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        model = self.model
        if self.output_path and model is None:
            raise RuntimeError("CheckpointingLayer cannot locate model in trainer pipeline.")
        for s in self.inner.fit(cases, epochs=epochs, val_cases=val_cases):
            if self.output_path and model:
                from model.weights import WeightMeta
                target = self.output_path
                meta = WeightMeta(epoch=s.epoch, train_loss=s.loss, val_loss=s.val_loss)
                p_target = Path(target)
                if s.val_loss is not None and s.val_loss < self.best_val_loss:
                    self.best_val_loss = s.val_loss
                    model.save(target, meta=meta)
                if s.batch == s.total_batches:
                    epoch_target = p_target if (p_target.suffix or p_target.parent != Path(".")) else f"{target}_v{s.epoch}"
                    model.save(epoch_target, meta=meta)
                    if s.val_loss is None:
                        model.save(target, meta=meta)
            yield s

    def evaluate(self, cases: Sequence[DecisionCase]) -> float:
        if self.inner is None:
            raise RuntimeError("CheckpointingLayer has no inner trainer.")
        return self.inner.evaluate(cases)

    def __or__(self, layer: ITrainerLayer) -> ITrainer:
        return layer(self)


// ============================================================================
// FILE: training/telemetry/metrics_layer.py (75 code lines, 89 total)
// ============================================================================

import time
from typing import Iterator, Optional, Protocol, Sequence, runtime_checkable
from rich.console import Console
from data.case import DecisionCase
from model.device import resolve_vram_gb
from training.stats import StepStats
from training.epoch_trainer import ITrainer, ITrainerLayer

@runtime_checkable
class ILogSink(Protocol):
    """Opinion Sub-Primitive: Stream sink for metric log messages."""
    def write(self, message: str) -> None: ...

class ConsoleSink(ILogSink):
    """Rich stdout logging sink with markup support."""
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def write(self, message: str) -> None:
        self.console.print(message)

class MetricsLayer(ITrainer):
    """Layer Decorator (lambda: P -> P): Real-time logging over the StepStats stream."""

    def __init__(
        self,
        inner: Optional[ITrainer] = None,
        sink: Optional[ILogSink] = None,
        log_interval_sec: float = 10.0,
    ):
        self.inner = inner
        self.sink = sink or ConsoleSink()
        self.log_interval_sec = log_interval_sec

    @property
    def model(self):
        return getattr(self.inner, "model", None)

    def __call__(self, inner: ITrainer) -> "MetricsLayer":
        self.inner = inner
        return self

    def fit(
        self,
        cases: Sequence[DecisionCase],
        epochs: int = 1,
        val_cases: Optional[Sequence[DecisionCase]] = None,
    ) -> Iterator[StepStats]:
        if self.inner is None:
            raise RuntimeError("MetricsLayer has no inner trainer.")
        start_time = time.perf_counter()
        last_log = start_time
        for s in self.inner.fit(cases, epochs=epochs, val_cases=val_cases):
            now = time.perf_counter()
            is_epoch_end = (s.batch == s.total_batches)
            is_checkpoint = (s.val_loss is not None)
            is_interval = (now - last_log >= self.log_interval_sec)

            if is_epoch_end or is_checkpoint or is_interval:
                el_s = int(now - start_time)
                el_str = f"{el_s // 60:02d}:{el_s % 60:02d}"
                rem_s = int((el_s / max(1, s.step)) * max(0, s.total_steps - s.step))
                eta_str = f"{rem_s // 60:02d}:{rem_s % 60:02d}"
                _v = resolve_vram_gb()
                vram = f" | VRAM: {_v:.1f}GB" if _v is not None else ""
                val = f" | [bold green]Val Loss: {s.val_loss:.4f}[/bold green]" if s.val_loss is not None else ""

                if is_epoch_end:
                    msg = f"[bold cyan]Epoch {s.epoch}/{s.total_epochs} Complete[/bold cyan] | [green]Train Loss: {s.loss:.4f}[/green]{val}{vram} | [dim]{s.elapsed_ms:.0f}ms/step[/dim] | Total: {el_str}"
                elif is_checkpoint:
                    pct = int((s.batch / max(1, s.total_batches)) * 100)
                    msg = f"[cyan]Epoch {s.epoch}/{s.total_epochs}[/cyan] Checkpoint ({pct}%) | [green]Train: {s.loss:.4f}[/green]{val}{vram} | [dim]Elapsed: {el_str} | ETA: {eta_str}[/dim]"
                else:
                    pct = int((s.step / max(1, s.total_steps)) * 100)
                    lr = f" | [cyan]LR: {s.lr:.2e}[/cyan]" if s.lr > 0 else ""
                    msg = f"[cyan]Epoch {s.epoch}/{s.total_epochs}[/cyan] | Step [bold]{s.step}[/bold]/{s.total_steps} ({pct}%) | [green]Loss: {s.loss:.4f}[/green]{lr}{vram} | [dim]{s.elapsed_ms:.0f}ms/step[/dim] | [dim]Elapsed: {el_str} | ETA: {eta_str}[/dim]"

                self.sink.write(msg)
                last_log = now

            yield s

    def evaluate(self, cases: Sequence[DecisionCase]) -> float:
        if self.inner is None:
            raise RuntimeError("MetricsLayer has no inner trainer.")
        return self.inner.evaluate(cases)

    def __or__(self, layer: ITrainerLayer) -> ITrainer:
        return layer(self)


