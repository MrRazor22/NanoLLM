import logging
import time
from typing import Mapping, Protocol

from inference.decision_engine import IDecisionEngine
from inference.options import ChoiceOption

logger = logging.getLogger(__name__)

class IPlayground(Protocol):
    def start(self, engines: Mapping[str, IDecisionEngine]) -> None: ...

class Playground(IPlayground):
    def start(self, engines: Mapping[str, IDecisionEngine]) -> None:
        if not engines:
            raise ValueError("No engines provided to playground.")

        names = ", ".join(engines.keys())
        print(f"\nPiqo Playground [{names}]. Enter context (or 'q' to quit):")

        while True:
            try:
                ctx = input("\nContext: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not ctx or ctx.lower() in ("q", "quit", "exit"):
                break
            try:
                raw_choices = input("Choices (comma-separated): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            opts = [o.strip() for o in raw_choices.split(",") if o.strip()]
            if not opts:
                continue

            opt = ChoiceOption(instruction=ctx, criteria={o: "" for o in opts})
            print(f"\nState: {ctx}\n" + "-" * 60)
            for name, eng in engines.items():
                try:
                    t0 = time.perf_counter()
                    ans = eng.decide(ctx, opt)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    print(f"[{name:<10}] {ans} | Lat: {dt_ms:.1f}ms")
                except Exception as e:
                    print(f"[{name:<10}] Error: {e}")
            print("-" * 60)
