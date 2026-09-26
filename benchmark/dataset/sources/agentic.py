import json
from typing import Any, Dict, List
from benchmark.dataset.dataset import ADAPTED_DIR, IDataSource

class AgenticSource(IDataSource):
    name = "agentic"
    display_name = "1. Agentic Decisions"
    category_labels: Dict[str, str] = {
        "agent_tool_routing": "Agent Tool Routing (100)",
        "negative_constraints": "Negative Constraints (100)",
        "safety_guardrails": "Safety & Guardrails (100)",
        "triage_incident": "Incident Triage (100)",
    }

    def extract(self) -> List[Dict[str, Any]]:
        raw_path = ADAPTED_DIR / "agentic.json"
        with open(raw_path, "r", encoding="utf-8") as f:
            return json.load(f)

__all__ = ["AgenticSource"]
