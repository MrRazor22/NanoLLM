from typing import Dict, List, Sequence
from nanollm.inference.layers.profiling import DecisionEngineLayer
from nanollm.inference.schema import Choice, DecisionResult, Question

class HierarchicalLayer(DecisionEngineLayer):
    def decide(self, state: str, questions: Sequence[Question]) -> DecisionResult:
        has_clusters = any(isinstance(q, Choice) and q.clusters for q in questions)
        if not has_clusters:
            return self.inner.decide(state, questions)

        total_latency = 0.0
        final_answers = {}
        for q in questions:
            if not (isinstance(q, Choice) and q.clusters):
                res = self.inner.decide(state, [q])
                final_answers[q.name] = res.answers[q.name]
                total_latency += res.latency_ms
                continue

            cluster_options = {c: f"Inquiries and operations regarding {c}" for c in q.clusters.keys()}
            pass1_q = Choice(
                name=f"{q.name}_cluster",
                options=cluster_options,
                instruction=f"Which high-level category applies: {q.instruction}" if q.instruction else "Which domain applies?",
            )
            res1 = self.inner.decide(state, [pass1_q])
            winning_cluster = res1.answers[f"{q.name}_cluster"].choice

            sub_options = q.clusters.get(winning_cluster, q.options)
            pass2_q = Choice(
                name=q.name,
                options=sub_options,
                instruction=q.instruction,
            )
            res2 = self.inner.decide(state, [pass2_q])
            final_answers[q.name] = res2.answers[q.name]
            total_latency += res1.latency_ms + res2.latency_ms

        return DecisionResult(answers=final_answers, latency_ms=total_latency)
