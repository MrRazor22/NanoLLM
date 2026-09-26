import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

DEFAULT_BASELINES_PATH = Path(__file__).resolve().parent / "baselines" / "competitor_cache.json"

@dataclass(frozen=True)
class BaselineEntry:
    """Pure DTO State ($S$): Immutable representation of competitor baseline performance."""
    overall: Optional[float] = None
    slices: Dict[str, float] = field(default_factory=dict)

class CompetitorScorecard:
    """ATA Primitive: Computes and renders competitive benchmark scorecards comparing candidate model results against baseline competitors."""

    def __init__(self, baselines_path: Optional[Path] = None, competitor_cache: Optional[Dict[str, Any]] = None):
        if competitor_cache is not None:
            self._raw_cache = competitor_cache
        else:
            p = baselines_path or DEFAULT_BASELINES_PATH
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    self._raw_cache = json.load(f)
            else:
                self._raw_cache = {}

        self.baselines: Dict[str, Dict[str, BaselineEntry]] = self._parse_baselines(self._raw_cache)
        self.p50_latencies: Dict[str, float] = self._raw_cache.get("p50_latencies_ms", {})

    @staticmethod
    def _parse_baselines(raw: Dict[str, Any]) -> Dict[str, Dict[str, BaselineEntry]]:
        """Pure Transform ($T$): Parses raw baseline JSON into clean typed domain states."""
        parsed: Dict[str, Dict[str, BaselineEntry]] = {}
        for trk_key, trk_val in raw.items():
            if not isinstance(trk_val, dict) or trk_key == "p50_latencies_ms":
                continue
            parsed[trk_key] = {}
            for comp_name, comp_val in trk_val.items():
                if isinstance(comp_val, dict):
                    parsed[trk_key][comp_name] = BaselineEntry(
                        overall=comp_val.get("overall"),
                        slices=comp_val.get("slices", {}),
                    )
        return parsed

    def render(
        self,
        reports: Dict[str, Any],
        track: str = "all",
        metadata: Optional[Dict[str, Any]] = None,
        model_name: str = "NanoLLM",
    ) -> None:
        meta = metadata or {}
        tracks_data = reports if track == "all" else {track: reports}

        # Discover competitors cleanly from typed baselines
        competitors: List[str] = []
        for trk_key in tracks_data:
            for comp_name in self.baselines.get(trk_key, {}):
                if comp_name not in competitors:
                    competitors.append(comp_name)

        label_width = 34
        count_width = 7
        candidate_width = max(len(model_name), 10)
        comp_col_width = 14
        delta_width = 15

        header_cols = [
            f"{'Benchmark Track / Sub-Slice':{label_width}s}",
            f"{'Count':>{count_width}s}",
            f"{model_name:>{candidate_width}s}",
        ]
        for c in competitors:
            header_cols.append(f"{c:>{comp_col_width}s}")
        if competitors:
            header_cols.append(f"{'Delta vs Best':>{delta_width}s}")

        header_line = " | ".join(header_cols)
        sep = "=" * len(header_line)
        dash = "-" * len(header_line)

        print("\n" + sep)
        title = f"{model_name.upper()} COMPETITIVE BENCHMARK: {track.upper()}"
        print(f"{title:^{len(header_line)}}\n" + sep)
        print(header_line + "\n" + dash)

        def _format_row(label: str, count: str, cand_acc: float, comp_vals: Dict[str, Optional[float]]) -> str:
            cols = [
                f"{label:{label_width}s}",
                f"{count:>{count_width}s}",
                f"{cand_acc * 100:>{candidate_width - 1}.1f}%",
            ]
            valid_comps = []
            for c in competitors:
                val = comp_vals.get(c)
                if val is not None:
                    cols.append(f"{val * 100:>{comp_col_width - 1}.1f}%")
                    valid_comps.append(val)
                else:
                    cols.append(f"{'-':>{comp_col_width}s}")

            if competitors:
                if valid_comps:
                    delta = (cand_acc - max(valid_comps)) * 100
                    cols.append(f"{delta:>+{delta_width - 1}.1f}%")
                else:
                    cols.append(f"{'Baseline':>{delta_width}s}")
            return " | ".join(cols)

        grand_c, grand_t, lats = 0, 0, []
        for trk_key, rep in tracks_data.items():
            trk_meta = meta.get(trk_key, {})
            trk_title = trk_meta.get("display_name") or trk_key.replace("_", " ").title()
            cat_labels = trk_meta.get("category_labels", {})

            trk_base = self.baselines.get(trk_key, {})
            comp_overall = {c: trk_base[c].overall for c in competitors if c in trk_base}

            tot_c = rep.get("total_correct", 0)
            tot_q = rep.get("total_questions", 0)
            grand_c += tot_c
            grand_t += tot_q
            if "p50_ms" in rep:
                lats.append(rep["p50_ms"])

            print(_format_row(trk_title, str(tot_q), rep.get("overall_acc", 0.0), comp_overall))

            for cat, cat_acc in sorted(rep.get("by_cat", {}).items()):
                cat_title = cat_labels.get(cat, cat.replace("_", " ").title())
                comp_slice = {c: trk_base[c].slices.get(cat) for c in competitors if c in trk_base}
                cat_count = str(rep.get("cat_counts", {}).get(cat, "-"))
                print(_format_row(f"  - {cat_title}", cat_count, cat_acc, comp_slice))
            print(dash)

        if track == "all" and grand_t > 0:
            macro_acc = (grand_c / grand_t) * 100.0
            comp_macro_vals = {}
            for c in competitors:
                c_overs = [self.baselines.get(tk, {}).get(c).overall for tk in tracks_data if c in self.baselines.get(tk, {}) and self.baselines[tk][c].overall is not None]
                comp_macro_vals[c] = float(np.mean(c_overs)) if c_overs else None

            summary_cols = [
                f"{'OVERALL MACRO ACCURACY':{label_width}s}",
                f"{str(grand_t):>{count_width}s}",
                f"{macro_acc:>{candidate_width - 1}.1f}%",
            ]
            valid_macros = []
            for c in competitors:
                mv = comp_macro_vals.get(c)
                if mv is not None:
                    summary_cols.append(f"{mv * 100:>{comp_col_width - 1}.1f}%")
                    valid_macros.append(mv * 100)
                else:
                    summary_cols.append(f"{'-':>{comp_col_width}s}")

            if competitors:
                d_macro = f"{macro_acc - max(valid_macros):>+{delta_width - 1}.1f}%" if valid_macros else f"{'Baseline':>{delta_width}s}"
                summary_cols.append(d_macro)
            print(" | ".join(summary_cols) + "\n" + dash)

            cand_p50_str = f"{float(np.mean(lats)):>{candidate_width - 4}.1f} ms" if lats else f"{'-':>{candidate_width}s}"
            lat_cols = [
                f"{'P50 INFERENCE LATENCY':{label_width}s}",
                f"{'-':>{count_width}s}",
                cand_p50_str,
            ]
            valid_lats = []
            for c in competitors:
                clat = self.p50_latencies.get(c)
                if clat is not None:
                    lat_cols.append(f"{clat:>{comp_col_width - 4}.1f} ms")
                    valid_lats.append(clat)
                else:
                    lat_cols.append(f"{'-':>{comp_col_width}s}")
            if competitors:
                if lats and valid_lats:
                    d_lat = f"{float(np.mean(lats)) - min(valid_lats):>+{delta_width - 4}.1f} ms"
                else:
                    d_lat = f"{'-':>{delta_width}s}"
                lat_cols.append(d_lat)
            print(" | ".join(lat_cols) + "\n" + sep + "\n")

__all__ = ["BaselineEntry", "CompetitorScorecard"]
