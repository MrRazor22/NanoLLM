import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

DEFAULT_BASELINES_PATH = Path(__file__).resolve().parent / "baselines" / "competitor_cache.json"

@dataclass(frozen=True)
class BaselineEntry:
    overall: Optional[float] = None
    slices: Dict[str, float] = field(default_factory=dict)

class ConsoleScorecardRenderer:
    """ATA Injected Presentation Policy (π): Formats and renders competitive matrix to the console."""

    def __init__(self, label_w: int = 34, count_w: int = 7, comp_w: int = 14, delta_w: int = 15):
        self.label_w, self.count_w, self.comp_w, self.delta_w = label_w, count_w, comp_w, delta_w

    def format_row(self, label: str, count: str, cand_val: str, comps: Dict[str, Optional[float]], competitors: List[str], delta_val: Optional[str] = None) -> str:
        cols = [f"{label:{self.label_w}s}", f"{count:>{self.count_w}s}", cand_val]
        valid_comps = []
        for c in competitors:
            val = comps.get(c)
            if val is not None:
                cols.append(f"{val * 100:>{self.comp_w - 1}.1f}%")
                valid_comps.append(val)
            else:
                cols.append(f"{'-':>{self.comp_w}s}")
        if competitors:
            cols.append(delta_val if delta_val is not None else "-")
        return " | ".join(cols)

    def render(self, tracks_data: Dict[str, Any], baselines: Dict[str, Dict[str, BaselineEntry]], p50s: Dict[str, float], track: str, meta: Dict[str, Any], model_name: str) -> None:
        competitors = [c for tk in tracks_data for c in baselines.get(tk, {}) if c not in []]
        competitors = list(dict.fromkeys(competitors))
        cand_w = max(len(model_name), 10)

        hdr = [f"{'Benchmark Track / Sub-Slice':{self.label_w}s}", f"{'Count':>{self.count_w}s}", f"{model_name:>{cand_w}s}"]
        for c in competitors: hdr.append(f"{c:>{self.comp_w}s}")
        if competitors: hdr.append(f"{'Delta vs Best':>{self.delta_w}s}")

        header_line = " | ".join(hdr)
        sep, dash = "=" * len(header_line), "-" * len(header_line)
        print(f"\n{sep}\n{f'{model_name.upper()} COMPETITIVE BENCHMARK: {track.upper()}':^{len(header_line)}}\n{sep}\n{header_line}\n{dash}")

        grand_c, grand_t, lats = 0, 0, []
        for trk_key, rep in tracks_data.items():
            t_meta = meta.get(trk_key, {})
            t_title = t_meta.get("display_name") or trk_key.replace("_", " ").title()
            t_base = baselines.get(trk_key, {})
            comp_overall = {c: t_base[c].overall for c in competitors if c in t_base}
            tot_c, tot_q = rep.get("total_correct", 0), rep.get("total_questions", 0)
            grand_c += tot_c; grand_t += tot_q
            if "p50_ms" in rep: lats.append(rep["p50_ms"])

            cand_acc = rep.get("overall_acc", 0.0)
            valid = [v for v in comp_overall.values() if v is not None]
            d_str = f"{(cand_acc - max(valid)) * 100:>+{self.delta_w - 1}.1f}%" if valid else f"{'Baseline':>{self.delta_w}s}"
            print(self.format_row(t_title, str(tot_q), f"{cand_acc * 100:>{cand_w - 1}.1f}%", comp_overall, competitors, d_str))

            cat_labels = t_meta.get("category_labels", {})
            for cat, cat_acc in sorted(rep.get("by_cat", {}).items()):
                cat_title = cat_labels.get(cat, cat.replace("_", " ").title())
                comp_slice = {c: t_base[c].slices.get(cat) for c in competitors if c in t_base}
                valid_sl = [v for v in comp_slice.values() if v is not None]
                sl_d = f"{(cat_acc - max(valid_sl)) * 100:>+{self.delta_w - 1}.1f}%" if valid_sl else f"{'Baseline':>{self.delta_w}s}"
                print(self.format_row(f"  - {cat_title}", str(rep.get("cat_counts", {}).get(cat, "-")), f"{cat_acc * 100:>{cand_w - 1}.1f}%", comp_slice, competitors, sl_d))
            print(dash)

        if track == "all" and grand_t > 0:
            macro_acc = (grand_c / grand_t) * 100.0
            comp_macro = {c: float(np.mean(overs)) for c in competitors if (overs := [baselines.get(tk, {}).get(c).overall for tk in tracks_data if c in baselines.get(tk, {}) and baselines[tk][c].overall is not None])}
            val_macros = list(comp_macro.values())
            d_mac = f"{macro_acc - max(val_macros) * 100:>+{self.delta_w - 1}.1f}%" if val_macros else f"{'Baseline':>{self.delta_w}s}"
            print(self.format_row("OVERALL MACRO ACCURACY", str(grand_t), f"{macro_acc:>{cand_w - 1}.1f}%", comp_macro, competitors, d_mac) + f"\n{dash}")

            cand_p50 = f"{float(np.mean(lats)):>{cand_w - 4}.1f} ms" if lats else f"{'-':>{cand_w}s}"
            val_lats = [cl for c in competitors if (cl := p50s.get(c)) is not None]
            d_lat = f"{float(np.mean(lats)) - min(val_lats):>+{self.delta_w - 4}.1f} ms" if (lats and val_lats) else f"{'-':>{self.delta_w}s}"
            lat_cols = [f"{'P50 INFERENCE LATENCY':{self.label_w}s}", f"{'-':>{self.count_w}s}", cand_p50]
            for c in competitors: lat_cols.append(f"{p50s[c]:>{self.comp_w - 4}.1f} ms" if c in p50s else f"{'-':>{self.comp_w}s}")
            if competitors: lat_cols.append(d_lat)
            print(" | ".join(lat_cols) + f"\n{sep}\n")

class CompetitorScorecard:
    """ATA Root Primitive (P): Computes and coordinates competitive benchmark scorecards."""

    def __init__(self, baselines_path: Optional[Path] = None, competitor_cache: Optional[Dict[str, Any]] = None, renderer: Optional[ConsoleScorecardRenderer] = None):
        p = baselines_path or DEFAULT_BASELINES_PATH
        self._raw_cache = competitor_cache if competitor_cache is not None else (json.load(open(p, "r", encoding="utf-8")) if p.exists() else {})
        self.baselines: Dict[str, Dict[str, BaselineEntry]] = {
            trk: {comp: BaselineEntry(val.get("overall"), val.get("slices", {})) for comp, val in tval.items() if isinstance(val, dict)}
            for trk, tval in self._raw_cache.items() if isinstance(tval, dict) and trk != "p50_latencies_ms"
        }
        self.p50_latencies: Dict[str, float] = self._raw_cache.get("p50_latencies_ms", {})
        self.renderer = renderer or ConsoleScorecardRenderer()

    def render(self, reports: Dict[str, Any], track: str = "all", metadata: Optional[Dict[str, Any]] = None, model_name: str = "NanoLLM") -> None:
        tracks_data = reports if track == "all" else {track: reports}
        self.renderer.render(tracks_data, self.baselines, self.p50_latencies, track, metadata or {}, model_name)

__all__ = ["BaselineEntry", "ConsoleScorecardRenderer", "CompetitorScorecard"]

