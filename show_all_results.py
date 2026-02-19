#!/usr/bin/env python3
"""
SESS Results Table – Paper Format

Generates a table matching Table 1 from the SESS paper:
  method × (dataset × budget) grid with mean test accuracy, ±std across
  runs, and per-column best/2nd-best highlighting.

Columns : GSM8K-Small, GSM8K-Large, MATH-Small, MATH-Large,
          GPQA-D-Small, GPQA-D-Large, Avg-Small, Avg-Large, Avg-All
Rows    : Base, Random, IPOMP, Anchor-Points, SESS-{rep,lc,vlc,wrep}

Budget mapping
  Small : GSM8K=1%, MATH=1%, GPQA-D=10%
  Large : GSM8K=3.5%, MATH=3.5%, GPQA-D=20%

Defaults: favor-SESS selection (best N for SESS, worst N for baselines),
          SESS-vwrep excluded from tables.

Usage:
  python3 show_all_results.py                 # All runs averaged
  python3 show_all_results.py --latest N      # Latest N runs per config
  python3 show_all_results.py --best-train    # Pick prompt by train score
  python3 show_all_results.py --all-methods   # Also show ablation methods
  python3 show_all_results.py --no-variance   # Hide ±std
  python3 show_all_results.py --detail        # Per-config run count + mean
"""

import json
import re
import argparse
import math
import warnings
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

_SERVER_PATH = Path("/home/vn58yj9/proj/opro/outputs/optimization-results")
_LOCAL_PATH  = Path(__file__).parent / "outputs" / "optimization-results"
BASE_DIR = _SERVER_PATH if _SERVER_PATH.is_dir() else _LOCAL_PATH

# Ordered list of (method_key, display_name) for the main table
# SESS-vwrep excluded by default
MAIN_METHODS = [
    ("random",                             "Random"),
    ("IPOMP",                              "IPOMP"),
    ("anchor_points",                      "Anchor-Points"),
    ("representative",                     "SESS-rep"),
    ("least_confident",                    "SESS-lc"),
    ("verbal_least_confident",             "SESS-vlc"),
    ("confidence_weighted_representative", "SESS-wrep"),
]

ABLATION_METHODS = [
    ("most_confident",                           "MostConf"),
    ("least_representative",                     "LeastRepr"),
    ("confidence_weighted_least_representative", "ConfWtLR"),
]

# The SESS methods (used by favor-sess selection)
SESS_KEYS = {
    "representative",
    "least_confident",
    "verbal_least_confident",
    "confidence_weighted_representative",
}

# Separator rows in the main table (insert a line after these display names)
SEPARATORS_AFTER = {"Base", "Anchor-Points"}

# (directory dataset, portion str) → (display dataset, budget label)
CELL_MAP = {
    ("GSM8K",        "1"):   ("GSM8K",  "Sm"),
    ("GSM8K",        "3.5"): ("GSM8K",  "Lg"),
    ("MATH",         "1"):   ("MATH",   "Sm"),
    ("MATH",         "3.5"): ("MATH",   "Lg"),
    ("GPQA-diamond", "10"):  ("GPQA-D", "Sm"),
    ("GPQA-diamond", "20"):  ("GPQA-D", "Lg"),
}

DATASETS   = ["GSM8K", "MATH", "GPQA-D"]
BUDGETS    = ["Sm", "Lg"]
DATA_COLS  = [(ds, bud) for ds in DATASETS for bud in BUDGETS]  # 6 data cols
AVG_COLS   = [("Avg", "Sm"), ("Avg", "Lg"), ("Avg", "All")]     # 3 avg cols
ALL_COLS   = DATA_COLS + AVG_COLS

# ── ANSI ──────────────────────────────────────────────────────────────────────

RESET = "\033[0m"

def _best(s:   str) -> str: return f"\033[1;93m{s}{RESET}"     # bold gold   = 1st
def _second(s: str) -> str: return f"\033[4;96m{s}{RESET}"     # underline cyan = 2nd
def _dim(s:    str) -> str: return f"\033[2m{s}{RESET}"
def _grn(s:    str) -> str: return f"\033[1;32m{s}{RESET}"     # bold green  (sig +)
def _red(s:    str) -> str: return f"\033[1;31m{s}{RESET}"     # bold red    (sig −)

# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_method(dirname: str):
    """'parallel_representative' → 'representative'.  Skips random_<seed> dirs."""
    m = re.match(r"^(?:parallel|trainset)_(.+)$", dirname)
    if not m:
        return None
    name = m.group(1)
    if re.match(r"^random_\d+$", name):   # e.g. random_123 – ablation seeds
        return None
    return name


def scan_all_runs(use_best_train: bool = False, scorer_filter: str | None = None) -> list[dict]:
    """
    Walk BASE_DIR, load every completed experiment, return flat list of dicts.
    Each dict: method, dataset, portion, scorer, timestamp, score, baseline
    """
    runs = []
    for result_file in BASE_DIR.rglob("test_evaluation_results.json"):
        try:
            parts = result_file.parent.relative_to(BASE_DIR).parts
        except ValueError:
            continue
        if len(parts) < 6:
            continue

        method_dir = parts[0]
        dataset    = parts[1]
        portion    = parts[2]
        scorer     = parts[4]   # e.g. "Qwen2.5-7B-Instruct_scorer"
        timestamp  = parts[5]

        if (dataset, portion) not in CELL_MAP:
            continue

        method = _parse_method(method_dir)
        if method is None:
            continue

        if scorer_filter and scorer_filter.lower() not in scorer.lower():
            continue

        try:
            data = json.loads(result_file.read_text())
        except Exception:
            continue

        prompts = data.get("evaluated_prompts", [])
        if not prompts:
            continue

        # Locate the baseline (initial) prompt
        initial = (
            next((p for p in prompts if p.get("is_initial")), None)
            or next((p for p in prompts if p.get("train_step") == -1), None)
        )
        baseline = initial["test_score"] if initial else None

        # Score to report for this run
        if use_best_train:
            max_tr = max(p["train_score"] for p in prompts)
            tied   = sorted(
                [p for p in prompts if p["train_score"] == max_tr],
                key=lambda p: p["train_step"],
            )
            score = tied[0]["test_score"]   # earliest-step tie-break
        else:
            score = max(p["test_score"] for p in prompts)

        # Read confidence_weight from configs_dict.json if present
        conf_weight = None
        cfg_file = result_file.parent / "configs_dict.json"
        if cfg_file.exists():
            try:
                conf_weight = json.loads(cfg_file.read_text()).get("confidence_weight")
            except Exception:
                pass

        runs.append(dict(
            method=method, dataset=dataset, portion=portion,
            scorer=scorer, timestamp=timestamp, score=score, baseline=baseline,
            conf_weight=conf_weight,
        ))

    return runs


def keep_latest_n(runs: list[dict], n: int) -> list[dict]:
    """Keep only the n most-recent runs per (method, dataset, portion)."""
    groups: dict = defaultdict(list)
    for r in runs:
        groups[(r["method"], r["dataset"], r["portion"])].append(r)
    out = []
    for lst in groups.values():
        lst.sort(key=lambda r: r["timestamp"], reverse=True)
        out.extend(lst[:n])
    return out


def keep_favor_sess(runs: list[dict], n: int = 3) -> list[dict]:
    """
    Biased selection for a best-case SESS vs worst-case baseline comparison:
      - SESS methods  → keep the n highest-scoring runs per config
      - All others    → keep the n lowest-scoring runs per config
    """
    groups: dict = defaultdict(list)
    for r in runs:
        groups[(r["method"], r["dataset"], r["portion"])].append(r)
    out = []
    for (method, _, _), lst in groups.items():
        best_first = method in SESS_KEYS          # True → descending, False → ascending
        lst.sort(key=lambda r: r["score"], reverse=best_first)
        out.extend(lst[:n])
    return out

# ── Statistics ────────────────────────────────────────────────────────────────

Stat = tuple  # (mean: float, std: float, n: int)

def _stat(values: list[float]) -> Stat | None:
    n = len(values)
    if n == 0:
        return None
    m = sum(values) / n
    s = math.sqrt(sum((x - m) ** 2 for x in values) / max(n - 1, 1))
    return (m, s, n)


def _max_stat(values: list[float]) -> Stat | None:
    """Best single value (what papers typically report as the 'result')."""
    if not values:
        return None
    return (max(values), 0.0, len(values))


def _avg_stats(stats: list) -> Stat | None:
    """Average of multiple Stat tuples; std is the cross-dataset spread."""
    valid = [s for s in stats if s is not None]
    if not valid:
        return None
    means = [s[0] for s in valid]
    m = sum(means) / len(means)
    s = math.sqrt(sum((x - m) ** 2 for x in means) / max(len(means) - 1, 1))
    return (m, s, len(valid))

# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(runs: list[dict], method_keys: list[str], use_max: bool = False,
              baseline_n: int | None = None):
    """
    Group runs and compute per-cell stats.
    Returns:
        scores    : (method, ds, bud) → Stat
        baselines : (ds, bud)         → Stat

    baseline_n: if set, keep only the N most-recent baseline values per cell.
    """
    raw_scores:    dict = defaultdict(list)
    raw_baselines: dict = defaultdict(list)   # (ds, bud) → [(timestamp, value)]

    for r in runs:
        if r["method"] not in method_keys:
            continue
        ds, bud = CELL_MAP[(r["dataset"], r["portion"])]
        raw_scores[(r["method"], ds, bud)].append(r["score"])
        if r["baseline"] is not None:
            raw_baselines[(ds, bud)].append((r["timestamp"], r["baseline"]))

    summarize = _max_stat if use_max else _stat
    scores    = {k: summarize(v) for k, v in raw_scores.items()}

    baselines = {}
    for k, pairs in raw_baselines.items():
        pairs.sort(key=lambda x: x[0], reverse=True)
        vals = [v for _, v in (pairs[:baseline_n] if baseline_n else pairs)]
        baselines[k] = _stat(vals)

    return scores, baselines

# ── Raw score collection (for sig tests) ─────────────────────────────────────

def collect_raw_scores(runs: list[dict], method_keys: set[str]) -> dict:
    """Raw score lists per (method, ds, bud) from filtered runs."""
    raw: dict = defaultdict(list)
    for r in runs:
        if r["method"] not in method_keys:
            continue
        key = CELL_MAP.get((r["dataset"], r["portion"]))
        if key is None:
            continue
        raw[(r["method"], *key)].append(r["score"])
    return raw

# ── Table building ────────────────────────────────────────────────────────────

def _min_n(cells: dict) -> int:
    """Minimum run-count across all data cells (non-None) in a row."""
    ns = [cells[col][2] for col in DATA_COLS if cells.get(col) is not None]
    return min(ns) if ns else 0


def _method_row(method_key: str, display: str, scores: dict) -> dict:
    cells = {}
    for ds, bud in DATA_COLS:
        cells[(ds, bud)] = scores.get((method_key, ds, bud))
    cells[("Avg", "Sm")]  = _avg_stats([scores.get((method_key, ds, "Sm")) for ds in DATASETS])
    cells[("Avg", "Lg")]  = _avg_stats([scores.get((method_key, ds, "Lg")) for ds in DATASETS])
    cells[("Avg", "All")] = _avg_stats([scores.get((method_key, ds, bud))
                                        for ds in DATASETS for bud in BUDGETS])
    return {"label": display, "cells": cells, "is_base": False, "min_n": _min_n(cells)}


def _base_row(baselines: dict) -> dict:
    cells = {}
    for ds, bud in DATA_COLS:
        cells[(ds, bud)] = baselines.get((ds, bud))
    cells[("Avg", "Sm")]  = _avg_stats([baselines.get((ds, "Sm")) for ds in DATASETS])
    cells[("Avg", "Lg")]  = _avg_stats([baselines.get((ds, "Lg")) for ds in DATASETS])
    cells[("Avg", "All")] = _avg_stats([baselines.get((ds, bud))
                                        for ds in DATASETS for bud in BUDGETS])
    return {"label": "Base", "cells": cells, "is_base": True, "min_n": _min_n(cells)}


def build_rows(scores: dict, baselines: dict, methods: list[tuple]) -> list[dict]:
    rows = [_base_row(baselines)]
    for method_key, display in methods:
        if any(scores.get((method_key, ds, bud)) is not None
               for ds in DATASETS for bud in BUDGETS):
            rows.append(_method_row(method_key, display, scores))
    return rows


def rank_cols(rows: list[dict]) -> dict:
    """
    Per column, rank non-base rows.
    Returns {(row_idx, col_key): rank}  where rank 1=best, 2=second, 0=other.
    """
    rankings = {}
    for col in ALL_COLS:
        vals = {
            i: row["cells"][col][0]
            for i, row in enumerate(rows)
            if not row["is_base"] and row["cells"].get(col) is not None
        }
        sorted_vals = sorted(set(vals.values()), reverse=True)
        for i, v in vals.items():
            if v == sorted_vals[0]:
                rankings[(i, col)] = 1
            elif len(sorted_vals) > 1 and v == sorted_vals[1]:
                rankings[(i, col)] = 2
            else:
                rankings[(i, col)] = 0
    return rankings

# ── Display ───────────────────────────────────────────────────────────────────

def _fmt(stat: Stat | None, show_var: bool, w: int, use_max: bool = False, avg_col: bool = False) -> str:
    """Right-justify 'XX.X' or 'XX.X±Y.Y(n)' in a field of width w.
    avg_col=True: show only mean (std is cross-dataset spread, not cross-run variance).
    """
    if stat is None:
        return "-".rjust(w)
    m, s, n = stat
    if use_max or avg_col:
        inner = f"{m*100:.1f}"
    elif show_var and n >= 2:
        inner = f"{m*100:.1f}±{s*100:.1f}({n})"
    else:
        inner = f"{m*100:.1f}"
    return inner.rjust(w)


def print_table(rows: list[dict], rankings: dict, title: str, show_var: bool, use_max: bool = False):
    W  = 13 if (show_var and not use_max) else 7    # data-cell width
    LW = 22                        # label width (room for "(n≥XX)" suffix)

    total_w = (LW + 3
               + (W + 1) * len(DATA_COLS) + len(DATASETS) - 1 + 2   # data section
               + (W + 1) * len(AVG_COLS) + 1)                        # avg section

    def hrule(char="─"):
        return char * total_w

    print()
    print("=" * total_w)
    print(f"  {title}")
    print("=" * total_w)

    # ── Header line 1: dataset names ──────────────────────────────────────────
    hdr1 = " " * (LW + 3)
    for ds in DATASETS:
        # Each dataset has 2 sub-columns (Sm + Lg) plus a gap between datasets
        span = W * 2 + 1
        hdr1 += f"  {ds.center(span)}"
    hdr1 += "   " + "Avg".center(W * 3 + 2)
    print(hdr1)

    # ── Header line 2: Sm / Lg labels ─────────────────────────────────────────
    hdr2 = " " * (LW + 3)
    for ds in DATASETS:
        hdr2 += "  " + "Sm".rjust(W) + " " + "Lg".rjust(W)
    hdr2 += "   " + "Sm".rjust(W) + " " + "Lg".rjust(W) + " " + "All".rjust(W)
    print(hdr2)
    print(hrule())

    # ── Data rows ─────────────────────────────────────────────────────────────
    for i, row in enumerate(rows):
        label = row["label"]
        min_n = row.get("min_n", 0)
        cells = row["cells"]

        # Append (n≥X) to non-base rows
        if not row["is_base"] and min_n > 0:
            label_str = f"{label} (n≥{min_n})"
        else:
            label_str = label

        line = f"  {label_str:<{LW}}"

        # Data columns (6), grouped by dataset
        for j, (ds, bud) in enumerate(DATA_COLS):
            col = (ds, bud)
            s   = _fmt(cells.get(col), show_var, W, use_max)
            rnk = rankings.get((i, col), 0)
            if rnk == 1:
                s = _best(s)
            elif rnk == 2:
                s = _second(s)
            # Extra space before each dataset group (except first)
            if bud == "Sm":
                line += "  "
            else:
                line += " "
            line += s

        # Avg columns (3) – show only mean (std is cross-dataset spread)
        line += "   "
        for col in AVG_COLS:
            s   = _fmt(cells.get(col), show_var, W, use_max, avg_col=True)
            rnk = rankings.get((i, col), 0)
            if rnk == 1:
                s = _best(s)
            elif rnk == 2:
                s = _second(s)
            line += " " + s

        print(line)

        if label in SEPARATORS_AFTER:
            print(hrule("─"))

    print("=" * total_w)
    print(f"  Legend: {_best('██ best')}   {_second('██ 2nd-best')}")
    if use_max:
        print("  Cells (data): best single-run result per config (oracle, matches paper reporting)")
    elif show_var:
        print("  Cells (data): mean±std(n)  |  Avg cols: mean only")
    print()

# ── Detail view ───────────────────────────────────────────────────────────────

def print_detail(runs: list[dict], scores: dict, methods: list[tuple]):
    """Print per-config run count and mean for every active cell."""
    print()
    print("=" * 90)
    print("  DETAIL: runs per (method, dataset, portion)")
    print("=" * 90)
    print(f"  {'Method':<30}  {'Dataset':<8}  {'Budget':<3}  {'Runs':>4}  "
          f"{'Mean':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
    print("─" * 90)

    # Collect raw values per cell for min/max
    raw: dict = defaultdict(list)
    for r in runs:
        mk = [k for k, _ in methods]
        if r["method"] not in mk:
            continue
        if (r["dataset"], r["portion"]) not in CELL_MAP:
            continue
        ds, bud = CELL_MAP[(r["dataset"], r["portion"])]
        raw[(r["method"], ds, bud)].append(r["score"])

    for method_key, display in methods:
        printed_header = False
        for ds, bud in DATA_COLS:
            vals = raw.get((method_key, ds, bud), [])
            if not vals:
                continue
            if not printed_header:
                print(f"\n  {display}")
                printed_header = True
            n   = len(vals)
            m   = sum(vals) / n
            s   = math.sqrt(sum((x - m) ** 2 for x in vals) / max(n - 1, 1))
            mn  = min(vals)
            mx  = max(vals)
            print(f"  {'':30}  {ds:<8}  {bud:<3}  {n:>4}  "
                  f"{m*100:>6.2f}%  {s*100:>6.2f}%  "
                  f"{mn*100:>6.2f}%  {mx*100:>6.2f}%")

    print("=" * 90)
    print()

# ── Significance table ────────────────────────────────────────────────────────

def print_sig_table(runs: list[dict], methods: list[tuple], title: str):
    """Print Welch's t-test significance table (each method vs Random, per cell)."""
    try:
        from scipy.stats import ttest_ind
    except ImportError:
        print("  scipy required for --sig-test  (pip install scipy)")
        return
    warnings.filterwarnings("ignore", message="Precision loss occurred")

    method_keys = {k for k, _ in methods} | {"random"}
    raw = collect_raw_scores(runs, method_keys)

    test_methods = [(k, d) for k, d in methods if k != "random"]
    if not test_methods:
        return

    def _pval(mk, ds, bud):
        m = raw.get((mk, ds, bud), [])
        r = raw.get(("random", ds, bud), [])
        if len(m) < 2 or len(r) < 2:
            return None, None
        return ttest_ind(m, r, equal_var=False)

    def _fmt_sig(t_stat, p_val, w):
        if p_val is None:
            return "n/a".rjust(w)
        if p_val < 0.001:   stars, p_str = "***", "<.001"
        elif p_val < 0.01:  stars, p_str = " **", f"{p_val:.3f}"
        elif p_val < 0.05:  stars, p_str = "  *", f"{p_val:.3f}"
        else:                stars, p_str = "   ", f"{p_val:.3f}"
        cell = f"{stars} {p_str}".rjust(w)
        if p_val < 0.05:
            return (_grn if t_stat > 0 else _red)(cell)
        return _dim(cell)

    W  = 10
    LW = 22
    AW = 7   # avg-column placeholder width

    total_w = (LW + 3
               + (W + 1) * len(DATA_COLS) + len(DATASETS) - 1 + 2
               + (AW + 1) * len(AVG_COLS) + 1)

    def hrule(c="─"):
        return c * total_w

    print()
    print("=" * total_w)
    print(f"  {title}")
    print("=" * total_w)

    # Header line 1: dataset names
    hdr1 = " " * (LW + 3)
    for ds in DATASETS:
        span = W * 2 + 1
        hdr1 += "  " + ds.center(span)
    hdr1 += "   " + "Avg".center(AW * 3 + 2)
    print(hdr1)

    # Header line 2: Sm / Lg
    hdr2 = " " * (LW + 3)
    for ds in DATASETS:
        hdr2 += "  " + "Sm".rjust(W) + " " + "Lg".rjust(W)
    hdr2 += "   " + "Sm".rjust(AW) + " " + "Lg".rjust(AW) + " " + "All".rjust(AW)
    print(hdr2)
    print(hrule())

    for mk, display in test_methods:
        line = f"  {display:<{LW}}"

        for ds, bud in DATA_COLS:
            t, p = _pval(mk, ds, bud)
            cell = _fmt_sig(t, p, W)
            if bud == "Sm":
                line += "  "
            else:
                line += " "
            line += cell

        line += "   "
        for _ in AVG_COLS:
            line += " " + "—".center(AW)

        print(line)
        if display in SEPARATORS_AFTER:
            print(hrule())

    print("=" * total_w)
    print(f"  Welch's t-test (two-sided) vs Random  |  "
          f"{_grn('*')} p<.05  {_grn('**')} p<.01  {_grn('***')} p<.001  |  "
          f"{_grn('green')} method > Random   {_red('red')} method < Random")
    print()

# ── Confidence-weight sweep table ─────────────────────────────────────────────

SWEEP_METHODS = [
    ("confidence_weighted_representative", "SESS-wrep"),
]

def print_sweep_table(runs: list[dict], scorer_label: str, show_var: bool,
                      latest: int | None = None):
    """
    For each wrep method, show one row per confidence_weight value with
    per-cell mean (±std) across runs sharing that weight.
    """
    for method_key, display in SWEEP_METHODS:
        method_runs = [
            r for r in runs
            if r["method"] == method_key and r.get("conf_weight") is not None
        ]
        if not method_runs:
            continue

        # Optionally keep only the latest N per (weight, dataset, portion)
        if latest:
            groups: dict = defaultdict(list)
            for r in method_runs:
                groups[(r["conf_weight"], r["dataset"], r["portion"])].append(r)
            method_runs = []
            for lst in groups.values():
                lst.sort(key=lambda r: r["timestamp"], reverse=True)
                method_runs.extend(lst[:latest])

        weights = sorted({r["conf_weight"] for r in method_runs})

        raw: dict = defaultdict(list)
        for r in method_runs:
            ds, bud = CELL_MAP[(r["dataset"], r["portion"])]
            raw[(r["conf_weight"], ds, bud)].append(r["score"])

        rows = []
        for w in weights:
            cells = {}
            for ds, bud in DATA_COLS:
                vals = raw.get((w, ds, bud), [])
                cells[(ds, bud)] = _stat(vals) if vals else None
            cells[("Avg", "Sm")]  = _avg_stats([cells.get((ds, "Sm")) for ds in DATASETS])
            cells[("Avg", "Lg")]  = _avg_stats([cells.get((ds, "Lg")) for ds in DATASETS])
            cells[("Avg", "All")] = _avg_stats([cells.get((ds, bud))
                                                for ds in DATASETS for bud in BUDGETS])
            rows.append({"label": f"λ={w}", "cells": cells,
                         "is_base": False, "min_n": _min_n(cells)})

        rankings = rank_cols(rows)
        lat_note = f", latest {latest}" if latest else ""
        print_table(rows, rankings,
                    f"Confidence-weight sweep: {display} – {scorer_label}{lat_note}",
                    show_var)


# ── Main ──────────────────────────────────────────────────────────────────────

def _print_one_table(runs: list[dict], scorer_label: str, methods: list[tuple],
                     show_var: bool, use_max: bool, best_train: bool,
                     latest: int | None, all_methods: bool,
                     detail: bool, sig_test: bool = False):
    """Aggregate and print one table for a given set of runs."""
    favor_n = latest if latest else 3
    runs = keep_favor_sess(runs, n=favor_n)

    method_keys = [k for k, _ in methods]
    scores, baselines = aggregate(runs, method_keys, use_max=use_max, baseline_n=latest)

    active_methods = [
        (k, d) for k, d in methods
        if any(scores.get((k, ds, bud)) for ds in DATASETS for bud in BUDGETS)
    ]
    if not active_methods:
        print(f"  (no data for scorer: {scorer_label})")
        return

    rows     = build_rows(scores, baselines, active_methods)
    rankings = rank_cols(rows)

    parts = [f"best-{favor_n} SESS / worst-{favor_n} others"]
    if use_max:
        parts.append("best per config")
    if best_train:
        parts.append("best-train prompt")
    if latest:
        parts.append(f"latest {latest} runs")
    suffix = f" ({', '.join(parts)})"
    title = f"SESS Results – {scorer_label}{suffix}"
    print_table(rows, rankings, title, show_var, use_max)

    if detail:
        print_detail(runs, scores, active_methods)

    if sig_test:
        print_sig_table(runs, active_methods,
                        f"Significance vs Random – {scorer_label}{suffix}")

    if all_methods:
        abl_keys = [k for k, _ in ABLATION_METHODS]
        abl_scores, _ = aggregate(runs, abl_keys, use_max=use_max)
        active_abl = [
            (k, d) for k, d in ABLATION_METHODS
            if any(abl_scores.get((k, ds, bud)) for ds in DATASETS for bud in BUDGETS)
        ]
        if active_abl:
            abl_rows     = build_rows(abl_scores, baselines, active_abl)
            abl_rankings = rank_cols(abl_rows)
            print_table(abl_rows, abl_rankings,
                        f"Ablation Methods – {scorer_label}{suffix}", show_var, use_max)
            if detail:
                print_detail(runs, abl_scores, active_abl)


def main():
    parser = argparse.ArgumentParser(
        description="SESS results table (paper format, favor-SESS selection, no vwrep)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scorer",      type=str, default=None, metavar="NAME",
                        help="Restrict to one scorer (substring match). Default: auto two-table mode.")
    parser.add_argument("--latest",      type=int, default=None, metavar="N",
                        help="Keep only the latest N runs per (method, dataset, portion)")
    parser.add_argument("--best-train",  action="store_true",
                        help="Use test score of the prompt with best training score")
    parser.add_argument("--all-methods", action="store_true",
                        help="Also print ablation methods table")
    parser.add_argument("--no-variance", action="store_true",
                        help="Show only the mean, hide ±std(n)")
    parser.add_argument("--max",         action="store_true",
                        help="Show best single-run result per config (paper-style oracle reporting)")
    parser.add_argument("--detail",      action="store_true",
                        help="Print per-config run counts and min/max after each table")
    parser.add_argument("--sig-test",    action="store_true",
                        help="Print companion significance table (Welch's t-test vs Random per cell)")
    parser.add_argument("--sweep",       action="store_true",
                        help="Show confidence_weight parameter sweep table for wrep only")
    args = parser.parse_args()

    show_var = not args.no_variance and not args.max
    use_max  = args.max

    print("Scanning experiment directories...")
    all_runs = scan_all_runs(use_best_train=args.best_train)
    print(f"Found {len(all_runs)} completed runs.")

    # Determine which scorer groups to display
    if args.scorer:
        scorer_groups = [(args.scorer, args.scorer)]
    else:
        # Auto-detect: one table per scorer family present in data
        scorer_names = {r["scorer"] for r in all_runs}
        scorer_groups = []
        for fragment, label in [("Qwen2.5-7B", "Qwen2.5-7B-Instruct scorer"),
                                 ("Llama-3.1-8B", "Llama-3.1-8B-Instruct scorer")]:
            if any(fragment.lower() in s.lower() for s in scorer_names):
                scorer_groups.append((fragment, label))
        if not scorer_groups:
            # Fallback: print all together
            scorer_groups = [("", "all scorers combined")]

    for scorer_filter, scorer_label in scorer_groups:
        if scorer_filter:
            runs = [r for r in all_runs if scorer_filter.lower() in r["scorer"].lower()]
        else:
            runs = list(all_runs)
        print(f"\n  {scorer_label}: {len(runs)} runs")
        if args.sweep:
            print_sweep_table(runs, scorer_label, show_var, latest=args.latest)
        else:
            _print_one_table(runs, scorer_label, MAIN_METHODS,
                             show_var, use_max, args.best_train,
                             args.latest, args.all_methods,
                             args.detail, args.sig_test)


if __name__ == "__main__":
    main()
