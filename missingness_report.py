"""Missingness inventory for the columns that go into the CHTC ensemble.

Reports every column in tiers.ALL_COLS -- the exact set datasets.py selects --
measured on the analysis cohort, meaning after the listwise drop of rows with an
unobservable PRIMARY_OUTCOME. Those are the rows the ensemble actually sees, so
these are the rates that decide indicator and config membership.

Sorted by missingness, worst first. The indicator column shows how datasets.py
actually carries each variable's missingness -- via a collapsed panel flag, via
its own "<col>_missing" column, or not at all -- so the consequence of a rate is
readable off the table without rebuilding.

    python missingness_report.py [path.csv]     # default: cohort/analysis_final.csv

Point it at a raw SQL export (cohort/cohort.csv) to inventory that instead; the
columns built later in Python are then reported as absent rather than missing.

Writes reports/missingness.csv and prints the same table.
"""

import os
import sys

import pandas as pd

from datasets import INPUT_PATH, load_data
from tiers import (
    ALL_COLS,
    ALWAYS_INDICATOR_COLS,
    BLOODWORK_PANEL_COLS,
    BLOODWORK_PANEL_INDICATOR,
    HIGH_MISSING_COLS,
    PRIMARY_OUTCOME,
    TIER_MAP,
    VITALS_PANEL_COLS,
    VITALS_PANEL_INDICATOR,
    get_feature_tier,
)

DEFAULT_INPUT = INPUT_PATH
OUT_PATH = "reports/missingness.csv"
RAW_OUT_PATH = "reports/missingness_raw.csv"

TIER_LABELS = {
    0: "0 baseline",
    1: "1 pre-sepsis",
    2: "2 post-onset 0-24h",
    3: "3 outcome 24-48h",
    4: "4 mortality",
}


def indicator_for(col):
    """How datasets.py carries this column's missingness.

    There is no threshold sweep any more -- datasets.py add_group_indicators()
    and add_sensitivity_indicators() key off the fixed tiers.py memberships, so
    this reports membership rather than a rate crossing a cutoff.
    """
    if col in BLOODWORK_PANEL_COLS:
        return f"via {BLOODWORK_PANEL_INDICATOR}"
    if col in VITALS_PANEL_COLS:
        return f"via {VITALS_PANEL_INDICATOR}"
    if col in ALWAYS_INDICATOR_COLS:
        return f"{col}_missing"
    if col in HIGH_MISSING_COLS:
        # Dropped from main entirely; only the sensitivity datasets carry it,
        # and there it gets its own indicator.
        return f"{col}_missing (sensitivity only)"
    return "-"


def build_report(df):
    """One row per ALL_COLS column present in the frame."""
    rows = []
    for col in ALL_COLS:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        rate = n_missing / len(df)
        rows.append({
            "column": col,
            "tier": TIER_LABELS.get(get_feature_tier(col), "unmapped"),
            "n": int(df[col].notna().sum()),
            "n_missing": n_missing,
            "pct_missing": round(rate * 100, 2),
            "indicator": indicator_for(col),
        })

    # The collapsed panel indicator is what the ensemble actually carries, so
    # report its rate alongside the six analytes it stands in for.
    panel = [c for c in BLOODWORK_PANEL_COLS if c in df.columns]
    if panel:
        panel_missing = df[panel].isna().any(axis=1)
        rate = panel_missing.mean()
        rows.append({
            "column": f"[{BLOODWORK_PANEL_INDICATOR}]",
            "tier": "1 pre-sepsis",
            "n": int((~panel_missing).sum()),
            "n_missing": int(panel_missing.sum()),
            "pct_missing": round(rate * 100, 2),
            "indicator": "(is an indicator)",
        })

    return (pd.DataFrame(rows)
            .sort_values("pct_missing", ascending=False)
            .reset_index(drop=True))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raw_mode = "--raw" in sys.argv[1:]
    path = args[0] if args else DEFAULT_INPUT
    if not os.path.exists(path):
        sys.exit(f"missing input: {path}")

    raw_cols = pd.read_csv(path, nrows=0).columns

    if raw_mode:
        # Every row in the export, with no outcome-based filtering at all --
        # not the recode, not the drop. This is the inventory of the cohort as
        # it comes out of the SQL, so PRIMARY_OUTCOME reports its own true
        # missingness here instead of being 0.00% by construction. Use it to
        # see the raw data; use the default mode for the rates the ensemble
        # actually sees.
        df = pd.read_csv(path)[[c for c in ALL_COLS if c in raw_cols]]
        out_path = RAW_OUT_PATH
        print(f"{path}: {len(df):,} rows, no rows dropped (--raw)\n")
    else:
        # Go through datasets.load_data() rather than a plain dropna on
        # PRIMARY_OUTCOME. Those are not the same row set: load_data() first
        # recodes the outcome for patients who died before the window (stage 4)
        # and for those discharged alive with no realistic chance of a draw
        # (stage 0), and only then drops what is left. A plain dropna discards
        # all of those instead -- ~1,100 rows, over a tenth of the cohort -- so
        # it measures missingness on a frame the ensemble never sees, and skews
        # it toward the sicker, longer-staying patients who survived the 48h.
        df = load_data(path)
        out_path = OUT_PATH
        print()

    report = build_report(df)
    print(report.to_string(index=False))

    absent = [c for c in ALL_COLS if c not in raw_cols]
    if absent:
        print(f"\nNOTE: {len(absent)} analysis column(s) not in this file "
              f"(built later in the pipeline): {absent}")

    extra = [c for c in raw_cols if c not in TIER_MAP]
    if extra:
        print(f"\n{len(extra)} column(s) in the file are not analysis variables "
              f"and are dropped by datasets.py: {sorted(extra)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    report.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
