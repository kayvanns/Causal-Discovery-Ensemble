"""Separate window-driven from ordering-driven missingness in tier 1.

A tier-1 variable can be missing for two very different reasons:

  window-driven   the baseline window (72h lookback ending at onset) sits
                  partly before icu_intime, and the stay_id-keyed MIMIC tables
                  hold nothing there. An artefact of when sepsis started
                  relative to ICU admission -- MAR given onset_to_icu_hrs,
                  which is itself a tier-0 variable in the graph.

  ordering-driven the test was not ordered. Plausibly severity-driven and
                  therefore MNAR, which is what HIGH_MISSING_COLS exists to
                  carve out so a sensitivity config can drop those columns.

Restricting to patients whose baseline window lies wholly inside the ICU stay
(onset_to_icu_hrs >= 72) removes the first mechanism. Whatever missingness
survives is the second. This is the check behind the rates quoted in tiers.py.

    python baseline_window_check.py [path.csv]   # default: cohort/analysis.csv

The subgroup is small (~3.6% of the cohort), so the threshold sweep is printed
alongside: a variable whose rate is genuinely ordering-driven stays high across
every threshold rather than turning at one of them.

Writes reports/baseline_window_check.csv.
"""

import os
import sys

import pandas as pd

from datasets import INPUT_PATH, load_data
from tiers import (
    ALWAYS_INDICATOR_COLS,
    HIGH_MISSING_COLS,
    PRESEPSIS_COLS,
    PRIMARY_OUTCOME,
)

DEFAULT_INPUT = INPUT_PATH
OUT_PATH = "reports/baseline_window_check.csv"

# Baseline lookback length in hours; onset this far after intime means the
# whole window is inside the stay. Matches the 72h lookback in the SQL.
WINDOW_HRS = 72
SWEEP_HRS = [0, 12, 24, 48, 72, 96, 120]

# Above this, in the wholly-inside-ICU subgroup, the gap cannot be blamed on
# the window and the variable is ordering-driven. Below it, the variable is
# recovered by ICU admission and its overall rate is a window artefact.
ORDERING_FLOOR = 0.20


def classify(rate):
    if rate >= ORDERING_FLOOR:
        return "ordering-driven"
    if rate >= 0.05:
        return "intermediate"
    return "window-driven"


def expected_membership(col):
    """Where tiers.py currently puts this column."""
    if col in HIGH_MISSING_COLS:
        return "HIGH_MISSING_COLS"
    if col in ALWAYS_INDICATOR_COLS:
        return "ALWAYS_INDICATOR_COLS"
    return "-"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    if not os.path.exists(path):
        sys.exit(f"missing input: {path}")

    # datasets.load_data(), not a plain dropna on PRIMARY_OUTCOME -- see the
    # note in missingness_report.main().
    df = load_data(path)
    hrs = df["onset_to_icu_hrs"]
    inside = df[hrs >= WINDOW_HRS]

    print(f"{path}: {len(df):,} rows, {len(inside):,} ({len(inside)/len(df):.1%}) "
          f"with the baseline window wholly inside the ICU stay "
          f"(onset_to_icu_hrs >= {WINDOW_HRS})\n")

    cols = [c for c in PRESEPSIS_COLS if c in df.columns]
    rows = []
    for col in cols:
        overall = float(df[col].isna().mean())
        restricted = float(inside[col].isna().mean())
        rows.append({
            "column": col,
            "pct_missing_overall": round(overall * 100, 2),
            f"pct_missing_inside_{WINDOW_HRS}h": round(restricted * 100, 2),
            "mechanism": classify(restricted),
            "tiers_py": expected_membership(col),
        })

    report = (pd.DataFrame(rows)
              .sort_values(f"pct_missing_inside_{WINDOW_HRS}h", ascending=False)
              .reset_index(drop=True))
    print(report.to_string(index=False))

    # Membership disagreements: HIGH_MISSING_COLS is for ordering-driven
    # columns only, so anything else in it -- or anything ordering-driven left
    # out of it -- is a claim the data no longer supports.
    print("\nDisagreements with tiers.py:")
    flagged = False
    for row in report.to_dict("records"):
        ordering = row["mechanism"] == "ordering-driven"
        in_high = row["tiers_py"] == "HIGH_MISSING_COLS"
        if ordering != in_high and (ordering or in_high):
            flagged = True
            print(f"  {row['column']}: {row['mechanism']} at "
                  f"{row[f'pct_missing_inside_{WINDOW_HRS}h']:.2f}% inside-ICU, "
                  f"but tiers.py has it in {row['tiers_py']}")
    if not flagged:
        print("  none")

    print(f"\nThreshold sweep (% missing at onset_to_icu_hrs >= t):")
    watch = [c for c in cols if c in HIGH_MISSING_COLS + ALWAYS_INDICATOR_COLS
             or c in ("platelet", "wbc", "creatinine")]
    header = f"{'t':>5}{'n':>8}" + "".join(f"{c:>13}" for c in watch)
    print(header)
    for threshold in SWEEP_HRS:
        sub = df[hrs >= threshold]
        line = f"{threshold:>5}{len(sub):>8,}"
        line += "".join(f"{sub[c].isna().mean() * 100:>13.2f}" for c in watch)
        print(line)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    report.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
