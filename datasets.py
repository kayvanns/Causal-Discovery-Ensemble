"""Build the dataset variants for the CHTC PC/FCI ensemble.

Column selection, tiering, and every indicator-membership decision live in
tiers.py -- this file only assembles datasets from those decisions and must
not encode column lists itself.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from tiers import (
    ALL_COLS,
    ALWAYS_INDICATOR_COLS,
    BASELINE_COLS,
    BLOODWORK_PANEL_COLS,
    BLOODWORK_PANEL_INDICATOR,
    HIGH_MISSING_COLS,
    MORTALITY_COLS,
    OUTCOME_COLS,
    PRESEPSIS_COLS,
    VITALS_PANEL_COLS,
    VITALS_PANEL_INDICATOR,
    WINDOW_0TO24H_COLS,
    get_feature_tier,
)

# Final cohort build (build_ed.py from sql/causal_discovery_features_ed.sql).
INPUT_PATH = "cohort/analysis_final.csv"
OUTPUT_DIR = "data"

# The primary outcome is the max AKI stage over onset+24h..onset+48h. A patient
# discharged alive partway through that span has an "eligible window": the part
# of it they were actually present for. Below this many hours there was no
# realistic opportunity to draw a creatinine, so a missing value is recoded to
# stage 0. Above it, the patient was present and alive long enough that a
# missing value means not-measured rather than measured-and-normal, and the row
# is dropped instead of being asserted as no-AKI.
MIN_ELIGIBLE_WINDOW_HRS = 6


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(path=INPUT_PATH):
    """Load the cohort CSV and select exactly the columns tiers.py knows
    about. ALL_COLS is the single source of truth for membership"""
    df = pd.read_csv(path)
    
    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"tiers.py expects columns not present in {path}: {missing}")
    # race -> is-white indicator. str.contains(na=False) maps NaN to False,
    # which would silently assert non-white for an unrecorded race, so the
    # originally-null rows are restored to NaN afterwards. Both the capture and
    # the restore must stay inside this block: hoisting the restore out raises
    # NameError on any export without a race column.
    if "race" in df.columns:
        race_original = df["race"]
        df["race"] = (
            df["race"]
            .astype("string")
            .str.upper()
            .str.contains("WHITE", na=False)
            .astype(float)
        )
        df.loc[race_original.isna(), "race"] = np.nan

    if "gender" in df.columns:
        df["gender"] = (
            df["gender"]
            .astype("string")
            .str.upper()
            .map({"F": 0.0, "M": 1.0})
        )
    onset = pd.to_datetime(df["sepsis_onset_time"])
    icu_outtime = pd.to_datetime(df["icu_outtime"])
    deathtime = pd.to_datetime(df["deathtime"])

    window_end = onset + pd.Timedelta(hours=48)

    # Death is judged against the full onset+48h bound, NOT against a bound
    # truncated by icu_outtime: a patient discharged from the ICU at 30h who
    # then dies at 40h died before the window could complete, so they are
    # group A. Truncating first would let them fall through to group B and be
    # recorded as stage 0 -- the best possible outcome for a patient who died.
    died_before_window = deathtime.notna() & (deathtime <= window_end)

    # Window was truncated by discharge/transfer before it could reach the
    # full onset+48h bound, and the patient did NOT die in that span (death
    # takes priority -- a row can satisfy both time conditions, but it's
    # group A, not group B).
    left_alive_before_window = (icu_outtime < window_end) & ~died_before_window

    # How much of the 24..48h outcome window the patient was actually present
    # for. Death never binds here -- left_alive_before_window already excludes
    # anyone who died by window_end -- so discharge is the only early stop.
    # Discharge before the window even opens gives <= 0 and falls under the
    # floor, which is what we want.
    eligible_hrs = (
        np.minimum(icu_outtime, window_end) - (onset + pd.Timedelta(hours=24))
    ).dt.total_seconds() / 3600

    no_real_window = left_alive_before_window & (eligible_hrs <= MIN_ELIGIBLE_WINDOW_HRS)
    unmeasured_window = left_alive_before_window & (eligible_hrs > MIN_ELIGIBLE_WINDOW_HRS)

    still_null = df["aki_24to48h_max_stage"].isna()

    n_a = (still_null & died_before_window).sum()
    n_b = (still_null & no_real_window).sum()
    n_c_partial = (still_null & unmeasured_window).sum()

    df.loc[still_null & died_before_window, "aki_24to48h_max_stage"] = 4
    df.loc[still_null & no_real_window, "aki_24to48h_max_stage"] = 0

    n_c = df["aki_24to48h_max_stage"].isna().sum()

    print(f"  Group A (died before window) recoded to stage 4: {n_a:,}")
    print(f"  Group B (discharged alive, <= {MIN_ELIGIBLE_WINDOW_HRS}h eligible "
          f"window) recoded to stage 0: {n_b:,}")
    print(f"  Group C (window present, no creatinine drawn) still missing: {n_c:,} "
          f"({n_c / len(df):.1%}) -- {n_c_partial:,} discharged alive with "
          f"> {MIN_ELIGIBLE_WINDOW_HRS}h eligible window, "
          f"{n_c - n_c_partial:,} with the window complete")
    n_before = len(df)
    df = df[df["aki_24to48h_max_stage"].notna()].copy()
    n_after = len(df)
    print(f"  Dropped {n_before - n_after:,} rows with unresolved missing "
          f"aki_24to48h_max_stage ({(n_before - n_after) / n_before:.1%})")
    df = df[ALL_COLS]

    print(f"  Rows: {len(df):,}")
    print(f"  Columns loaded: {len(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Column set builders
# ---------------------------------------------------------------------------

def get_main_value_cols(include_mortality=False):
    """Value columns for the main datasets: everything in the tiers except
    lactate/bilirubin (HIGH_MISSING_COLS), which main drops entirely.
    MORTALITY_COLS is opt-in via include_mortality, off by default -- see
    build_main_with_mortality() for the one-off runtime-cost check."""
    presepsis = [c for c in PRESEPSIS_COLS if c not in HIGH_MISSING_COLS]
    cols = BASELINE_COLS + presepsis + WINDOW_0TO24H_COLS + OUTCOME_COLS
    if include_mortality:
        cols = cols + MORTALITY_COLS
    return cols


def get_sensitivity_value_cols():
    """Sensitivity adds HIGH_MISSING_COLS back in."""
    return get_main_value_cols() + HIGH_MISSING_COLS


def get_cols_by_tier(value_cols):
    """Partition a column list by tier, for per-tier KNN imputation."""
    by_tier = {}
    for col in value_cols:
        tier = get_feature_tier(col)
        by_tier.setdefault(tier, []).append(col)
    return by_tier


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def add_group_indicators(df):
    df = df.copy()
    df[BLOODWORK_PANEL_INDICATOR] = df[BLOODWORK_PANEL_COLS].isna().any(axis=1).astype(int)
    df[VITALS_PANEL_INDICATOR] = df[VITALS_PANEL_COLS].isna().any(axis=1).astype(int)
    for col in ALWAYS_INDICATOR_COLS:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def add_sensitivity_indicators(df):
    df = df.copy()
    for col in HIGH_MISSING_COLS:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def build_raw(df, value_cols):
    """mv_fisherz only. No imputation, no indicators -- NaNs preserved as-is."""
    return df[value_cols].copy()


def simple_impute(df, cols_by_tier):
    """Median (numeric) / mode (binary) imputation."""
    df = df.copy()
    for cols in cols_by_tier.values():
        for col in cols:
            if df[col].dropna().isin([0, 1]).all():
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    return df


def knn_impute_by_tier(df, cols_by_tier, n_neighbors=10):
    """Fit separately per tier using only that tier's own columns plus tier 0
    (baseline is always legitimate to condition on). Prevents a later tier's
    signal leaking into an earlier tier's imputed values ahead of the
    search."""
    df = df.copy()
    baseline_cols = cols_by_tier.get(0, [])
    for tier, cols in cols_by_tier.items():
        predictor_cols = cols if tier == 0 else list(dict.fromkeys(cols + baseline_cols))
        block = df[predictor_cols]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(block)
        imputed_scaled = KNNImputer(n_neighbors=n_neighbors).fit_transform(scaled)
        imputed = scaler.inverse_transform(imputed_scaled)
        imputed_df = pd.DataFrame(imputed, columns=predictor_cols, index=df.index)

        df[cols] = imputed_df[cols]
    return df


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_main(df):
    value_cols = get_main_value_cols()
    cols_by_tier = get_cols_by_tier(value_cols)

    df_raw = build_raw(df, value_cols)

    df_ind = add_group_indicators(df)
    indicator_cols = [BLOODWORK_PANEL_INDICATOR, VITALS_PANEL_INDICATOR] + [
        f"{c}_missing" for c in ALWAYS_INDICATOR_COLS
    ]

    df_simple = simple_impute(df_ind, cols_by_tier)[value_cols + indicator_cols]
    df_knn = knn_impute_by_tier(df_ind, cols_by_tier)[value_cols + indicator_cols]

    return df_raw, df_simple, df_knn


def build_main_with_mortality(df):
    """One-off variant: adds MORTALITY_COLS to the simple-imputed main
    dataset only, to measure whether the extra tier-4 sink column adds
    meaningfully to FCI runtime. Not part of the core ensemble."""
    value_cols = get_main_value_cols(include_mortality=True)
    cols_by_tier = get_cols_by_tier(value_cols)

    df_ind = add_group_indicators(df)
    indicator_cols = [BLOODWORK_PANEL_INDICATOR, VITALS_PANEL_INDICATOR] + [
        f"{c}_missing" for c in ALWAYS_INDICATOR_COLS
    ]

    return simple_impute(df_ind, cols_by_tier)[value_cols + indicator_cols]


def build_sensitivity(df):
    value_cols = get_sensitivity_value_cols()
    cols_by_tier = get_cols_by_tier(value_cols)

    df_ind = add_sensitivity_indicators(add_group_indicators(df))
    indicator_cols = (
        [BLOODWORK_PANEL_INDICATOR, VITALS_PANEL_INDICATOR]
        + [f"{c}_missing" for c in ALWAYS_INDICATOR_COLS]
        + [f"{c}_missing" for c in HIGH_MISSING_COLS]
    )

    df_simple = simple_impute(df_ind, cols_by_tier)[value_cols + indicator_cols]
    df_knn = knn_impute_by_tier(df_ind, cols_by_tier)[value_cols + indicator_cols]

    return df_simple, df_knn


def main():
    df = load_data()

    print("Building MAIN datasets...")
    main_raw, main_simple, main_knn = build_main(df)
    main_raw.to_csv(f"{OUTPUT_DIR}/main_raw.csv", index=False)
    main_simple.to_csv(f"{OUTPUT_DIR}/main_simple.csv", index=False)
    main_knn.to_csv(f"{OUTPUT_DIR}/main_knn.csv", index=False)
    print(f"  main_raw:    {main_raw.shape}")
    print(f"  main_simple: {main_simple.shape}")
    print(f"  main_knn:    {main_knn.shape}")

    print("Building mortality runtime-check dataset...")
    main_with_mortality = build_main_with_mortality(df)
    main_with_mortality.to_csv(f"{OUTPUT_DIR}/main_simple_with_mortality.csv", index=False)
    print(f"  main_simple_with_mortality: {main_with_mortality.shape}")

    print("Building SENSITIVITY datasets...")
    sens_simple, sens_knn = build_sensitivity(df)
    sens_simple.to_csv(f"{OUTPUT_DIR}/sens_simple.csv", index=False)
    sens_knn.to_csv(f"{OUTPUT_DIR}/sens_knn.csv", index=False)
    print(f"  sens_simple: {sens_simple.shape}")
    print(f"  sens_knn:    {sens_knn.shape}")


if __name__ == "__main__":
    main()