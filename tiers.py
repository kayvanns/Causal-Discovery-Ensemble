"""Shared column groups and temporal tier assignments."""

BASELINE_COLS = [
    "age",
    "gender",
    "race",
    "charlson_comorbidity_index",
    "ckd_baseline",
    "creatinine_baseline",
    "onset_to_icu_hrs",
]

# sofa_pre24h and fio2 were dropped here. Both are stay_id-keyed and simply not
# computed before ICU admission.
PRESEPSIS_COLS = [
    "heart_rate",
    "blood_pressure",
    "spo2",
    "temperature_c",
    "bicarbonate",
    "sodium",
    "potassium",
    "glucose",
    "bun",
    "creatinine",
    "platelet",
    "hemoglobin",
    "wbc",
    "inr",
    "lactate",
    "bilirubin",
    # mechvent_pre_onset removed: the final cohort excludes patients ventilated
    # before onset (sql/causal_discovery_features_ed.sql, change #9)
]

# fluid_balance_pre24h_ml is deliberately NOT in PRESEPSIS_COLS. Excluded because it is
# 74.8% missing on the analysis cohort -- worse than anything retained -- which


# Every 0-24h post-onset variable sits in one tier: treatments and the
# physiology they act on share the same window, so no sub-ordering is
# defensible.
WINDOW_0TO24H_COLS = [
    "antibiotics_0to24h",
    "vasopressors_0to24h",
    "diuretics_0to24h",
    "corticosteroids_0to24h",
    "mechvent_0to24h",
    "sofa_0to24h_max",
    "heart_rate_0to24h_max",
    "blood_pressure_0to24h_min",
    "spo2_0to24h_min",
    "temperature_c_0to24h_max",
    "lactate_0to24h_max",
    "creatinine_0to24h_max",
    "wbc_0to24h_max",
    "fluid_balance_0to24h_ml",
    "aki_0to24h_max_stage",
    "crrt_0to24h",
]

# The renal outcome is the 24-48h max AKI stage
OUTCOME_COLS = [
    "aki_24to48h_max_stage",
    "mechvent_24to48h",
    "crrt_24to48h",
]

MORTALITY_COLS = [
    "mortality_28day",
]

ROOT_NODES = ["age", "gender", "race"]

PRIMARY_OUTCOME = "aki_24to48h_max_stage"




HIGH_MISSING_COLS = ["lactate", "bilirubin"]


ALWAYS_INDICATOR_COLS = ["inr", "lactate_0to24h_max"]


BLOODWORK_PANEL_COLS = ["bicarbonate", "sodium", "potassium", "glucose",
                         "bun", "creatinine", "platelet", "wbc", "hemoglobin"]
BLOODWORK_PANEL_INDICATOR = "bloodwork_panel_missing"

# Vitals: different mechanism (bedside monitoring cadence, not lab-order
# batching), and only r ~ 0.16 against the chemistry/bloodwork panel, so kept
# as its own group.
VITALS_PANEL_COLS = ["heart_rate", "blood_pressure", "spo2", "temperature_c"]
VITALS_PANEL_INDICATOR = "vitals_panel_missing"

# group indicator name -> member columns, for the two grouped variants that
# datasets.py builds. Anything not in here (HIGH_MISSING_COLS,
# ALWAYS_INDICATOR_COLS) gets an individual "<col>_missing" indicator instead.
INDICATOR_GROUPS = {
    BLOODWORK_PANEL_INDICATOR: BLOODWORK_PANEL_COLS,
    VITALS_PANEL_INDICATOR: VITALS_PANEL_COLS,
}

TIER_MAP = {
    **{col: 0 for col in BASELINE_COLS},
    **{col: 1 for col in PRESEPSIS_COLS},
    **{col: 2 for col in WINDOW_0TO24H_COLS},
    **{col: 3 for col in OUTCOME_COLS},
    **{col: 4 for col in MORTALITY_COLS},
}

ALL_COLS = list(TIER_MAP.keys())

# Indicators that are not derived from a single TIER_MAP column and so cannot
# be resolved by stripping a suffix -- only the two GROUP indicators need
# registering here. Individual indicators (inr_missing,
# lactate_0to24h_max_missing, lactate_missing, bilirubin_missing) all resolve
# automatically: strip "_missing" and the base name is already in TIER_MAP.
INDICATOR_ONLY_TIERS = {
    BLOODWORK_PANEL_INDICATOR: 1,
    VITALS_PANEL_INDICATOR: 1,
}


def get_base_feature_name(column_name):
    """Strip a missingness-indicator suffix."""
    for suffix in ("_missing_indicator", "_missing"):
        if column_name.endswith(suffix):
            return column_name[: -len(suffix)]
    return None


def get_feature_tier(column_name):
    """Tier for a feature or its indicator."""
    if column_name in TIER_MAP:
        return TIER_MAP[column_name]

    if column_name in INDICATOR_ONLY_TIERS:
        return INDICATOR_ONLY_TIERS[column_name]

    base_feature = get_base_feature_name(column_name)
    if base_feature is not None and base_feature in TIER_MAP:
        return TIER_MAP[base_feature]

    return None