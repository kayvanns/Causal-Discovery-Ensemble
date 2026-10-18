import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
import features_ed as fe

MIMIC_HOSP = "/Users/kayvans/Documents/mimic-iv-3.1/hosp"
MIMIC_ICU  = "/Users/kayvans/Documents/mimic-iv-3.1/icu"

DEFAULT_COHORT = os.path.join(os.path.dirname(__file__), "cohort", "cohort.csv")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "cohort", "analysis_final.csv")



USECOLS = {
    "patients": ["subject_id", "gender"],
    "admissions":  ["subject_id", "hadm_id", "admittime", "dischtime",
                    "race", "hospital_expire_flag", "deathtime"],
    "labevents":   ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
    "pharmacy":    ["subject_id", "hadm_id", "medication", "starttime"],
    "omr":         ["subject_id", "result_name", "result_value", "chartdate"],
    "icustays":    ["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    "inputevents": ["subject_id", "hadm_id", "stay_id", "itemid",
                    "starttime", "amount", "amountuom"],
    "outputevents":["subject_id", "hadm_id", "stay_id", "itemid",
                    "charttime", "value", "valueuom"],
}


def load_all() -> dict:
    def read(path, name):
        print(f"  Loading {name}...")
        return pd.read_csv(f"{path}/{name}.csv.gz", usecols=USECOLS.get(name))

    print("Loading MIMIC tables...")
    tables = {
        "patients": read(MIMIC_HOSP, "patients"),
        "admissions":   read(MIMIC_HOSP, "admissions"),
        "labs":         read(MIMIC_HOSP, "labevents"),
        "pharmacy":     read(MIMIC_HOSP, "pharmacy"),
        "omr":          read(MIMIC_HOSP, "omr"),
        "icustays":     read(MIMIC_ICU,  "icustays"),
        "inputevents":  read(MIMIC_ICU,  "inputevents"),
        "outputevents": read(MIMIC_ICU,  "outputevents"),
    }
    print(f"Done. {len(tables)} tables loaded.\n")
    return tables


def filter_to_cohort(tables: dict, cohort: pd.DataFrame) -> dict:
    stay_ids = set(cohort["stay_id"])
    hadm_ids = set(cohort["hadm_id"])
    subj_ids = set(cohort["subject_id"])
    return {
        "patients":     tables["patients"][tables["patients"]["subject_id"].isin(subj_ids)],
        "admissions":   tables["admissions"][tables["admissions"]["hadm_id"].isin(hadm_ids)],
        # Filtered by SUBJECT, not hadm_id. Roughly half of labevents rows carry
        # a NULL hadm_id -- specimens drawn before the inpatient admission was
        # registered, which for an ED->ICU patient is most of the baseline
        # window, since sepsis onset lands a median 1.8h BEFORE admittime.
        # Filtering on hadm_id here discarded them before get_labs() ever saw
        # them. The episode-start guard that keeps this from pulling in an
        # unrelated earlier encounter lives in features.get_labs().
        "labs":         tables["labs"][tables["labs"]["subject_id"].isin(subj_ids)],
        "pharmacy":     tables["pharmacy"][tables["pharmacy"]["hadm_id"].isin(hadm_ids)],
        "omr":          tables["omr"][tables["omr"]["subject_id"].isin(subj_ids)],
        "icustays":     tables["icustays"][tables["icustays"]["stay_id"].isin(stay_ids)],
        "inputevents":  tables["inputevents"][tables["inputevents"]["stay_id"].isin(stay_ids)],
        "outputevents": tables["outputevents"][tables["outputevents"]["stay_id"].isin(stay_ids)],
    }


def build_base(cohort: pd.DataFrame, tables: dict) -> pd.DataFrame:
    df = cohort.copy()
    for col in ["icu_intime", "icu_outtime", "sepsis_onset_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.merge(tables["patients"][["subject_id", "gender"]], on="subject_id", how="left")
    df = df.merge(
        tables["admissions"][["hadm_id", "admittime", "dischtime",
                               "race", "hospital_expire_flag","deathtime"]],
        on="hadm_id", how="left"
    )
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    df["dischtime"] = pd.to_datetime(df["dischtime"], errors="coerce")

    # Start of this care episode, used by get_labs() to decide whether a
    # NULL-hadm_id lab belongs to this encounter. ED arrival where we have it,
    # because admittime is registered hours AFTER the ED workup that produced
    # the baseline labs; the earlier of the two otherwise.
    if "ed_intime" in df.columns:
        ed_intime = pd.to_datetime(df["ed_intime"], errors="coerce")
        df["episode_start"] = ed_intime.where(
            ed_intime.notna() & (ed_intime <= df["admittime"]), df["admittime"])
    else:
        df["episode_start"] = df["admittime"]

    # hospital LOS before sepsis onset
    df["hosp_los_pre_sepsis_hrs"] = (
        (df["sepsis_onset_time"] - df["admittime"]).dt.total_seconds() / 3600
    ).clip(0, 720)

    return df


def build_features(base: pd.DataFrame, tables: dict) -> pd.DataFrame:
    df = base.copy()

    df = fe.add_causal_windows(df)
    df = fe.get_mortality_28day(df)

    # baseline labs (latest, <= onset): lactate, bilirubin, platelet, inr, hemoglobin, wbc
    df = fe.get_labs(df, labs=tables["labs"])

    # at-sepsis-onset extreme labs (1-24h post onset): lactate max, wbc max
    df = fe.get_labs_extremes(df, labs=tables["labs"])

    # medications
    df = fe.get_medications(df, pharmacy=tables["pharmacy"])


    # fluid balance (window)
    df = fe.get_pre24h_fluid_balance(
        df, inputevents=tables["inputevents"], outputevents=tables["outputevents"]
    )

    # post-24h fluid balance (mediation variable)
    df = fe.get_0_24h_fluid_balance(
        df, inputevents=tables["inputevents"], outputevents=tables["outputevents"]
    )

    return df


if __name__ == "__main__":
    cohort_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COHORT
    output_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    cohort = pd.read_csv(cohort_path)
    print(f"Cohort loaded: {cohort.shape[0]} stays\n")

    tables = load_all()
    tables = filter_to_cohort(tables, cohort)

    base   = build_base(cohort, tables)
    result = build_features(base, tables)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}  {result.shape}")
