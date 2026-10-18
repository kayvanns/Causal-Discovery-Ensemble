import pandas as pd
from datetime import timedelta
import numpy as np

antibiotic_patterns = [
    'adoxa', 'ala-tet', 'alodox', 'amikacin', 'amikin', 'amoxicill', 'amphotericin',
    'anidulafungin', 'ancef', 'clavulanate', 'ampicillin', 'augmentin', 'avelox',
    'avidoxy', 'azactam', 'azithromycin', 'aztreonam', 'axetil', 'bactocill', 'bactrim',
    'bactroban', 'bethkis', 'biaxin', 'bicillin l-a', 'cayston', 'cefazolin', 'cedax',
    'cefoxitin', 'ceftazidime', 'cefaclor', 'cefadroxil', 'cefdinir', 'cefditoren',
    'cefepime', 'cefotan', 'cefotetan', 'cefotaxime', 'ceftaroline', 'cefpodoxime',
    'cefpirome', 'cefprozil', 'ceftibuten', 'ceftin', 'ceftriaxone', 'cefuroxime',
    'cephalexin', 'cephalothin', 'cephapririn', 'chloramphenicol', 'cipro', 'ciprofloxacin',
    'claforan', 'clarithromycin', 'cleocin', 'clindamycin', 'cubicin', 'dicloxacillin',
    'dirithromycin', 'doryx', 'doxycy', 'duricef', 'dynacin', 'ery-tab', 'eryped', 'eryc',
    'erythrocin', 'erythromycin', 'factive', 'flagyl', 'fortaz', 'furadantin', 'garamycin',
    'gentamicin', 'kanamycin', 'keflex', 'kefzol', 'ketek', 'levaquin', 'levofloxacin',
    'lincocin', 'linezolid', 'macrobid', 'macrodantin', 'maxipime', 'mefoxin',
    'metronidazole', 'meropenem', 'methicillin', 'minocin', 'minocycline', 'monodox',
    'monurol', 'morgidox', 'moxatag', 'moxifloxacin', 'mupirocin', 'myrac', 'nafcillin',
    'neomycin', 'nicazel doxy 30', 'nitrofurantoin', 'norfloxacin', 'noroxin', 'ocudox',
    'ofloxacin', 'omnicef', 'oracea', 'oraxyl', 'oxacillin', 'pc pen vk', 'pce dispertab',
    'panixine', 'pediazole', 'penicillin', 'periostat', 'pfizerpen', 'piperacillin',
    'tazobactam', 'primsol', 'proquin', 'raniclor', 'rifadin', 'rifampin', 'rocephin',
    'smz-tmp', 'septra', 'septra ds', 'solodyn', 'spectracef', 'streptomycin',
    'sulfadiazine', 'sulfamethoxazole', 'trimethoprim', 'sulfatrim', 'sulfisoxazole',
    'suprax', 'synercid', 'tazicef', 'tetracycline', 'timentin', 'tobramycin', 'unasyn',
    'vancocin', 'vancomycin', 'vantin', 'vibativ', 'vibra-tabs', 'vibramycin', 'zinacef',
    'zithromax', 'zosyn', 'zyvox'
]

vasoactive_patterns = [
    'norepinephrine',
    'epinephrine',
    'vasopressin',
    'phenylephrine',
    'dopamine',
'dobutamine',
'milrinone']

corticosteroids_patterns = ['hydrocortisone','prednisone','prednisolone','methylprednisolone','dexamethasone']
diuretic_patterns = ['furosemide', 'lasix', 'bumetanide', 'bumex', 'torsemide']

labevents = {
    "lactate":    {'itemid': [50813, 53154], 'valid': (0, 50)},
    "bilirubin":  {'itemid': [50885, 53089], 'valid': (0, 150)},
    "platelet":   {'itemid': [51704, 51265], 'valid': (0, 3000)},
    "inr":        {'itemid': [51675, 51237], 'valid': (0.5, 20)},
    "hemoglobin": {'itemid': [50811, 51222], 'valid': (0, 30)},
    "wbc":        {'itemid': [51301, 51516], 'valid': (0, 500)},
}

onset_extreme_labs = {
    "lactate": {'itemid': [50813, 53154], 'agg': 'max', 'valid': (0, 50)},
    "wbc":     {'itemid': [51301, 51516], 'agg': 'max', 'valid': (0, 500)},
}

def add_causal_windows(df):
 
    df = df.copy()
 
    onset = pd.to_datetime(df["sepsis_onset_time"])
 
    # Physiological predictors -- baseline / "latest value" window.
    # Ends AT onset, not after it: these labs are tier 1 in tiers.py, so they
    # must not be able to carry post-onset information. Mirrors the SQL
    # baseline windows in causal_discovery_features.sql.
    df["physiology_start"] = onset - pd.Timedelta(hours=72)
    df["physiology_end"] = onset
 
    # At-sepsis-onset extreme-value window (0-24h post onset). Starts at onset
    # so it meets the baseline window exactly: no gap, no overlap.
    df["onset_extreme_start"] = onset
    df["onset_extreme_end"] = onset + pd.Timedelta(hours=24)
 
    df["pre_start"] = onset - pd.Timedelta(hours=24)
    df["pre_end"] = onset

    # Hours from ICU admission to sepsis onset; negative when the patient was
    # already septic on arrival at the unit (56% of this cohort, median -1.7h).
    # This is the variable that governs tier-1 missingness: derived.vitalsign,
    # ventilator_setting and sofa are all stay_id-keyed and therefore hold
    # nothing before icu_intime, so a baseline window sitting left of intime
    # comes back empty regardless of how the patient was actually monitored.
    # Carried as a real tier-0 variable rather than a nuisance term -- it also
    # separates pre-ICU deterioration from ICU-acquired sepsis, which is a
    # clinically meaningful distinction in its own right.
    df["onset_to_icu_hrs"] = (
        (onset - pd.to_datetime(df["icu_intime"])).dt.total_seconds() / 3600
    ).clip(-720, 720)
 
    # Treatment/exposure period
    df["treatment_start"] = onset
    df["treatment_end"] = onset + pd.Timedelta(hours=24)
    return df

 
def _link_labs(df, labs, window_cols):
    """Attach labevents rows to cohort stays, keyed on SUBJECT not hadm_id.

    Why not hadm_id: about half of labevents carries a NULL hadm_id -- the
    specimen was drawn before the inpatient admission was registered. For an
    ED->ICU patient that is most of the baseline window, because sepsis onset
    lands a median 1.8h BEFORE admittime and 73.8% of them are septic before
    they are admitted at all. Joining on hadm_id dropped 59.5% of their
    in-window draws (against 13.1% for everyone else), which is the whole of
    the apparent "ED patients have no baseline labs" effect.

    The guard that replaces the hadm_id join: a lab counts for this stay if it
    either carries this stay's hadm_id, or carries no hadm_id AND was drawn at
    or after episode_start (ED arrival, else admittime). A lab from a DIFFERENT
    admission is excluded outright -- subject_id alone would let an unrelated
    earlier encounter inside the 72h window, the same hazard the ed_stays CTE
    comment in the SQL spells out for vitals.
    """
    l = labs.copy()
    l["charttime"] = pd.to_datetime(l["charttime"])
    keys = ["subject_id", "hadm_id", "stay_id", "episode_start"] + window_cols
    merged = df[keys].merge(l, on="subject_id", how="left", suffixes=("", "_lab"))

    same_admission = merged["hadm_id_lab"] == merged["hadm_id"]
    this_episode = (merged["hadm_id_lab"].isna()
                    & (merged["charttime"] >= merged["episode_start"]))
    return merged[same_admission | this_episode]


def get_labs(df, labs):
    """Baseline labs: latest value at or before onset (72h lookback)."""
    df = df.copy()
    merged = _link_labs(df, labs, ["physiology_start", "physiology_end"])
    # Inclusive of onset, matching the SQL baseline windows' "<= onset". The
    # extreme window below is strictly "> onset", so a value charted exactly at
    # onset lands in the baseline only -- no gap, no double-count.
    mask   = (merged["physiology_start"] <= merged["charttime"]) & \
             (merged["charttime"]   <= merged["physiology_end"])
    merged = merged[mask]
    for name, info in labevents.items():
        subset = merged[merged["itemid"].isin(info["itemid"])]
        lo, hi = info["valid"]
        subset = subset[(subset["valuenum"] >= lo) & (subset["valuenum"] <= hi)]
        latest = (subset.sort_values("charttime").groupby("stay_id").last()["valuenum"].reset_index(name=name))
        df     = df.merge(latest, on="stay_id", how="left")
    return df

 
def get_labs_extremes(df, labs):
    """At-sepsis-onset extreme labs: max lactate and max wbc, strictly within
    (onset, onset+24h]. Mirrors get_labs() but aggregates by info['agg']
    over the onset_extreme_start/end window instead of taking the latest value."""
    df = df.copy()
    merged = _link_labs(df, labs, ["onset_extreme_start", "onset_extreme_end"])
    mask   = (merged["onset_extreme_start"] < merged["charttime"]) & \
             (merged["charttime"] <= merged["onset_extreme_end"])
    merged = merged[mask]
    for name, info in onset_extreme_labs.items():
        subset = merged[merged["itemid"].isin(info["itemid"])]
        lo, hi = info["valid"]
        subset = subset[(subset["valuenum"] >= lo) & (subset["valuenum"] <= hi)]
        agg_col = f"{name}_0to24h_{info['agg']}"
        extreme = (subset.groupby("stay_id")["valuenum"].agg(info["agg"]).reset_index(name=agg_col))
        df = df.merge(extreme, on="stay_id", how="left")
    return df

def get_medications(df, pharmacy):
    df = df.copy()
    p  = pharmacy.copy()
    p["starttime"] = pd.to_datetime(p["starttime"], errors="coerce")
    merged = p.merge(df[["hadm_id", "treatment_start", "treatment_end","stay_id"]], on="hadm_id", how="inner")
    mask   = (merged["starttime"] >= merged["treatment_start"]) & \
             (merged["starttime"] < merged["treatment_end"])
    merged = merged[mask].copy()
 
    med_lower = merged["medication"].fillna("").str.lower()
 
    merged["is_antibiotic"]    = med_lower.apply(
        lambda x: any(p in x for p in antibiotic_patterns))
    merged["is_vaso"]          = med_lower.apply(lambda x: any(p in x for p in vasoactive_patterns))
    merged["is_corticosteroid"] = med_lower.apply(
        lambda x: any(p in x for p in corticosteroids_patterns))
    merged["is_diuretic"]      = med_lower.apply(
        lambda x: any(p in x for p in diuretic_patterns))
 
    for col, flag, label in [
        ("is_antibiotic",    "ab_count",     "antibiotics_0to24h"),
        ("is_vaso",          "vaso_count",   "vasopressors_0to24h"),
        ("is_corticosteroid","steroid_count","corticosteroids_0to24h"),
        ("is_diuretic",      "diur_count",   "diuretics_0to24h")]:
        counts = merged[merged[col]].groupby("stay_id").size().rename(flag)
        df     = df.merge(counts, on="stay_id", how="left")
        df[label] = df[flag].notna().astype(int)
        df = df.drop(columns=[flag])
 
    return df

_VOLUME_UOMS = {"ml", "l"}

def _fluid_balance(df, inputevents, outputevents, start_col, end_col, out_col):
    df = df.copy()
    i = inputevents.copy()
    o = outputevents.copy()

    i = i.merge(df[["stay_id", start_col, end_col]], on="stay_id", how="inner")
    o = o.merge(df[["stay_id", start_col, end_col]], on="stay_id", how="inner")

    i["starttime"] = pd.to_datetime(i["starttime"])
    o["charttime"] = pd.to_datetime(o["charttime"])

    i = i[(i["starttime"] >= i[start_col]) & (i["starttime"] < i[end_col])]
    o = o[(o["charttime"] >= o[start_col]) & (o["charttime"] < o[end_col])]

    # Keep volumetric records only: amounts charted in mg/mcg/units/mEq/dose are
    # not fluid and must not be summed as millilitres.
    i_uom = i["amountuom"].astype(str).str.strip().str.lower()
    o_uom = o["valueuom"].astype(str).str.strip().str.lower()
    i = i[i_uom.isin(_VOLUME_UOMS)]
    o = o[o_uom.isin(_VOLUME_UOMS)]
    i_uom = i_uom[i_uom.isin(_VOLUME_UOMS)]
    o_uom = o_uom[o_uom.isin(_VOLUME_UOMS)]

    i.loc[i_uom == "l", "amount"] *= 1000
    o.loc[o_uom == "l", "value"] *= 1000

    i = i[(i["amount"] >= 0) & (i["amount"] <= 50000)]
    o = o[(o["value"] >= 0) & (o["value"] <= 50000)]

    total_in = i.groupby("stay_id")["amount"].sum().reset_index(name="total_fluid_input")
    total_out = o.groupby("stay_id")["value"].sum().reset_index(name="total_fluid_output")

    # Outer merge so a stay with only one side is kept; the absent side is a
    # true zero, not a missing value. Stays with no records at all never appear
    # here and stay NaN after the left merge below.
    balance = total_in.merge(total_out, on="stay_id", how="outer")
    balance[["total_fluid_input", "total_fluid_output"]] = balance[
        ["total_fluid_input", "total_fluid_output"]].fillna(0)

    balance[out_col] = (
        balance["total_fluid_input"] - balance["total_fluid_output"]
    ).clip(-20000, 20000)
    balance = balance.drop(columns=["total_fluid_input", "total_fluid_output"])

    return df.merge(balance, on="stay_id", how="left")


def get_pre24h_fluid_balance(df, inputevents, outputevents):
    return _fluid_balance(df, inputevents, outputevents,
                          "pre_start", "pre_end", "fluid_balance_pre24h_ml")


def get_0_24h_fluid_balance(df, inputevents, outputevents):
    return _fluid_balance(df, inputevents, outputevents,
                          "treatment_start", "treatment_end", "fluid_balance_0to24h_ml")

def get_mortality_28day(df):
    df = df.copy()
    df["sepsis_onset_time"] = pd.to_datetime(df["sepsis_onset_time"], errors="coerce")
    df["deathtime"] = pd.to_datetime(df["deathtime"], errors="coerce")
    days_to_death = (df["deathtime"] - df["sepsis_onset_time"]).dt.total_seconds() / (24 * 60 * 60)
    df["mortality_28day"] = days_to_death.between(0,28,inclusive = "both")
    return df
