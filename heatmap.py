import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle

INPUT = "results/heatmap_table.csv"
OUTPUT = "results/heatmap.png"
KEY_CAUSE, KEY_EFFECT = "mechvent_0to24h", "aki_24to48h_max_stage"

# Display labels, in tier order: baseline -> 0-24h -> 24-48h.
TIERS = {
    "Baseline": {
        "age": "Age", "gender": "Gender", "race": "Race",
        "charlson_comorbidity_index": "Charlson Index", "ckd_baseline": "CKD",
        "creatinine_baseline": "Creatinine (baseline)", "onset_to_icu_hrs": "Onset to ICU (h)",
        "heart_rate": "Heart Rate", "blood_pressure": "Blood Pressure",
        "temperature_c": "Temperature", "spo2": "SpO2", "wbc": "WBC",
        "platelet": "Platelet", "hemoglobin": "Hemoglobin", "inr": "INR", "bun": "BUN",
        "creatinine": "Creatinine", "glucose": "Glucose", "sodium": "Sodium",
        "potassium": "Potassium", "bicarbonate": "Bicarbonate",
        "vitals_panel_missing": "Vitals Missing", "bloodwork_panel_missing": "Bloodwork Missing",
        "inr_missing": "INR Missing",
    },
    "0-24h": {
        "heart_rate_0to24h_max": "Heart Rate (max)", "blood_pressure_0to24h_min": "Blood Pressure (min)",
        "temperature_c_0to24h_max": "Temperature (max)", "spo2_0to24h_min": "SpO2 (min)",
        "wbc_0to24h_max": "WBC (max)", "lactate_0to24h_max": "Lactate (max)",
        "lactate_0to24h_max_missing": "Lactate Missing", "creatinine_0to24h_max": "Creatinine (max)",
        "sofa_0to24h_max": "SOFA (max)", "fluid_balance_0to24h_ml": "Fluid Balance",
        "antibiotics_0to24h": "Antibiotics", "vasopressors_0to24h": "Vasopressors",
        "corticosteroids_0to24h": "Corticosteroids", "diuretics_0to24h": "Diuretics",
        "crrt_0to24h": "CRRT (0-24h)", "mechvent_0to24h": "Mech. Vent (0-24h)",
        "aki_0to24h_max_stage": "AKI Stage (0-24h)",
    },
    "24-48h": {
        "mechvent_24to48h": "Mech. Vent (24-48h)", "crrt_24to48h": "CRRT (24-48h)",
        "aki_24to48h_max_stage": "AKI Stage (24-48h)",
    },
}
LABELS = {raw: label for tier in TIERS.values() for raw, label in tier.items()}

df = pd.read_csv(INPUT)
n_runs = int(df["n_runs"].max())
print(df.head())

# Order: known variables in tier order, then any others (e.g. sens-only) alphabetically.
all_vars = set(df["cause"]) | set(df["effect"])
known = [v for v in LABELS if v in all_vars]
full_order = known + sorted(all_vars - set(known))

matrix = df.pivot_table(index="effect", columns="cause", values="agreement_score",
                        aggfunc="max", fill_value=0)
matrix = matrix.reindex(index=full_order, columns=full_order, fill_value=0)
matrix = matrix.rename(index=LABELS, columns=LABELS)

# Colour bins: 0 | 1-2 | 3-4 | 5-6 | 7-8 | 9 (for 9 runs); adapts to other run counts.
bounds = sorted({0, 1, *range(3, n_runs, 2), n_runs, n_runs + 1})
colors = ["#FFFFFF", "#FADBD8", "#F1948A", "#EF6657", "#F04E3C", "#C5050C", "#9B0000"]
colors = [colors[0]] + colors[-(len(bounds) - 2):]
cmap = LinearSegmentedColormap.from_list("uw", colors, N=len(colors))
norm = BoundaryNorm(bounds, ncolors=len(colors))

fig, ax = plt.subplots(figsize=(24, 22))
sns.heatmap(matrix, ax=ax, cmap=cmap, norm=norm, square=True,
            linewidths=0.5, linecolor="#E1E5E7",
            cbar_kws={"label": f"Runs with edge (of {n_runs})", "shrink": 0.6})

# Colour bar: label each bin by its range of run counts.
cbar = ax.collections[0].colorbar
cbar.set_ticks([(lo + hi) / 2 for lo, hi in zip(bounds[:-1], bounds[1:])])
cbar.set_ticklabels([str(lo) if hi - lo == 1 else f"{lo}-{hi - 1}"
                     for lo, hi in zip(bounds[:-1], bounds[1:])])
cbar.ax.set_ylabel(f"Runs with edge (of {n_runs})", fontsize=20)
cbar.ax.tick_params(labelsize=18)

# Tier separators.
pos = 0
for tier in list(TIERS.values())[:-1]:
    pos += sum(v in all_vars for v in tier)
    ax.axhline(pos, color="#121212", linewidth=2)
    ax.axvline(pos, color="#121212", linewidth=2)

# Outline the key exposure -> outcome cell.
if KEY_CAUSE in full_order and KEY_EFFECT in full_order:
    col, row = full_order.index(KEY_CAUSE), full_order.index(KEY_EFFECT)
    ax.add_patch(Rectangle((col, row), 1, 1, fill=False, edgecolor="#1F4E79", linewidth=4))

ax.set_xlabel("From", color="#121212", fontsize=25, fontweight="bold")
ax.set_ylabel("To", color="#121212", fontsize=25, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
plt.savefig(OUTPUT, dpi=300)