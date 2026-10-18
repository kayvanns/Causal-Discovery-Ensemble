# Sepsis causal discovery ensemble

Constraint-based causal discovery (PC and FCI) over a MIMIC-IV sepsis cohort,
run as an ensemble across imputation strategies on CHTC. Outcome is
`aki_24to48h_max_stage`, the max AKI stage in onset+24h..onset+48h.

## Pipeline

```
sql/causal_discovery_features_ed.sql   (run by hand in BigQuery)
        |
        v
cohort/cohort.csv --build_ed.py--> cohort/analysis_final.csv --datasets.py--> data/*.csv
        |                                                                        |
        v                                                              chtc/bundle.sh -> CHTC
sql/cohort_attrition.sql -> cohort/cohort_attrition.csv                          |
                                                                                 v
                    results/*.csv <-- summarize_discovery_runs.py <-- results/graphs/*.pkl
                    results/heatmap.png <-- heatmap.py
```

```bash
python build_ed.py cohort/cohort.csv cohort/analysis_final.csv
python datasets.py                    # writes data/ -- KNN is the slow step
chtc/bundle.sh                        # stage chtc_bundle/, then submit (see CHTC below)
python summarize_discovery_runs.py    # reads results/graphs/, writes results/
python heatmap.py
```

`build_ed.py` defaults to writing `cohort/analysis.csv`; pass the output path
explicitly. `cohort/analysis.csv` is an older build that keyed baseline labs on
`hadm_id` and so dropped the NULL-`hadm_id` ED draws — don't use it.

## Temporal tiers

Background knowledge is a tier ordering defined once in `tiers.py`. Edges may
only run forward in tier; `age`, `gender`, `race` take no incoming edges.

| Tier | Contents |
| --- | --- |
| 0 | Baseline demographics, comorbidity, `onset_to_icu_hrs` |
| 1 | Pre-sepsis physiology (lookback ends **at** onset) |
| 2 | All 0-24h post-onset variables |
| 3 | 24-48h outcomes |
| 4 | 28-day mortality |

`onset_to_icu_hrs` is tier 0, not a nuisance term. It governs tier-1
missingness: the vitals/SOFA source tables are `stay_id`-keyed and hold nothing
before `icu_intime`, and 86% of this cohort is already septic on arrival. Leave
it out and that structure becomes a latent common cause of every tier-1
indicator, which PC can only express as a spurious clique.

Tier 2 is flat. Treatments given in the first 24h and the physiology measured
over that window overlap in time, so no sub-ordering is defensible.

## Outcome recode

1,129 rows (12.5%) have no 24-48h creatinine. `datasets.py` splits them:

- **died before onset+48h** -> stage 4. Judged against the full 48h bound, not
  one truncated by `icu_outtime`, so a patient discharged at 30h who dies at
  40h is counted here rather than falling through to the next case.
- **discharged alive with <= 6h of the window** -> stage 0. No realistic chance
  of a draw, so missing means no AKI.
- **everything else** -> dropped (48 rows). The patient was present and alive
  long enough that missing means not-measured, and asserting stage 0 would be
  a free pass.

Leaves 8,958 rows. Use `missingness_report.py --raw` to see rates before this
step.

## Missingness

Indicators are carried per mechanism, not per column. Membership and the
evidence for it are in `tiers.py`; regenerate with `indicator_grouping.py` and
`baseline_window_check.py`.

| Handling | Columns | Why |
| --- | --- | --- |
| `bloodwork_panel_missing` | 9 chemistry/heme analytes | Ordered as one panel; within-group r 0.71-0.99 |
| `vitals_panel_missing` | heart_rate, blood_pressure, spo2, temperature_c | Monitoring cadence, a different mechanism — only r~0.22 against bloodwork |
| `inr_missing` | inr (40.9%) | r=0.434 against the panel, so not the panel's mechanism |
| `lactate_0to24h_max_missing` | lactate_0to24h_max (14.2%) | ~7x the next tier-2 column |
| sensitivity only | lactate (78.6%), bilirubin (52.6%) | Ordering-driven, plausibly MNAR — dropped from main entirely |

The lactate/bilirubin split rests on the `onset_to_icu_hrs >= 72` subgroup,
where the window mechanism is removed and both stay above 70% while the panels
fall under 1%. Nine tier-2 columns carry <2% missingness with no indicator:
deliberate, since an indicator that flags 2 rows is near-constant and pushes
the Fisher-Z correlation matrix toward singularity.

## Datasets

`datasets.py` writes six files. Imputation is fit per tier, using only that
tier's columns plus tier 0, so a later tier can't leak into an earlier one.

| File | Contents |
| --- | --- |
| `main_raw.csv` | Values only, NaNs preserved — for `mv_fisherz` |
| `main_simple.csv` | Median/mode imputed + indicators |
| `main_knn.csv` | KNN imputed + indicators |
| `sens_simple.csv` / `sens_knn.csv` | Adds lactate + bilirubin and their indicators |
| `main_simple_with_mortality.csv` | One-off FCI runtime check, not core |

## Runs

The final ensemble is 12 runs, 7 PC and 5 FCI, in `results/graphs/`. Nine
use the main cohort; three are sensitivity runs on the `sens_*` datasets,
which add the high-missingness lactate and bilirubin columns and their
indicators. Raw data pairs with `mv_fisherz` (test-wise deletion); imputed
data with Fisher-Z (linear) or RCIT (nonparametric).

| Run | Algorithm | Test | Data | n |
| --- | --- | --- | --- | --- |
| `pc_mv_main` | PC | mv_fisherz | `main_raw` | full |
| `pc_fisherz_main_simple` | PC | Fisher-Z | `main_simple` | full |
| `pc_fisherz_main_knn` | PC | Fisher-Z | `main_knn` | full |
| `pc_rcit_main_simple` | PC | RCIT | `main_simple` | 7,500 |
| `pc_rcit_main_knn` | PC | RCIT | `main_knn` | 7,500 |
| `fci_mv_main` | FCI | mv_fisherz | `main_raw` | 5,000 |
| `fci_fisherz_main_simple` | FCI | Fisher-Z | `main_simple` | 5,000 |
| `fci_fisherz_main_knn` | FCI | Fisher-Z | `main_knn` | 5,000 |
| `fci_rcit_main_simple` | FCI | RCIT | `main_simple` | 5,000 |
| `pc_fisherz_sens_simple` | PC | Fisher-Z | `sens_simple` | full |
| `pc_fisherz_sens_knn` | PC | Fisher-Z | `sens_knn` | full |
| `fci_fisherz_sens_simple` | FCI | Fisher-Z | `sens_simple` | 5,000 |

The last three rows are the sensitivity runs. "Full" is all 8,958 rows.

| | PC | FCI |
| --- | --- | --- |
| alpha | 0.01 | 0.01 |
| depth | unlimited | 4 |
| starting memory | 8GB | 16GB (both double on each retry) |

## CHTC

CHTC jobs run from a single flat directory on the access point, but the repo
keeps the cluster files in `chtc/` and the shared modules at the root.
`chtc/bundle.sh` bridges the two by copying everything a job needs into a
gitignored `chtc_bundle/`:

```
chtc_bundle/
├── causal_discovery.py  tiers.py  causal_discovery.sh
├── pc.sub  fci.sub  pc_runs.txt  fci_runs.txt
├── data/*.csv
└── logs/
```

```bash
chtc/bundle.sh
scp -r chtc_bundle/ <netid>@<ap>.chtc.wisc.edu:~/sepsis
# on the access point
cd ~/sepsis && condor_submit pc.sub && condor_submit fci.sub
# back here, once jobs finish
scp '<netid>@<ap>.chtc.wisc.edu:~/sepsis/*.{pkl,png}' results/graphs/
```

Each job returns `<run_name>.pkl` (pickled `(graph, col_names)`) and
`<run_name>.png`. Edit the submit files and run lists in `chtc/`, never in the
bundle, then re-run `bundle.sh`.

## Layout

| Path | Contents |
| --- | --- |
| root `*.py` | Flat on purpose: scripts import `tiers`/`datasets` by bare name, and `causal_discovery.py` + `tiers.py` must sit side by side on CHTC |
| `sql/` | BigQuery extracts, run by hand |
| `chtc/` | Submit files, job wrapper, run lists, `bundle.sh` |
| `results/` | Ensemble summary tables and `heatmap.png` |
| `results/graphs/` | Final ensemble: one `.pkl` + `.png` per run, same base name |
| `reports/` | Missingness and grouping inventories (CSVs gitignored) |
| `cohort/`, `data/` | Gitignored. Credentialed MIMIC-IV data, never commit |
| `chtc_bundle/`, `logs/` | Gitignored staging and job logs |
