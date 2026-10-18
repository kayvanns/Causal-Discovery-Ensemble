-- ============================================================================
-- Causal discovery feature build (SQL portion)
-- Changes from prior version:
--   1. mechvent_pre_onset: dropped "endtime > onset" condition -- now "ever
--      ventilated before onset," matching the target trial exclusion logic,
--      not "actively mid-ventilation at the instant of onset."
--   2. Baseline ("latest value") windows tightened from onset+12h to onset
--      across sofa_pre24h, vitals_window, fio2_window, chemistry_window.
--      They end AT onset, not after it: a tier-1 "pre-sepsis" variable must
--      not be able to carry post-onset information.
--   3. aki_24to48h window bounded to LEAST(onset+48h, icu_outtime) instead of
--      open-ended to icu_outtime, to keep the outcome proximal to the 0-24h
--      exposure window rather than picking up unrelated late-ICU-stay events.
--   4. NEW: at-sepsis-onset extreme-value CTEs (0-24h post onset, i.e.
--      > onset AND <= onset+24h) for SOFA, HR, BP (MAP), SpO2, temp,
--      creatinine -- one direction each, chosen for clinical relevance.
--      (Lactate and wbc extremes are built in Python -- see features.py.)
--      These meet the baseline windows exactly at onset: no gap, no overlap.
--      These are named _0to24h_ after the window they actually cover. The
--      earlier _1to24h_ names were a leftover from the pre-shift windows and
--      described the wrong interval; they were renamed together with the
--      24-48h outcome columns (formerly post24h, which read as open-ended but
--      is bounded to onset+48h or icu_outtime). Any pc_*.pkl / fci_*.pkl graph
--      saved before that rename uses the old labels and is not comparable.
--   5. NEW: ED backfill of baseline vitals -- see ed_vitals_window below.
--   6. NEW: ED provenance columns (came_via_ed, ed_intime, ed_outtime,
--      n_ed_stays, ed_arrival_to_icu_hrs, ed_depart_to_icu_hrs,
--      admission_location) -- see ed_link below. These FLAG, they do not
--      filter: every row the parent query returns is still returned here, so
--      candidate ED->ICU cutoffs can be swept in Python off one export
--      instead of one BigQuery run per cutoff. No WHERE clause was added.
--   7. Baseline VITALS lookback shortened from 72h to 24h (vitals_window and
--      ed_vitals_window). A 72h-old vital is not a pre-sepsis baseline in a
--      population selected for reaching the ICU within a day or two of ED
--      arrival, and the completeness numbers have to be measured on the
--      window we will actually run. Vitals only: fio2_window and
--      chemistry_window stay at 72h, as does the Python lab lookback in
--      features.py get_labs(), because labs are ordered on a much slower
--      cadence and that lookback was tuned separately (probe_lab_lookback.sql).
--      Consequence: the four baseline vitals are strictly less complete here
--      than in cohort/base_ed.csv. That is the window, not the ED linkage --
--      do not read it as an ED effect.
--   8. Baseline/extreme chemistry keyed on subject_id + an episode-start guard
--      instead of hadm_id -- see chemistry_window. Labs drawn before the
--      inpatient admission is registered carry a NULL hadm_id and were being
--      dropped by the join, which hit ED->ICU patients hardest because their
--      sepsis onset precedes admittime. Measured on labevents directly,
--      baseline creatinine coverage in the ED<=24h stratum goes 23% -> 83%.
--   9. FINAL COHORT. This file now returns the analysis cohort, not all
--      sepsis3 stays. Criteria, applied in this order (sql/cohort_attrition.sql
--      reports the count after each):
--        1. linked pre-ICU ED stay                        (ed_link)
--        2. ED arrival -> icu_intime <= 24h
--        3. no ESRD / chronic dialysis                    (esrd_dialysis)
--        4. not mechanically ventilated before onset      (mechvent_pre_onset)
--        5. first stay per patient that passes 1-4        (cohort_included)
--      Applied as an inner join in the final SELECT; every feature CTE is
--      still computed per stay, so no feature value depends on the filter.
--      mechvent_pre_onset is constant (0) in the output as a consequence of
--      criterion 4 and is removed from tiers.py.
--      Also: chemistry_window ties on charttime are broken by specimen_id, so
--      repeated runs return identical values (they did not before).
-- ============================================================================

WITH sepsis AS (
SELECT
  s.subject_id,
  i.hadm_id,
  s.stay_id,
  i.intime  AS icu_intime,
  i.outtime AS icu_outtime,
  s.suspected_infection_time AS sepsis_onset_time
FROM `physionet-data.mimiciv_3_1_derived.sepsis3` s
JOIN `physionet-data.mimiciv_3_1_icu.icustays` i
  ON s.stay_id = i.stay_id
WHERE s.sepsis3 = TRUE
),

/* ---------- AKI during the 24h before sepsis onset ---------- */
aki_pre24h AS (
SELECT
  k.stay_id,
  MAX(k.aki_stage_smoothed) AS aki_pre24h_max_stage,
  MAX(CASE WHEN k.aki_stage_smoothed >= 1 THEN 1 ELSE 0 END) AS aki_pre24h_any
FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages` k
JOIN sepsis s ON k.stay_id = s.stay_id
WHERE k.charttime >= TIMESTAMP_SUB(s.sepsis_onset_time, INTERVAL 24 HOUR)
  AND k.charttime < s.sepsis_onset_time
GROUP BY k.stay_id
),

/* ---------- AKI within 0-24h ---------- */
aki_0to24h AS (
SELECT
  k.stay_id,
  MAX(k.aki_stage_smoothed) AS aki_0to24h_max_stage,
  MAX(CASE WHEN k.aki_stage_smoothed >= 1 THEN 1 ELSE 0 END) AS aki_0to24h_any
FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages` k
JOIN sepsis s ON k.stay_id = s.stay_id
WHERE k.charttime >= s.sepsis_onset_time
  AND k.charttime <  TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR)
GROUP BY k.stay_id
),

/* ---------- AKI after 24h, bounded to a 48h-post-onset window (or discharge, whichever is first) ---------- */
aki_24to48h AS (
SELECT
  k.stay_id,
  MAX(k.aki_stage_smoothed) AS aki_24to48h_max_stage,
  MAX(CASE WHEN k.aki_stage_smoothed >= 1 THEN 1 ELSE 0 END) AS aki_24to48h_any
FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages` k
JOIN sepsis s ON k.stay_id = s.stay_id
WHERE k.charttime >= TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR)
  AND k.charttime <= LEAST(
        TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 48 HOUR),
        s.icu_outtime
      )
GROUP BY k.stay_id
),

/* ---------- Mechanical ventilation ever present before sepsis onset ---------- */
mechvent_pre_onset AS (
SELECT
  v.stay_id,
  1 AS mechvent_pre_onset_any
FROM `physionet-data.mimiciv_3_1_derived.ventilation` v
JOIN sepsis s ON v.stay_id = s.stay_id
WHERE v.ventilation_status IN ('InvasiveVent', 'Tracheostomy')
  AND v.starttime < s.sepsis_onset_time
GROUP BY v.stay_id
),

/* ---------- New mechanical ventilation initiation during 0-24h ---------- */
mechvent_0to24h AS (
SELECT
  v.stay_id,
  1 AS mechvent_0to24h_any
FROM `physionet-data.mimiciv_3_1_derived.ventilation` v
JOIN sepsis s ON v.stay_id = s.stay_id
WHERE v.ventilation_status IN ('InvasiveVent', 'Tracheostomy')
  AND v.starttime >= s.sepsis_onset_time
  AND v.starttime < TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR)
GROUP BY v.stay_id
),

/* ---------- New mechanical ventilation initiation after 24h ---------- */
mechvent_24to48h AS (
SELECT
  v.stay_id,
  1 AS mechvent_24to48h_any
FROM `physionet-data.mimiciv_3_1_derived.ventilation` v
JOIN sepsis s ON v.stay_id = s.stay_id
WHERE v.ventilation_status IN ('InvasiveVent', 'Tracheostomy')
  AND v.starttime >= TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR)
  AND v.starttime <= LEAST(
        TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 48 HOUR),
        s.icu_outtime
      )
GROUP BY v.stay_id
),

/* ---------- CRRT after 24h from sepsis onset ---------- */
crrt_first_time AS (
SELECT
  stay_id,
  MIN(charttime) AS first_crrt_time
FROM `physionet-data.mimiciv_3_1_derived.crrt`
WHERE crrt_mode IS NOT NULL
GROUP BY stay_id
),

crrt_24to48h AS (
SELECT
  c.stay_id,
  1 AS crrt_24to48h_any
FROM crrt_first_time c
JOIN sepsis s ON c.stay_id = s.stay_id
WHERE c.first_crrt_time >= TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR)
  AND c.first_crrt_time <= LEAST(
        TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 48 HOUR),
        s.icu_outtime
      )
),

/* ---------- CRRT within 24h of sepsis onset ---------- */
crrt_0to24h AS (
SELECT
  c.stay_id,
  1 AS crrt_0to24h_any
FROM crrt_first_time c
JOIN sepsis s ON c.stay_id = s.stay_id
WHERE c.first_crrt_time >= s.sepsis_onset_time
  AND c.first_crrt_time < TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR)
),

/* ---------- SOFA: baseline (latest, <= onset, 24h lookback) ----------
  The 24h here is not a 72h window that was forgotten. derived.sofa is only
  computed from ICU intime onward, and a large share of this cohort has a
  sepsis onset that PRECEDES ICU admission, so their baseline window contains
  no sofa rows at any lookback length. Widening 24h -> 72h would not recover
  those stays; sofa_pre24h is missingness-dominated either way. Left at 24h. */
sofa_pre24h AS (
SELECT
  stay_id,
  sofa_24hours AS sofa_pre24h,
  endtime AS sofa_window_end
FROM (
  SELECT
    s.stay_id,
    sf.sofa_24hours,
    sf.endtime,
    ROW_NUMBER() OVER (PARTITION BY s.stay_id ORDER BY sf.endtime DESC) AS rn
  FROM sepsis s
  JOIN `physionet-data.mimiciv_3_1_derived.sofa` sf ON sf.stay_id = s.stay_id
  WHERE sf.endtime >= DATETIME_SUB(DATETIME(s.sepsis_onset_time), INTERVAL 24 HOUR)
    AND sf.endtime <= DATETIME(s.sepsis_onset_time)
)
WHERE rn = 1
),

/* ---------- SOFA: at-onset extreme (max, 0-24h post onset) ---------- */
sofa_onset_extreme AS (
SELECT
  s.stay_id,
  MAX(sf.sofa_24hours) AS sofa_max
FROM sepsis s
JOIN `physionet-data.mimiciv_3_1_derived.sofa` sf ON sf.stay_id = s.stay_id
WHERE sf.endtime > s.sepsis_onset_time
  AND sf.endtime <= TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR)
GROUP BY s.stay_id
),

/* ---------- CKD at baseline ---------- */
ckd_baseline AS (
SELECT
  d.hadm_id,
  1 AS ckd_baseline
FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
JOIN sepsis s ON d.hadm_id = s.hadm_id
WHERE (
  SUBSTR(d.icd_code, 1, 3) IN ('N18', 'N19')
  OR SUBSTR(d.icd_code, 1, 4) IN ('I120', 'I131', 'N250', 'Z940', 'Z992',
                                    'Z490', 'Z491', 'Z492')
  OR SUBSTR(d.icd_code, 1, 3) IN ('582', '585', '586')
  OR SUBSTR(d.icd_code, 1, 4) IN ('5880', 'V420', 'V451')
)
AND d.icd_version IN (9, 10)
GROUP BY d.hadm_id
),

/* ---------- ESRD / chronic dialysis (exclusion) ----------
  The outcome is AKI stage, which is not meaningful for a patient on
  maintenance dialysis. Deliberately narrower than ckd_baseline, which covers
  every CKD stage and stays in the model as a variable. Prefix matches, since
  diagnoses_icd stores codes without dots.
    ICD-10  N186 ESRD | Z992 dialysis dependence | Z491, Z492 dialysis encounter
    ICD-9   5856 ESRD | V451 renal dialysis status (V4511, V4512)
            V560, V568 dialysis encounter */
esrd_dialysis AS (
SELECT d.hadm_id
FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
JOIN sepsis s ON d.hadm_id = s.hadm_id
WHERE (d.icd_version = 10 AND (STARTS_WITH(d.icd_code, 'N186')
                                OR STARTS_WITH(d.icd_code, 'Z992')
                                OR STARTS_WITH(d.icd_code, 'Z491')
                                OR STARTS_WITH(d.icd_code, 'Z492')))
    OR (d.icd_version = 9  AND (STARTS_WITH(d.icd_code, '5856')
                                OR STARTS_WITH(d.icd_code, 'V451')
                                OR STARTS_WITH(d.icd_code, 'V560')
                                OR STARTS_WITH(d.icd_code, 'V568')))
GROUP BY d.hadm_id
),

/* ---------- ED vitals: source rows for the baseline backfill ----------
  MIMIC-IV-ED is a separate PhysioNet module but sits in the same BigQuery
  project and shares the subject_id / hadm_id namespace.

  WHY: derived.vitalsign is chartevents-derived and keyed by ICU stay_id, so
  the part of the -72h baseline window that falls before ICU intime matches no
  rows at all. For ED-admitted sepsis patients that is usually most of the
  window, and it is the main driver of baseline vitals missingness.

  LINKAGE -- the link is edstays.hadm_id = s.hadm_id, which pins the ED visit to the exact
  hospital admission this cohort row belongs to; subject_id is kept as a
  redundant guard. The ED stay_id is a DIFFERENT namespace from the ICU
  stay_id and the two are never equated. */
ed_stays AS (
SELECT
  s.stay_id     AS icu_stay_id,
  e.stay_id     AS ed_stay_id,
  e.intime      AS ed_intime,
  e.outtime     AS ed_outtime
FROM sepsis s
JOIN `physionet-data.mimiciv_ed.edstays` e
  ON  e.hadm_id    = s.hadm_id
  AND e.subject_id = s.subject_id
),

/* ---------- ED provenance: one row per ICU stay ----------
  Which ED stay: the latest one starting at or before icu_intime, i.e. the
  visit the patient actually came to the unit from. 

  A cohort row with no qualifying ED stay gets no row here, which is exactly
  came_via_ed = 0 in the final select. n_ed_stays counts only the qualifying
  ones, so > 1 means the patient had multiple pre-ICU ED visits on this
  admission and the most recent was taken. */
ed_link AS (
SELECT
  s.stay_id,
  COUNT(*) AS n_ed_stays,
  ARRAY_AGG(es.ed_intime  ORDER BY es.ed_intime DESC LIMIT 1)[SAFE_OFFSET(0)] AS ed_intime,
  ARRAY_AGG(es.ed_outtime ORDER BY es.ed_intime DESC LIMIT 1)[SAFE_OFFSET(0)] AS ed_outtime
FROM sepsis s
JOIN ed_stays es ON es.icu_stay_id = s.stay_id
WHERE es.ed_intime <= s.icu_intime
GROUP BY s.stay_id
),

/* ---------- Final cohort: per-stay criteria ----------
  One 0/1 column per criterion, so sql/cohort_attrition.sql can count each
  step from exactly the same definitions. KEEP THIS CTE AND cohort_included
  IDENTICAL TO THE COPIES IN sql/cohort_attrition.sql -- the attrition query's
  last step must equal this file's row count, and that check only means
  something if the logic is shared. */
cohort_flags AS (
SELECT
  s.stay_id,
  s.subject_id,
  s.icu_intime,
  IF(el.stay_id IS NOT NULL
      AND TIMESTAMP_DIFF(s.icu_intime, el.ed_intime, MINUTE) / 60 BETWEEN 0 AND 24,
      1, 0)                                         AS ed_within_24h,
  IF(esrd.hadm_id IS NULL, 1, 0)                   AS no_esrd_dialysis,
  IF(mpre.stay_id IS NULL, 1, 0)                   AS no_mechvent_pre_onset
FROM sepsis s
LEFT JOIN ed_link            el   ON el.stay_id   = s.stay_id
LEFT JOIN esrd_dialysis      esrd ON esrd.hadm_id = s.hadm_id
LEFT JOIN mechvent_pre_onset mpre ON mpre.stay_id = s.stay_id
),

/* First stay per patient among the stays that pass every other criterion,
  so a patient whose earliest sepsis stay fails (say, came via the ward) is
  still represented by a later qualifying stay. stay_id breaks icu_intime
  ties so the choice is deterministic. */
cohort_included AS (
SELECT stay_id
FROM (
  SELECT
    stay_id,
    ROW_NUMBER() OVER (PARTITION BY subject_id
                        ORDER BY icu_intime, stay_id) AS rn
  FROM cohort_flags
  WHERE ed_within_24h = 1
    AND no_esrd_dialysis = 1
    AND no_mechvent_pre_onset = 1
)
WHERE rn = 1
),

/* ---------- Start of this care episode ----------
  Needed by the chemistry CTEs below to decide whether a lab with no hadm_id
  belongs to this encounter. ED arrival where there is one, because admittime
  is registered hours AFTER the ED workup that produced the baseline labs --
  for this cohort sepsis onset lands a median 1.8h BEFORE admittime. */
episode AS (
SELECT
  s.stay_id,
  s.subject_id,
  s.hadm_id,
  s.sepsis_onset_time,
  LEAST(COALESCE(el.ed_intime, adm.admittime), adm.admittime) AS episode_start
FROM sepsis s
LEFT JOIN ed_link el ON el.stay_id = s.stay_id
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` adm ON adm.hadm_id = s.hadm_id
),

/* Two ED sources unioned into one timeline:
    - ed.vitalsign: repeated observations, has its own charttime
    - ed.triage:    the single arrival observation, no charttime of its own,
                    so edstays.intime is used. Often the ONLY pre-ICU vital
                    set on record, which is exactly the gap being filled.
  Column names differ from derived.vitalsign (heartrate/o2sat, no mbp), MAP is
  derived as (sbp + 2*dbp)/3, and temperature is in FAHRENHEIT here while
  derived.vitalsign is Celsius -- unconverted it would silently corrupt
  temperature_c. ED values are raw rather than cleaned the way the derived
  table is, so each is range-guarded to NULL rather than trusted. */
ed_vitals_src AS (
SELECT
  es.icu_stay_id,
  v.charttime,
  1 AS src_rank,   -- prefer a real vitalsign row over triage at the same time
  IF(v.heartrate BETWEEN 20 AND 300, v.heartrate, NULL) AS heart_rate,
  IF(v.sbp BETWEEN 40 AND 300 AND v.dbp BETWEEN 20 AND 200,
      (v.sbp + 2 * v.dbp) / 3, NULL)                     AS blood_pressure,
  IF(v.o2sat BETWEEN 50 AND 100, v.o2sat, NULL)         AS spo2,
  IF((v.temperature - 32) * 5 / 9 BETWEEN 25 AND 45,
      (v.temperature - 32) * 5 / 9, NULL)                AS temperature_c
FROM ed_stays es
JOIN `physionet-data.mimiciv_ed.vitalsign` v ON v.stay_id = es.ed_stay_id

UNION ALL

SELECT
  es.icu_stay_id,
  es.ed_intime AS charttime,
  0 AS src_rank,
  IF(t.heartrate BETWEEN 20 AND 300, t.heartrate, NULL) AS heart_rate,
  IF(t.sbp BETWEEN 40 AND 300 AND t.dbp BETWEEN 20 AND 200,
      (t.sbp + 2 * t.dbp) / 3, NULL)                     AS blood_pressure,
  IF(t.o2sat BETWEEN 50 AND 100, t.o2sat, NULL)         AS spo2,
  IF((t.temperature - 32) * 5 / 9 BETWEEN 25 AND 45,
      (t.temperature - 32) * 5 / 9, NULL)                AS temperature_c
FROM ed_stays es
JOIN `physionet-data.mimiciv_ed.triage` t ON t.stay_id = es.ed_stay_id
),

/* ---------- ED vitals: baseline (latest non-null, <= onset, 24h lookback) ----------
  Same window and same latest-non-null idiom as vitals_window below, so the
  two are directly substitutable -- and they must stay that way, so this
  window was shortened to 24h together with vitals_window (change #7). The
  lower bound matters here too: a triage observation from a visit weeks
  earlier must never become a baseline.
  SAFE_OFFSET rather than OFFSET because ED coverage is patchy -- a stay with
  no usable value at all for one vital yields an empty array, and plain
  OFFSET(0) would fail the whole query rather than return NULL.
  Grouping to s.stay_id also collapses multiple ED stays under one hadm_id, so
  this can never fan out cohort rows. */
ed_vitals_window AS (
SELECT
  s.stay_id,
  ARRAY_AGG(ev.heart_rate     IGNORE NULLS ORDER BY ev.charttime DESC, ev.src_rank DESC LIMIT 1)[SAFE_OFFSET(0)] AS heart_rate,
  ARRAY_AGG(ev.blood_pressure IGNORE NULLS ORDER BY ev.charttime DESC, ev.src_rank DESC LIMIT 1)[SAFE_OFFSET(0)] AS blood_pressure,
  ARRAY_AGG(ev.spo2           IGNORE NULLS ORDER BY ev.charttime DESC, ev.src_rank DESC LIMIT 1)[SAFE_OFFSET(0)] AS spo2,
  ARRAY_AGG(ev.temperature_c  IGNORE NULLS ORDER BY ev.charttime DESC, ev.src_rank DESC LIMIT 1)[SAFE_OFFSET(0)] AS temperature_c
FROM sepsis s
JOIN ed_vitals_src ev ON ev.icu_stay_id = s.stay_id
WHERE ev.charttime >= TIMESTAMP_SUB(s.sepsis_onset_time, INTERVAL 24 HOUR)
  AND ev.charttime <= s.sepsis_onset_time
GROUP BY s.stay_id
),

/* ---------- Vitals: baseline (latest non-null, <= onset, 24h lookback) ---------- */
vitals_window AS (
SELECT
  s.stay_id,
  ARRAY_AGG(v.heart_rate IGNORE NULLS ORDER BY v.charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS heart_rate,
  ARRAY_AGG(v.mbp        IGNORE NULLS ORDER BY v.charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS blood_pressure,
  ARRAY_AGG(v.spo2       IGNORE NULLS ORDER BY v.charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS spo2,
  ARRAY_AGG(v.temperature IGNORE NULLS ORDER BY v.charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS temperature_c
FROM sepsis s
JOIN `physionet-data.mimiciv_3_1_derived.vitalsign` v ON s.stay_id = v.stay_id
WHERE v.charttime >= TIMESTAMP_SUB(s.sepsis_onset_time, INTERVAL 24 HOUR)
  AND v.charttime <= s.sepsis_onset_time
GROUP BY s.stay_id
),

/* ---------- Vitals: at-onset extremes (0-24h post onset) ----------
  HR: max (tachycardia)      BP (MAP): min (hypotension)
  SpO2: min (hypoxemia)      Temp: max (fever) -- flagged as a judgment
  call with no existing precedent in the codebase; flip to MIN if you'd
  rather flag hypothermia, which is arguably the worse prognostic sign
  in septic shock. */
vitals_onset_extreme AS (
SELECT
  s.stay_id,
  MAX(v.heart_rate)   AS heart_rate_max,
  MIN(v.mbp)           AS blood_pressure_min,
  MIN(v.spo2)          AS spo2_min,
  MAX(v.temperature)   AS temperature_c_max
FROM sepsis s
JOIN `physionet-data.mimiciv_3_1_derived.vitalsign` v ON s.stay_id = v.stay_id
WHERE v.charttime > s.sepsis_onset_time
  AND v.charttime <= TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR)
GROUP BY s.stay_id
),

/* ---------- FiO2: baseline (latest, <= onset, 72h lookback) ---------- */
fio2_window AS (
SELECT
  s.stay_id,
  ARRAY_AGG(vs.fio2 IGNORE NULLS ORDER BY vs.charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS fio2
FROM sepsis s
JOIN `physionet-data.mimiciv_3_1_derived.ventilator_setting` vs ON s.stay_id = vs.stay_id
WHERE vs.charttime >= TIMESTAMP_SUB(s.sepsis_onset_time, INTERVAL 72 HOUR)
  AND vs.charttime <= s.sepsis_onset_time
GROUP BY s.stay_id
),

/* ---------- Chemistry: baseline (latest, <= onset, 72h lookback) ----------
  Keyed on subject_id, NOT hadm_id. About half of labevents -- which
  derived.chemistry is built from -- carries a NULL hadm_id, because the
  specimen was drawn before the inpatient admission was registered. For an
  ED->ICU patient that is most of the baseline window: 73.8% of them are
  septic before they are admitted at all, and the hadm_id join dropped 59.5%
  of their in-window draws against 13.1% for everyone else. That single join
  was the whole of the apparent "ED patients have no baseline labs" effect.

  The guard replacing it: a lab counts if it carries THIS admission's hadm_id,
  or carries none and was drawn at or after episode_start. A lab belonging to
  a DIFFERENT admission is excluded outright -- subject_id alone would let an
  unrelated earlier encounter inside the 72h window, the same hazard the
  ed_stays comment above spells out for vitals. */
chemistry_window AS (
SELECT
  s.stay_id,
  ARRAY_AGG(c.bicarbonate IGNORE NULLS ORDER BY c.charttime DESC, c.specimen_id DESC LIMIT 1)[SAFE_OFFSET(0)] AS bicarbonate,
  ARRAY_AGG(c.sodium      IGNORE NULLS ORDER BY c.charttime DESC, c.specimen_id DESC LIMIT 1)[SAFE_OFFSET(0)] AS sodium,
  ARRAY_AGG(c.potassium   IGNORE NULLS ORDER BY c.charttime DESC, c.specimen_id DESC LIMIT 1)[SAFE_OFFSET(0)] AS potassium,
  ARRAY_AGG(c.glucose     IGNORE NULLS ORDER BY c.charttime DESC, c.specimen_id DESC LIMIT 1)[SAFE_OFFSET(0)] AS glucose,
  ARRAY_AGG(c.bun         IGNORE NULLS ORDER BY c.charttime DESC, c.specimen_id DESC LIMIT 1)[SAFE_OFFSET(0)] AS bun,
  ARRAY_AGG(c.creatinine  IGNORE NULLS ORDER BY c.charttime DESC, c.specimen_id DESC LIMIT 1)[SAFE_OFFSET(0)] AS creatinine
FROM episode s
JOIN `physionet-data.mimiciv_3_1_derived.chemistry` c ON c.subject_id = s.subject_id
WHERE c.charttime >= TIMESTAMP_SUB(s.sepsis_onset_time, INTERVAL 72 HOUR)
  AND c.charttime <= s.sepsis_onset_time
  AND (c.hadm_id = s.hadm_id
        OR (c.hadm_id IS NULL AND c.charttime >= s.episode_start))
GROUP BY s.stay_id
),

/* ---------- Creatinine: at-onset extreme (max, 0-24h post onset) ----------
  NOTE: overlaps conceptually with aki_0to24h_any / aki_progression_24to48h,
  since creatinine is the primary input to KDIGO staging. Kept because it
  was explicitly named, but be aware a strong edge to the AKI flags in the
  discovery graph may partly reflect this definitional overlap rather than
  a novel finding. */
chemistry_onset_extreme AS (
SELECT
  s.stay_id,
  MAX(c.creatinine) AS creatinine_max
FROM episode s
JOIN `physionet-data.mimiciv_3_1_derived.chemistry` c ON c.subject_id = s.subject_id
WHERE c.charttime > s.sepsis_onset_time
  AND c.charttime <= TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR)
  AND (c.hadm_id = s.hadm_id
        OR (c.hadm_id IS NULL AND c.charttime >= s.episode_start))
GROUP BY s.stay_id
)

SELECT
s.*,

/* ED provenance. Flags only -- nothing here filters. came_via_ed is the
    presence of a qualifying pre-ICU ED stay on this admission;
    admission_location is hosp.admissions' independent account of where the
    patient came from, kept as a cross-check because two provenance
    definitions that disagree is something to know before picking one.
    ed_arrival_to_icu_hrs is the sweep variable. ed_depart_to_icu_hrs
    separates long ED boarding from a ward stopover, which is the first
    question any proposed cutoff invites. */
IF(el.stay_id IS NULL, 0, 1) AS came_via_ed,
el.ed_intime,
el.ed_outtime,
el.n_ed_stays,
TIMESTAMP_DIFF(s.icu_intime, el.ed_intime,  MINUTE) / 60 AS ed_arrival_to_icu_hrs,
TIMESTAMP_DIFF(s.icu_intime, el.ed_outtime, MINUTE) / 60 AS ed_depart_to_icu_hrs,
adm.admission_location,

/* AKI worsening during the first 24h after sepsis onset compared with the preceding 24h */
CASE
  WHEN apre.aki_pre24h_max_stage IS NULL OR a24.aki_0to24h_max_stage IS NULL THEN NULL
  WHEN a24.aki_0to24h_max_stage > apre.aki_pre24h_max_stage THEN 1
  ELSE 0
END AS aki_progression_0to24h,

/* AKI worsening after 24h (bounded to 48h post-onset or discharge) compared with the first 24h */
CASE
  WHEN a24.aki_0to24h_max_stage IS NULL OR ap.aki_24to48h_max_stage IS NULL THEN NULL
  WHEN ap.aki_24to48h_max_stage > a24.aki_0to24h_max_stage THEN 1
  ELSE 0
END AS aki_progression_24to48h,

a.age,
cc.charlson_comorbidity_index,
cb.scr_baseline AS creatinine_baseline,

apre.aki_pre24h_max_stage,
apre.aki_pre24h_any,
a24.aki_0to24h_max_stage,
a24.aki_0to24h_any,
ap.aki_24to48h_max_stage,
ap.aki_24to48h_any,

COALESCE(mpre.mechvent_pre_onset_any, 0) AS mechvent_pre_onset,
COALESCE(m24.mechvent_0to24h_any, 0)   AS mechvent_0to24h,
COALESCE(mp.mechvent_24to48h_any, 0)     AS mechvent_24to48h,

COALESCE(ckd.ckd_baseline, 0) AS ckd_baseline,
COALESCE(cp.crrt_24to48h_any, 0) AS crrt_24to48h,
COALESCE(c24.crrt_0to24h_any, 0) AS crrt_0to24h,

sp.sofa_pre24h,
soe.sofa_max                  AS sofa_0to24h_max,

/* ED backfill: keep the ICU-derived value wherever it exists, fall back to
    the ED value only where the ICU baseline is null. Values that are already
    present are never overwritten, so this reduces missingness without
    perturbing rows that were already complete. */
COALESCE(vw.heart_rate,     ev.heart_rate)     AS heart_rate,
COALESCE(vw.blood_pressure, ev.blood_pressure) AS blood_pressure,
COALESCE(vw.spo2,           ev.spo2)           AS spo2,
COALESCE(vw.temperature_c,  ev.temperature_c)  AS temperature_c,

voe.heart_rate_max            AS heart_rate_0to24h_max,
voe.blood_pressure_min        AS blood_pressure_0to24h_min,
voe.spo2_min                  AS spo2_0to24h_min,
voe.temperature_c_max         AS temperature_c_0to24h_max,

fw.fio2,

cw.bicarbonate,
cw.sodium,
cw.potassium,
cw.glucose,
cw.bun,
cw.creatinine,

coe.creatinine_max            AS creatinine_0to24h_max

FROM sepsis s
JOIN cohort_included ci ON ci.stay_id = s.stay_id   -- final cohort filter (change #9)
LEFT JOIN `physionet-data.mimiciv_3_1_derived.age`                 a   ON a.hadm_id = s.hadm_id
LEFT JOIN `physionet-data.mimiciv_3_1_derived.charlson`            cc  ON cc.hadm_id = s.hadm_id
LEFT JOIN `physionet-data.mimiciv_3_1_derived.creatinine_baseline` cb  ON cb.hadm_id = s.hadm_id
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions`             adm ON adm.hadm_id = s.hadm_id
LEFT JOIN ed_link el ON el.stay_id = s.stay_id
LEFT JOIN aki_pre24h apre ON s.stay_id = apre.stay_id
LEFT JOIN aki_0to24h a24 ON s.stay_id = a24.stay_id
LEFT JOIN aki_24to48h ap ON s.stay_id = ap.stay_id
LEFT JOIN crrt_24to48h cp ON s.stay_id = cp.stay_id
LEFT JOIN crrt_0to24h c24 ON c24.stay_id = s.stay_id
LEFT JOIN mechvent_pre_onset mpre ON mpre.stay_id = s.stay_id
LEFT JOIN mechvent_0to24h m24   ON m24.stay_id = s.stay_id
LEFT JOIN mechvent_24to48h mp     ON mp.stay_id = s.stay_id
LEFT JOIN ckd_baseline ckd ON ckd.hadm_id = s.hadm_id
LEFT JOIN vitals_window vw ON vw.stay_id = s.stay_id
LEFT JOIN ed_vitals_window ev ON ev.stay_id = s.stay_id
LEFT JOIN vitals_onset_extreme voe ON voe.stay_id = s.stay_id
LEFT JOIN fio2_window fw ON fw.stay_id = s.stay_id
LEFT JOIN chemistry_window cw ON cw.stay_id = s.stay_id
LEFT JOIN chemistry_onset_extreme coe ON coe.stay_id = s.stay_id
LEFT JOIN sofa_pre24h sp ON sp.stay_id = s.stay_id
LEFT JOIN sofa_onset_extreme soe ON soe.stay_id = s.stay_id
