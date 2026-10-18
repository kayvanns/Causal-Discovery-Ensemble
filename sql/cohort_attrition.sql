-- ============================================================================
-- Cohort attrition (CONSORT) counts for sql/causal_discovery_features_ed.sql
--
-- One row per inclusion/exclusion step, cumulative, in the order the feature
-- SQL applies them. Every CTE below is a VERBATIM COPY from
-- causal_discovery_features_ed.sql -- change the criteria there and here
-- together.
--
-- DRIFT CHECK: step 5 n_stays must equal the row count of the feature SQL's
-- export. If they differ, the two files have diverged.
--
-- The last CONSORT box (primary outcome observable) is not here: that recode
-- and drop live in datasets.py, which prints its own counts.
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
)

SELECT 0 AS step, 'sepsis3 ICU stays' AS criterion,
       COUNT(*) AS n_stays, COUNT(DISTINCT subject_id) AS n_patients
FROM cohort_flags
UNION ALL
SELECT 1, 'with linked pre-ICU ED stay',
       COUNT(*), COUNT(DISTINCT cf.subject_id)
FROM cohort_flags cf JOIN ed_link el ON el.stay_id = cf.stay_id
UNION ALL
SELECT 2, 'ED arrival to ICU <= 24h',
       COUNT(*), COUNT(DISTINCT subject_id)
FROM cohort_flags WHERE ed_within_24h = 1
UNION ALL
SELECT 3, 'excluding ESRD / chronic dialysis',
       COUNT(*), COUNT(DISTINCT subject_id)
FROM cohort_flags WHERE ed_within_24h = 1 AND no_esrd_dialysis = 1
UNION ALL
SELECT 4, 'excluding ventilated before onset',
       COUNT(*), COUNT(DISTINCT subject_id)
FROM cohort_flags
WHERE ed_within_24h = 1 AND no_esrd_dialysis = 1 AND no_mechvent_pre_onset = 1
UNION ALL
SELECT 5, 'first qualifying stay per patient',
       COUNT(*), COUNT(DISTINCT cf.subject_id)
FROM cohort_included ci JOIN cohort_flags cf ON cf.stay_id = ci.stay_id
ORDER BY step
