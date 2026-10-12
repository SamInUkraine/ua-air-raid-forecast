# Forecasting air raid alert onsets in Ukraine

KSE AI Agentic School 2026 pet project: Python time-series and tabular forecasting with strict no-leakage evaluation.
**Scope:** this is a statistical exercise on *recorded alert messages*. It does not predict attacks and must not be
used for warning or safety decisions.

## Task
For every oblast and every hour `t` (UTC grid) in which **no alert is active** in that oblast, estimate the probability
that a **new** alert (oblast- or raion-level) **starts** within the next `k` hours, `k = 1, 3, 6`:

`onset_k(t) = 1` if any alert in the oblast starts in `(t, t+k]`, using only information available at `t`.

- An alert that starts exactly at `t` makes the oblast active at `t`, so `t` is not a forecast origin.
- An alert that starts exactly at `t+k` counts.
- Hours with an active alert are excluded, so the model is not a "current status" classifier.
- Overlapping raion alerts count once per hour: the label is "any start".
- Calendar features use Kyiv local time (DST handled by `Europe/Kyiv`).

## Data, and what the source really contains
[Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), which is updated daily.
The dataset is built from Telegram messages (`processors/*.py` in that repo): 🔴 opens an alert, 🟢 closes it.
Three properties of the source matter for leakage. Each was verified in the processor code and in the data:

1. **The CSV contains only finished alerts.** Alerts still open when the processor last ran are kept in
   `processors/pkl/*_active_alerts.pkl`. As a result, the last hours of the CSV are missing alert *starts*. Subtracting
   6 h from the last timestamp therefore does **not** make the labels complete.
   *Fix:* the open-alert snapshot from the same clone is attached (read with a restricted unpickler that creates no
   custom objects), and labels are kept only for `t + 6h <= data_cut`. Without the snapshot, the pipeline falls back to
   a conservative cut-off of `data_cut - 54h` and records a warning.
2. **Some end times are fake and were written retrospectively.** If a new 🔴 arrives for a location whose alert has been
   open for more than 3 h, the processor writes the old alert with `end = start + 1h` (official file) or
   `start + 30 min, naive=True` (volunteer file). In real time those hours looked **active**. The fake end creates fake
   "inactive" forecast origins that are almost always followed by an onset.
   - Evidence: in the official file, 255 rows last exactly 3600 s, compared with about 20 at 3599 s and 3601 s.
   - 248 of those 255 rows are followed by a new start at the same location more than 3 h later.
   - In the test period alone, 36 such alerts create about 2,350 fake inactive raion-hours.

   *Fix (`end_policy='causal'`):* the effective end of an imputed alert is the next start at the same location, which
   is the moment the processor itself learned it.
3. **The official feed stopped on 2026-09-07.** The last official CSV change was committed on 2026-09-08, and the
   processor's message counter has not moved since. The volunteer (eTryvoga) file is still updated (latest event
   2026-10-12 04:49 UTC), but it is a different source:
   - oblast level only;
   - parsed from keywords;
   - hourly agreement with the official data is weak on the frontline (Cohen's κ 0.04–0.25);
   - `naive` share is 5.0% overall, 9.6% in October 2026, and up to 48% for some regions in the last 6 weeks
     (`results/sources_check.txt`).

Cleaning (`features.load_events`):
- the official file contains a second full dump, so 113,845 exact duplicate rows are dropped;
- hromada-level alerts (long or permanent) and Luhanska oblast are excluded;
- rows with a missing or malformed time are counted and dropped (there are none in the current file);
- the panel starts on 2025-11-01, when raion-level alerting began to dominate.

All counts are written to `run_info.json`.

## No-leakage design
- **Features at `t`** use only:
  - starts in `(t-k, t]`;
  - the "active at t" state, meaning a 🔴 seen and no 🟢 yet (known live; fake ends removed, see above);
  - finishes `<= t`;
  - Kyiv calendar.

  The features are `hours_since_last_finish`, `starts_{1,3,6,24,168}h`, `oblast`, `nat_active_oblasts_cnt`,
  `nat_starts_{1,3}h`, `hour_of_day_kyiv` and `day_of_week_kyiv`. `is_active` is only the eligibility filter, never a
  model input.
- **Splits are derived from the data, with no hard-coded dates:**
  - test = the last 42 days ending at the last complete label;
  - validation = the 28 days before that;
  - train = everything earlier;
  - 24 h gaps between splits, longer than the 6 h horizon.
- **What uses which split:**
  - LightGBM early stopping, the choice of best baseline and the F1 thresholds use Val only;
  - the final model is refit on Train+Val with the Val-selected number of rounds;
  - Test is used once, for reporting.
- **Baselines:**
  - A: oblast × Kyiv hour rate;
  - B: rolling 168 h onset rate built from labels lagged by `k`, so only labels already resolved at `t` are used;
  - C: logistic regression on time since the last alert;
  - D: oblast rate;
  - Always-positive.
- **Tests** (`tests/test_no_leakage.py`, 30 tests, all calling the real functions):
  - censoring and future-perturbation invariance;
  - a **source-replay test** that rebuilds the CSV and snapshot "as of T" from a simulated message stream and checks
    that features at `t <= T` and labels up to the last valid origin are identical to the full data;
  - **negative controls** showing that the pre-audit behaviour (fake ends, no snapshot) fails that replay;
  - boundary semantics at `t`, `t+k` and the rolling-window edges;
  - baseline-B lag;
  - Kyiv time and DST;
  - split isolation;
  - malformed, duplicate and naive rows;
  - stale-data flag;
  - a censoring test on the real data.

## Results (executed 2026-10-12; numbers come from `results*/` and `results/sensitivity_summary.md`)
BSS is computed as `1 - Brier_LightGBM / Brier_baseline`, where the baseline is the one chosen on Val (here Baseline B).
CIs are day-block bootstrap 95% intervals (1,000 resamples) of LightGBM minus baseline.

**Official data, test 2026-07-27 00:00 – 2026-09-07 00:00 UTC (latest complete period of this source), Rear oblasts:**

| Horizon | N | Prevalence | AP LGB / B | Brier LGB / B | BSS | ΔAP 95% CI | ΔBrier 95% CI |
|---|---|---|---|---|---|---|---|
| 1 h | 15,351 | 0.112 | 0.374 / 0.379 | 0.0794 / 0.0799 | +0.6% | [-0.030, 0.022] | [-0.0023, 0.0011] |
| 3 h | 15,351 | 0.230 | 0.691 / 0.675 | 0.1075 / 0.1113 | +3.4% | [-0.010, 0.042] | [-0.0075, -0.0002] |
| 6 h | 15,351 | 0.317 | 0.818 / 0.799 | 0.1126 / 0.1193 | +5.6% | [-0.003, 0.044] | [-0.0124, -0.0007] |

**Volunteer data, test 2026-08-31 00:00 – 2026-10-11 23:00 UTC (the most recent period available anywhere), all oblasts:**

| Horizon | N | Prevalence | AP LGB / B | Brier LGB / B | BSS | ΔAP 95% CI | ΔBrier 95% CI |
|---|---|---|---|---|---|---|---|
| 1 h | 17,277 | 0.266 | 0.546 / 0.525 | 0.1583 / 0.1603 | +1.2% | [0.010, 0.032] | [-0.0033, -0.0006] |
| 3 h | 17,277 | 0.522 | 0.833 / 0.825 | 0.1661 / 0.1671 | +0.6% | [0.001, 0.016] | [-0.0048, 0.0023] |
| 6 h | 17,277 | 0.673 | 0.938 / 0.931 | 0.1217 / 0.1220 | +0.2% | [0.001, 0.014] | [-0.0060, 0.0043] |

The two tables use different sources and different target populations, so their numbers must not be compared with each
other.

**Honest reading:**
- LightGBM is at best a **modest** improvement over a simple 7-day-rate baseline. On official data the Brier
  improvement is distinguishable from zero only at 3 h and 6 h. At 1 h the model is no better than the baseline.
- On the freshest (volunteer) data the gain is small at every horizon. Brier is distinguishable from zero only at 1 h.
- **Frontline oblasts** (inactive less than 40% of Train hours) have prevalence of 0.70–0.96, which makes the task
  nearly trivial. At 1 h LightGBM is worse than the baseline there (BSS -5.0%, CI includes 0). See
  `results/metrics_summary.csv`.
- **Sensitivity checks:**
  - Keeping the fake end times (A0) inflates the official Rear BSS to +4.3 / +8.5 / +12.8%. Most of the gain reported
    before the audit (+4.7 / +9.8 / +13.0%) was this artifact.
  - Dropping `nat_active_oblasts_cnt` (C) gives slightly *better* test numbers (Rear BSS +1.0 / +4.8 / +8.8%), so the
    feature adds nothing. The main feature set was fixed before the test was run and was **not** changed after seeing
    this.
  - Volunteer variants V0 (fake naive ends kept) and V1 (naive rows dropped) change N and prevalence substantially.
    V0's fake ends add about 3,900 test origins.
- Feature importance (gain, `results/feature_importance.csv`) is dominated by `oblast`, followed by time since the
  last alert, the 168 h count and national activity. Importance is not causality.

## Reproduce
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset.git   # CSV + open-alert snapshot, same clone
python -m pytest -q                                   # 30 tests
python run_pipeline.py --run-tests                    # official run -> results/ (+ run_info.json with tests)
python volunteer_check.py                             # latest-period run on volunteer data -> results_volunteer/
python ablation_no_nat_active.py                      # -> results_ablation/
python run_pipeline.py --end-policy as_recorded --results_dir results_sensitivity/official_as_recorded_ends
python volunteer_check.py --naive-policy as_recorded --out results_sensitivity/volunteer_naive_as_recorded
python volunteer_check.py --naive-policy exclude --out results_sensitivity/volunteer_naive_excluded
python sensitivity_summary.py                         # results/sensitivity_summary.{csv,md}
python results_table.py                               # results/results_tables.md
python plot_figures.py                                # figures/
python sources_check.py > results/sources_check.txt   # source comparison and naive statistics
```
- Each run builds an hourly panel in about 3 minutes. The panel is cached in `.cache/`, keyed by a hash of the cleaned
  events and the code.
- Re-cloning the dataset refreshes the data automatically.
- `run_info.json` in every results folder records:
  - the source commit, the SHA-256 of the data and the snapshot, and the data cut;
  - the last valid label origin and the rule used to compute it;
  - the splits, with row counts and prevalence;
  - features, parameters and the seed;
  - headline metrics;
  - a `STALE DATA` warning when the source is more than 2 days old.

  The official source is flagged as stale (35 days).

## Files
- `features.py` — loading, causal end policy, open-alert snapshot, label cut-off and panel.
- `run_pipeline.py` — splits, baselines, LightGBM, metrics, bootstrap and manifest.
- `volunteer_check.py`, `ablation_no_nat_active.py`, `sensitivity_summary.py`.
- `sources_check.py`, `results_table.py`, `plot_figures.py`.
- `tests/test_no_leakage.py`.

## Limitations
- The official feed ends on 2026-09-07, so the official test cannot be newer than that. The newest evaluation (to
  2026-10-11) relies on the lower-quality volunteer feed.
- The heuristic for imputed ends (exactly 3600 s and a next start more than 3 h later) probably also catches about 10
  genuine one-hour alerts (estimated from the 3599 s and 3601 s neighbours).
- When a second 🔴 arrives within 3 h of an earlier one with no 🟢 in between, the processor overwrites the earlier
  alert. These cases cannot be detected from the CSV.
- The test window is only 6 weeks, and the alert process is non-stationary.
- Thresholds are chosen on Val predictions of the Train-only model but applied to the Train+Val refit.
- No news or Telegram text features are used (they were optional in the assignment).
