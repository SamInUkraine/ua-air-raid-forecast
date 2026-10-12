# Forecasting air raid alert onsets in Ukraine

Stage 2 pet project (KSE AI Agentic School). Python, time-series / tabular forecasting with strict no-leakage evaluation.

## Task
For every oblast and every hour `t` in which the oblast is **not** under alert, predict whether a **new** alert starts within the next `k` hours (`k = 1, 3, 6`):
`onset_k(t) = 1` if a new alert (oblast- or raion-level) starts in `(t, t+k]`.
Hours with an active alert are excluded (the question "will an alert start" is meaningless then).

## Data
[Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), `official_data_en.csv` (data until 2026-09-07).

```bash
git clone https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset.git
```
Cleaning (`features.load_events`): the file contains two dumps, so 113,845 exact duplicates are dropped; hromada-level alerts (multi-day/permanent) and Luhanska oblast (2 rows) are excluded; result: 164,617 events, 24 oblasts, hourly UTC panel from 2025-11-01 (the period of raion-level alerts).

## No-leakage design
- Features at time `t` use only events with `started_at <= t` and `finished_at <= t` (a still-running alert is "active", its end is unknown).
- Labels are only built where the whole horizon is covered by data (`last_valid_label_time = data end - 6h`), otherwise the tail would get false zeros.
- Chronological split with 24 h gaps (longer than the label horizon): Train 2025-11-01..2026-06-28, Val 2026-06-30..2026-07-26, Test 2026-07-28..2026-09-06 (the latest 6 weeks available).
- Baseline B (frequency in the last 7 days) is shifted by `k` hours, so only already-resolved labels are used.
- Baselines/threshold/baseline choice use Train/Val only; the final LightGBM is refit on Train+Val with the iteration count found on Val.
- `tests/test_no_leakage.py`: censoring test — removing all events after `T` (and marking alerts running at `T` as unfinished) must not change any feature for `t <= T`; run on periodic and random overlapping events; the test asserts that it really exercises censoring.

## Features
Local: `is_active`-filtered hours since last finish, alert starts in the last 1/3/6/24/168 h, oblast. National: number of active oblasts, number of oblasts with new alerts in the last 1/3 h. Calendar: hour of day and day of week (Kyiv time).

## Models and evaluation
LightGBM (early stopping on Val) vs baselines: A (oblast x hour rate), B (rate over the last 7 days), C (logistic regression on time since last alert), D (oblast base rate), Always-Positive.
Metrics: Average Precision (not trapezoidal AUC), Lift, Brier score, Brier Skill Score vs the best baseline chosen **on Val**, F1 with a Val-chosen threshold. Uncertainty: day-block bootstrap (1000) for the LightGBM-minus-best-baseline difference.
Oblasts are split into **Frontline** (inactive less than 40% of Train time: Chernihivska, Dnipropetrovska, Donetska, Kharkivska, Sumska, Zaporizka) and **Rear**, because in the frontline group almost every hour is followed by an alert (prevalence 70-97%) and the task is trivial there.

## Results (Test, 2026-07-28..2026-09-06)
Rear regions, LightGBM vs best baseline (B):

| Horizon | Prevalence | AP LGB / B | Brier LGB / B | BSS | 95% CI of Brier diff |
|---|---|---|---|---|---|
| 1 h | 0.110 | 0.381 / 0.350 | 0.0774 / 0.0812 | +4.7% | [-0.0072, -0.0010] |
| 3 h | 0.225 | 0.689 / 0.625 | 0.1075 / 0.1191 | +9.8% | [-0.0193, -0.0047] |
| 6 h | 0.310 | 0.810 / 0.751 | 0.1149 / 0.1321 | +13.0% | [-0.0266, -0.0088] |

Honest reading:
- The model beats the best simple baseline on Brier for all Rear horizons (CI excludes 0); the improvement grows with the horizon. For 1 h the AP difference is **not** significant (CI [-0.007, 0.073]).
- Frontline: at 1 h LightGBM is **worse** than baseline B (BSS -5.6%, CI above 0); at 3 h it is not significantly different; at 6 h it is better than the constant baseline D. Prevalence is so high there that Always-Positive is a strong reference.
- The 7-day baseline B is strong: this is an autocorrelated process, so "what happened this week" explains much; the real gain is moderate, not dramatic.
- Most important feature groups: oblast, time since last alert, national activity, hour of day (`results/feature_importance.csv`).

Figures are in `figures/`, all numeric tables in `results/` (`metrics_summary.csv`, `bootstrap_ci.csv`, `rear_by_region.csv`, `feature_importance.csv`, `test_predictions.csv`, `results_tables.md`).

## Limitations
- Official data ends 2026-09-07, so "latest period" means the last 6 weeks of this file, not October.
- A second source (volunteer eTryvoga feed, up to 2026-10-12) was examined (sources_check.py, results/sources_check.txt) and not used for the main result: it has oblast-level alerts only (the target is not comparable; frontline oblasts are active 17-21% of the time vs 65-90% in the official data), hourly agreement with the official data is weak on the frontline (Cohen’s kappa 0.04-0.25), and ~5% of end times are imputed (naive, 9.6% in October). As a robustness check on the latest period (volunteer_check.py, test 2026-09-07..2026-10-11) the model is better than the best baseline in AP at all horizons, but better in Brier only at 1 h (BSS +1.1%, +0.8%, -1.2% at 1/3/6 h), i.e. the gain on this target is small.
- Test period is short (6 weeks); behaviour of the threat changes over time, so absolute numbers can shift.
- No news/Telegram text data was used (optional in the task).

## Reproduce
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset.git
python -m pytest -q                      # leakage tests
python run_pipeline.py                   # panel (cached in .cache/), models, metrics -> results/
python results_table.py                  # results/results_tables.md
python plot_figures.py                   # figures/
python volunteer_check.py               # robustness check on the freshest (volunteer) data -> results_volunteer/
```
The first run builds the hourly panel and takes several minutes; later runs use the cache.

## Files
`features.py` (cleaning, panel, leakage-safe features), `run_pipeline.py` (splits, models, baselines, bootstrap), `results_table.py`, `plot_figures.py`, `sources_check.py` (comparison of data sources), `tests/test_no_leakage.py`, `volunteer_check.py`.
