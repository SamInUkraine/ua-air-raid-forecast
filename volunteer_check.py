"""Latest-period evaluation on the volunteer (eTryvoga) feed - the only source that is still updated.

Same panel, features, baselines, LightGBM, splits rule and bootstrap as the main pipeline (run_pipeline.run_experiment),
but a DIFFERENT target population: oblast-level alerts reconstructed by keyword-parsing the eTryvoga Telegram channel
(no raion level). Metrics are NOT comparable with results/ (official data).

naive rows (no 'end' message; source writes end = start + 30 min, decided only when the next alert starts):
  --naive-policy causal       (default) end := next start in the same region (what a live listener believed)
  --naive-policy as_recorded  keep the fake 30-min end (pre-audit behaviour; sensitivity only)
  --naive-policy exclude      drop naive rows entirely (sensitivity only; also deletes their genuine starts)
Usage: python volunteer_check.py [--naive-policy causal] [--out results_volunteer]
"""
import argparse
import datetime
import os

from features import load_volunteer_events, DEFAULT_DATA_DIR
from run_pipeline import FEATURES, run_experiment, source_info, write_manifest, run_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=f'{DEFAULT_DATA_DIR}/datasets/volunteer_data_en.csv')
    p.add_argument('--snapshot', default=f'{DEFAULT_DATA_DIR}/processors/pkl/volunteer_active_alerts.pkl')
    p.add_argument('--naive-policy', default='causal', choices=['causal', 'as_recorded', 'exclude'])
    p.add_argument('--drop-features', nargs='*', default=[])
    p.add_argument('--out', default='results_volunteer')
    p.add_argument('--run-tests', action='store_true')
    a = p.parse_args()
    run_time = datetime.datetime.now(datetime.timezone.utc)
    snap = a.snapshot if a.snapshot and os.path.exists(a.snapshot) else None
    ev = load_volunteer_events(a.data, snapshot_path=snap, naive_policy=a.naive_policy)
    feats = [f for f in FEATURES if f not in a.drop_features]
    exp = run_experiment(ev, a.out, features=feats, use_groups=False, label=os.path.basename(a.out))
    naive_by_month = (ev[~ev['open_at_cut']].assign(m=lambda d: d['started_at_dt'].dt.strftime('%Y-%m'))
                      .groupby('m')['naive'].mean().tail(8).round(4).to_dict()) if a.naive_policy != 'exclude' else None
    extra = {'experiment': {'dataset': 'volunteer (eTryvoga, oblast level only)', 'naive_policy': a.naive_policy,
                            'dropped_features': a.drop_features, 'naive_share_by_month_last8': naive_by_month},
             'warnings': ['Different source and target population than results/ (official); do not compare directly.']
                         + ([] if snap else ['open-alert snapshot not used: label cut-off is conservative'])}
    write_manifest(os.path.join(a.out, 'run_info.json'), run_time, ev, source_info(a.data, snap), exp, extra,
                   run_tests() if a.run_tests else None)
    print(f"Done. Results in '{a.out}/'.")


if __name__ == '__main__':
    main()
