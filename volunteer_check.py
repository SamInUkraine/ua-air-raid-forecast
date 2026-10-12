"""Robustness check on the freshest data (volunteer eTryvoga feed, up to 2026-10-12).

Same panel, features, baselines, LightGBM and bootstrap as the main pipeline, but:
 * source: volunteer_data_en.csv (oblast-level alerts only, no raion level),
 * test = the latest period (2026-09-07 .. last valid label time), val/train are earlier,
 * no Frontline/Rear split (here every oblast ends up in the 'Rear' group).
NOT comparable with results/ (different target definition) - this is a separate sanity check.
Usage: python volunteer_check.py [--data PATH] [--out results_volunteer]
"""
import argparse, os, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import brier_score_loss
from features import generate_panel_data, get_panel, last_valid_label_time, START_UTC
from run_pipeline import FEATURES, BASE_COLS, MODELS, compute_baselines, day_block_bootstrap, ap

SPL = {'train': ('2025-11-01 00:00:00+00:00', '2026-08-07 23:00:00+00:00'),
       'val': ('2026-08-09 00:00:00+00:00', '2026-09-05 23:00:00+00:00'),
       'test_start': '2026-09-07 00:00:00+00:00'}


def load_volunteer(path):
    df = pd.read_csv(path).rename(columns={'region': 'oblast'})
    df['started_at_dt'] = pd.to_datetime(df['started_at'], utc=True)
    df['finished_at_dt'] = pd.to_datetime(df['finished_at'], utc=True)
    df = df.dropna(subset=['started_at_dt', 'finished_at_dt'])
    df = df[df['oblast'] != 'Luhanska oblast']
    df = df.sort_values('started_at_dt').drop_duplicates(subset=['oblast', 'started_at_dt'], keep='last')
    df['naive'] = df['naive'].astype(str).str.lower().eq('true')
    return df.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='ukrainian-air-raid-sirens-dataset/datasets/volunteer_data_en.csv')
    p.add_argument('--out', default='results_volunteer')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ev = load_volunteer(a.data)
    oblasts = sorted(ev['oblast'].unique())
    test_end = last_valid_label_time(ev, 6)
    print(f"events={len(ev):,} naive={int(ev['naive'].sum()):,} test_end={test_end}")
    panel = get_panel(ev, a.data, START_UTC, test_end)
    sl = lambda x, y: panel[(panel['t'] >= pd.Timestamp(x)) & (panel['t'] <= pd.Timestamp(y))].copy()
    dtr, dva, dte = sl(*SPL['train']), sl(*SPL['val']), sl(SPL['test_start'], test_end)
    cat = pd.CategoricalDtype(categories=oblasts)
    prep = lambda d: d[FEATURES].assign(oblast=d['oblast'].astype(cat))
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05, 'verbose': -1, 'seed': 42}
    rng = np.random.default_rng(42)
    rows, boots = [], []
    for k in (1, 3, 6):
        tgt = f'onset_{k}'
        sel = lambda d: d[d['is_active'] == 0].dropna(subset=[tgt] + FEATURES).copy()
        tr, va, te = sel(dtr), sel(dva), sel(dte)
        va = compute_baselines(va, dtr, panel, k)
        te = compute_baselines(te, pd.concat([dtr, dva]), panel, k)
        d1 = lgb.Dataset(prep(tr), label=tr[tgt], categorical_feature=['oblast'])
        d2 = lgb.Dataset(prep(va), label=va[tgt], reference=d1, categorical_feature=['oblast'])
        m = lgb.train(params, d1, 500, valid_sets=[d2], callbacks=[lgb.early_stopping(50, verbose=False)])
        full = pd.concat([tr, va])
        mf = lgb.train(params, lgb.Dataset(prep(full), label=full[tgt], categorical_feature=['oblast']), m.best_iteration)
        va['p_LightGBM'] = m.predict(prep(va), num_iteration=m.best_iteration)
        te['p_LightGBM'] = mf.predict(prep(te))
        best = min(BASE_COLS, key=lambda c: brier_score_loss(va[tgt], va[c]))
        yt = te[tgt].values
        bb = brier_score_loss(yt, te[best])
        for col in ['p_LightGBM'] + BASE_COLS:
            b = brier_score_loss(yt, te[col])
            rows.append({'Target': tgt, 'Model': MODELS[col], 'N': len(te), 'Prevalence': round(float(yt.mean()), 4),
                         'AP': round(ap(yt, te[col]), 4), 'Brier': round(b, 4), 'BSS (%)': round((1 - b / bb) * 100, 2),
                         'Best_Baseline_Used': MODELS[best]})
        boots.append({'Target': tgt, 'Baseline': MODELS[best], **day_block_bootstrap(te, tgt, best, rng)})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'metrics_summary.csv'), index=False)
    pd.DataFrame(boots).to_csv(os.path.join(a.out, 'bootstrap_ci.csv'), index=False)
    json.dump({'source': 'volunteer', 'events': len(ev), 'naive_rows': int(ev['naive'].sum()),
               'test_start': SPL['test_start'], 'test_end': str(test_end)},
              open(os.path.join(a.out, 'run_info.json'), 'w'), indent=2)
    print(pd.DataFrame(rows)[lambda d: d.Model.isin(['LightGBM', 'Baseline_B'])].to_string(index=False))
    print(pd.DataFrame(boots)[['Target', 'Baseline', 'AP_Diff_Mean', 'AP_Diff_CI_Low', 'AP_Diff_CI_High',
                               'Brier_Diff_Mean', 'Brier_Diff_CI_Low', 'Brier_Diff_CI_High']].to_string(index=False))


if __name__ == '__main__':
    main()
