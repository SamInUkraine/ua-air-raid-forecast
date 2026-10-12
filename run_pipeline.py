"""Air raid alert ONSET forecasting: splits, baselines, LightGBM, metrics, bootstrap, run manifest.

Main run (official dataset):   python run_pipeline.py
Everything is derived from the data: the test window ends at the last forecast origin whose 6 h label is complete.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve

from features import (load_events, get_panel, last_valid_label_time, START_UTC, MAX_HORIZON_H, TARGETS,
                      DEFAULT_DATA_DIR, events_fingerprint)

warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
N_BOOT = 1000
SEED = 42
HORIZONS = (1, 3, 6)
TEST_DAYS, VAL_DAYS, GAP_H = 42, 28, 24        # fixed config, not tuned; GAP_H > max horizon (6 h)
FRONTLINE_INACTIVE_SHARE = 0.40               # oblast is 'Frontline' if inactive < 40% of TRAIN hours

FEATURES = ['oblast', 'hours_since_last_finish', 'starts_1h', 'starts_3h', 'starts_6h', 'starts_24h',
            'starts_168h', 'nat_active_oblasts_cnt', 'nat_starts_1h', 'nat_starts_3h',
            'hour_of_day_kyiv', 'day_of_week_kyiv']
MODELS = {'p_LightGBM': 'LightGBM', 'p_base_a': 'Baseline_A', 'p_base_b': 'Baseline_B',
          'p_base_c': 'Baseline_C', 'p_base_d': 'Baseline_D', 'p_always_pos': 'Always_Positive'}
BASE_COLS = ['p_base_a', 'p_base_b', 'p_base_c', 'p_base_d']
LGB_PARAMS = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05, 'verbose': -1,
              'seed': SEED, 'deterministic': True, 'force_col_wise': True, 'num_threads': 1}


def ap(y, p):
    """Average precision (sklearn step-wise AP, not trapezoidal PR-AUC, not ROC-AUC)."""
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if y.sum() > 0 else np.nan


def make_splits(last_valid, start_utc=START_UTC, test_days=TEST_DAYS, val_days=VAL_DAYS, gap_h=GAP_H):
    """Deterministic chronological split anchored on the last complete label (no hard-coded dates).
    Test = last `test_days` days ending at last_valid; Val = `val_days` before it; gaps of `gap_h` hours between."""
    last_valid = pd.Timestamp(last_valid)
    test_end = last_valid
    test_start = (test_end - pd.Timedelta(days=test_days)).ceil('D')
    val_end = test_start - pd.Timedelta(hours=gap_h + 1)
    val_start = val_end + pd.Timedelta(hours=1) - pd.Timedelta(days=val_days)
    train_end = val_start - pd.Timedelta(hours=gap_h + 1)
    train_start = pd.Timestamp(start_utc)
    if not train_start < train_end:
        raise ValueError('Not enough history for the requested split')
    return {'train': (train_start, train_end), 'val': (val_start, val_end), 'test': (test_start, test_end)}


def compute_baselines(df_target, df_fit, df_panel, k):
    """A, C, D are fitted on df_fit only (strictly earlier than df_target).
    B = onset rate over the last 168 h using only labels already RESOLVED at t: onset_k(t-k) covers (t-k, t]."""
    tgt = f'onset_{k}'
    res = df_target.copy()
    fit_inactive = df_fit[df_fit['is_active'] == 0]

    rate_d = fit_inactive.groupby('oblast')[tgt].mean().to_dict()
    res['p_base_d'] = res['oblast'].map(rate_d).astype(float).clip(1e-4, 1 - 1e-4)

    rate_a = fit_inactive.groupby(['oblast', 'hour_of_day_kyiv'])[tgt].mean().reset_index().rename(columns={tgt: 'p_base_a'})
    res = res.merge(rate_a, on=['oblast', 'hour_of_day_kyiv'], how='left', validate='many_to_one')
    res['p_base_a'] = res['p_base_a'].fillna(res['p_base_d']).clip(1e-4, 1 - 1e-4)

    res = res.merge(baseline_b_series(df_panel, k), on=['t', 'oblast'], how='left', validate='one_to_one')
    res['p_base_b'] = res['p_base_b_calc'].fillna(res['p_base_d']).clip(1e-4, 1 - 1e-4)
    res = res.drop(columns=['p_base_b_calc'])

    fit_c = fit_inactive.dropna(subset=['hours_since_last_finish', tgt])
    lr = LogisticRegression().fit(fit_c[['hours_since_last_finish']], fit_c[tgt])
    p_c = np.full(len(res), res['p_base_d'].mean())
    ok = res['hours_since_last_finish'].notna().values
    if ok.any():
        p_c[ok] = lr.predict_proba(res.loc[ok, ['hours_since_last_finish']])[:, 1]
    res['p_base_c'] = np.clip(p_c, 1e-4, 1 - 1e-4)
    return res


def baseline_b_series(df_panel, k):
    """Rolling 168 h onset rate per oblast, lagged by k hours so that every label used is resolved by t."""
    tgt = f'onset_{k}'
    ds = df_panel[['t', 'oblast', 'is_active', tgt]].sort_values(['oblast', 't']).copy()
    # the panel must be a complete hourly grid per oblast, otherwise shift(k) would not mean "k hours"
    step = ds.groupby('oblast')['t'].diff().dropna()
    assert (step == pd.Timedelta(hours=1)).all(), 'panel is not a contiguous hourly grid'
    ds['target_valid'] = np.where(ds['is_active'] == 0, ds[tgt], np.nan)
    ds['inactive_flag'] = (ds['is_active'] == 0).astype(int)
    ds['target_shifted'] = ds.groupby('oblast')['target_valid'].shift(k)
    ds['inactive_shifted'] = ds.groupby('oblast')['inactive_flag'].shift(k)
    roll_pos = ds.groupby('oblast')['target_shifted'].transform(lambda x: x.rolling(168, min_periods=1).sum())
    roll_inact = ds.groupby('oblast')['inactive_shifted'].transform(lambda x: x.rolling(168, min_periods=1).sum())
    ds['p_base_b_calc'] = np.where(roll_inact > 0, roll_pos / roll_inact, np.nan)
    return ds[['t', 'oblast', 'p_base_b_calc']]


def day_block_bootstrap(te_grp, tgt, base_col, rng):
    """Resample whole Kyiv-calendar days (keeps within-day autocorrelation together)."""
    dates = te_grp['t'].dt.tz_convert('Europe/Kyiv').dt.date
    day_groups = {d: g for d, g in te_grp.groupby(dates)}
    keys = list(day_groups)
    ap_lgb, d_ap, d_brier = [], [], []
    for _ in range(N_BOOT):
        s = pd.concat([day_groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        y = s[tgt].values
        if y.sum() == 0 or y.sum() == len(y):
            continue
        a_l, a_b = ap(y, s['p_LightGBM']), ap(y, s[base_col])
        ap_lgb.append(a_l)
        d_ap.append(a_l - a_b)
        d_brier.append(brier_score_loss(y, s['p_LightGBM']) - brier_score_loss(y, s[base_col]))
    q = lambda a, p: round(float(np.percentile(a, p)), 4)
    return {'N_days': len(keys), 'N_boot_used': len(d_ap),
            'AP_LGB_Mean': round(float(np.mean(ap_lgb)), 4), 'AP_LGB_CI_Low': q(ap_lgb, 2.5), 'AP_LGB_CI_High': q(ap_lgb, 97.5),
            'AP_Diff_Mean': round(float(np.mean(d_ap)), 4), 'AP_Diff_CI_Low': q(d_ap, 2.5), 'AP_Diff_CI_High': q(d_ap, 97.5),
            'Brier_Diff_Mean': round(float(np.mean(d_brier)), 4), 'Brier_Diff_CI_Low': q(d_brier, 2.5), 'Brier_Diff_CI_High': q(d_brier, 97.5)}


def run_experiment(events, results_dir, features=None, use_groups=True, label='main'):
    """Shared by the main run, the volunteer run and all sensitivity runs (identical methodology)."""
    features = list(features or FEATURES)
    os.makedirs(results_dir, exist_ok=True)
    oblasts = sorted(events['oblast'].unique())
    last_valid = last_valid_label_time(events, horizon_h=MAX_HORIZON_H)
    sp = make_splits(last_valid)
    print(f'[{label}] last valid label origin {last_valid}; splits: ' + ', '.join(f'{k}={a}..{b}' for k, (a, b) in sp.items()))

    df_panel = get_panel(events, START_UTC, sp['test'][1])
    sl = lambda a, b: df_panel[(df_panel['t'] >= a) & (df_panel['t'] <= b)].copy()
    df_train, df_val, df_test = sl(*sp['train']), sl(*sp['val']), sl(*sp['test'])
    assert df_train['t'].max() < df_val['t'].min() and df_val['t'].max() < df_test['t'].min()

    frontline = ([o for o, g in df_train.groupby('oblast') if (g['is_active'] == 0).mean() < FRONTLINE_INACTIVE_SHARE]
                 if use_groups else [])
    obl_type = pd.CategoricalDtype(categories=oblasts)
    prep = lambda d: d[features].assign(**({'oblast': d['oblast'].astype(obl_type)} if 'oblast' in features else {}))
    cat = ['oblast'] if 'oblast' in features else []
    assert not set(features) & set(TARGETS + ['is_active', 'hours_since_current_start']), 'target/eligibility in X'

    metrics, boots, rear_rows, fi_rows, preds, split_rows = [], [], [], [], [], []
    rng = np.random.default_rng(SEED)

    for k in HORIZONS:
        tgt = f'onset_{k}'
        sel = lambda d: d[d['is_active'] == 0].dropna(subset=[tgt] + features).copy()
        tr, va, te = sel(df_train), sel(df_val), sel(df_test)
        for name, d, full in (('train', tr, df_train), ('val', va, df_val), ('test', te, df_test)):
            split_rows.append({'Target': tgt, 'Split': name, 'Start': str(full['t'].min()), 'End': str(full['t'].max()),
                               'Panel_rows': len(full), 'Active_rows_excluded': int((full['is_active'] == 1).sum()),
                               'Warmup_NaN_dropped': int(((full['is_active'] == 0)).sum() - len(d)),
                               'Eligible_rows': len(d), 'Positives': int(d[tgt].sum()),
                               'Prevalence': round(float(d[tgt].mean()), 4)})

        va = compute_baselines(va, df_train, df_panel, k)                                # Val: fitted on Train
        te = compute_baselines(te, pd.concat([df_train, df_val]), df_panel, k)           # Test: fitted on Train+Val

        dtr = lgb.Dataset(prep(tr), label=tr[tgt], categorical_feature=cat)
        dva = lgb.Dataset(prep(va), label=va[tgt], reference=dtr, categorical_feature=cat)
        model = lgb.train(LGB_PARAMS, dtr, num_boost_round=500, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        best_iter = model.best_iteration
        full = pd.concat([tr, va])                                                       # refit on Train+Val
        model_full = lgb.train(LGB_PARAMS, lgb.Dataset(prep(full), label=full[tgt], categorical_feature=cat),
                               num_boost_round=best_iter)

        for feat, gain in zip(model_full.feature_name(), model_full.feature_importance(importance_type='gain')):
            fi_rows.append({'Target': tgt, 'Feature': feat, 'Gain': round(float(gain), 2), 'BestIter': best_iter})

        va['p_LightGBM'] = model.predict(prep(va), num_iteration=best_iter)              # Val: thresholds/baseline choice
        te['p_LightGBM'] = model_full.predict(prep(te))
        va['p_always_pos'] = 1.0
        te['p_always_pos'] = 1.0
        te['Group'] = np.where(te['oblast'].isin(frontline), 'Frontline', 'Rear')
        va['Group'] = np.where(va['oblast'].isin(frontline), 'Frontline', 'Rear')
        preds.append(te[['t', 'oblast', 'Group', tgt, 'p_LightGBM'] + BASE_COLS].rename(columns={tgt: 'y'}).assign(Target=tgt))

        groups = {'ALL': (va, te)}
        if frontline:
            groups.update({g: (va[va['Group'] == g], te[te['Group'] == g]) for g in ('Frontline', 'Rear')})
        for gname, (va_g, te_g) in groups.items():
            if len(te_g) == 0 or len(va_g) == 0:
                continue
            yv, yt = va_g[tgt].values, te_g[tgt].values
            best_base = min(BASE_COLS, key=lambda c: brier_score_loss(yv, va_g[c]))     # chosen on Val ONLY
            brier_best = brier_score_loss(yt, te_g[best_base])
            prev = float(yt.mean())
            for col, name in MODELS.items():
                prec, rec, thr = precision_recall_curve(yv, va_g[col])
                f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
                best_t = float(thr[np.argmax(f1)]) if len(thr) else 0.5
                pred_bin = (te_g[col] >= best_t).astype(int)
                a, b = ap(yt, te_g[col]), brier_score_loss(yt, te_g[col])
                metrics.append({'Target': tgt, 'Group': gname, 'Model': name, 'N': len(te_g), 'Positives': int(yt.sum()),
                                'Prevalence': round(prev, 4), 'AP': round(a, 4), 'Lift': round(a / prev, 3) if prev > 0 else np.nan,
                                'Brier': round(b, 4), 'BSS (%)': round((1 - b / brier_best) * 100, 2),
                                'Precision_Test': round(float(((pred_bin == 1) & (yt == 1)).sum() / max(pred_bin.sum(), 1)), 4),
                                'Recall_Test': round(float(((pred_bin == 1) & (yt == 1)).sum() / max(yt.sum(), 1)), 4),
                                'F1_Test': round(f1_score(yt, pred_bin), 4),
                                'Thresh_Val': round(best_t, 4), 'Best_Baseline_Used': MODELS[best_base]})
            boots.append({'Target': tgt, 'Group': gname, 'Baseline': MODELS[best_base],
                          **day_block_bootstrap(te_g, tgt, best_base, rng)})

        if k in (1, 3):
            for obl, g in te[te['Group'] == 'Rear'].groupby('oblast'):
                y, p = g[tgt].values, g['p_LightGBM'].values
                a, prev = ap(y, p), float(y.mean())
                rear_rows.append({'Target': tgt, 'Oblast': obl, 'N': len(g), 'Positives': int(y.sum()),
                                  'Prevalence': round(prev, 4), 'AP': round(a, 4) if a == a else np.nan,
                                  'Lift': round(a / prev, 3) if prev > 0 and a == a else np.nan,
                                  'Brier': round(brier_score_loss(y, p), 4)})

    out = lambda n: os.path.join(results_dir, n)
    m = pd.DataFrame(metrics)
    m.to_csv(out('metrics_summary.csv'), index=False)
    pd.DataFrame(boots).to_csv(out('bootstrap_ci.csv'), index=False)
    pd.DataFrame(rear_rows).to_csv(out('rear_by_region.csv'), index=False)
    pd.DataFrame(fi_rows).to_csv(out('feature_importance.csv'), index=False)
    pd.DataFrame(split_rows).to_csv(out('splits.csv'), index=False)
    pd.concat(preds).to_csv(out('test_predictions.csv'), index=False)

    headline = []
    for (t, g), d in m.groupby(['Target', 'Group'], sort=False):
        bb = d['Best_Baseline_Used'].iloc[0]
        l, b = d[d.Model == 'LightGBM'].iloc[0], d[d.Model == bb].iloc[0]
        headline.append({'target': t, 'group': g, 'n': int(l.N), 'prevalence': float(l.Prevalence),
                         'best_baseline_on_val': bb, 'ap_lightgbm': float(l.AP), 'ap_baseline': float(b.AP),
                         'brier_lightgbm': float(l.Brier), 'brier_baseline': float(b.Brier),
                         'bss_pct_vs_baseline': float(l['BSS (%)'])})
    return {'last_valid_label_origin_utc': str(last_valid),
            'last_valid_rule': ('t + 6h <= data_cut (open-at-cut alerts attached from the source snapshot)'
                                if events.attrs.get('open_snapshot_used') else
                                't + 6h <= data_cut - 48h (no open-alert snapshot; conservative)'),
            'splits_utc': {k: [str(a), str(b)] for k, (a, b) in sp.items()},
            'split_config': {'test_days': TEST_DAYS, 'val_days': VAL_DAYS, 'gap_hours': GAP_H},
            'split_rows': split_rows, 'frontline_oblasts': frontline,
            'frontline_rule': f'inactive share in TRAIN < {FRONTLINE_INACTIVE_SHARE}' if use_groups else 'not used',
            'features': features, 'targets': {f'onset_{k}': f'1 if any alert in the oblast STARTS in (t, t+{k}h]; '
                                              f'defined only when no alert is active in the oblast at t' for k in HORIZONS},
            'model': {'lightgbm_params': LGB_PARAMS, 'early_stopping': 'on Val, patience 50, max 500 rounds',
                      'final_fit': 'Train+Val with the Val-selected number of rounds'},
            'baselines': {'A': 'oblast x Kyiv-hour onset rate (fit data)', 'B': 'rolling 168h onset rate, labels lagged by k',
                          'C': 'logistic regression on hours_since_last_finish', 'D': 'oblast onset rate (fit data)',
                          'selection': 'best Brier on Val, per horizon and group'},
            'thresholds': 'max-F1 threshold chosen on Val predictions of the Train-only model',
            'bootstrap': {'type': 'day-block (Kyiv calendar days)', 'n': N_BOOT, 'seed': SEED},
            'headline_metrics_test': headline}


# --------------------------------------------------------------------------------------------------------- manifest
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(repo, *args):
    try:
        return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:
        return None


def source_info(data_path, snapshot_path):
    repo = os.path.dirname(os.path.dirname(os.path.realpath(data_path)))
    rel = os.path.relpath(os.path.realpath(data_path), repo)
    info = {'source_repo': 'https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset',
            'data_file': rel, 'data_sha256': sha256(data_path),
            'data_file_mtime_utc (approx. retrieval time)': datetime.datetime.fromtimestamp(
                os.path.getmtime(data_path), datetime.timezone.utc).isoformat(timespec='seconds'),
            'dataset_git_head': _git(repo, 'rev-parse', 'HEAD'),
            'dataset_git_head_time': _git(repo, 'log', '-1', '--format=%cI'),
            'data_file_last_changed_in_commit_time': _git(repo, 'log', '-1', '--format=%cI', '--', rel)}
    if snapshot_path:
        info['open_alert_snapshot'] = os.path.relpath(os.path.realpath(snapshot_path), repo)
        info['open_alert_snapshot_sha256'] = sha256(snapshot_path)
    return info


def freshness(events, run_time, max_age_days=2):
    cut = events.attrs['data_cut']
    age = (pd.Timestamp(run_time) - cut).total_seconds() / 86400
    w = []
    if age > max_age_days:
        w.append(f'STALE DATA: last event in {events.attrs["cleaning"]["source_file"]} is {age:.1f} days older than '
                 f'the run time; the source stopped updating or the local copy is old (re-clone to refresh).')
    return {'latest_raw_start_utc': str(events.loc[~events['open_at_cut'], 'started_at_dt'].max()),
            'data_cut_utc': str(cut), 'data_age_days_at_run': round(age, 2)}, w


def run_tests():
    try:
        r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider'], capture_output=True,
                           text=True, timeout=1800)
        return {'command': 'python -m pytest -q', 'exit_code': r.returncode, 'summary': r.stdout.strip().splitlines()[-1]}
    except Exception as e:
        return {'command': 'python -m pytest -q', 'error': repr(e)}


def write_manifest(path, run_time, events, src, exp, extra=None, tests=None):
    fresh, warns = freshness(events, run_time)
    manifest = {'run_time_utc': run_time.isoformat(timespec='seconds'),
                'run_time_kyiv': run_time.astimezone(ZoneInfo('Europe/Kyiv')).isoformat(timespec='seconds'),
                'command': ' '.join([os.path.basename(sys.executable)] + sys.argv),
                'environment': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
                                'scikit-learn': sklearn.__version__, 'lightgbm': lgb.__version__},
                'source': src, 'freshness': fresh, 'cleaning_and_filtering': events.attrs['cleaning'],
                'events_fingerprint_sha256': events_fingerprint(events), 'seed': SEED, **exp,
                'tests': tests or 'not run by this command (use --run-tests)',
                'warnings': warns + (extra or {}).pop('warnings', []), **(extra or {})}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    for w in manifest['warnings']:
        print('WARNING:', w)
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Air raid sirens: onset forecasting pipeline (official dataset)')
    parser.add_argument('--data', default=f'{DEFAULT_DATA_DIR}/datasets/official_data_en.csv')
    parser.add_argument('--snapshot', default=f'{DEFAULT_DATA_DIR}/processors/pkl/official_active_alerts.pkl',
                        help="open-alert snapshot of the same clone; pass '' to disable (conservative label cut-off)")
    parser.add_argument('--end-policy', default='causal', choices=['causal', 'as_recorded'])
    parser.add_argument('--drop-features', nargs='*', default=[])
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--run-tests', action='store_true', help='run pytest and store the summary in run_info.json')
    args = parser.parse_args()
    run_time = datetime.datetime.now(datetime.timezone.utc)
    if not os.path.exists(args.data):
        sys.exit(f'Data file not found: {args.data}\nClone it first: git clone '
                 f'https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset.git')
    snap = args.snapshot if args.snapshot and os.path.exists(args.snapshot) else None
    if args.snapshot and not snap:
        print(f'WARNING: snapshot {args.snapshot} not found; using the conservative label cut-off')

    events = load_events(args.data, snapshot_path=snap, end_policy=args.end_policy)
    feats = [f for f in FEATURES if f not in args.drop_features]
    exp = run_experiment(events, args.results_dir, features=feats, label=os.path.basename(args.results_dir))
    tests = run_tests() if args.run_tests else None
    extra = {'experiment': {'dataset': 'official', 'end_policy': args.end_policy, 'dropped_features': args.drop_features},
             'warnings': [] if snap else ['open-alert snapshot not used: label cut-off is conservative (cut - 54h)']}
    write_manifest(os.path.join(args.results_dir, 'run_info.json'), run_time, events, source_info(args.data, snap),
                   exp, extra, tests)
    print(f"Done. Results in '{args.results_dir}/'.")


if __name__ == '__main__':
    main()
