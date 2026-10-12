import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve

from features import load_events, get_panel, last_valid_label_time, START_UTC

warnings.filterwarnings('ignore')
N_BOOT = 1000

TRAIN = ('2025-11-01 00:00:00+00:00', '2026-06-28 23:00:00+00:00')
VAL = ('2026-06-30 00:00:00+00:00', '2026-07-26 23:00:00+00:00')      # 24h буфер перед Val
TEST_START = '2026-07-28 00:00:00+00:00'                              # 24h буфер перед Test
TEST_END_PLANNED = '2026-09-06 23:00:00+00:00'

FEATURES = ['oblast', 'hours_since_last_finish', 'starts_1h', 'starts_3h', 'starts_6h', 'starts_24h',
            'starts_168h', 'nat_active_oblasts_cnt', 'nat_starts_1h', 'nat_starts_3h',
            'hour_of_day_kyiv', 'day_of_week_kyiv']
MODELS = {'p_LightGBM': 'LightGBM', 'p_base_a': 'Baseline_A', 'p_base_b': 'Baseline_B',
          'p_base_c': 'Baseline_C', 'p_base_d': 'Baseline_D', 'p_always_pos': 'Always_Positive'}
BASE_COLS = ['p_base_a', 'p_base_b', 'p_base_c', 'p_base_d']


def ap(y, p):
    """Average precision (НЕ трапецоїдальний auc(R,P), який завищує PR-AUC)."""
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if y.sum() > 0 else np.nan


def compute_baselines(df_target, df_fit, df_panel, k):
    """A, C, D навчаються лише на df_fit (минуле відносно df_target); B — лише на минулому вікні t-168h..t-k."""
    tgt = f'onset_{k}'
    res = df_target.copy()
    fit_inactive = df_fit[df_fit['is_active'] == 0]

    rate_d = fit_inactive.groupby('oblast')[tgt].mean().to_dict()
    res['p_base_d'] = res['oblast'].map(rate_d).clip(1e-4, 1 - 1e-4)

    rate_a = fit_inactive.groupby(['oblast', 'hour_of_day_kyiv'])[tgt].mean().reset_index().rename(columns={tgt: 'p_base_a'})
    res = res.merge(rate_a, on=['oblast', 'hour_of_day_kyiv'], how='left')
    res['p_base_a'] = res['p_base_a'].fillna(res['p_base_d']).clip(1e-4, 1 - 1e-4)

    ds = df_panel.sort_values(['oblast', 't']).copy()
    ds['target_valid'] = np.where(ds['is_active'] == 0, ds[tgt], np.nan)
    ds['inactive_flag'] = (ds['is_active'] == 0).astype(int)
    ds['target_shifted'] = ds.groupby('oblast')['target_valid'].shift(k)      # onset_k(t-k) покриває (t-k, t] -> відомо в t
    ds['inactive_shifted'] = ds.groupby('oblast')['inactive_flag'].shift(k)
    roll_pos = ds.groupby('oblast')['target_shifted'].transform(lambda x: x.rolling(168, min_periods=1).sum())
    roll_inact = ds.groupby('oblast')['inactive_shifted'].transform(lambda x: x.rolling(168, min_periods=1).sum())
    ds['p_base_b_calc'] = np.where(roll_inact > 0, roll_pos / roll_inact, np.nan)
    res = res.merge(ds[['t', 'oblast', 'p_base_b_calc']], on=['t', 'oblast'], how='left')
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


def day_block_bootstrap(te_grp, tgt, base_col, rng):
    dates = te_grp['t'].dt.date
    day_groups = {d: g for d, g in te_grp.groupby(dates)}
    keys = list(day_groups)
    ap_lgb, d_ap, d_brier = [], [], []
    for _ in range(N_BOOT):
        s = pd.concat([day_groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        y = s[tgt].values
        if y.sum() == 0:
            continue
        a_l, a_b = ap(y, s['p_LightGBM']), ap(y, s[base_col])
        ap_lgb.append(a_l)
        d_ap.append(a_l - a_b)
        d_brier.append(brier_score_loss(y, s['p_LightGBM']) - brier_score_loss(y, s[base_col]))
    q = lambda a, p: round(float(np.percentile(a, p)), 4)
    return {'AP_LGB_Mean': round(float(np.mean(ap_lgb)), 4), 'AP_LGB_CI_Low': q(ap_lgb, 2.5), 'AP_LGB_CI_High': q(ap_lgb, 97.5),
            'AP_Diff_Mean': round(float(np.mean(d_ap)), 4), 'AP_Diff_CI_Low': q(d_ap, 2.5), 'AP_Diff_CI_High': q(d_ap, 97.5),
            'Brier_Diff_Mean': round(float(np.mean(d_brier)), 4), 'Brier_Diff_CI_Low': q(d_brier, 2.5), 'Brier_Diff_CI_High': q(d_brier, 97.5)}


def main():
    parser = argparse.ArgumentParser(description="Air raid sirens: onset forecasting pipeline")
    parser.add_argument('--data', default='ukrainian-air-raid-sirens-dataset/datasets/official_data_en.csv')
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    events = load_events(args.data)
    oblasts = sorted(events['oblast'].unique())
    last_valid = last_valid_label_time(events, horizon_h=6)
    test_end = min(pd.Timestamp(TEST_END_PLANNED), last_valid)
    print(f"Кінець даних дає валідні мітки до {last_valid}; test_end = {test_end}")

    df_panel = get_panel(events, args.data, START_UTC, test_end)
    sl = lambda a, b: df_panel[(df_panel['t'] >= pd.Timestamp(a)) & (df_panel['t'] <= pd.Timestamp(b))].copy()
    df_train, df_val, df_test = sl(*TRAIN), sl(*VAL), sl(TEST_START, test_end)

    frontline = [o for o, g in df_train.groupby('oblast') if (g['is_active'] == 0).mean() < 0.40]
    obl_type = pd.CategoricalDtype(categories=oblasts)
    prep = lambda d: d[FEATURES].assign(oblast=d['oblast'].astype(obl_type))
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05, 'verbose': -1, 'seed': 42}

    metrics, boots, rear_rows, fi_rows, preds = [], [], [], [], []
    rng = np.random.default_rng(42)

    for k in (1, 3, 6):
        tgt = f'onset_{k}'
        sel = lambda d: d[d['is_active'] == 0].dropna(subset=[tgt] + FEATURES).copy()
        tr, va, te = sel(df_train), sel(df_val), sel(df_test)

        va = compute_baselines(va, df_train, df_panel, k)                                # Val: бейзлайни з Train
        te = compute_baselines(te, pd.concat([df_train, df_val]), df_panel, k)           # Test: бейзлайни з Train+Val

        dtr = lgb.Dataset(prep(tr), label=tr[tgt], categorical_feature=['oblast'])
        dva = lgb.Dataset(prep(va), label=va[tgt], reference=dtr, categorical_feature=['oblast'])
        model = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        best_iter = model.best_iteration
        full = pd.concat([tr, va])                                                       # перенавчання на Train+Val
        model_full = lgb.train(params, lgb.Dataset(prep(full), label=full[tgt], categorical_feature=['oblast']),
                               num_boost_round=best_iter)

        for feat, gain in zip(model_full.feature_name(), model_full.feature_importance(importance_type='gain')):
            fi_rows.append({'Target': tgt, 'Feature': feat, 'Gain': round(float(gain), 2), 'BestIter': best_iter})

        va['p_LightGBM'] = model.predict(prep(va), num_iteration=best_iter)              # для порогів/вибору бейзлайна
        te['p_LightGBM'] = model_full.predict(prep(te))
        va['p_always_pos'] = 1.0
        te['p_always_pos'] = 1.0

        te['Group'] = np.where(te['oblast'].isin(frontline), 'Frontline', 'Rear')
        va['Group'] = np.where(va['oblast'].isin(frontline), 'Frontline', 'Rear')
        preds.append(te[['t', 'oblast', 'Group', tgt, 'p_LightGBM'] + BASE_COLS].rename(columns={tgt: 'y'}).assign(Target=tgt))

        groups = {'ALL': (va, te), 'Frontline': (va[va['Group'] == 'Frontline'], te[te['Group'] == 'Frontline']),
                  'Rear': (va[va['Group'] == 'Rear'], te[te['Group'] == 'Rear'])}
        for gname, (va_g, te_g) in groups.items():
            if len(te_g) == 0 or len(va_g) == 0:
                continue
            yv, yt = va_g[tgt].values, te_g[tgt].values
            best_base = min(BASE_COLS, key=lambda c: brier_score_loss(yv, va_g[c]))     # вибір ЛИШЕ на Val
            brier_best = brier_score_loss(yt, te_g[best_base])
            prev = float(yt.mean())

            for col, name in MODELS.items():
                prec, rec, thr = precision_recall_curve(yv, va_g[col])
                f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
                best_t = float(thr[np.argmax(f1)]) if len(thr) else 0.5
                a, b = ap(yt, te_g[col]), brier_score_loss(yt, te_g[col])
                metrics.append({'Target': tgt, 'Group': gname, 'Model': name, 'N': len(te_g),
                                'Prevalence': round(prev, 4), 'AP': round(a, 4), 'Lift': round(a / prev, 3) if prev > 0 else np.nan,
                                'Brier': round(b, 4), 'BSS (%)': round((1 - b / brier_best) * 100, 2),
                                'F1_Test': round(f1_score(yt, (te_g[col] >= best_t).astype(int)), 4),
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

    out = lambda n: os.path.join(args.results_dir, n)
    pd.DataFrame(metrics).to_csv(out('metrics_summary.csv'), index=False)
    pd.DataFrame(boots).to_csv(out('bootstrap_ci.csv'), index=False)
    pd.DataFrame(rear_rows).to_csv(out('rear_by_region.csv'), index=False)
    pd.DataFrame(fi_rows).to_csv(out('feature_importance.csv'), index=False)
    pd.concat(preds).to_csv(out('test_predictions.csv'), index=False)
    with open(out('run_info.json'), 'w', encoding='utf-8') as f:
        json.dump({'cleaning': events.attrs.get('cleaning'), 'last_valid_label_time': str(last_valid),
                   'test_end_used': str(test_end), 'frontline_oblasts': frontline}, f, ensure_ascii=False, indent=2)
    print(f"Готово. Результати у '{args.results_dir}/'.")


if __name__ == '__main__':
    main()