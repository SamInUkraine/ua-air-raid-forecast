import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from features import load_events, START_UTC, DEFAULT_DATA_DIR


def count_episodes(df):
    """Епізоди тривог на рівні області: перекриті raion-тривоги зливаються в одну."""
    out = {}
    for obl, g in df.sort_values('started_at_dt').groupby('oblast'):
        s = g['started_at_dt'].values.astype('int64')
        f = g['finished_at_dt'].values.astype('int64')
        cm = np.maximum.accumulate(f)
        new = np.ones(len(s), dtype=bool)
        new[1:] = s[1:] > cm[:-1]
        out[obl] = int(new.sum())
    return pd.Series(out).sort_values(ascending=False)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--data', default=f'{DEFAULT_DATA_DIR}/datasets/official_data_en.csv')
    ap_.add_argument('--snapshot', default=f'{DEFAULT_DATA_DIR}/processors/pkl/official_active_alerts.pkl')
    ap_.add_argument('--results_dir', default='results')
    ap_.add_argument('--output_dir', default='figures')
    a = ap_.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    m = pd.read_csv(os.path.join(a.results_dir, 'metrics_summary.csv'))
    b = pd.read_csv(os.path.join(a.results_dir, 'bootstrap_ci.csv'))
    p = pd.read_csv(os.path.join(a.results_dir, 'test_predictions.csv'))

    # Fig 1: Brier, Rear
    rear = m[(m['Group'] == 'Rear') & m['Model'].isin(['LightGBM', 'Baseline_A', 'Baseline_B', 'Baseline_D'])]
    piv = rear.pivot(index='Target', columns='Model', values='Brier')
    ax = piv.plot(kind='bar', figsize=(8, 4.5), rot=0)
    ax.set_title('Fig 1: Brier score, Rear regions (lower is better)')
    ax.set_ylabel('Brier')
    plt.tight_layout(); plt.savefig(os.path.join(a.output_dir, 'fig1_brier_rear.png'), dpi=200); plt.close()

    # Fig 2: bootstrap CI різниць (LGB - найкращий бейзлайн)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    labels = [f"{r.Target}\n{r.Group}" for r in b.itertuples()]
    for ax, key, title in [(axes[0], 'Brier_Diff', 'Δ Brier (LGB − best baseline); <0 = LGB better'),
                           (axes[1], 'AP_Diff', 'Δ AP (LGB − best baseline); >0 = LGB better')]:
        for i, r in b.iterrows():
            ax.errorbar(i, r[f'{key}_Mean'], yerr=[[r[f'{key}_Mean'] - r[f'{key}_CI_Low']], [r[f'{key}_CI_High'] - r[f'{key}_Mean']]],
                        fmt='o', capsize=4, color='black')
        ax.axhline(0, ls='--', color='gray')
        ax.set_xticks(range(len(b))); ax.set_xticklabels(labels, fontsize=7); ax.set_title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(a.output_dir, 'fig2_ci_vs_baseline.png'), dpi=200); plt.close()

    # Fig 3: калібрування (Rear)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, tgt in zip(axes, ['onset_1', 'onset_3', 'onset_6']):
        d = p[(p['Target'] == tgt) & (p['Group'] == 'Rear')]
        for col, name in [('p_LightGBM', 'LightGBM'), ('p_base_b', 'Baseline B')]:
            fp, mp = calibration_curve(d['y'], d[col], n_bins=10, strategy='quantile')
            ax.plot(mp, fp, marker='o', label=name)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title(f'Fig 3: reliability, Rear, {tgt}'); ax.set_xlabel('Predicted'); ax.set_ylabel('Observed'); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(a.output_dir, 'fig3_calibration_rear.png'), dpi=200); plt.close()

    # Fig 4: топ-10 областей за кількістю епізодів тривог (з сирих подій, не з панелі)
    ev = load_events(a.data, snapshot_path=a.snapshot if os.path.exists(a.snapshot) else None)
    ev = ev[(ev['started_at_dt'] >= pd.Timestamp(START_UTC)) & ~ev['open_at_cut']]
    top = count_episodes(ev).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top.index, top.values, color='steelblue')
    ax.set_title(f'Fig 4: Top-10 oblasts by alert episodes since {START_UTC[:10]} (overlapping raion alerts merged)', fontsize=9)
    ax.set_xlabel('Episodes')
    plt.tight_layout(); plt.savefig(os.path.join(a.output_dir, 'fig4_top10_regions.png'), dpi=200); plt.close()
    print(f"Figures saved to {a.output_dir}/")


if __name__ == '__main__':
    main()