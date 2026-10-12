"""Collects the headline metrics of every executed run (from each run_info.json) into one table.

Usage: python sensitivity_summary.py  ->  results/sensitivity_summary.csv and .md
Runs from different SOURCES (official vs volunteer) have different target populations: compare only within a source.
"""
import json
import os

import pandas as pd

RUNS = [('A  official, main (causal ends, open-alert snapshot)', 'results'),
        ('A0 official, as-recorded fake ends (pre-audit data semantics)', 'results_sensitivity/official_as_recorded_ends'),
        ('C  official, without nat_active_oblasts_cnt', 'results_ablation'),
        ('V  volunteer, main (naive end := next start)', 'results_volunteer'),
        ('V0 volunteer, naive fake 30-min ends kept', 'results_sensitivity/volunteer_naive_as_recorded'),
        ('V1 volunteer, naive rows excluded', 'results_sensitivity/volunteer_naive_excluded')]


def main():
    rows = []
    for name, d in RUNS:
        p = os.path.join(d, 'run_info.json')
        if not os.path.exists(p):
            print(f'missing (not run): {p}')
            continue
        info = json.load(open(p, encoding='utf-8'))
        test = info['splits_utc']['test']
        for h in info['headline_metrics_test']:
            if h['group'] not in ('ALL', 'Rear'):
                continue
            rows.append({'Run': name, 'Test (UTC)': f"{test[0][:13]} .. {test[1][:13]}", 'Target': h['target'],
                         'Group': h['group'], 'N': h['n'], 'Prevalence': h['prevalence'],
                         'Baseline': h['best_baseline_on_val'], 'AP LGB': h['ap_lightgbm'], 'AP base': h['ap_baseline'],
                         'Brier LGB': h['brier_lightgbm'], 'Brier base': h['brier_baseline'],
                         'BSS %': h['bss_pct_vs_baseline']})
    df = pd.DataFrame(rows)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/sensitivity_summary.csv', index=False)
    lines = ['| ' + ' | '.join(df.columns) + ' |', '|' + '---|' * len(df.columns)]
    lines += ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in df.itertuples(index=False)]
    open('results/sensitivity_summary.md', 'w', encoding='utf-8').write(
        '### Sensitivity runs (test period; BSS = 1 - Brier_LGB / Brier_best-baseline-on-Val)\n\n' + '\n'.join(lines) + '\n')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
