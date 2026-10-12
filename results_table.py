"""Генерує таблиці для README з CSV, щоб жодних цифр не вписувати руками."""
import os
import argparse
import pandas as pd


def md(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join('' if pd.isna(v) else str(v) for v in r) + ' |')
    return '\n'.join(lines)


ap = argparse.ArgumentParser()
ap.add_argument('--results_dir', default='results')
a = ap.parse_args()
m = pd.read_csv(os.path.join(a.results_dir, 'metrics_summary.csv'))
b = pd.read_csv(os.path.join(a.results_dir, 'bootstrap_ci.csv'))

keep = []
for (t, g), d in m.groupby(['Target', 'Group'], sort=False):
    best = d['Best_Baseline_Used'].iloc[0]
    keep.append(d[d['Model'].isin(['LightGBM', best, 'Always_Positive'])])
tbl = pd.concat(keep)[['Target', 'Group', 'Model', 'N', 'Prevalence', 'AP', 'Lift', 'Brier', 'BSS (%)', 'Precision_Test', 'Recall_Test', 'F1_Test']]

out = ['### Test metrics (AP = average precision; BSS relative to best baseline chosen on Val)', '', md(tbl), '',
       '### Block-bootstrap (by day) 95% CI of LightGBM − best baseline', '', md(b), '']
path = os.path.join(a.results_dir, 'results_tables.md')
open(path, 'w', encoding='utf-8').write('\n'.join(out))
print('Saved', path)