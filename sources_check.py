"""Diagnostic: how well does volunteer_data (eTryvoga, oblast level) agree with official_data?
Usage: python sources_check.py --official <official_data_en.csv> --volunteer <volunteer_data_en.csv>
Does not modify the project; prints tables used in the README section 'Why volunteer_data was not used'."""
import argparse
import numpy as np
import pandas as pd

KEYS = ['oblast', 'raion', 'hromada', 'level', 'started_at']
START = '2025-11-01 00:00+00:00'   # overlap window end = last full hour of the official data (derived below)


def to_ns(s):
    return s.values.astype('datetime64[ns]').astype('int64')   # explicit unit (pandas 3 uses us)


def merge_intervals(d):
    d = d.sort_values('s')
    s, f = to_ns(d['s']), to_ns(d['f'])
    cm = np.maximum.accumulate(f)
    new = np.ones(len(s), bool)
    new[1:] = s[1:] > cm[:-1]
    idx = np.flatnonzero(new)
    ends = np.r_[idx[1:], len(s)] - 1
    return s[idx], cm[ends]


def state(ms, me, grid):
    i = np.searchsorted(ms, grid, side='right') - 1
    out = np.zeros(len(grid), int)
    ok = i >= 0
    out[ok] = (grid[ok] < me[i[ok]]).astype(int)
    return out


def kappa(x, y):
    po, px, py = (x == y).mean(), x.mean(), y.mean()
    pe = px * py + (1 - px) * (1 - py)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--official', default='ukrainian-air-raid-sirens-dataset/datasets/official_data_en.csv')
    ap.add_argument('--volunteer', default='ukrainian-air-raid-sirens-dataset/datasets/volunteer_data_en.csv')
    a = ap.parse_args()
    o, v = pd.read_csv(a.official), pd.read_csv(a.volunteer)
    o = o.drop_duplicates(KEYS, keep='last')                       # official file contains two dumps
    for d in (o, v):
        d['s'] = pd.to_datetime(d['started_at'], utc=True)
        d['f'] = pd.to_datetime(d['finished_at'], utc=True)
    o = o[(o.level != 'hromada') & (o.oblast != 'Luhanska oblast')]
    v = v.rename(columns={'region': 'oblast'})

    print('official max started/finished:', o.s.max(), o.f.max())
    print('volunteer max started/finished:', v.s.max(), v.f.max())
    print('volunteer episodes after 2026-09-07:', int((v.s > pd.Timestamp('2026-09-07', tz='UTC')).sum()))
    print('names only in official:', sorted(set(o.oblast) - set(v.oblast)),
          '| only in volunteer:', sorted(set(v.oblast) - set(o.oblast)))
    exact30 = (v.f - v.s) == pd.Timedelta(minutes=30)
    print(f'volunteer rows: {len(v):,}; naive: {int(v.naive.sum()):,} ({v.naive.mean():.1%}); '
          f'exactly 30 min: {int(exact30.sum()):,} (of which naive {int(exact30[v.naive].sum()):,})')
    print('naive end = start + 30 min exactly:', bool(((v.f - v.s)[v.naive] == pd.Timedelta(minutes=30)).all()))
    print('naive share by month (last 6):')
    print(v.groupby(v.s.dt.strftime('%Y-%m')).naive.mean().tail(6).round(3).to_string())
    rec = v[v.s >= v.s.max() - pd.Timedelta(days=42)]
    print(f'naive share by region, last 42 days ({len(rec):,} rows), top 8:')
    print(rec.groupby('oblast').naive.agg(['mean', 'size']).sort_values('mean', ascending=False).head(8).round(3).to_string())
    d = (o.f - o.s).dt.total_seconds()
    print('official rows with duration exactly 3599/3600/3601 s:', [int((d == x).sum()) for x in (3599, 3600, 3601)],
          '(the 3600 s spike = processor-imputed ends, see features.py)')

    END = o.f.max().floor('h') - pd.Timedelta(hours=1)
    grid = pd.date_range(START, END, freq='1h', tz='UTC').values.astype('datetime64[ns]').astype('int64')
    rows, om, ot, vm, vt, diffs = [], 0, 0, 0, 0, []
    for ob in sorted(set(o.oblast) & set(v.oblast)):
        ao, bo = merge_intervals(o[o.oblast == ob]), merge_intervals(v[v.oblast == ob])
        xa, xb = state(*ao, grid), state(*bo, grid)
        rows.append((ob, xa.mean(), xb.mean(), (xa == xb).mean(), kappa(xa, xb)))
        sa = ao[0][(ao[0] >= grid[0]) & (ao[0] <= grid[-1])]
        sb = bo[0][(bo[0] >= grid[0]) & (bo[0] <= grid[-1])]
        if len(sa) and len(sb):
            da = np.abs(sa[:, None] - sb[None, :]).min(1) / 6e10     # ns -> minutes
            db = np.abs(sb[:, None] - sa[None, :]).min(1) / 6e10
            om += int((da <= 5).sum()); ot += len(sa); vm += int((db <= 5).sum()); vt += len(sb); diffs += list(da)
    r = pd.DataFrame(rows, columns=['oblast', 'official_active_share', 'volunteer_active_share',
                                    'hourly_agreement', 'cohen_kappa']).round(3)
    print(f'\nHourly state, {START}..{END} (official = union of oblast+raion alerts, hromada excluded):')
    print(r.sort_values('cohen_kappa').to_string(index=False))
    print(f'\nmedian kappa {r.cohen_kappa.median():.3f}; mean hourly agreement {r.hourly_agreement.mean():.3f}')
    print(f'official episodes with a volunteer start within 5 min: {om}/{ot} ({om/ot:.1%})')
    print(f'volunteer episodes with an official start within 5 min: {vm}/{vt} ({vm/vt:.1%})')
    print(f'median distance from official episode start to nearest volunteer start: {np.median(diffs):.1f} min')


if __name__ == '__main__':
    main()
