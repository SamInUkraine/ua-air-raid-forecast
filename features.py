"""Event loading, cleaning and leakage-safe hourly panel construction.

How the source is produced (verified by reading the dataset repo's `processors/*.py`):
  * Alerts are reconstructed from Telegram messages (official: @air_alert_ua; volunteer: eTryvoga).
    A 🔴 message opens an alert at a location, a 🟢 message closes it. Both are published in real time,
    so "started_at <= t < finished_at" is exactly the state a live listener knows at t.
  * A row is written to the CSV only once the alert is CLOSED. Alerts still open when the processor last ran are
    kept in `processors/pkl/*_active_alerts.pkl`, NOT in the CSV.  => the CSV tail is incomplete (missing starts).
  * If a new 🔴 arrives for a location that is still open and >3 h old, the processor writes the old alert with a
    FAKE end: started_at + 1 h (official) or started_at + 30 min with naive=True (volunteer).  That fake end is
    created only when the next alert starts, i.e. it is retrospective information: in real time the location
    looked active until that next 🔴.  Default policy 'causal' therefore sets the effective end of an imputed
    alert to the next start at the same location.
"""
import argparse
import datetime
import hashlib
import os
import pickle
import warnings

import numpy as np
import pandas as pd

START_UTC = '2025-11-01 00:00:00+00:00'      # raion-level alerting dominates from Nov 2025 (see README)
MAX_HORIZON_H = 6
IMPUTE_GAP_H = 3                             # processor rule: imputation only if next 🔴 is >3 h after the start
OPEN_STALE_H = 48                            # open-at-cut alerts older than this are treated as stale (dropped)
FAR_FUTURE = pd.Timestamp('2262-01-01', tz='UTC')
TARGETS = ['onset_1', 'onset_3', 'onset_6']
DEFAULT_DATA_DIR = 'ukrainian-air-raid-sirens-dataset'


# ----------------------------------------------------------------------------------------------- snapshot of open alerts
class _Stub:
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else (state[1] or {}))


class _SafeUnpickler(pickle.Unpickler):
    """Only datetime objects and a stub for the processor's dataclass may be created (no arbitrary code)."""
    def find_class(self, module, name):
        if module == 'datetime' and name in ('datetime', 'timezone', 'timedelta'):
            return getattr(datetime, name)
        if module.startswith('processors.'):
            return _Stub
        raise pickle.UnpicklingError(f'blocked global {module}.{name}')


def read_open_snapshot(path):
    """Alerts that were still open when the source processor last ran (Ukrainian names)."""
    with open(path, 'rb') as f:
        d = _SafeUnpickler(f).load()
    rows = []
    for v in d.values():
        a = v.__dict__
        rows.append({'oblast_uk': a.get('oblast') or a.get('region'), 'raion_uk': a.get('raion') or '',
                     'hromada_uk': a.get('hromada') or '', 'level': a.get('level') or 'oblast',
                     'started_at_dt': pd.Timestamp(a['started_at']).tz_convert('UTC')})
    cols = ['oblast_uk', 'raion_uk', 'hromada_uk', 'level', 'started_at_dt']
    return pd.DataFrame(rows, columns=cols).astype({'started_at_dt': 'datetime64[ns, UTC]'})


def _uk_en_map(en_path, uk_path, cols):
    """The *_en.csv and *_uk.csv files are row-aligned translations; build a uk->en name map."""
    en = pd.read_csv(en_path, usecols=cols).fillna('')
    uk = pd.read_csv(uk_path, usecols=cols).fillna('')
    if len(en) != len(uk):
        raise ValueError(f'{en_path} and {uk_path} are not row-aligned ({len(en)} vs {len(uk)})')
    m = {}
    for c in cols:
        pairs = pd.DataFrame({'uk': uk[c], 'en': en[c]}).drop_duplicates()
        if pairs['uk'].duplicated().any():
            raise ValueError(f'ambiguous uk->en mapping in column {c}')
        m[c] = dict(zip(pairs['uk'], pairs['en']))
    return m


# ------------------------------------------------------------------------------------------------------- event loading
def _causal_end(df, key_cols, imputed):
    """Effective end of an imputed alert = next start at the same location (when the processor itself learned it)."""
    df = df.sort_values('started_at_dt', kind='stable')
    nxt = df.groupby(key_cols, dropna=False)['started_at_dt'].shift(-1)
    imputed = imputed.reindex(df.index)
    ok = imputed & nxt.notna()
    df.loc[ok, 'finished_at_dt'] = nxt[ok]
    return df, int(ok.sum()), int((imputed & nxt.isna()).sum())


def _data_cut(df):
    """Lower bound of the last message the source processor handled (computed before any row filtering)."""
    return max(df['started_at_dt'].max(), df['finished_at_raw'].max())


def _finalize(df, info, snapshot, data_cut):
    info['data_cut_utc'] = str(data_cut)
    df = df.sort_values('started_at_dt', kind='stable').reset_index(drop=True)
    df.attrs['cleaning'] = info
    df.attrs['data_cut'] = data_cut
    df.attrs['open_snapshot_used'] = snapshot is not None
    print(f"[load] {info}")
    return df


def _attach_open(df, snap, info, key_cols):
    """Add open-at-cut alerts as events with unknown end (active until the end of the data)."""
    if snap is None or len(snap) == 0:
        return df
    cut = max(df['started_at_dt'].max(), df['finished_at_raw'].max(), snap['started_at_dt'].max())
    stale = snap['started_at_dt'] < cut - pd.Timedelta(hours=OPEN_STALE_H)
    info['open_at_cut_total'] = int(len(snap))
    info['open_at_cut_stale_dropped'] = int(stale.sum())
    snap = snap[~stale].copy()
    snap['finished_at_dt'] = FAR_FUTURE
    snap['finished_at_raw'] = pd.NaT
    snap['end_imputed'] = False
    snap['open_at_cut'] = True
    info['open_at_cut_used'] = int(len(snap))
    out = pd.concat([df, snap[df.columns.intersection(snap.columns)]], ignore_index=True)
    for c in ('started_at_dt', 'finished_at_dt', 'finished_at_raw'):
        out[c] = pd.to_datetime(out[c], utc=True)
    out['end_imputed'] = out['end_imputed'].astype(bool)
    out['open_at_cut'] = out['open_at_cut'].astype(bool)
    return out


def load_events(path, snapshot_path=None, uk_path=None, end_policy='causal'):
    """Official dataset (oblast/raion/hromada levels). Single place where the official data is cleaned.

    end_policy: 'causal' (default) - imputed 1 h ends replaced by the next start at the same location;
                'as_recorded'      - keep the processor's fake ends (the pre-audit behaviour, for sensitivity only).
    snapshot_path: processors/pkl/official_active_alerts.pkl (open alerts at the data cut); None = not used.
    """
    assert end_policy in ('causal', 'as_recorded')
    df = pd.read_csv(path)
    info = {'source_file': os.path.basename(path), 'raw_rows': len(df), 'end_policy': end_policy}
    n0 = len(df)
    df = df.drop_duplicates().copy()                          # the file contains a second full dump (exact copies)
    info['exact_duplicates_dropped'] = n0 - len(df)
    df['started_at_dt'] = pd.to_datetime(df['started_at'], utc=True, errors='coerce')
    df['finished_at_dt'] = pd.to_datetime(df['finished_at'], utc=True, errors='coerce')
    bad = df['started_at_dt'].isna() | df['finished_at_dt'].isna()
    info['missing_or_malformed_time_dropped'] = int(bad.sum())
    df = df[~bad]
    neg = df['finished_at_dt'] < df['started_at_dt']
    info['negative_duration_dropped'] = int(neg.sum())
    df = df[~neg].copy()
    for c in ('raion', 'hromada'):
        df[c] = df[c].fillna('')
    key = ['oblast', 'raion', 'hromada', 'level']

    df['finished_at_raw'] = df['finished_at_dt']
    df['end_imputed'] = False
    df['open_at_cut'] = False

    snap = None
    if snapshot_path:
        snap = read_open_snapshot(snapshot_path)
        uk_path = uk_path or path.replace('_en.csv', '_uk.csv')
        mp = _uk_en_map(path, uk_path, ['oblast', 'raion', 'hromada'])
        snap['oblast'] = snap['oblast_uk'].map(mp['oblast'])
        snap['raion'] = snap['raion_uk'].map(lambda x: mp['raion'].get(x, x) if x else '')
        snap['hromada'] = snap['hromada_uk'].map(lambda x: mp['hromada'].get(x, x) if x else '')
        info['open_at_cut_unmapped_oblast'] = int(snap['oblast'].isna().sum())
        snap = snap.dropna(subset=['oblast'])
    df = _attach_open(df, snap, info, key)        # open alerts first: they can be the "next start" of an imputed one

    # imputed end = exactly 1 h AND the next start at the same location is > 3 h after the start (processor rule)
    df = df.sort_values('started_at_dt', kind='stable')
    dur = (df['finished_at_raw'] - df['started_at_dt']).dt.total_seconds()
    nxt = df.groupby(key)['started_at_dt'].shift(-1)
    imputed = (dur == 3600) & ((nxt - df['started_at_dt']) > pd.Timedelta(hours=IMPUTE_GAP_H))
    df['end_imputed'] = imputed
    info['imputed_end_rows_all_levels'] = int(imputed.sum())
    if end_policy == 'causal':
        df, n_fix, _ = _causal_end(df, key, imputed)
        info['imputed_end_replaced_by_next_start'] = n_fix

    cut = _data_cut(df)
    keep = (df['level'] != 'hromada') & (df['oblast'] != 'Luhanska oblast')
    info['hromada_or_luhanska_dropped'] = int((~keep).sum())
    df = df[keep]
    info['clean_rows'] = int(len(df))
    info['end_imputed_in_scope'] = int(df['end_imputed'].sum())
    return _finalize(df, info, snap, cut)


def load_volunteer_events(path, snapshot_path=None, uk_path=None, naive_policy='causal'):
    """Volunteer (eTryvoga) dataset: oblast-level only, `naive=True` rows have a fake end = start + 30 min.

    naive_policy: 'causal' (default) - effective end of a naive alert = next start in the same region;
                  'as_recorded' - keep fake ends;  'exclude' - drop naive rows entirely (sensitivity only:
                  this also deletes their genuine starts).
    """
    assert naive_policy in ('causal', 'as_recorded', 'exclude')
    df = pd.read_csv(path).rename(columns={'region': 'oblast'})
    info = {'source_file': os.path.basename(path), 'raw_rows': len(df), 'naive_policy': naive_policy}
    n0 = len(df)
    df = df.drop_duplicates().copy()
    info['exact_duplicates_dropped'] = n0 - len(df)
    df['started_at_dt'] = pd.to_datetime(df['started_at'], utc=True, errors='coerce')
    df['finished_at_dt'] = pd.to_datetime(df['finished_at'], utc=True, errors='coerce')
    bad = df['started_at_dt'].isna() | df['finished_at_dt'].isna()
    info['missing_or_malformed_time_dropped'] = int(bad.sum())
    df = df[~bad].copy()
    df['naive'] = df['naive'].astype(str).str.lower().eq('true')
    df['level'] = 'oblast'
    dup = df.duplicated(subset=['oblast', 'started_at_dt'], keep='last')
    info['same_region_same_start_dropped'] = int(dup.sum())
    df = df[~dup]
    df['end_imputed'] = df['naive']
    df['finished_at_raw'] = df['finished_at_dt']
    df['open_at_cut'] = False
    info['naive_rows'] = int(df['naive'].sum())

    snap = None
    if snapshot_path:
        snap = read_open_snapshot(snapshot_path)
        uk_path = uk_path or path.replace('_en.csv', '_uk.csv')
        mp = _uk_en_map(path, uk_path, ['region'])['region']
        snap['oblast'] = snap['oblast_uk'].map(mp)
        snap['level'] = 'oblast'
        snap['naive'] = False
        info['open_at_cut_unmapped_oblast'] = int(snap['oblast'].isna().sum())
        snap = snap.dropna(subset=['oblast'])
    df = _attach_open(df, snap, info, ['oblast'])
    df['naive'] = df['naive'].fillna(False).astype(bool)
    if naive_policy == 'causal':
        df, n_fix, n_nofix = _causal_end(df, ['oblast'], df['naive'])
        info['naive_end_replaced_by_next_start'] = n_fix
        info['naive_without_next_start_kept_as_recorded'] = n_nofix
    cut = _data_cut(df)
    if naive_policy == 'exclude':
        info['naive_rows_excluded'] = int(df['naive'].sum())
        df = df[~df['naive']]
    df = df[df['oblast'] != 'Luhanska oblast']
    info['clean_rows'] = int(len(df))
    return _finalize(df, info, snap, cut)


# -------------------------------------------------------------------------------------------- label-completeness rule
def last_valid_label_time(df_events, horizon_h=MAX_HORIZON_H):
    """Last forecast origin t whose label onset_h(t) is fully observable.

    onset_h(t) needs every alert START in (t, t+h].  The CSV holds only CLOSED alerts, so starts of alerts still open
    at the data cut are missing unless the open-alert snapshot is attached.
      * with snapshot: starts are complete up to the cut  ->  t + h <= cut
      * without:       an open alert may have started up to OPEN_STALE_H before the cut -> t + h <= cut - OPEN_STALE_H
    The cut itself is a lower bound of the last processed message (max timestamp seen), so the rule is conservative.
    """
    cut = df_events.attrs.get('data_cut')
    if cut is None:                                            # synthetic data in tests
        cut = max(df_events['started_at_dt'].max(), df_events['finished_at_dt'][df_events['finished_at_dt'] < FAR_FUTURE].max())
        return (cut - pd.Timedelta(hours=horizon_h)).floor('h')
    if df_events.attrs.get('open_snapshot_used'):
        return (cut - pd.Timedelta(hours=horizon_h)).floor('h')
    warnings.warn('No open-alert snapshot: using conservative label cut-off (cut - horizon - %dh)' % OPEN_STALE_H)
    return (cut - pd.Timedelta(hours=horizon_h + OPEN_STALE_H)).floor('h')


# ------------------------------------------------------------------------------------------------------------ panel
def generate_panel_data(df_events, start_utc=START_UTC, end_utc=None):
    """Hourly UTC grid x oblast.

    Features at origin t use only: alert starts in (t-k, t]; alerts active at t (started <= t < finished, i.e. a 🔴
    seen and no 🟢 yet); finishes <= t.  Targets: onset_k(t) = 1 if any alert (oblast or raion level) in the oblast
    STARTS in (t, t+k]; defined only when the oblast is inactive at t (else NaN).
    """
    if end_utc is None:
        end_utc = last_valid_label_time(df_events)
    time_grid = pd.date_range(start=start_utc, end=end_utc, freq='1h', tz='UTC')
    oblasts = sorted(df_events['oblast'].unique())

    events_by_obl = {
        obl: {'starts': g['started_at_dt'].values, 'finishes': g['finished_at_dt'].values}
        for obl, g in df_events.groupby('oblast')
    }
    # national arrays: only events that can be active/recent inside the grid (exactly equivalent, just faster)
    lo = pd.Timestamp(start_utc) - pd.Timedelta(hours=4)
    nat = df_events[df_events['finished_at_dt'] > lo]
    all_starts = nat['started_at_dt'].values
    all_finishes = nat['finished_at_dt'].values
    all_oblasts = nat['oblast'].values

    h = lambda k: np.timedelta64(k, 'h')
    records = []
    for t in time_grid:
        t_np = t.to_datetime64()
        nat_active_cnt = len(set(all_oblasts[(all_starts <= t_np) & (all_finishes > t_np)]))
        nat_s1 = len(set(all_oblasts[(all_starts > t_np - h(1)) & (all_starts <= t_np)]))
        nat_s3 = len(set(all_oblasts[(all_starts > t_np - h(3)) & (all_starts <= t_np)]))
        t_kyiv = t.tz_convert('Europe/Kyiv')

        for obl in oblasts:
            starts = events_by_obl[obl]['starts']
            finishes = events_by_obl[obl]['finishes']
            active_mask = (starts <= t_np) & (finishes > t_np)
            is_active = int(np.any(active_mask))
            hours_start = (t_np - np.min(starts[active_mask])) / h(1) if is_active else 0.0
            past_fin = finishes[finishes <= t_np]
            hours_finish = (t_np - np.max(past_fin)) / h(1) if len(past_fin) > 0 else np.nan
            cnt = lambda k: int(np.sum((starts > t_np - h(k)) & (starts <= t_np)))

            if is_active == 0:
                fut = lambda k: int(np.any((starts > t_np) & (starts <= t_np + h(k))))
                o1, o3, o6 = fut(1), fut(3), fut(6)
            else:
                o1 = o3 = o6 = np.nan

            records.append({
                't': t, 'oblast': obl, 'is_active': is_active,
                'hours_since_current_start': round(hours_start, 4),
                'hours_since_last_finish': round(hours_finish, 4) if not np.isnan(hours_finish) else np.nan,
                'starts_1h': cnt(1), 'starts_3h': cnt(3), 'starts_6h': cnt(6),
                'starts_24h': cnt(24), 'starts_168h': cnt(168),
                'nat_active_oblasts_cnt': nat_active_cnt,
                'nat_starts_1h': nat_s1, 'nat_starts_3h': nat_s3,
                'hour_of_day_kyiv': t_kyiv.hour, 'day_of_week_kyiv': t_kyiv.dayofweek,
                'onset_1': o1, 'onset_3': o3, 'onset_6': o6,
            })
    return pd.DataFrame(records)


def events_fingerprint(df_events):
    cols = ['oblast', 'started_at_dt', 'finished_at_dt']
    return hashlib.sha256(pd.util.hash_pandas_object(df_events[cols], index=False).values.tobytes()).hexdigest()


def get_panel(df_events, start_utc, end_utc, cache_dir='.cache'):
    """Cache keyed by the cleaned events themselves + this file's code + bounds (stale cache impossible)."""
    hsh = hashlib.sha256(events_fingerprint(df_events).encode())
    with open(__file__, 'rb') as f:
        hsh.update(f.read())
    hsh.update(f'{start_utc}|{end_utc}'.encode())
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'panel_{hsh.hexdigest()[:16]}.pkl')
    if os.path.exists(path):
        print(f'[panel] cache hit {path}')
        return pd.read_pickle(path)
    print(f'[panel] building {start_utc} .. {end_utc} (several minutes)')
    panel = generate_panel_data(df_events, start_utc=start_utc, end_utc=end_utc)
    panel.to_pickle(path)
    return panel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build panel features dataset (official data)')
    parser.add_argument('--data', default=f'{DEFAULT_DATA_DIR}/datasets/official_data_en.csv')
    parser.add_argument('--snapshot', default=f'{DEFAULT_DATA_DIR}/processors/pkl/official_active_alerts.pkl')
    parser.add_argument('--output', default='panel.pkl')
    args = parser.parse_args()
    events = load_events(args.data, snapshot_path=args.snapshot if os.path.exists(args.snapshot) else None)
    generate_panel_data(events).to_pickle(args.output)
    print(f'Panel saved to {args.output}')
