"""Leakage, label-completeness and data-handling tests. All tests call the REAL project functions.

The central end-to-end test (`test_source_replay_*`) replays how the source processor builds the dataset:
at a cut time T it only knows alerts whose end was revealed by T (CSV) plus alerts still open (snapshot).
Features computed from that "as-of-T" data must equal features computed from the full, later data.
"""
import os
import pickle
import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from features import (generate_panel_data, load_events, load_volunteer_events, last_valid_label_time,
                      FAR_FUTURE, TARGETS, MAX_HORIZON_H)
from run_pipeline import FEATURES, make_splits, baseline_b_series, freshness

FEATURE_COLS = ['is_active', 'hours_since_current_start', 'hours_since_last_finish',
                'starts_1h', 'starts_3h', 'starts_6h', 'starts_24h', 'starts_168h',
                'nat_active_oblasts_cnt', 'nat_starts_1h', 'nat_starts_3h',
                'hour_of_day_kyiv', 'day_of_week_kyiv']
UTC = lambda s: pd.Timestamp(s, tz='UTC')
REAL_DATA = os.path.join('ukrainian-air-raid-sirens-dataset', 'datasets', 'official_data_en.csv')
REAL_SNAP = os.path.join('ukrainian-air-raid-sirens-dataset', 'processors', 'pkl', 'official_active_alerts.pkl')


def _cmp(a, b, cols):
    a = a.sort_values(['t', 'oblast'])[['t', 'oblast'] + cols].reset_index(drop=True)
    b = b.sort_values(['t', 'oblast'])[['t', 'oblast'] + cols].reset_index(drop=True)
    assert len(a) == len(b)
    return int((a.fillna(-999) != b.fillna(-999)).sum().sum())


# ------------------------------------------------------------------ 1. censoring / future-perturbation invariance
def assert_features_causal(df_events, T, start):
    """Features for t<=T must not change if events after T are removed and alerts open at T get an unknown end."""
    full = generate_panel_data(df_events, start_utc=start, end_utc=T)
    cens = df_events[df_events['started_at_dt'] <= T].copy()
    future_finish = cens['finished_at_dt'] > T
    assert future_finish.sum() > 0, f'censoring not exercised: no active alerts at T={T}'
    cens.loc[future_finish, 'finished_at_dt'] = FAR_FUTURE
    assert _cmp(full, generate_panel_data(cens, start_utc=start, end_utc=T), FEATURE_COLS) == 0


@pytest.mark.parametrize('test_t', ['2026-02-15 13:00:00+00:00', '2026-05-10 01:00:00+00:00',
                                    '2026-08-10 17:00:00+00:00'])
def test_periodic_events(test_t):
    dates = pd.date_range('2025-11-01', '2026-09-07', freq='4h', tz='UTC')
    ev = pd.DataFrame({'oblast': ['Kyivska oblast' if i % 2 == 0 else 'Kharkivska oblast' for i in range(len(dates))],
                       'started_at_dt': dates, 'finished_at_dt': dates + pd.Timedelta(hours=2)})
    assert_features_causal(ev, pd.Timestamp(test_t), '2025-11-01 00:00:00+00:00')


def _random_events(seed, n=500, oblasts=('A', 'B', 'C')):
    rng = np.random.default_rng(seed)
    t0 = UTC('2026-01-01')
    rows = []
    for _ in range(n):
        s = t0 + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 60)))
        rows.append({'oblast': str(rng.choice(list(oblasts))), 'started_at_dt': s,
                     'finished_at_dt': s + pd.Timedelta(minutes=int(rng.integers(5, 900)))})
    return pd.DataFrame(rows), rng, t0


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_random_overlapping_events(seed):
    ev, rng, t0 = _random_events(seed)
    T = (t0 + pd.Timedelta(hours=int(rng.integers(24 * 12, 24 * 50)))).floor('h')
    assert_features_causal(ev, T, '2026-01-01 00:00:00+00:00')


@pytest.mark.parametrize('seed', [3, 4])
def test_future_perturbation_invariance(seed):
    """Adding, deleting and changing events strictly after T changes nothing at t<=T (features) and
    nothing at t<=T-6h (labels: they only look into (t, t+6h])."""
    ev, rng, t0 = _random_events(seed)
    T = t0 + pd.Timedelta(days=30)
    start = '2026-01-20 00:00:00+00:00'
    base = generate_panel_data(ev, start_utc=start, end_utc=T)
    pert = ev[~((ev['started_at_dt'] > T) & (rng.random(len(ev)) < 0.5))].copy()           # delete future starts
    late = pert['finished_at_dt'] > T                                                        # move future ends
    pert.loc[late, 'finished_at_dt'] = pert.loc[late, 'finished_at_dt'] + pd.Timedelta(hours=5)
    extra = pd.DataFrame({'oblast': ['A', 'B', 'C'] * 20,                                    # add future events
                          'started_at_dt': [T + pd.Timedelta(minutes=17 * i + 1) for i in range(60)]})
    extra['finished_at_dt'] = extra['started_at_dt'] + pd.Timedelta(minutes=30)
    pert = pd.concat([pert, extra], ignore_index=True)
    new = generate_panel_data(pert, start_utc=start, end_utc=T)
    assert _cmp(base, new, FEATURE_COLS) == 0
    early = lambda d: d[d['t'] <= T - pd.Timedelta(hours=MAX_HORIZON_H)]
    assert _cmp(early(base), early(new), TARGETS) == 0
    assert _cmp(base, new, TARGETS) > 0, 'perturbation did not touch any label - test not exercised'


# ------------------------------------------------------------------------------------ 2. target exclusion
def test_targets_not_in_model_inputs():
    leaky = set(TARGETS) | {'is_active', 'hours_since_current_start', 'y'} | {c for c in FEATURES if c.startswith('onset')}
    assert not set(FEATURES) & leaky
    assert len(FEATURES) == len(set(FEATURES))


# -------------------------------------------------------------------- 3/5. window and label boundary semantics
def _one(starts_finishes, oblast='A'):
    return pd.DataFrame({'oblast': oblast, 'started_at_dt': [UTC(s) for s, _ in starts_finishes],
                         'finished_at_dt': [UTC(f) for _, f in starts_finishes]})


def _row(panel, t, oblast='A'):
    return panel[(panel['t'] == UTC(t)) & (panel['oblast'] == oblast)].iloc[0]


def test_label_boundaries():
    # history event so that hours_since_last_finish exists; next start exactly at t+3h
    ev = _one([('2026-03-01 00:00', '2026-03-01 00:10'), ('2026-03-01 13:00', '2026-03-01 13:20')])
    p = generate_panel_data(ev, start_utc='2026-03-01 10:00:00+00:00', end_utc=UTC('2026-03-01 12:00'))
    r = _row(p, '2026-03-01 10:00')
    assert (r.onset_1, r.onset_3, r.onset_6) == (0, 1, 1)          # start at exactly t+3h is INCLUDED in (t, t+3h]
    r = _row(p, '2026-03-01 12:00')
    assert (r.onset_1, r.onset_3) == (1, 1)                         # start at exactly t+1h included
    ev2 = _one([('2026-03-01 00:00', '2026-03-01 00:10'), ('2026-03-01 13:00:01', '2026-03-01 13:20')])
    p2 = generate_panel_data(ev2, start_utc='2026-03-01 10:00:00+00:00', end_utc=UTC('2026-03-01 10:00'))
    r = _row(p2, '2026-03-01 10:00')
    assert (r.onset_1, r.onset_3, r.onset_6) == (0, 0, 1)          # one second after t+3h is EXCLUDED


def test_start_exactly_at_t_is_active_not_label():
    ev = _one([('2026-03-01 00:00', '2026-03-01 00:10'), ('2026-03-01 10:00', '2026-03-01 10:30')])
    p = generate_panel_data(ev, start_utc='2026-03-01 09:00:00+00:00', end_utc=UTC('2026-03-01 11:00'))
    r9, r10, r11 = (_row(p, f'2026-03-01 {h}:00') for h in ('09', '10', '11'))
    assert r9.onset_1 == 1                                          # (09:00, 10:00] contains 10:00
    assert r10.is_active == 1 and np.isnan(r10.onset_1)            # alert starting at t => active, not eligible
    assert r10.starts_1h == 1                                       # and counted as a past start: (t-1h, t]
    assert r11.is_active == 0 and r11.hours_since_last_finish == 0.5


def test_finish_exactly_at_t_and_rolling_edges():
    ev = _one([('2026-03-01 07:00', '2026-03-01 08:00'), ('2026-03-01 09:00', '2026-03-01 09:30')])
    p = generate_panel_data(ev, start_utc='2026-03-01 08:00:00+00:00', end_utc=UTC('2026-03-01 10:00'))
    r8 = _row(p, '2026-03-01 08:00')
    assert r8.is_active == 0 and r8.hours_since_last_finish == 0    # finished at t => not active ([start, finish))
    r10 = _row(p, '2026-03-01 10:00')
    assert r10.starts_1h == 0 and r10.starts_3h == 1                # starts at exactly t-1h / t-3h are OUTSIDE (t-k, t]
    assert r10.starts_6h == 2


def test_rolling_counts_ignore_future_starts():
    ev = _one([('2026-02-25 00:00', '2026-02-25 00:10')] + [(f'2026-03-02 {h:02d}:30', f'2026-03-02 {h:02d}:40')
                                                            for h in range(24)])
    p = generate_panel_data(ev, start_utc='2026-03-01 12:00:00+00:00', end_utc=UTC('2026-03-02 00:00'))
    assert (p[['starts_1h', 'starts_3h', 'starts_6h', 'starts_24h']] == 0).all().all()
    assert p['starts_168h'].max() == 1


# ----------------------------------------------------------------------------------------- 4. baseline alignment
@pytest.mark.parametrize('k', [1, 3, 6])
def test_baseline_b_uses_only_resolved_labels(k):
    ev, rng, t0 = _random_events(7, n=250)
    T = t0 + pd.Timedelta(days=30)
    start = '2026-01-10 00:00:00+00:00'
    base = generate_panel_data(ev, start_utc=start, end_utc=T)
    pert = pd.concat([ev, pd.DataFrame({'oblast': ['A', 'B', 'C'] * 10,
                                        'started_at_dt': [T + pd.Timedelta(minutes=13 * i + 1) for i in range(30)]})
                      .assign(finished_at_dt=lambda d: d['started_at_dt'] + pd.Timedelta(minutes=20))])
    new = generate_panel_data(pert, start_utc=start, end_utc=T)
    assert _cmp(base, new, [f'onset_{k}']) > 0                     # labels near T did change ...
    b0, b1 = baseline_b_series(base, k), baseline_b_series(new, k)
    m = b0.merge(b1, on=['t', 'oblast'])
    assert np.allclose(m['p_base_b_calc_x'].fillna(-1), m['p_base_b_calc_y'].fillna(-1))   # ... baseline B did not


def test_baseline_b_lag_is_necessary():
    """Negative control: a k-1 lag WOULD use an unresolved label (shows the alignment test has teeth)."""
    ev = _one([('2026-03-01 00:00', '2026-03-01 00:10'), ('2026-03-01 12:30', '2026-03-01 12:40')])
    p = generate_panel_data(ev, start_utc='2026-03-01 06:00:00+00:00', end_utc=UTC('2026-03-01 12:00'))
    b = baseline_b_series(p, 3).set_index('t')['p_base_b_calc']
    assert b.loc[UTC('2026-03-01 12:00')] == 0                       # onset at 12:30 not yet visible at 12:00
    assert p.set_index('t').loc[UTC('2026-03-01 10:00'), 'onset_3'] == 1   # though label at 10:00 already contains it


# -------------------------------------------------------------------------------------------- 6. time zones
def test_kyiv_calendar_features_and_dst():
    ev = _one([('2025-12-31 00:00', '2025-12-31 00:10')])
    for t, hour, dow in [('2026-01-15 10:00', 12, 3),               # winter UTC+2, Thursday
                         ('2026-07-15 10:00', 13, 2),               # summer UTC+3, Wednesday
                         ('2026-03-29 00:00', 2, 6),                # just before DST switch (01:00 UTC)
                         ('2026-03-29 01:00', 4, 6),                # just after DST switch
                         ('2026-01-11 23:00', 1, 0)]:               # Sunday in UTC, Monday in Kyiv
        p = generate_panel_data(ev, start_utc=UTC(t), end_utc=UTC(t))
        assert (p.hour_of_day_kyiv.iloc[0], p.day_of_week_kyiv.iloc[0]) == (hour, dow), t


# ------------------------------------------------------------------------------------------- 7. split isolation
@pytest.mark.parametrize('last_valid', ['2026-09-07 00:00', '2026-10-11 23:00', '2026-08-01 05:00'])
def test_splits_chronological_with_gaps(last_valid):
    sp = make_splits(UTC(last_valid))
    (a0, a1), (b0, b1), (c0, c1) = sp['train'], sp['val'], sp['test']
    assert a0 < a1 < b0 <= b1 < c0 <= c1 == UTC(last_valid)
    for end, nxt in ((a1, b0), (b1, c0)):
        assert nxt - end > pd.Timedelta(hours=MAX_HORIZON_H)         # a label at `end` cannot see into next split
    assert make_splits(UTC(last_valid)) == sp                        # deterministic


# ------------------------------------------------------- 8/9/10. source replay: open alerts, imputed ends, dups
class _FakeAlert:
    pass


def _write_snapshot(path, open_rows):
    mod = types.ModuleType('processors.tg_dataclasses')
    _FakeAlert.__module__ = 'processors.tg_dataclasses'
    _FakeAlert.__qualname__ = 'OfficialAirRaidAlertChannelAlert'
    mod.OfficialAirRaidAlertChannelAlert = _FakeAlert
    sys.modules.setdefault('processors', types.ModuleType('processors'))
    sys.modules['processors.tg_dataclasses'] = mod
    d = {}
    for i, r in enumerate(open_rows):
        a = _FakeAlert()
        a.__dict__.update({'oblast': r['oblast'], 'raion': r['raion'], 'hromada': '', 'level': r['level'],
                           'started_at': r['start'].to_pydatetime(), 'finished_at': None, 'source': 'official'})
        d[f'#{i}'] = a
    with open(path, 'wb') as f:
        pickle.dump(d, f)


def _simulate_source(seed, n_per_key=60):
    """Ground truth alerts per location; ~15% have no 'end' message (processor imputes start+1h when the next
    start for that location comes >3h later). Returns rows with the time each record became known (reveal)."""
    rng = np.random.default_rng(seed)
    keys = [('Aoblast', 'R1', 'raion'), ('Aoblast', 'R2', 'raion'), ('Boblast', '', 'oblast'), ('Boblast', 'R3', 'raion')]
    rows = []
    for ob, ra, lvl in keys:
        t = UTC('2026-01-01') + pd.Timedelta(minutes=int(rng.integers(0, 600)))
        for _ in range(n_per_key):
            no_end = rng.random() < 0.15
            if no_end:
                gap = pd.Timedelta(minutes=int(rng.integers(3 * 60 + 5, 20 * 60)))
                rows.append(dict(oblast=ob, raion=ra, level=lvl, start=t, fin=t + pd.Timedelta(hours=1), reveal=t + gap))
                t = t + gap
            else:
                m = int(rng.integers(5, 400))
                dur = pd.Timedelta(minutes=61 if m == 60 else m)   # a genuine exactly-1h alert is indistinguishable
                rows.append(dict(oblast=ob, raion=ra, level=lvl, start=t, fin=t + dur, reveal=t + dur))
                t = t + dur + pd.Timedelta(minutes=int(rng.integers(10, 900)))
    return pd.DataFrame(rows)


def _write_csv(path, rows):
    df = pd.DataFrame({'oblast': rows['oblast'], 'raion': rows['raion'].replace('', np.nan), 'hromada': np.nan,
                       'level': rows['level'], 'started_at': rows['start'].astype(str),
                       'finished_at': rows['fin'].astype(str), 'source': 'official'})
    df.to_csv(path, index=False)
    df.to_csv(path.replace('_en.csv', '_uk.csv'), index=False)       # identical names -> identity uk->en map


def _pick_T(truth):
    """A cut 7 h after the start of an alert that has no 'end' message yet (open at T, started before T-6h)."""
    r = truth[(truth['reveal'] - truth['start'] > pd.Timedelta(hours=9))
              & (truth['start'] > truth['start'].min() + pd.Timedelta(days=5))].iloc[0]
    return (r['start'] + pd.Timedelta(hours=7)).floor('h')


def _source_as_of(tmp_path, truth, T, tag):
    known = truth[truth['reveal'] <= T]
    still_open = truth[(truth['start'] <= T) & (truth['reveal'] > T)]
    csv = str(tmp_path / f'{tag}_en.csv')
    snap = str(tmp_path / f'{tag}.pkl')
    _write_csv(csv, known)
    _write_snapshot(snap, still_open.to_dict('records'))
    return csv, snap


@pytest.mark.parametrize('seed', [0, 1])
def test_source_replay_features_and_labels(tmp_path, seed):
    truth = _simulate_source(seed)
    full_csv, full_snap = _source_as_of(tmp_path, truth, UTC('2030-01-01'), 'full')
    T = _pick_T(truth)
    cut_csv, cut_snap = _source_as_of(tmp_path, truth, T, 'cut')
    start = (T - pd.Timedelta(days=4)).floor('h')

    ev_full = load_events(full_csv, snapshot_path=full_snap)
    ev_cut = load_events(cut_csv, snapshot_path=cut_snap)
    assert ev_full['end_imputed'].sum() > 0 and ev_cut['open_at_cut'].sum() > 0, 'scenario not exercised'
    p_full = generate_panel_data(ev_full, start_utc=start, end_utc=T)
    p_cut = generate_panel_data(ev_cut, start_utc=start, end_utc=T)
    assert _cmp(p_full, p_cut, FEATURE_COLS) == 0                    # causal end policy: as-of-T features identical

    lv = last_valid_label_time(ev_cut)
    assert lv <= T - pd.Timedelta(hours=MAX_HORIZON_H)
    early = lambda d: d[d['t'] <= lv]
    assert _cmp(early(p_full), early(p_cut), TARGETS) == 0           # labels up to last_valid are complete


@pytest.mark.parametrize('seed', [0, 1])
def test_negative_controls_as_recorded_and_no_snapshot(tmp_path, seed):
    """Pre-audit behaviour fails the same replay: fake ends leak, and without the snapshot tail labels are wrong."""
    truth = _simulate_source(seed)
    full_csv, full_snap = _source_as_of(tmp_path, truth, UTC('2030-01-01'), 'full')
    T = _pick_T(truth)
    cut_csv, cut_snap = _source_as_of(tmp_path, truth, T, 'cut')
    start = (truth['start'].min() + pd.Timedelta(days=1)).floor('h')

    p_cut = generate_panel_data(load_events(cut_csv, snapshot_path=cut_snap), start_utc=start, end_utc=T)
    p_full_rec = generate_panel_data(load_events(full_csv, end_policy='as_recorded'), start_utc=start, end_utc=T)
    assert _cmp(p_full_rec, p_cut, ['is_active', 'hours_since_last_finish']) > 0

    ev_nosnap = load_events(cut_csv)                                 # CSV only: starts of open alerts missing
    p_nosnap = generate_panel_data(ev_nosnap, start_utc=start, end_utc=T)
    naive_lv = T - pd.Timedelta(hours=MAX_HORIZON_H)                 # old rule: (data end - 6h)
    tail = lambda d: d[d['t'] <= naive_lv]
    p_full = generate_panel_data(load_events(full_csv, snapshot_path=full_snap), start_utc=start, end_utc=T)
    assert _cmp(tail(p_full), tail(p_nosnap), TARGETS + ['is_active']) > 0
    with pytest.warns(UserWarning):
        assert last_valid_label_time(ev_nosnap) <= ev_nosnap.attrs['data_cut'] - pd.Timedelta(hours=54)


def test_imputed_end_detection_rule(tmp_path):
    rows = pd.DataFrame([
        dict(oblast='Aoblast', raion='R1', level='raion', start=UTC('2026-03-01 00:00'), fin=UTC('2026-03-01 01:00')),
        dict(oblast='Aoblast', raion='R1', level='raion', start=UTC('2026-03-01 05:00'), fin=UTC('2026-03-01 05:20')),
        dict(oblast='Aoblast', raion='R2', level='raion', start=UTC('2026-03-01 00:00'), fin=UTC('2026-03-01 01:00')),
        dict(oblast='Aoblast', raion='R2', level='raion', start=UTC('2026-03-01 02:00'), fin=UTC('2026-03-01 02:30')),
    ])
    csv = str(tmp_path / 'x_en.csv')
    _write_csv(csv, rows)
    ev = load_events(csv).sort_values(['raion', 'started_at_dt'])
    assert ev['end_imputed'].tolist() == [True, False, False, False]    # R2: next start within 3h -> genuine 1h alert
    assert ev.iloc[0]['finished_at_dt'] == UTC('2026-03-01 05:00')       # causal end = next start
    assert ev.iloc[2]['finished_at_dt'] == UTC('2026-03-01 01:00')
    rec = load_events(csv, end_policy='as_recorded').sort_values(['raion', 'started_at_dt'])
    assert rec.iloc[0]['finished_at_dt'] == UTC('2026-03-01 01:00')


def test_malformed_and_duplicate_rows(tmp_path):
    csv = str(tmp_path / 'y_en.csv')
    lines = ['oblast,raion,hromada,level,started_at,finished_at,source',
             'Aoblast,,,oblast,2026-03-01 00:00:00+00:00,2026-03-01 00:30:00+00:00,official',
             'Aoblast,,,oblast,2026-03-01 00:00:00+00:00,2026-03-01 00:30:00+00:00,official',     # exact duplicate
             'Aoblast,,,oblast,2026-03-01 02:00:00+00:00,,official',                               # missing end
             'Aoblast,,,oblast,not-a-date,2026-03-01 03:00:00+00:00,official',                     # malformed
             'Aoblast,,,oblast,2026-03-01 05:00:00+00:00,2026-03-01 04:00:00+00:00,official',      # negative
             'Aoblast,R1,H1,hromada,2026-03-01 06:00:00+00:00,2026-03-01 07:00:00+00:00,official',  # out of scope
             'Aoblast,,,oblast,2026-03-01 08:00:00+00:00,2026-03-01 08:10:00+00:00,official']
    open(csv, 'w').write('\n'.join(lines) + '\n')
    ev = load_events(csv)
    c = ev.attrs['cleaning']
    assert (c['exact_duplicates_dropped'], c['missing_or_malformed_time_dropped'], c['negative_duration_dropped'],
            c['hromada_or_luhanska_dropped'], c['clean_rows']) == (1, 2, 1, 1, 2)
    # a dropped row with an unknown end is NOT turned into a fabricated label for the 02:00 start
    p = generate_panel_data(ev, start_utc='2026-03-01 01:00:00+00:00', end_utc=UTC('2026-03-01 01:00'))
    assert p.iloc[0]['onset_1'] == 0


def test_volunteer_naive_policies(tmp_path):
    csv = str(tmp_path / 'v_en.csv')
    df = pd.DataFrame({'region': ['Aoblast'] * 3,
                       'started_at': ['2026-03-01 00:00:00+00:00', '2026-03-01 06:00:00+00:00', '2026-03-01 09:00:00+00:00'],
                       'finished_at': ['2026-03-01 00:30:00+00:00', '2026-03-01 06:20:00+00:00', '2026-03-01 09:10:00+00:00'],
                       'naive': [True, False, False]})
    df.to_csv(csv, index=False)
    df.to_csv(csv.replace('_en.csv', '_uk.csv'), index=False)
    c = load_volunteer_events(csv, naive_policy='causal')
    assert c.iloc[0]['finished_at_dt'] == UTC('2026-03-01 06:00')
    assert load_volunteer_events(csv, naive_policy='as_recorded').iloc[0]['finished_at_dt'] == UTC('2026-03-01 00:30')
    assert len(load_volunteer_events(csv, naive_policy='exclude')) == 2
    p = generate_panel_data(c, start_utc='2026-03-01 03:00:00+00:00', end_utc=UTC('2026-03-01 03:00'))
    assert p.iloc[0]['is_active'] == 1                               # no 'end' message yet -> still active at 03:00


# --------------------------------------------------------------------------------------------- 11. freshness
def test_stale_data_is_flagged():
    ev = _one([('2026-09-07 00:00', '2026-09-07 00:10')])
    ev.attrs.update({'data_cut': UTC('2026-09-07 06:09'), 'cleaning': {'source_file': 'x.csv'}})
    ev['open_at_cut'] = False
    _, w = freshness(ev, UTC('2026-10-12 15:00'))
    assert w and 'STALE' in w[0]
    _, w = freshness(ev, UTC('2026-09-08 00:00'))
    assert not w


# ------------------------------------------------------------------------------- 12. real data (skipped if absent)
@pytest.mark.skipif(not (os.path.exists(REAL_DATA) and os.path.exists(REAL_SNAP)), reason='dataset clone not present')
def test_real_data_censoring_at_label_cutoff():
    """On the real official data: features over the last 12 h before T are unchanged when all information
    revealed after T (later starts, later ends) is hidden."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ev = load_events(REAL_DATA, snapshot_path=REAL_SNAP)
    T = last_valid_label_time(ev) - pd.Timedelta(days=10)
    start = T - pd.Timedelta(hours=12)
    win = ev[ev['finished_at_dt'] > start - pd.Timedelta(days=60)]
    assert_features_causal(win, T, start)
    assert ev['end_imputed'].sum() > 0
    imp = ev[ev['end_imputed']]
    assert (imp['finished_at_dt'] > imp['finished_at_raw']).all()   # causal ends are later than the fake ends
