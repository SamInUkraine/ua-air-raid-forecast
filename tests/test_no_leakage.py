import numpy as np
import pandas as pd
import pytest

from features import generate_panel_data

FEATURE_COLS = ['is_active', 'hours_since_current_start', 'hours_since_last_finish',
                'starts_1h', 'starts_3h', 'starts_6h', 'starts_24h', 'starts_168h',
                'nat_active_oblasts_cnt', 'nat_starts_1h', 'nat_starts_3h',
                'hour_of_day_kyiv', 'day_of_week_kyiv']


def assert_features_causal(df_events, T, start):
    """Ознаки для t<=T не змінюються, якщо видалити події після T і
    позначити всі незавершені на T тривоги як 'ще тривають' (finished_at = 2262)."""
    full = generate_panel_data(df_events, start_utc=start, end_utc=T)

    cens = df_events[df_events['started_at_dt'] <= T].copy()
    future_finish = cens['finished_at_dt'] > T
    assert future_finish.sum() > 0, f"Цензурування не випробувано: немає активних тривог у T={T}"
    cens.loc[future_finish, 'finished_at_dt'] = pd.Timestamp('2262-01-01', tz='UTC')
    cens_panel = generate_panel_data(cens, start_utc=start, end_utc=T)

    a = full.sort_values(['t', 'oblast'])[['t', 'oblast'] + FEATURE_COLS].reset_index(drop=True)
    b = cens_panel.sort_values(['t', 'oblast'])[['t', 'oblast'] + FEATURE_COLS].reset_index(drop=True)
    diff = (a.fillna(-999) != b.fillna(-999)).sum().sum()
    assert diff == 0, f"{diff} розбіжностей у ознаках для T={T}"


@pytest.mark.parametrize("test_t", ['2026-02-15 13:00:00+00:00', '2026-05-10 01:00:00+00:00',
                                    '2026-08-10 17:00:00+00:00'])
def test_periodic_events(test_t):
    dates = pd.date_range('2025-11-01', '2026-09-07', freq='4h', tz='UTC')
    ev = pd.DataFrame({'oblast': ['Kyivska oblast' if i % 2 == 0 else 'Kharkivska oblast' for i in range(len(dates))],
                       'started_at_dt': dates, 'finished_at_dt': dates + pd.Timedelta(hours=2)})
    assert_features_causal(ev, pd.Timestamp(test_t), '2025-11-01 00:00:00+00:00')


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_overlapping_events(seed):
    """Перекриваються тривоги (як raion-рівень усередині області), довільні тривалості."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp('2026-01-01', tz='UTC')
    rows = []
    for _ in range(500):
        s = t0 + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 60)))
        rows.append({'oblast': str(rng.choice(['A', 'B', 'C'])), 'started_at_dt': s,
                     'finished_at_dt': s + pd.Timedelta(minutes=int(rng.integers(5, 900)))})
    ev = pd.DataFrame(rows)
    T = (t0 + pd.Timedelta(hours=int(rng.integers(24 * 12, 24 * 50)))).floor('h')
    assert_features_causal(ev, T, '2026-01-01 00:00:00+00:00')