import argparse
import hashlib
import os

import numpy as np
import pandas as pd

DEDUP_KEYS = ['oblast', 'raion', 'hromada', 'level', 'started_at']
START_UTC = '2025-11-01 00:00:00+00:00'


def load_events(path):
    """Єдине місце очищення даних (використовують і pipeline, і plot, і CLI)."""
    df = pd.read_csv(path)
    n_raw = len(df)
    df = df.drop_duplicates(subset=DEDUP_KEYS, keep='last').copy()
    df['started_at_dt'] = pd.to_datetime(df['started_at'], utc=True)
    df['finished_at_dt'] = pd.to_datetime(df['finished_at'], utc=True)

    # Події без часу початку/кінця виключаємо ЯВНО (інакше finishes > t завжди False,
    # і тривога мовчки "зникає"). Кількість друкуємо для звітності.
    n_bad = int(df['started_at_dt'].isna().sum() + df['finished_at_dt'].isna().sum())
    df = df.dropna(subset=['started_at_dt', 'finished_at_dt'])

    df = df[(df['level'] != 'hromada') & (df['oblast'] != 'Luhanska oblast')]
    df = df.sort_values('started_at_dt').reset_index(drop=True)
    df.attrs['cleaning'] = {'raw_rows': n_raw, 'rows_with_missing_time_dropped': n_bad, 'clean_rows': len(df)}
    print(f"[load_events] raw={n_raw:,} -> clean={len(df):,}; dropped (NaT)={n_bad}")
    return df


def last_valid_label_time(df_events, horizon_h=6):
    """Остання година t, для якої onset_k (до horizon_h) ще повністю покритий даними.
    Далі мітки хибно дорівнювали б 0 (правостороннє цензурування)."""
    data_end = max(df_events['started_at_dt'].max(), df_events['finished_at_dt'].max())
    return data_end.floor('h') - pd.Timedelta(hours=horizon_h)


def generate_panel_data(df_events, start_utc=START_UTC, end_utc=None):
    """
    Часова сітка (UTC) × області: локальні/національні ознаки на момент t
    (лише події зі started_at <= t, finished_at <= t) та цілі onset_k (лише is_active == 0).
    """
    if end_utc is None:
        end_utc = last_valid_label_time(df_events)
    time_grid = pd.date_range(start=start_utc, end=end_utc, freq='1h', tz='UTC')
    oblasts = sorted(df_events['oblast'].unique())

    events_by_obl = {
        obl: {
            'starts': df_events[df_events['oblast'] == obl]['started_at_dt'].values,
            'finishes': df_events[df_events['oblast'] == obl]['finished_at_dt'].values,
        }
        for obl in oblasts
    }

    all_starts = df_events['started_at_dt'].values
    all_finishes = df_events['finished_at_dt'].values
    all_oblasts = df_events['oblast'].values

    records = []
    for t in time_grid:
        t_np = t.to_datetime64()

        nat_active_mask = (all_starts <= t_np) & (all_finishes > t_np)
        nat_active_cnt = len(set(all_oblasts[nat_active_mask]))

        s1_mask = (all_starts > (t_np - np.timedelta64(1, 'h'))) & (all_starts <= t_np)
        s3_mask = (all_starts > (t_np - np.timedelta64(3, 'h'))) & (all_starts <= t_np)
        nat_s1 = len(set(all_oblasts[s1_mask]))
        nat_s3 = len(set(all_oblasts[s3_mask]))

        t_kyiv = t.tz_convert('Europe/Kyiv')

        for obl in oblasts:
            starts = events_by_obl[obl]['starts']
            finishes = events_by_obl[obl]['finishes']

            active_mask = (starts <= t_np) & (finishes > t_np)
            is_active = int(np.any(active_mask))

            hours_start = (t_np - np.min(starts[active_mask])) / np.timedelta64(1, 'h') if is_active else 0.0

            past_fin = finishes[finishes <= t_np]
            hours_finish = (t_np - np.max(past_fin)) / np.timedelta64(1, 'h') if len(past_fin) > 0 else np.nan

            s_1h = np.sum((starts > (t_np - np.timedelta64(1, 'h'))) & (starts <= t_np))
            s_3h = np.sum((starts > (t_np - np.timedelta64(3, 'h'))) & (starts <= t_np))
            s_6h = np.sum((starts > (t_np - np.timedelta64(6, 'h'))) & (starts <= t_np))
            s_24h = np.sum((starts > (t_np - np.timedelta64(24, 'h'))) & (starts <= t_np))
            s_168h = np.sum((starts > (t_np - np.timedelta64(168, 'h'))) & (starts <= t_np))

            if is_active == 0:
                o1 = int(np.any((starts > t_np) & (starts <= (t_np + np.timedelta64(1, 'h')))))
                o3 = int(np.any((starts > t_np) & (starts <= (t_np + np.timedelta64(3, 'h')))))
                o6 = int(np.any((starts > t_np) & (starts <= (t_np + np.timedelta64(6, 'h')))))
            else:
                o1, o3, o6 = np.nan, np.nan, np.nan

            records.append({
                't': t, 'oblast': obl, 'is_active': is_active,
                'hours_since_current_start': round(hours_start, 4),
                'hours_since_last_finish': round(hours_finish, 4) if not np.isnan(hours_finish) else np.nan,
                'starts_1h': s_1h, 'starts_3h': s_3h, 'starts_6h': s_6h,
                'starts_24h': s_24h, 'starts_168h': s_168h,
                'nat_active_oblasts_cnt': nat_active_cnt,
                'nat_starts_1h': nat_s1, 'nat_starts_3h': nat_s3,
                'hour_of_day_kyiv': t_kyiv.hour, 'day_of_week_kyiv': t_kyiv.dayofweek,
                'onset_1': o1, 'onset_3': o3, 'onset_6': o6,
            })
    return pd.DataFrame(records)


def get_panel(df_events, data_path, start_utc, end_utc, cache_dir='.cache'):
    """Кеш ключується хешем даних + коду features.py + меж, тож застарілий кеш неможливий."""
    h = hashlib.md5()
    with open(data_path, 'rb') as f:
        h.update(f.read())
    with open(__file__, 'rb') as f:
        h.update(f.read())
    h.update(f'{start_utc}|{end_utc}'.encode())
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'panel_{h.hexdigest()[:12]}.pkl')
    if os.path.exists(path):
        return pd.read_pickle(path)
    panel = generate_panel_data(df_events, start_utc=start_utc, end_utc=end_utc)
    panel.to_pickle(path)
    return panel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build panel features dataset")
    parser.add_argument('--data', default='ukrainian-air-raid-sirens-dataset/datasets/official_data_en.csv')
    parser.add_argument('--output', default='panel.pkl')
    args = parser.parse_args()
    events = load_events(args.data)
    generate_panel_data(events).to_pickle(args.output)
    print(f"Панель збережено в {args.output}")