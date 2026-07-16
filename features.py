"""
Leakage-safe feature engineering for the horse-race winner-prediction task.

Core rule: every "history" feature for a given horse/jockey/trainer entry on race day D is computed
from an expanding window over STRICTLY EARLIER rows only (sorted by date, then race_number as an
intra-day tiebreaker), via `groupby(...).shift(1)` + `expanding()`. The row's own outcome never
leaks into its own features.
"""
import numpy as np
import pandas as pd


def _expanding_rate(df, group_cols, indicator_col, min_periods=1):
    """Expanding mean of indicator_col within group_cols, using only PRIOR rows (shift(1))."""
    g = df.groupby(group_cols, sort=False)[indicator_col]
    shifted = g.shift(1)
    return shifted.groupby([df[c] for c in group_cols]).expanding(min_periods=min_periods).mean().reset_index(level=list(range(len(group_cols))), drop=True)


def _expanding_count(df, group_cols):
    g = df.groupby(group_cols, sort=False).cumcount()
    return g  # number of PRIOR rows in group (0-indexed count is already "prior" count)


def _expanding_mean_lastn(series_by_group, n):
    return series_by_group.shift(1).rolling(n, min_periods=1).mean()


def build_feature_table(chart_df, pp_df):
    df = chart_df.copy()

    # --- attach morning-line odds (pre-race market feature; PP files are the only pre-race odds source) ---
    pp_key = pp_df[['date', 'track_code', 'race_number', 'horse_name',
                    'morning_line_decimal_odds', 'morning_line_implied_prob']].copy()
    df = df.merge(pp_key, on=['date', 'track_code', 'race_number', 'horse_name'], how='left')

    # --- target ---
    df['won'] = (df['post_official_fin'] == 1).astype(int)
    df['in_the_money'] = (df['post_official_fin'] <= 3).astype(int)

    # --- global chronological order (date, then race_number as an intra-day tiebreaker) ---
    df = df.sort_values(['date', 'track_code', 'race_number']).reset_index(drop=True)
    df['race_key'] = df['date'].astype(str) + '_' + df['track_code'] + '_' + df['race_number'].astype(str)
    df['_order'] = np.arange(len(df))

    # distance bucket for suitability features / subgroup analysis
    df['sprint_route'] = np.where(df['distance_furlongs'] <= 7, 'Sprint', 'Route')

    # ============ HORSE history (sorted per horse by chronological order) ============
    df = df.sort_values(['horse_name', '_order'])
    horse_g = df.groupby('horse_name', sort=False)

    df['horse_prior_starts'] = horse_g.cumcount()
    df['horse_prior_wins'] = horse_g['won'].apply(lambda s: s.shift(1).cumsum()).values
    df['horse_win_pct'] = (df['horse_prior_wins'] / df['horse_prior_starts'].replace(0, np.nan))
    df['horse_prior_itm'] = horse_g['in_the_money'].apply(lambda s: s.shift(1).cumsum()).values
    df['horse_itm_pct'] = (df['horse_prior_itm'] / df['horse_prior_starts'].replace(0, np.nan))

    df['horse_avg_speed_last3'] = horse_g['post_speed_rating'].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).values
    df['horse_avg_speed_last5'] = horse_g['post_speed_rating'].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).values
    df['horse_best_speed_lifetime'] = horse_g['post_speed_rating'].apply(lambda s: s.shift(1).cummax()).values
    df['horse_avg_finish_last5'] = horse_g['post_official_fin'].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).values

    df['horse_prev_race_date'] = horse_g['date'].shift(1)
    df['horse_days_since_last_race'] = (df['date'] - df['horse_prev_race_date']).dt.days
    df['horse_is_debut'] = (df['horse_prior_starts'] == 0).astype(int)

    # horse x surface / distance-bucket / track suitability (win rate in that context, prior only)
    for ctx_col, out_prefix in [('surface', 'horse_surface'), ('sprint_route', 'horse_distbucket'), ('track_code', 'horse_track')]:
        gcols = ['horse_name', ctx_col]
        tmp = df.groupby(gcols, sort=False)['won'].apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
        df[f'{out_prefix}_win_pct'] = tmp.values
        tmp_n = df.groupby(gcols, sort=False).cumcount()
        df[f'{out_prefix}_starts'] = tmp_n.values

    # ============ JOCKEY history ============
    df = df.sort_values(['jockey_key', '_order'])
    jock_g = df.groupby('jockey_key', sort=False)
    df['jockey_prior_mounts'] = jock_g.cumcount()
    df['jockey_prior_wins'] = jock_g['won'].apply(lambda s: s.shift(1).cumsum()).values
    df['jockey_win_pct'] = df['jockey_prior_wins'] / df['jockey_prior_mounts'].replace(0, np.nan)
    df['jockey_prior_itm'] = jock_g['in_the_money'].apply(lambda s: s.shift(1).cumsum()).values
    df['jockey_itm_pct'] = df['jockey_prior_itm'] / df['jockey_prior_mounts'].replace(0, np.nan)
    df['jockey_win_pct_last50'] = jock_g['won'].apply(lambda s: s.shift(1).rolling(50, min_periods=5).mean()).values

    # ============ TRAINER history ============
    df = df.sort_values(['trainer_key', '_order'])
    tr_g = df.groupby('trainer_key', sort=False)
    df['trainer_prior_starts'] = tr_g.cumcount()
    df['trainer_prior_wins'] = tr_g['won'].apply(lambda s: s.shift(1).cumsum()).values
    df['trainer_win_pct'] = df['trainer_prior_wins'] / df['trainer_prior_starts'].replace(0, np.nan)
    df['trainer_win_pct_last50'] = tr_g['won'].apply(lambda s: s.shift(1).rolling(50, min_periods=5).mean()).values

    # ============ trainer-horse combo (has this trainer won with this horse before) ============
    df = df.sort_values(['horse_name', 'trainer_key', '_order'])
    th_g = df.groupby(['horse_name', 'trainer_key'], sort=False)
    df['trainer_horse_prior_starts'] = th_g.cumcount()
    df['trainer_horse_prior_wins'] = th_g['won'].apply(lambda s: s.shift(1).cumsum()).values

    df = df.sort_values('_order').reset_index(drop=True)
    return df


PRE_RACE_FEATURE_COLUMNS = [
    # race conditions (known before the race)
    'purse', 'distance_furlongs', 'class_rating', 'field_size',
    'track_condition', 'surface', 'race_class', 'sprint_route',
    # entry conditions (known before the race)
    'post_pos', 'weight', 'age', 'sex', 'meds', 'equip', 'claim_price',
    # market feature
    'morning_line_decimal_odds', 'morning_line_implied_prob',
    # horse history
    'horse_prior_starts', 'horse_win_pct', 'horse_itm_pct',
    'horse_avg_speed_last3', 'horse_avg_speed_last5', 'horse_best_speed_lifetime',
    'horse_avg_finish_last5', 'horse_days_since_last_race', 'horse_is_debut',
    'horse_surface_win_pct', 'horse_surface_starts',
    'horse_distbucket_win_pct', 'horse_distbucket_starts',
    'horse_track_win_pct', 'horse_track_starts',
    # jockey history
    'jockey_prior_mounts', 'jockey_win_pct', 'jockey_itm_pct', 'jockey_win_pct_last50',
    # trainer history
    'trainer_prior_starts', 'trainer_win_pct', 'trainer_win_pct_last50',
    # trainer-horse combo
    'trainer_horse_prior_starts', 'trainer_horse_prior_wins',
]

CATEGORICAL_COLUMNS = ['track_condition', 'surface', 'race_class', 'sprint_route', 'sex', 'meds', 'equip']

LEAKAGE_COLUMNS = [
    'post_official_fin', 'post_speed_rating', 'post_finish_time',
    'post_final_call_pos', 'post_final_call_lengths', 'post_dollar_odds', 'post_win_payoff',
]
