"""Race-grouped evaluation metrics for the winner-prediction task."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss


def evaluate_predictions(eval_df, prob_col='pred_prob', race_col='race_key',
                          fin_col='post_official_fin', payoff_col='post_win_payoff', bet_size=2.0):
    """
    eval_df: one row per horse-race entry, already restricted to the evaluation split.
    Returns a dict of scalar metrics plus a per-race detail frame (for subgroup slicing / plots).
    """
    df = eval_df.copy()
    df['won'] = (df[fin_col] == 1).astype(int)

    metrics = {}
    metrics['log_loss'] = log_loss(df['won'], df[prob_col].clip(1e-6, 1 - 1e-6))
    metrics['brier_score'] = brier_score_loss(df['won'], df[prob_col])
    try:
        metrics['roc_auc'] = roc_auc_score(df['won'], df[prob_col])
    except ValueError:
        metrics['roc_auc'] = np.nan

    # rank within each race by predicted probability (1 = model's top pick)
    df['pred_rank'] = df.groupby(race_col)[prob_col].rank(ascending=False, method='first')

    per_race_rows = []
    for race_key, g in df.groupby(race_col):
        g = g.sort_values('pred_rank')
        top_pick = g.iloc[0]
        actual_winner_rank = g.loc[g['won'] == 1, 'pred_rank']
        winner_rank = actual_winner_rank.iloc[0] if len(actual_winner_rank) else np.nan

        top1_correct = int(top_pick['won'] == 1)
        top3_hit = int((g.loc[g['pred_rank'] <= 3, 'won'] == 1).any())

        # NDCG@3: graded relevance from true finish position (3/2/1/0 for 1st/2nd/3rd/other)
        fin = g[fin_col].fillna(99)
        relevance = fin.map(lambda f: {1: 3, 2: 2, 3: 1}.get(f, 0))
        g = g.assign(relevance=relevance.values)
        top3_pred = g.sort_values('pred_rank').head(3)
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(top3_pred['relevance']))
        ideal = g.sort_values('relevance', ascending=False).head(3)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal['relevance']))
        ndcg3 = dcg / idcg if idcg > 0 else np.nan

        payoff = top_pick.get(payoff_col, np.nan)
        bet_return = (payoff if pd.notna(payoff) and payoff > 0 else 0.0) - bet_size

        per_race_rows.append({
            'race_key': race_key, 'field_size': len(g), 'top1_correct': top1_correct,
            'top3_hit': top3_hit, 'ndcg3': ndcg3, 'winner_pred_rank': winner_rank,
            'bet_return': bet_return, 'track_code': g['track_code'].iloc[0] if 'track_code' in g else None,
            'surface': g['surface'].iloc[0] if 'surface' in g else None,
            'sprint_route': g['sprint_route'].iloc[0] if 'sprint_route' in g else None,
            'race_class': g['race_class'].iloc[0] if 'race_class' in g else None,
        })
    per_race = pd.DataFrame(per_race_rows)

    metrics['n_races'] = len(per_race)
    metrics['top_rank_win_rate'] = per_race['top1_correct'].mean()
    metrics['top3_hit_rate'] = per_race['top3_hit'].mean()
    metrics['ndcg_at_3'] = per_race['ndcg3'].mean()
    metrics['roi'] = per_race['bet_return'].sum() / (bet_size * len(per_race))
    return metrics, per_race


def metrics_table(named_metrics):
    """named_metrics: dict[model_name] -> metrics dict from evaluate_predictions."""
    rows = []
    for name, m in named_metrics.items():
        rows.append({
            'model': name,
            'log_loss': m['log_loss'],
            'brier_score': m['brier_score'],
            'roc_auc': m['roc_auc'],
            'top_rank_win_rate': m['top_rank_win_rate'],
            'top3_hit_rate': m['top3_hit_rate'],
            'ndcg_at_3': m['ndcg_at_3'],
            'roi': m['roi'],
            'n_races': m['n_races'],
        })
    return pd.DataFrame(rows).set_index('model')
