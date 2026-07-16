"""Preprocessing, baselines, GBM, and the race-grouped neural network model."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def time_split(df, train_end, val_end, test_end, date_col='date'):
    train = df[df[date_col] <= train_end]
    val = df[(df[date_col] > train_end) & (df[date_col] <= val_end)]
    test = df[(df[date_col] > val_end) & (df[date_col] <= test_end)]
    return train, val, test


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', numeric_pipe, numeric_cols),
        ('cat', categorical_pipe, categorical_cols),
    ])


def favorite_baseline_probs(df):
    """Baseline 1: rule-based, pick the morning-line favorite. Probability = normalized implied prob."""
    implied = 1.0 / df['morning_line_decimal_odds'].replace(0, np.nan)
    norm = implied.groupby(df['race_key']).transform(lambda s: s / s.sum())
    return norm.fillna(0.0)


# ----------------------------- Neural network: per-race softmax (Plackett-Luce top-1) -----------------------------

class RaceGroupedTensors:
    """Pads variable field-size races into fixed-size (max_field) tensors with a validity mask."""

    def __init__(self, X, race_ids, won, max_field=20):
        self.max_field = max_field
        self.n_features = X.shape[1]
        race_index = pd.Index(pd.unique(race_ids))
        self.race_keys = race_index
        n_races = len(race_index)

        Xp = np.zeros((n_races, max_field, self.n_features), dtype=np.float32)
        mask = np.zeros((n_races, max_field), dtype=np.float32)
        winner_idx = np.zeros(n_races, dtype=np.int64)
        row_race_pos = np.zeros(len(X), dtype=np.int64)  # which (race, slot) each original row maps to, for scoring later
        row_race_slot = np.zeros(len(X), dtype=np.int64)

        race_pos = {k: i for i, k in enumerate(race_index)}
        slot_counter = {}
        won_arr = np.asarray(won)
        for i in range(len(X)):
            r = race_ids[i]
            ridx = race_pos[r]
            slot = slot_counter.get(r, 0)
            if slot >= max_field:
                continue  # extremely large field (rare); drop overflow entries
            slot_counter[r] = slot + 1
            Xp[ridx, slot, :] = X[i]
            mask[ridx, slot] = 1.0
            if won_arr[i] == 1:
                winner_idx[ridx] = slot
            row_race_pos[i] = ridx
            row_race_slot[i] = slot

        self.X = torch.tensor(Xp)
        self.mask = torch.tensor(mask)
        self.winner_idx = torch.tensor(winner_idx)
        self.row_race_pos = row_race_pos
        self.row_race_slot = row_race_slot


class WinnerMLP(nn.Module):
    def __init__(self, n_features, hidden=(64, 32), dropout=0.2):
        super().__init__()
        layers = []
        d = n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (n_races, max_field, n_features) -> (n_races, max_field)
        return self.net(x).squeeze(-1)


def masked_softmax_nll(logits, mask, winner_idx):
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(mask == 0, neg_inf)
    log_probs = torch.log_softmax(masked_logits, dim=1)
    picked = log_probs.gather(1, winner_idx.view(-1, 1)).squeeze(1)
    return -picked.mean()


def train_winner_mlp(train_rt, val_rt, n_features, hidden=(64, 32), dropout=0.2,
                      lr=1e-3, weight_decay=1e-5, batch_size=256, max_epochs=200, patience=15,
                      seed=0, verbose=True):
    torch.manual_seed(seed)
    model = WinnerMLP(n_features, hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    n_train_races = train_rt.X.shape[0]
    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train_races)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train_races, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            logits = model(train_rt.X[idx])
            loss = masked_softmax_nll(logits, train_rt.mask[idx], train_rt.winner_idx[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_logits = model(val_rt.X)
            val_loss = masked_softmax_nll(val_logits, val_rt.mask, val_rt.winner_idx).item()
        sched.step(val_loss)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch}, best val_loss={best_val:.4f}')
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_probs_from_rt(model, rt):
    model.eval()
    with torch.no_grad():
        logits = model(rt.X)
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(rt.mask == 0, neg_inf)
        probs = torch.softmax(masked_logits, dim=1).numpy()
    n = len(rt.row_race_pos)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        out[i] = probs[rt.row_race_pos[i], rt.row_race_slot[i]]
    return out
