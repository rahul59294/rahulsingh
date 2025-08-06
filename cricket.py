# ipl_fantasy_model.py
"""
Template pipeline to predict IPL fantasy cricket teams (Best, Optimal, Least‑Risk) from historical data.
Fill in paths, scoring weights, role definitions, and constraints according to your fantasy platform rules.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import pulp as pl

# =========================================
# 1) DATA LOADING HELPERS – EDIT PATHS
# =========================================

def load_batting(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

# Repeat analogous loaders for bowling, partnership, fielding, dismissals, match data, etc.

def load_scoring_table(path: Path) -> Dict[str, float]:
    scoring_df = pd.read_csv(path)
    return dict(zip(scoring_df['event'], scoring_df['points']))

# =========================================
# 2) FANTASY SCORING & FEATURE ENGINEERING
# =========================================

def apply_fantasy_scoring(df: pd.DataFrame, scoring_weights: Dict[str, float]) -> pd.DataFrame:
    """Add `fantasy_points` column given per‑event weights."""
    points = (
        df['Runs'] * scoring_weights.get('run', 0)
        + df['4s'] * scoring_weights.get('four', 0)
        + df['6s'] * scoring_weights.get('six', 0)
        + df['wkts'] * scoring_weights.get('wicket', 0)
        + df['ct'] * scoring_weights.get('catch', 0)
        # … add more events as required
    )
    df = df.copy()
    df['fantasy_points'] = points
    return df


def build_player_match_features(row: pd.Series) -> pd.Series:
    """Transform a single match row into model features (venue, opposition, form, role, etc.)."""
    return pd.Series({
        'venue_id': row['ground_id'],
        'opposition_id': row['opposition_id'],
        'bat_avg': row['career_bat_avg'],
        'bowl_avg': row['career_bowl_avg'],
        'recent_form': row['recent_points_mean_5'],
        'role_BAT': 1 if row['role'] == 'BAT' else 0,
        'role_BWL': 1 if row['role'] == 'BWL' else 0,
        'role_AR' : 1 if row['role'] == 'AR'  else 0,
        'role_WK' : 1 if row['role'] == 'WK'  else 0,
    })


# =========================================
# 3) MODEL TRAINING
# =========================================

def train_xgb_regressor(features: pd.DataFrame, target: pd.Series) -> XGBRegressor:
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)
    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    model = GridSearchCV(
        XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"Validation MAE: {mean_absolute_error(y_val, y_pred):.2f}")
    return model.best_estimator_


# =========================================
# 4) TEAM OPTIMISATION (ILP)
# =========================================

ROLE_MAX = {'BAT': 5, 'BWL': 5, 'AR': 3, 'WK': 1}  # Example – edit as needed
ROLE_MIN = {'BAT': 3, 'BWL': 3, 'AR': 1, 'WK': 1}
TEAM_LIMIT = 4                # max players from a single IPL franchise
CREDIT_CAP = 100              # fantasy platform credit cap


def build_ilp_team(players_df: pd.DataFrame, objective: str = 'best', risk_weight: float = 0.0) -> Tuple[pd.DataFrame, float]:
    """Return chosen XI and expected points.

    objective ∈ {'best', 'optimal', 'least_risk'}
    - 'best': maximise total expected points.
    - 'least_risk': minimise variance subject to expected pts ≥ percentile threshold.
    - 'optimal': maximise (pts − λ·stdev).
    """
    players = players_df.copy()
    n = len(players)

    prob = pl.LpProblem("FantasyXI", pl.LpMaximize if objective != 'least_risk' else pl.LpMinimize)
    x = pl.LpVariable.dicts('pick', players.index, cat='Binary')

    # Objective
    if objective == 'best':
        prob += pl.lpSum(x[i] * players.loc[i, 'exp_points'] for i in players.index)
    elif objective == 'least_risk':
        threshold = players['exp_points'].quantile(0.75)
        prob += pl.lpSum(x[i] * players.loc[i, 'var_points'] for i in players.index)  # minimise variance
        prob += pl.lpSum(x[i] * players.loc[i, 'exp_points'] for i in players.index) >= threshold
    else:  # optimal risk‑adjusted
        prob += pl.lpSum(x[i] * (players.loc[i, 'exp_points'] - risk_weight * np.sqrt(players.loc[i, 'var_points'])) for i in players.index)

    # Exactly 11 players
    prob += pl.lpSum(x.values()) == 11

    # Role constraints
    for role in ROLE_MIN.keys():
        prob += pl.lpSum(x[i] for i in players.index if players.loc[i, 'role'] == role) >= ROLE_MIN[role]
        prob += pl.lpSum(x[i] for i in players.index if players.loc[i, 'role'] == role) <= ROLE_MAX[role]

    # Team constraint
    for tm in players['ipl_team'].unique():
        prob += pl.lpSum(x[i] for i in players.index if players.loc[i, 'ipl_team'] == tm) <= TEAM_LIMIT

    # Credit cap
    prob += pl.lpSum(x[i] * players.loc[i, 'credits'] for i in players.index) <= CREDIT_CAP

    # Solve
    prob.solve(pl.PULP_CBC_CMD(msg=0))

    chosen = players.loc[[i for i in players.index if pl.value(x[i]) == 1]].copy()
    total_exp = chosen['exp_points'].sum()
    return chosen, total_exp


# =========================================
# 5) CLI ENTRY POINT
# =========================================

def main():
    parser = argparse.ArgumentParser(description="Predict IPL fantasy XI teams for a given fixture.")
    parser.add_argument('--team1', required=True)
    parser.add_argument('--team2', required=True)
    parser.add_argument('--venue', required=True)
    parser.add_argument('--scoring', required=True, help="Path to CSV with scoring system")
    args = parser.parse_args()

    # Load fantasy scoring system
    scoring_weights = load_scoring_table(Path(args.scoring))

    # TODO: Load master datasets
    # batting = load_batting(Path('data/batting.csv'))
    # ...

    # TODO: Build current squad list for the two franchises playing (manual or via API)

    # TODO: Aggregate career + recent stats, venue effects, opponent match‑ups, etc.
    # features_df = ...
    # model = train_xgb_regressor(features_df.drop('fantasy_points', axis=1), features_df['fantasy_points'])

    # TODO: Predict upcoming fixture expected points + variance per player
    # pred_df = ...

    # Choose teams
    # best_team, best_pts       = build_ilp_team(pred_df, 'best')
    # optimal_team, optimal_pts = build_ilp_team(pred_df, 'optimal', risk_weight=0.15)
    # safe_team, safe_pts       = build_ilp_team(pred_df, 'least_risk')

    # print or save to CSV


if __name__ == "__main__":
    main()
