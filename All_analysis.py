#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced US Open 2025 Prediction System
Author: Rishabh Jain

Uses 5+ years of ATP data for Elo ratings + ML model to predict outcomes.
Organized into logical sections: Config, Cleaning, Feature Engineering,
Modeling, Simulation, and Execution.
"""

# ==============================
# Imports
# ==============================
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import timedelta

# ==============================
# Configuration
# ==============================
DATA_FILES = [
    "atp_matches_2019.csv",
    "atp_matches_2020.csv", 
    "atp_matches_2021.csv",
    "atp_matches_2022.csv",
    "atp_matches_2023.csv",
    "atp_matches_2024.csv"
]
OUTPUT_DRAW_FILE = "usopen_2025_complete_draw.csv"
NUM_SIMULATIONS = 10000
BAD_TOKENS = ("W/O","ABN","ABD","DEF","RET","w/o","walkover","ret.")

# ==============================
# Step 1 – Data Loading & Cleaning
# ==============================
def load_and_clean(files):
    all_data = []
    for file in files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
            print(f"  Loaded {file}: {len(df)} matches")
        except Exception as e:
            print(f"  Warning: Could not load {file}: {e}")
    df_all = pd.concat(all_data, ignore_index=True)

    df_all["tourney_date"] = pd.to_datetime(df_all["tourney_date"], format='%Y%m%d', errors="coerce")
    df_all["winner_age"].fillna(df_all["winner_age"].median(), inplace=True)
    df_all["loser_age"].fillna(df_all["loser_age"].median(), inplace=True)
    df_all["winner_rank_points"].fillna(0, inplace=True)
    df_all["loser_rank_points"].fillna(0, inplace=True)

    mask_bad = df_all["score"].fillna("").str.contains("|".join(BAD_TOKENS), case=False, regex=True)
    df_all = df_all.loc[~mask_bad].copy()
    return df_all

# ==============================
# Step 2 – Elo Ratings
# ==============================
def build_comprehensive_elo(matches, k=32, decay_rate=0.95):
    elo_overall = defaultdict(lambda: 1500.0)
    elo_hard = defaultdict(lambda: 1500.0)
    elo_usopen = defaultdict(lambda: 1500.0)

    matches = matches.sort_values("tourney_date")
    current_date = pd.Timestamp.now()

    for _, match in matches.iterrows():
        if pd.isna(match['winner_name']) or pd.isna(match['loser_name']):
            continue
        w, l = match['winner_name'], match['loser_name']

        if pd.notna(match['tourney_date']):
            days_ago = (current_date - match['tourney_date']).days
            time_factor = decay_rate ** (days_ago / 365)
        else:
            time_factor = 0.5

        if 'Grand Slam' in str(match.get('tourney_level', '')):
            k_adj = k * 1.5 * time_factor
        elif 'Masters' in str(match.get('tourney_name', '')):
            k_adj = k * 1.2 * time_factor
        else:
            k_adj = k * time_factor

        exp_w = 1 / (1 + 10**((elo_overall[l] - elo_overall[w]) / 400))
        elo_overall[w] += k_adj * (1 - exp_w)
        elo_overall[l] += k_adj * (0 - exp_w)

        if str(match['surface']).lower() == 'hard':
            exp_w_h = 1 / (1 + 10**((elo_hard[l] - elo_hard[w]) / 400))
            elo_hard[w] += k_adj * (1 - exp_w_h)
            elo_hard[l] += k_adj * (0 - exp_w_h)

            if 'US Open' in str(match.get('tourney_name', '')):
                k_us = k_adj * 1.5
                exp_w_us = 1 / (1 + 10**((elo_usopen[l] - elo_usopen[w]) / 400))
                elo_usopen[w] += k_us * (1 - exp_w_us)
                elo_usopen[l] += k_us * (0 - exp_w_us)

    return elo_overall, elo_hard, elo_usopen

# ==============================
# Step 3 – Player Statistics
# ==============================
def calculate_player_stats(matches, player_name):
    won = matches[matches['winner_name'] == player_name]
    lost = matches[matches['loser_name'] == player_name]

    cutoff = pd.Timestamp.now() - timedelta(days=180)
    recent_won = won[won['tourney_date'] > cutoff]
    recent_lost = lost[lost['tourney_date'] > cutoff]

    total_matches = len(won) + len(lost)
    win_rate = len(won) / total_matches if total_matches else 0
    recent_matches = len(recent_won) + len(recent_lost)
    recent_win_rate = len(recent_won) / recent_matches if recent_matches else win_rate

    hard_won = won[won['surface'].str.lower() == 'hard']
    hard_lost = lost[lost['surface'].str.lower() == 'hard']
    hard_wr = len(hard_won) / (len(hard_won)+len(hard_lost)) if (len(hard_won)+len(hard_lost)) else win_rate

    gs_won = won[won['tourney_level'] == 'G']
    gs_lost = lost[lost['tourney_level'] == 'G']
    gs_wr = len(gs_won) / (len(gs_won)+len(gs_lost)) if (len(gs_won)+len(gs_lost)) else win_rate

    avg_opp_rank = pd.concat([
        won['loser_rank_points'],
        lost['winner_rank_points']
    ]).mean()

    return {
        'total_matches': total_matches,
        'win_rate': win_rate,
        'recent_form': recent_win_rate,
        'hard_win_rate': hard_wr,
        'gs_win_rate': gs_wr,
        'avg_opp_strength': avg_opp_rank if not pd.isna(avg_opp_rank) else 0,
    }

# ==============================
# Step 4 – Feature Engineering
# ==============================
def prepare_match_features(matches, elo_overall, elo_hard, elo_usopen, stats):
    features, labels = [], []
    for _, match in matches.iterrows():
        if pd.isna(match['winner_name']) or pd.isna(match['loser_name']):
            continue
        w, l = match['winner_name'], match['loser_name']
        if w not in elo_overall or l not in elo_overall:
            continue

        feat = {
            'elo_diff': elo_overall[w] - elo_overall[l],
            'elo_hard_diff': elo_hard[w] - elo_hard[l],
            'elo_usopen_diff': elo_usopen[w] - elo_usopen[l],
            'form_diff': stats[w]['recent_form'] - stats[l]['recent_form'],
            'hard_wr_diff': stats[w]['hard_win_rate'] - stats[l]['hard_win_rate'],
            'gs_wr_diff': stats[w]['gs_win_rate'] - stats[l]['gs_win_rate'],
            'matches_diff': np.log1p(stats[w]['total_matches']) - np.log1p(stats[l]['total_matches']),
            'age_diff': match.get('winner_age', 25) - match.get('loser_age', 25),
            'best_of_5': 1 if match.get('best_of', 3) == 5 else 0,
            'is_hard': 1 if str(match.get('surface', '')).lower() == 'hard' else 0
        }

        features.append(list(feat.values())); labels.append(1)
        feat_rev = {k: -v if 'diff' in k else v for k,v in feat.items()}
        features.append(list(feat_rev.values())); labels.append(0)
    return np.array(features), np.array(labels)

# ==============================
# Step 5 – Model Training
# ==============================
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)

    print(f"Train acc: {model.score(X_train_sc, y_train):.3f}, Test acc: {model.score(X_test_sc, y_test):.3f}")
    return model, scaler

# ==============================
# Step 6 – Simulation
# ==============================
def simulate_match_ml(p1, p2, model, scaler, elo, elo_hard, elo_usopen, stats, ratings):
    if model is not None and scaler is not None:
        features = np.array([[
            elo.get(p1,1500) - elo.get(p2,1500),
            elo_hard.get(p1,1500) - elo_hard.get(p2,1500),
            elo_usopen.get(p1,1500) - elo_usopen.get(p2,1500),
            stats.get(p1,{}).get('recent_form',0.5) - stats.get(p2,{}).get('recent_form',0.5),
            stats.get(p1,{}).get('hard_win_rate',0.5) - stats.get(p2,{}).get('hard_win_rate',0.5),
            stats.get(p1,{}).get('gs_win_rate',0.5) - stats.get(p2,{}).get('gs_win_rate',0.5),
            np.log1p(stats.get(p1,{}).get('total_matches',0)) - np.log1p(stats.get(p2,{}).get('total_matches',0)),
            0,1,1
        ]])
        prob = model.predict_proba(scaler.transform(features))[0,1]
    else:
        r1, r2 = ratings.get(p1,1500), ratings.get(p2,1500)
        prob = 1 / (1 + 10**((r2-r1)/400))

    prob = prob*0.8 + np.random.random()*0.2
    return p1 if np.random.random() < prob else p2

def simulate_tournament(draw_df, model, scaler, elo, elo_hard, elo_usopen, stats, ratings):
    results = Counter()
    for _, match in draw_df.iterrows():
        winner = simulate_match_ml(match['player1'], match['player2'],
                                   model, scaler, elo, elo_hard, elo_usopen, stats, ratings)
        results[winner] += 1
    return results

# ==============================
# Execution (Main)
# ==============================
if __name__ == "__main__":
    print("="*70)
    print("ENHANCED US OPEN 2025 PREDICTION SYSTEM")
    print("Using Historical Data 2019-2024")
    print("="*70)

    # Step 1
    print("\nStep 1: Loading & cleaning data...")
    df_all = load_and_clean(DATA_FILES)
    print("Matches after cleaning:", len(df_all))

    # Step 2
    print("\nStep 2: Building Elo ratings...")
    elo_overall, elo_hard, elo_usopen = build_comprehensive_elo(df_all)

    # Step 3
    print("\nStep 3: Calculating player stats...")
    player_stats = {p: calculate_player_stats(df_all, p) for p in elo_overall.keys()}

    # Step 4 & 5
    print("\nStep 4: Preparing features & training model...")
    train_df = df_all[df_all['tourney_date'] > '2021-01-01']
    X, y = prepare_match_features(train_df, elo_overall, elo_hard, elo_usopen, player_stats)
    model, scaler = train_model(X, y)

    # Simulation (plug in your draw_df here)
    # results = simulate_tournament(draw_df, model, scaler, elo_overall, elo_hard, elo_usopen, player_stats, elo_overall)
    # print(results.most_common(10))
