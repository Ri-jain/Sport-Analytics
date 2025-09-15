#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:31:19 2025

@author: rishabhjain
"""

#!/usr/bin/env python3
"""
Enhanced US Open 2025 Prediction System
Uses 5+ years of ATP data for more accurate predictions
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Configuration
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

print("="*70)
print("ENHANCED US OPEN 2025 PREDICTION SYSTEM")
print("Using Historical Data 2019-2024")
print("="*70)

# Step 1: Load all historical data
print("\nStep 1: Loading historical ATP data...")
all_data = []
for file in DATA_FILES:
    try:
        df = pd.read_csv(file)
        all_data.append(df)
        print(f"  Loaded {file}: {len(df)} matches")
    except Exception as e:
        print(f"  Warning: Could not load {file}: {e}")

# Combine all data
df_all = pd.concat(all_data, ignore_index=True)
print(f"\nTotal matches loaded: {len(df_all)}")

# Clean combined data
df_all["tourney_date"] = pd.to_datetime(df_all["tourney_date"], format='%Y%m%d', errors="coerce")
df_all["winner_age"] = df_all["winner_age"].fillna(df_all["winner_age"].median())
df_all["loser_age"] = df_all["loser_age"].fillna(df_all["loser_age"].median())
df_all["winner_rank_points"] = df_all["winner_rank_points"].fillna(0)
df_all["loser_rank_points"] = df_all["loser_rank_points"].fillna(0)

# Drop non-played matches
bad_tokens = ("W/O", "ABN", "ABD", "DEF", "RET", "w/o", "walkover", "ret.")
mask_bad = df_all["score"].fillna("").str.contains("|".join(bad_tokens), case=False, regex=True)
df_all = df_all.loc[~mask_bad].copy()

print(f"Matches after cleaning: {len(df_all)}")

# Step 2: Build comprehensive Elo ratings with time decay
print("\nStep 2: Building time-weighted Elo ratings...")

def build_comprehensive_elo(matches, k=32, decay_rate=0.95):
    """Build Elo ratings with time decay for more recent matches"""
    # Initialize ratings
    elo_overall = defaultdict(lambda: 1500.0)
    elo_hard = defaultdict(lambda: 1500.0)
    elo_usopen = defaultdict(lambda: 1500.0)
    
    # Sort by date
    matches = matches.sort_values("tourney_date")
    current_date = pd.Timestamp.now()
    
    for _, match in matches.iterrows():
        if pd.isna(match['winner_name']) or pd.isna(match['loser_name']):
            continue
            
        w, l = match['winner_name'], match['loser_name']
        
        # Calculate time decay factor (more recent matches matter more)
        if pd.notna(match['tourney_date']):
            days_ago = (current_date - match['tourney_date']).days
            time_factor = decay_rate ** (days_ago / 365)  # Decay per year
        else:
            time_factor = 0.5
        
        # Adjusted K-factor based on tournament importance and recency
        if 'Grand Slam' in str(match.get('tourney_level', '')):
            k_adj = k * 1.5 * time_factor
        elif 'Masters' in str(match.get('tourney_name', '')):
            k_adj = k * 1.2 * time_factor
        else:
            k_adj = k * time_factor
        
        # Overall Elo
        exp_w = 1 / (1 + 10**((elo_overall[l] - elo_overall[w])/400))
        elo_overall[w] += k_adj * (1 - exp_w)
        elo_overall[l] += k_adj * (0 - exp_w)
        
        # Hard court Elo
        if str(match['surface']).lower() == 'hard':
            exp_w_hard = 1 / (1 + 10**((elo_hard[l] - elo_hard[w])/400))
            elo_hard[w] += k_adj * (1 - exp_w_hard)
            elo_hard[l] += k_adj * (0 - exp_w_hard)
            
            # US Open specific Elo
            if 'US Open' in str(match.get('tourney_name', '')):
                k_usopen = k_adj * 1.5  # Extra weight for US Open matches
                exp_w_uso = 1 / (1 + 10**((elo_usopen[l] - elo_usopen[w])/400))
                elo_usopen[w] += k_usopen * (1 - exp_w_uso)
                elo_usopen[l] += k_usopen * (0 - exp_w_uso)
    
    return elo_overall, elo_hard, elo_usopen

elo_overall, elo_hard, elo_usopen = build_comprehensive_elo(df_all)
print(f"Built ratings for {len(elo_overall)} players")

# Step 3: Calculate additional player statistics
print("\nStep 3: Calculating player statistics...")

def calculate_player_stats(matches, player_name):
    """Calculate comprehensive stats for a player"""
    # Filter matches for this player
    won = matches[matches['winner_name'] == player_name]
    lost = matches[matches['loser_name'] == player_name]
    
    # Recent form (last 6 months)
    cutoff_date = pd.Timestamp.now() - timedelta(days=180)
    recent_won = won[won['tourney_date'] > cutoff_date]
    recent_lost = lost[lost['tourney_date'] > cutoff_date]
    
    # Calculate stats
    total_matches = len(won) + len(lost)
    win_rate = len(won) / total_matches if total_matches > 0 else 0
    recent_matches = len(recent_won) + len(recent_lost)
    recent_win_rate = len(recent_won) / recent_matches if recent_matches > 0 else win_rate
    
    # Surface-specific stats
    hard_won = won[won['surface'].str.lower() == 'hard']
    hard_lost = lost[lost['surface'].str.lower() == 'hard']
    hard_matches = len(hard_won) + len(hard_lost)
    hard_win_rate = len(hard_won) / hard_matches if hard_matches > 0 else win_rate
    
    # Grand Slam performance
    gs_won = won[won['tourney_level'] == 'G']
    gs_lost = lost[lost['tourney_level'] == 'G']
    gs_matches = len(gs_won) + len(gs_lost)
    gs_win_rate = len(gs_won) / gs_matches if gs_matches > 0 else win_rate
    
    # Average opponent strength faced
    avg_opp_rank = pd.concat([
        won['loser_rank_points'],
        lost['winner_rank_points']
    ]).mean()
    
    # Best wins (highest ranked opponents beaten)
    if len(won) > 0:
        best_wins = won.nlargest(5, 'loser_rank_points')['loser_rank_points'].mean()
    else:
        best_wins = 0
    
    return {
        'total_matches': total_matches,
        'win_rate': win_rate,
        'recent_form': recent_win_rate,
        'hard_win_rate': hard_win_rate,
        'gs_win_rate': gs_win_rate,
        'avg_opp_strength': avg_opp_rank if not pd.isna(avg_opp_rank) else 0,
        'best_wins_quality': best_wins,
        'matches_2024': len(won[won['tourney_date'].dt.year == 2024]) + len(lost[lost['tourney_date'].dt.year == 2024])
    }

# Calculate stats for top players
top_players = ['Jannik Sinner', 'Carlos Alcaraz', 'Novak Djokovic', 'Alexander Zverev', 
               'Taylor Fritz', 'Daniil Medvedev', 'Jack Draper', 'Ben Shelton']

player_stats = {}
for player in set(elo_overall.keys()):
    player_stats[player] = calculate_player_stats(df_all, player)

# Step 4: Train ML model on historical data
print("\nStep 4: Training machine learning model...")

# Prepare training data
def prepare_match_features(matches, elo_overall, elo_hard, elo_usopen, player_stats):
    """Prepare features for ML model"""
    features = []
    labels = []
    
    for _, match in matches.iterrows():
        if pd.isna(match['winner_name']) or pd.isna(match['loser_name']):
            continue
            
        w, l = match['winner_name'], match['loser_name']
        
        # Skip if we don't have data for both players
        if w not in elo_overall or l not in elo_overall:
            continue
        
        # Features
        feat = {
            # Elo differences
            'elo_diff': elo_overall[w] - elo_overall[l],
            'elo_hard_diff': elo_hard[w] - elo_hard[l],
            'elo_usopen_diff': elo_usopen[w] - elo_usopen[l],
            
            # Form differences
            'form_diff': player_stats[w]['recent_form'] - player_stats[l]['recent_form'],
            'hard_wr_diff': player_stats[w]['hard_win_rate'] - player_stats[l]['hard_win_rate'],
            'gs_wr_diff': player_stats[w]['gs_win_rate'] - player_stats[l]['gs_win_rate'],
            
            # Experience difference
            'matches_diff': np.log1p(player_stats[w]['total_matches']) - np.log1p(player_stats[l]['total_matches']),
            
            # Age difference (if available)
            'age_diff': match.get('winner_age', 25) - match.get('loser_age', 25),
            
            # Best of 5 indicator
            'best_of_5': 1 if match.get('best_of', 3) == 5 else 0,
            
            # Surface is hard
            'is_hard': 1 if str(match.get('surface', '')).lower() == 'hard' else 0
        }
        
        # Add both perspectives (winner and loser)
        features.append(list(feat.values()))
        labels.append(1)  # Winner won
        
        # Reverse perspective
        feat_rev = {k: -v if 'diff' in k else v for k, v in feat.items()}
        features.append(list(feat_rev.values()))
        labels.append(0)  # Loser lost
    
    return np.array(features), np.array(labels)

# Filter to recent matches for training
train_matches = df_all[df_all['tourney_date'] > '2021-01-01']
X, y = prepare_match_features(train_matches, elo_overall, elo_hard, elo_usopen, player_stats)

# Train model
if len(X) > 0:
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"  Model accuracy - Train: {train_score:.3f}, Test: {test_score:.3f}")
else:
    print("  Warning: Not enough data for ML model, using Elo only")
    model = None
    scaler = None

# Step 5: Create enhanced rating system
print("\nStep 5: Creating final player ratings...")

# Combine all factors into final ratings
player_ratings = {}
for player in elo_overall:
    # Base rating from Elo
    base_rating = (elo_overall[player] + elo_hard[player] * 1.2 + elo_usopen[player] * 0.8) / 3
    
    # Adjust for recent form and stats
    stats = player_stats.get(player, {})
    form_bonus = (stats.get('recent_form', 0.5) - 0.5) * 200
    hard_bonus = (stats.get('hard_win_rate', 0.5) - 0.5) * 150
    experience_bonus = np.log1p(stats.get('total_matches', 0)) * 10
    
    # Combine
    player_ratings[player] = base_rating + form_bonus + hard_bonus + experience_bonus

# Manual adjustments for known current form
current_form_adjustments = {
    "Jannik Sinner": 50,      # Dominant 2024
    "Carlos Alcaraz": 30,      # Strong but inconsistent
    "Novak Djokovic": 20,      # Still elite at majors
    "Daniil Medvedev": -10,    # Slight dip
    "Taylor Fritz": 10,        # Improving
    "Jack Draper": 15,         # Rising star
}

for player, adj in current_form_adjustments.items():
    if player in player_ratings:
        player_ratings[player] += adj

# Step 6: Load draw and run simulation
print("\nStep 6: Loading US Open 2025 draw...")

# [Previous draw creation code remains the same]
complete_matches = [
    # Quarter 1 (Sinner's quarter)
    {"match": 1, "p1": "Jannik Sinner", "p2": "Vit Kopriva", "s1": 1, "s2": None},
    {"match": 2, "p1": "Alexei Popyrin", "p2": "Emil Ruusuvuori", "s1": None, "s2": None},
    {"match": 3, "p1": "Valentin Royer", "p2": "Yunchaokete Bu", "s1": None, "s2": None},
    {"match": 4, "p1": "Marton Fucsovics", "p2": "Denis Shapovalov", "s1": None, "s2": 27},
    {"match": 5, "p1": "Alexander Bublik", "p2": "Marin Cilic", "s1": 23, "s2": None},
    {"match": 6, "p1": "Lorenzo Sonego", "p2": "Tristan Schoolkate", "s1": None, "s2": None},
    {"match": 7, "p1": "Nuno Borges", "p2": "Brandon Holt", "s1": None, "s2": None},
    {"match": 8, "p1": "Elmer Moller", "p2": "Tommy Paul", "s1": None, "s2": 14},
    {"match": 9, "p1": "Lorenzo Musetti", "p2": "Giovanni Mpetshi Perricard", "s1": 10, "s2": None},
    {"match": 10, "p1": "Quentin Halys", "p2": "David Goffin", "s1": None, "s2": None},
    {"match": 11, "p1": "Jenson Brooksby", "p2": "Aleksandar Vukic", "s1": None, "s2": None},
    {"match": 12, "p1": "Francesco Passaro", "p2": "Flavio Cobolli", "s1": None, "s2": 24},
    {"match": 13, "p1": "Gabriel Diallo", "p2": "Damir Dzumhur", "s1": 31, "s2": None},
    {"match": 14, "p1": "Jaume Munar", "p2": "Jaime Faria", "s1": None, "s2": None},
    {"match": 15, "p1": "Zizou Bergs", "p2": "Chun-Hsin Tseng", "s1": None, "s2": None},
    {"match": 16, "p1": "Federico Agustin Gomez", "p2": "Jack Draper", "s1": None, "s2": 5},
    
    # Quarter 2 (Zverev's quarter)
    {"match": 17, "p1": "Alexander Zverev", "p2": "Alejandro Tabilo", "s1": 3, "s2": None},
    {"match": 18, "p1": "Roberto Bautista Agut", "p2": "Jacob Fearnley", "s1": None, "s2": None},
    {"match": 19, "p1": "Gael Monfils", "p2": "Roman Safiullin", "s1": None, "s2": None},
    {"match": 20, "p1": "Billy Harris", "p2": "Felix Auger-Aliassime", "s1": None, "s2": 25},
    {"match": 21, "p1": "Ugo Humbert", "p2": "Adam Walton", "s1": 22, "s2": None},
    {"match": 22, "p1": "Aleksandar Kovacevic", "p2": "Coleman Wong", "s1": None, "s2": None},
    {"match": 23, "p1": "James Duckworth", "p2": "Tristan Boyer", "s1": None, "s2": None},
    {"match": 24, "p1": "Dino Prizmic", "p2": "Andrey Rublev", "s1": None, "s2": 15},
    {"match": 25, "p1": "Karen Khachanov", "p2": "Nishesh Basavareddy", "s1": 9, "s2": None},
    {"match": 26, "p1": "Hugo Dellien", "p2": "Kamil Majchrzak", "s1": None, "s2": None},
    {"match": 27, "p1": "Leandro Riedi", "p2": "Pedro Martinez", "s1": None, "s2": None},
    {"match": 28, "p1": "Matteo Arnaldi", "p2": "Francisco Cerundolo", "s1": None, "s2": 19},
    {"match": 29, "p1": "Stefanos Tsitsipas", "p2": "Alexandre Muller", "s1": 26, "s2": None},
    {"match": 30, "p1": "Daniel Altmaier", "p2": "Hamad Medjedovic", "s1": None, "s2": None},
    {"match": 31, "p1": "Hugo Gaston", "p2": "Shintaro Mochizuki", "s1": None, "s2": None},
    {"match": 32, "p1": "Christopher O'Connell", "p2": "Alex De Minaur", "s1": None, "s2": 8},
    
    # Quarter 3 (Djokovic's quarter)
    {"match": 33, "p1": "Novak Djokovic", "p2": "Learner Tien", "s1": 7, "s2": None},
    {"match": 34, "p1": "Zachary Svajda", "p2": "Zsombor Piros", "s1": None, "s2": None},
    {"match": 35, "p1": "Cameron Norrie", "p2": "Sebastian Korda", "s1": None, "s2": None},
    {"match": 36, "p1": "Francisco Comesana", "p2": "Alex Michelsen", "s1": None, "s2": 28},
    {"match": 37, "p1": "Frances Tiafoe", "p2": "Yoshihito Nishioka", "s1": 17, "s2": None},
    {"match": 38, "p1": "Martin Damm", "p2": "Darwin Blanch", "s1": None, "s2": None},
    {"match": 39, "p1": "Jan-Lennard Struff", "p2": "Mackenzie Mcdonald", "s1": None, "s2": None},
    {"match": 40, "p1": "Botic Van De Zandschulp", "p2": "Holger Rune", "s1": None, "s2": 11},
    {"match": 41, "p1": "Jakub Mensik", "p2": "Nicolas Jarry", "s1": 16, "s2": None},
    {"match": 42, "p1": "Ugo Blanchet", "p2": "Fabian Marozsan", "s1": None, "s2": None},
    {"match": 43, "p1": "Joao Fonseca", "p2": "Miomir Kecmanovic", "s1": None, "s2": None},
    {"match": 44, "p1": "Luca Nardi", "p2": "Tomas Machac", "s1": None, "s2": 21},
    {"match": 45, "p1": "Brandon Nakashima", "p2": "Jesper De Jong", "s1": 30, "s2": None},
    {"match": 46, "p1": "Jerome Kym", "p2": "Ethan Quinn", "s1": None, "s2": None},
    {"match": 47, "p1": "Sebastian Baez", "p2": "Lloyd Harris", "s1": None, "s2": None},
    {"match": 48, "p1": "Emilio Nava", "p2": "Taylor Fritz", "s1": None, "s2": 4},
    
    # Quarter 4 (Alcaraz's quarter)
    {"match": 49, "p1": "Ben Shelton", "p2": "Ignacio Buse", "s1": 6, "s2": None},
    {"match": 50, "p1": "Pablo Carreno Busta", "p2": "Pablo Llamas Ruiz", "s1": None, "s2": None},
    {"match": 51, "p1": "Jordan Thompson", "p2": "Corentin Moutet", "s1": None, "s2": None},
    {"match": 52, "p1": "Adrian Mannarino", "p2": "Tallon Griekspoor", "s1": None, "s2": 29},
    {"match": 53, "p1": "Jiri Lehecka", "p2": "Borna Coric", "s1": 20, "s2": None},
    {"match": 54, "p1": "Camilo Ugo Carabelli", "p2": "Tomas Martin Etcheverry", "s1": None, "s2": None},
    {"match": 55, "p1": "Daniel Elahi Galan", "p2": "Raphael Collignon", "s1": None, "s2": None},
    {"match": 56, "p1": "Sebastian Ofner", "p2": "Casper Ruud", "s1": None, "s2": 12},
    {"match": 57, "p1": "Daniil Medvedev", "p2": "Benjamin Bonzi", "s1": 13, "s2": None},
    {"match": 58, "p1": "Mariano Navone", "p2": "Marcos Giron", "s1": None, "s2": None},
    {"match": 59, "p1": "Roberto Carballes Baena", "p2": "Arthur Rinderknech", "s1": None, "s2": None},
    {"match": 60, "p1": "Alexander Shevchenko", "p2": "Alejandro Davidovich Fokina", "s1": None, "s2": 18},
    {"match": 61, "p1": "Luciano Darderi", "p2": "Rinky Hijikata", "s1": 32, "s2": None},
    {"match": 62, "p1": "Stefan Dostanic", "p2": "Eliot Spizzirri", "s1": None, "s2": None},
    {"match": 63, "p1": "Mattia Bellucci", "p2": "Juncheng Shang", "s1": None, "s2": None},
    {"match": 64, "p1": "Reilly Opelka", "p2": "Carlos Alcaraz", "s1": None, "s2": 2},
]

draw_df = pd.DataFrame(complete_matches)
draw_df = draw_df.rename(columns={'p1': 'player1', 'p2': 'player2', 's1': 'seed1', 's2': 'seed2'})

# Step 7: Enhanced match simulation
print("\nStep 7: Running enhanced tournament simulation...")

def simulate_match_ml(p1, p2):
    """Simulate match using ML model if available, else Elo"""
    if model is not None and scaler is not None:
        # Prepare features
        features = np.array([[
            elo_overall.get(p1, 1500) - elo_overall.get(p2, 1500),
            elo_hard.get(p1, 1500) - elo_hard.get(p2, 1500),
            elo_usopen.get(p1, 1500) - elo_usopen.get(p2, 1500),
            player_stats.get(p1, {}).get('recent_form', 0.5) - player_stats.get(p2, {}).get('recent_form', 0.5),
            player_stats.get(p1, {}).get('hard_win_rate', 0.5) - player_stats.get(p2, {}).get('hard_win_rate', 0.5),
            player_stats.get(p1, {}).get('gs_win_rate', 0.5) - player_stats.get(p2, {}).get('gs_win_rate', 0.5),
            np.log1p(player_stats.get(p1, {}).get('total_matches', 0)) - np.log1p(player_stats.get(p2, {}).get('total_matches', 0)),
            0,  # Age diff not available for future
            1,  # Best of 5
            1   # Hard court
        ]])
        
        features_scaled = scaler.transform(features)
        prob = model.predict_proba(features_scaled)[0, 1]
    else:
        # Fallback to rating difference
        r1 = player_ratings.get(p1, 1500)
        r2 = player_ratings.get(p2, 1500)
        prob = 1 / (1 + 10**((r2 - r1) / 400))
    
    # Add match-day variance
    prob = prob * 0.8 + np.random.random() * 0.2
    
    return p1 if np.random.random() < prob else p2

def simulate_tournament_enhanced():
    """Simulate tournament with enhanced predictions"""
    # Round 1
    r1_winners = []
    for _, match in draw_df.iterrows():
        winner = simulate_match_ml(match['player1'], match['player2'])
        r1_winners.append(winner)
    
    # Subsequent rounds
    current_round = r1_winners
    
    while len(current_round) > 1:
        next_round = []
        for i in range(0, len(current_round), 2):
            winner = simulate_match_ml(current_round[i], current_round[i+1])
            next_round.append(winner)
        current_round = next_round
    
    return current_round[0] if current_round else None

# Run simulations
results = Counter()
print(f"\nRunning {NUM_SIMULATIONS} simulations...")

for sim in range(NUM_SIMULATIONS):
    if sim % 1000 == 0:
        print(f"  Progress: {sim}/{NUM_SIMULATIONS}")
    
    champion = simulate_tournament_enhanced()
    if champion:
        results[champion] += 1

# Display results
print("\n" + "="*70)
print("US OPEN 2025 PREDICTIONS - ENHANCED MODEL")
print("="*70)

# Sort by probability
champion_probs = [(p, c/NUM_SIMULATIONS*100) for p, c in results.most_common(25)]

print(f"\n🏆 PREDICTED CHAMPION: {champion_probs[0][0]} ({champion_probs[0][1]:.1f}%)")

print("\n" + "-"*60)
print("TOP 25 CHAMPIONSHIP PROBABILITIES")
print("-"*60)
print(f"{'Rank':<5} {'Player':<25} {'Win %':>8} {'Elo':>8} {'Hard WR':>8}")
print("-"*60)

for rank, (player, win_prob) in enumerate(champion_probs[:25], 1):
    elo = int(elo_overall.get(player, 1500))
    hard_wr = player_stats.get(player, {}).get('hard_win_rate', 0) * 100
    print(f"{rank:<5} {player:<25} {win_prob:>7.1f}% {elo:>8} {hard_wr:>7.1f}%")

# Summary statistics
print("\n" + "-"*60)
print("MODEL INSIGHTS")
print("-"*60)
print(f"Total unique champions: {len(results)}")
print(f"Using ML model: {'Yes' if model is not None else 'No'}")
print(f"Players with historical data: {len(player_ratings)}")

# Top players comparison
print("\n" + "-"*60)
print("TOP SEEDS ANALYSIS")
print("-"*60)
print(f"{'Player':<20} {'Win %':>8} {'Elo':>8} {'Form':>8} {'US Open Elo':>12}")
print("-"*60)

for player in ['Jannik Sinner', 'Carlos Alcaraz', 'Novak Djokovic', 'Alexander Zverev', 
               'Taylor Fritz', 'Daniil Medvedev']:
    win_pct = next((prob for p, prob in champion_probs if p == player), 0)
    elo = int(elo_overall.get(player, 1500))
    form = player_stats.get(player, {}).get('recent_form', 0) * 100
    uso_elo = int(elo_usopen.get(player, 1500))
    print(f"{player:<20} {win_pct:>7.1f}% {elo:>8} {form:>7.1f}% {uso_elo:>12}")

print("\n" + "="*70)