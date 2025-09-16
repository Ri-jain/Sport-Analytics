#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 23:17:53 2025

@author: rishabhjainxa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced US Open 2025 Prediction System (v2)
- Multi-year ATP data with time-decay Elo
- Serve/return aggregates with decay (overall + hard)
- Head-to-head feature
- Time-based split, calibrated Logistic Regression & HistGradientBoosting
- Elo + ML ensemble
- Reproducible tournament simulation with small, zero-mean noise
---------------------------------------------------------------------
Requires: pandas, numpy, scikit-learn
pip install -U pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import unicodedata, re, warnings

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
DATA_FILES = [
    "atp_matches_2019.csv",
    "atp_matches_2020.csv",
    "atp_matches_2021.csv",
    "atp_matches_2022.csv",
    "atp_matches_2023.csv",
    "atp_matches_2024.csv",
]
NUM_SIMULATIONS = 10000
DRAW_OUTPUT_CSV = "usopen_2025_complete_draw.csv"

# Ensemble weights
W_LR  = 0.4
W_HGB = 0.4
W_ELO = 0.2

# Time-decay / form windows
ELO_K = 32
DECAY_RATE = 0.95       # per year
RECENT_DAYS = 180       # recent form window

# Time-based split cutoff for model training
TIME_CUTOFF = pd.Timestamp("2024-06-01")

# Reproducibility
np.random.seed(42)

# =========================
# Utilities
# =========================
def norm(s):
    """Normalize names/strings (strip accents, collapse spaces)."""
    if pd.isna(s): return s
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_date_col(series):
    """Robust parse for tourney_date across mixed formats."""
    # Try yyyymmdd first, then general parse
    s_str = series.astype(str)
    dt1 = pd.to_datetime(s_str, format="%Y%m%d", errors="coerce")
    dt2 = pd.to_datetime(s_str, errors="coerce")
    out = dt1.fillna(dt2)
    return out

def is_grand_slam(level, name):
    level = str(level)
    name  = str(name)
    return (level == "G") or ("Grand Slam" in name) or ("US Open" in name)

def safe_div(num, den, default=0.0):
    return float(num) / float(den) if den and den != 0 else float(default)

# =========================
# Load & Clean
# =========================
print("="*70)
print("ENHANCED US OPEN 2025 PREDICTION SYSTEM (v2)")
print("Using Historical Data 2019–2024")
print("="*70)

print("\nStep 1: Loading historical ATP data...")
frames = []
for f in DATA_FILES:
    try:
        df = pd.read_csv(f)
        frames.append(df)
        print(f"  Loaded {f}: {len(df)} rows")
    except Exception as e:
        print(f"  WARNING: Could not load {f}: {e}")

df_all = pd.concat(frames, ignore_index=True)
print(f"\nTotal matches loaded: {len(df_all)}")

# Normalize key columns
for col in ["winner_name","loser_name","tourney_name","surface","tourney_level"]:
    if col in df_all.columns:
        df_all[col] = df_all[col].map(norm)

# Dates
if "tourney_date" in df_all.columns:
    df_all["tourney_date"] = parse_date_col(df_all["tourney_date"])
else:
    raise ValueError("Missing 'tourney_date' column in data.")

# Basic fills
for col in ["winner_age","loser_age","winner_rank_points","loser_rank_points"]:
    if col in df_all.columns:
        if "age" in col:
            df_all[col] = df_all[col].fillna(df_all[col].median())
        else:
            df_all[col] = df_all[col].fillna(0)

# Drop non-played matches
bad_tokens = ("W/O","ABN","ABD","DEF","RET","w/o","walkover","ret.")
mask_bad = df_all["score"].fillna("").str.contains("|".join(bad_tokens), case=False, regex=True)
df_all = df_all.loc[~mask_bad].copy()

# Drop rows with missing names
df_all = df_all[(df_all["winner_name"].notna()) & (df_all["loser_name"].notna())].copy()

print(f"Matches after cleaning: {len(df_all)}")

# =========================
# Elo with time decay
# =========================
print("\nStep 2: Building time-weighted Elo ratings...")

def build_comprehensive_elo(matches, k=ELO_K, decay_rate=DECAY_RATE):
    elo_overall = defaultdict(lambda: 1500.0)
    elo_hard    = defaultdict(lambda: 1500.0)
    elo_usopen  = defaultdict(lambda: 1500.0)

    matches = matches.sort_values("tourney_date")
    now = pd.Timestamp.now()

    for _, m in matches.iterrows():
        w, l = m["winner_name"], m["loser_name"]
        if pd.isna(w) or pd.isna(l): 
            continue
        d = m["tourney_date"]
        days_ago = (now - d).days if pd.notna(d) else 3650
        tf = decay_rate ** (days_ago / 365.0)

        # Adjust K by importance
        name = m.get("tourney_name","")
        level = m.get("tourney_level","")
        if is_grand_slam(level, name):
            k_adj = k * 1.5 * tf
        elif "Masters" in str(name):
            k_adj = k * 1.2 * tf
        else:
            k_adj = k * tf

        # Overall
        exp_w = 1.0/(1.0 + 10.0**((elo_overall[l]-elo_overall[w])/400.0))
        elo_overall[w] += k_adj * (1.0 - exp_w)
        elo_overall[l] += k_adj * (0.0 - (1.0 - exp_w))

        # Hard
        if str(m.get("surface","")).lower() == "hard":
            exp_w_h = 1.0/(1.0 + 10.0**((elo_hard[l]-elo_hard[w])/400.0))
            elo_hard[w] += k_adj * (1.0 - exp_w_h)
            elo_hard[l] += k_adj * (0.0 - (1.0 - exp_w_h))

            # US Open specific boost
            if "US Open" in str(name):
                k_uso = k_adj * 1.5
                exp_w_u = 1.0/(1.0 + 10.0**((elo_usopen[l]-elo_usopen[w])/400.0))
                elo_usopen[w] += k_uso * (1.0 - exp_w_u)
                elo_usopen[l] += k_uso * (0.0 - (1.0 - exp_w_u))

    return elo_overall, elo_hard, elo_usopen

elo_overall, elo_hard, elo_usopen = build_comprehensive_elo(df_all)
print(f"Built ratings for {len(elo_overall)} players")

# =========================
# Serve/Return aggregates (decayed)
# =========================
print("\nStep 3: Calculating serve/return stats with decay...")

SERVE_COLS_W = ["w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_bpSaved","w_bpFaced","w_SvGms"]
SERVE_COLS_L = ["l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_bpSaved","l_bpFaced","l_SvGms"]

def aggregate_srvret(matches, decay_rate=DECAY_RATE):
    now = pd.Timestamp.now()
    agg_all  = defaultdict(lambda: defaultdict(float))
    agg_hard = defaultdict(lambda: defaultdict(float))

    def add(player, row, weight, hard):
        # Totals (weighted)
        ace = row.get("ace",0); dfv=row.get("df",0)
        svpt=row.get("svpt",0); s1i=row.get("1stIn",0)
        s1w=row.get("1stWon",0); s2w=row.get("2ndWon",0)
        bps=row.get("bpSaved",0); bpf=row.get("bpFaced",0)
        svg = row.get("SvGms",0)

        target = agg_hard if hard else agg_all
        target[player]["ace"]     += weight * (ace or 0)
        target[player]["df"]      += weight * (dfv or 0)
        target[player]["svpt"]    += weight * (svpt or 0)
        target[player]["s1in"]    += weight * (s1i or 0)
        target[player]["s1won"]   += weight * (s1w or 0)
        target[player]["s2won"]   += weight * (s2w or 0)
        target[player]["bpsaved"] += weight * (bps or 0)
        target[player]["bpfaced"] += weight * (bpf or 0)
        target[player]["svgms"]   += weight * (svg or 0)

    for _, m in matches.iterrows():
        d = m["tourney_date"]
        days_ago = (now - d).days if pd.notna(d) else 3650
        w = decay_rate ** (days_ago/365.0)
        hard = str(m.get("surface","")).lower()=="hard"

        # Winner add
        wrow = {
            "ace": m.get("w_ace",0), "df": m.get("w_df",0), "svpt": m.get("w_svpt",0),
            "1stIn": m.get("w_1stIn",0), "1stWon": m.get("w_1stWon",0), "2ndWon": m.get("w_2ndWon",0),
            "bpSaved": m.get("w_bpSaved",0), "bpFaced": m.get("w_bpFaced",0), "SvGms": m.get("w_SvGms",0),
        }
        add(m["winner_name"], wrow, w, hard)
        # Loser add
        lrow = {
            "ace": m.get("l_ace",0), "df": m.get("l_df",0), "svpt": m.get("l_svpt",0),
            "1stIn": m.get("l_1stIn",0), "1stWon": m.get("l_1stWon",0), "2ndWon": m.get("l_2ndWon",0),
            "bpSaved": m.get("l_bpSaved",0), "bpFaced": m.get("l_bpFaced",0), "SvGms": m.get("l_SvGms",0),
        }
        add(m["loser_name"], lrow, w, hard)

    def finalize(agg):
        out = {}
        for p, t in agg.items():
            svpt  = t["svpt"]; s1i=t["s1in"]; s1w=t["s1won"]; s2w=t["s2won"]
            bpsv  = t["bpsaved"]; bpf=t["bpfaced"]; svg=t["svgms"]
            out[p] = {
                "srv_aces_pg": safe_div(t["ace"], svg, 0.0),
                "srv_df_pg":   safe_div(t["df"],  svg, 0.0),
                "srv_1stin":   safe_div(s1i, svpt, 0.0),
                "srv_1stwon":  safe_div(s1w, s1i,  0.0),
                "srv_2ndwon":  safe_div(s2w, (svpt - s1i), 0.0),
                "srv_bpsave":  safe_div(bpsv, bpf, 0.5),
            }
        return out

    return finalize(agg_all), finalize(agg_hard)

srv_all, srv_hard = aggregate_srvret(df_all)

# =========================
# Player Stats (win rates etc.)
# =========================
print("\nStep 4: Calculating player stats (form/hard/GS)...")

def player_stats(matches, player):
    won  = matches[matches["winner_name"]==player]
    lost = matches[matches["loser_name"]==player]

    cutoff = pd.Timestamp.now() - timedelta(days=RECENT_DAYS)
    r_won  = won[won["tourney_date"]>cutoff]
    r_lost = lost[lost["tourney_date"]>cutoff]

    total = len(won) + len(lost)
    wr    = len(won) / total if total>0 else 0.0

    r_total = len(r_won) + len(r_lost)
    r_wr    = len(r_won) / r_total if r_total>0 else wr

    hw  = won[won["surface"].str.lower()=="hard"]
    hl  = lost[lost["surface"].str.lower()=="hard"]
    htot= len(hw) + len(hl)
    hwr = len(hw) / htot if htot>0 else wr

    gsw = won[ (won["tourney_level"]=="G") | (won["tourney_name"].str.contains("US Open|Grand Slam", case=False, na=False)) ]
    gsl = lost[ (lost["tourney_level"]=="G") | (lost["tourney_name"].str.contains("US Open|Grand Slam", case=False, na=False)) ]
    gst = len(gsw)+len(gsl)
    gsr = len(gsw)/gst if gst>0 else wr

    return {
        "total_matches": total,
        "win_rate": wr,
        "recent_form": r_wr,
        "hard_win_rate": hwr,
        "gs_win_rate": gsr
    }

players = set(elo_overall.keys())
pstats = {p: player_stats(df_all, p) for p in players}

# =========================
# Head-to-Head
# =========================
print("\nStep 5: Building head-to-head matrix...")

def build_h2h(matches):
    h2h = defaultdict(lambda: defaultdict(int))
    for _, m in matches.iterrows():
        w, l = m["winner_name"], m["loser_name"]
        h2h[w][l] += 1
    return h2h

H2H = build_h2h(df_all)

# =========================
# Features for ML
# =========================
print("\nStep 6: Preparing training features...")

def match_feature_row(w, l, m, use_hard=True):
    # Elo
    elo_d     = elo_overall.get(w,1500) - elo_overall.get(l,1500)
    elo_h_d   = elo_hard.get(w,1500)    - elo_hard.get(l,1500)
    elo_uso_d = elo_usopen.get(w,1500)  - elo_usopen.get(l,1500)

    # Stats
    ps_w = pstats.get(w, {})
    ps_l = pstats.get(l, {})
    form_d  = ps_w.get("recent_form",0.5)   - ps_l.get("recent_form",0.5)
    hard_d  = ps_w.get("hard_win_rate",0.5) - ps_l.get("hard_win_rate",0.5)
    gs_d    = ps_w.get("gs_win_rate",0.5)   - ps_l.get("gs_win_rate",0.5)
    exp_d   = np.log1p(ps_w.get("total_matches",0)) - np.log1p(ps_l.get("total_matches",0))

    # Age
    aw = m.get("winner_age", np.nan)
    al = m.get("loser_age", np.nan)
    if pd.isna(aw): aw = 25.0
    if pd.isna(al): al = 25.0
    age_d = aw - al

    # Serve/Return (use hard set if available)
    s_w = (srv_hard if use_hard else srv_all).get(w, {})
    s_l = (srv_hard if use_hard else srv_all).get(l, {})
    for k in ["srv_aces_pg","srv_df_pg","srv_1stin","srv_1stwon","srv_2ndwon","srv_bpsave"]:
        s_w.setdefault(k, np.nan); s_l.setdefault(k, np.nan)

    # Replace NaNs with global medians (computed on the fly)
    # (Simple, fast; you can precompute for speed if needed.)
    def med(key):
        pool = [v.get(key) for v in (srv_hard if use_hard else srv_all).values() if not pd.isna(v.get(key))]
        return float(np.median(pool)) if pool else 0.5

    feat = {
        "elo_diff":        elo_d,
        "elo_hard_diff":   elo_h_d,
        "elo_usopen_diff": elo_uso_d,
        "form_diff":       form_d,
        "hard_wr_diff":    hard_d,
        "gs_wr_diff":      gs_d,
        "matches_diff":    exp_d,
        "age_diff":        age_d,
        "best_of_5":       1 if m.get("best_of",3)==5 else 0,
        "is_hard":         1 if str(m.get("surface","")).lower()=="hard" else 0,
        "h2h_diff":        H2H[w][l] - H2H[l][w],
        # Serve/return diffs (hard by default)
        "srv_aces_pg_diff":  (s_w["srv_aces_pg"] if not pd.isna(s_w["srv_aces_pg"]) else med("srv_aces_pg"))
                           - (s_l["srv_aces_pg"] if not pd.isna(s_l["srv_aces_pg"]) else med("srv_aces_pg")),
        "srv_df_pg_diff":    (s_w["srv_df_pg"] if not pd.isna(s_w["srv_df_pg"]) else med("srv_df_pg"))
                           - (s_l["srv_df_pg"] if not pd.isna(s_l["srv_df_pg"]) else med("srv_df_pg")),
        "srv_1stin_diff":    (s_w["srv_1stin"] if not pd.isna(s_w["srv_1stin"]) else med("srv_1stin"))
                           - (s_l["srv_1stin"] if not pd.isna(s_l["srv_1stin"]) else med("srv_1stin")),
        "srv_1stwon_diff":   (s_w["srv_1stwon"] if not pd.isna(s_w["srv_1stwon"]) else med("srv_1stwon"))
                           - (s_l["srv_1stwon"] if not pd.isna(s_l["srv_1stwon"]) else med("srv_1stwon")),
        "srv_2ndwon_diff":   (s_w["srv_2ndwon"] if not pd.isna(s_w["srv_2ndwon"]) else med("srv_2ndwon"))
                           - (s_l["srv_2ndwon"] if not pd.isna(s_l["srv_2ndwon"]) else med("srv_2ndwon")),
        "srv_bpsave_diff":   (s_w["srv_bpsave"] if not pd.isna(s_w["srv_bpsave"]) else med("srv_bpsave"))
                           - (s_l["srv_bpsave"] if not pd.isna(s_l["srv_bpsave"]) else med("srv_bpsave")),
    }
    return feat

FEATURE_ORDER = [
    "elo_diff","elo_hard_diff","elo_usopen_diff",
    "form_diff","hard_wr_diff","gs_wr_diff",
    "matches_diff","age_diff","best_of_5","is_hard","h2h_diff",
    "srv_aces_pg_diff","srv_df_pg_diff","srv_1stin_diff",
    "srv_1stwon_diff","srv_2ndwon_diff","srv_bpsave_diff"
]

def prepare_features(matches):
    X, y = [], []
    rows = []
    for _, m in matches.iterrows():
        w, l = m["winner_name"], m["loser_name"]
        if (w not in elo_overall) or (l not in elo_overall):
            continue
        feat = match_feature_row(w,l,m, use_hard=True)
        X.append([feat[k] for k in FEATURE_ORDER])
        y.append(1)  # winner perspective
        # reverse
        feat_rev = {k: (-v if "diff" in k else v) for k,v in feat.items()}
        X.append([feat_rev[k] for k in FEATURE_ORDER])
        y.append(0)
        rows.append(m)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)

# Time-based split
train_mask = df_all["tourney_date"] < TIME_CUTOFF
train_df = df_all.loc[train_mask]
test_df  = df_all.loc[~train_mask]
X_train, y_train = prepare_features(train_df)
X_test,  y_test  = prepare_features(test_df)

print(f"  Train size (pairs): {len(y_train)}  | Test size (pairs): {len(y_test)}")

# =========================
# Models & Calibration
# =========================
print("\nStep 7: Training models with calibration...")

# Logistic Regression + scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

base_lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr = CalibratedClassifierCV(base_lr, method="isotonic", cv=3)
lr.fit(X_train_s, y_train)

# HistGradientBoosting + calibration
base_hgb = HistGradientBoostingClassifier(
    max_depth=None, learning_rate=0.07, max_iter=400, l2_regularization=0.0,
    random_state=42
)
hgb = CalibratedClassifierCV(base_hgb, method="isotonic", cv=3)
hgb.fit(X_train, y_train)

# Evaluate on test
def eval_model(name, proba, y_true):
    pred = (proba >= 0.5).astype(int)
    acc  = accuracy_score(y_true, pred)
    ll   = log_loss(y_true, proba, labels=[0,1])
    br   = brier_score_loss(y_true, proba)
    print(f"  {name:<12} Acc={acc:.3f}  LogLoss={ll:.3f}  Brier={br:.3f}")

p_lr  = lr.predict_proba(X_test_s)[:,1]
p_hgb = hgb.predict_proba(X_test)[:,1]

# Simple Elo-only probability on test (from feature elo_diff only)
elo_idx = FEATURE_ORDER.index("elo_diff")
elo_diff_test = X_test[:, elo_idx]
p_elo = 1.0 / (1.0 + 10.0 ** (-(elo_diff_test)/400.0))

# Blended
p_blend = W_LR*p_lr + W_HGB*p_hgb + W_ELO*p_elo

print("\nTest-set metrics:")
eval_model("LogReg", p_lr, y_test)
eval_model("HGB",    p_hgb, y_test)
eval_model("EloOnly",p_elo, y_test)
eval_model("Ensemble", p_blend, y_test)

# =========================
# Final Player Ratings (for fallback/priors)
# =========================
print("\nStep 8: Creating final player ratings...")

player_ratings = {}
for p in players:
    base = (elo_overall[p] + 1.2*elo_hard[p] + 0.8*elo_usopen[p]) / 3.0
    st = pstats.get(p, {})
    form_bonus = (st.get("recent_form",0.5)-0.5)*200
    hard_bonus = (st.get("hard_win_rate",0.5)-0.5)*150
    exp_bonus  = np.log1p(st.get("total_matches",0))*10
    player_ratings[p] = base + form_bonus + hard_bonus + exp_bonus

# =========================
# Draw (64 first-round matches)
# =========================
print("\nStep 9: Loading US Open 2025 draw...")
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
draw_df = pd.DataFrame(complete_matches).rename(columns={"p1":"player1","p2":"player2","s1":"seed1","s2":"seed2"})
draw_df.to_csv(DRAW_OUTPUT_CSV, index=False)

# =========================
# Simulation
# =========================
print("\nStep 10: Running tournament simulations...")

def match_prob(p1, p2):
    """Blend calibrated LR + calibrated HGB + Elo-only as prior."""
    # Build feature vector in the same order
    # For future match: best_of_5=1, is_hard=1, age diff unknown -> 0
    feat = np.array([[ 
        elo_overall.get(p1,1500) - elo_overall.get(p2,1500),       # elo_diff
        elo_hard.get(p1,1500)    - elo_hard.get(p2,1500),          # elo_hard_diff
        elo_usopen.get(p1,1500)  - elo_usopen.get(p2,1500),        # elo_usopen_diff
        pstats.get(p1,{}).get("recent_form",0.5) - pstats.get(p2,{}).get("recent_form",0.5),   # form_diff
        pstats.get(p1,{}).get("hard_win_rate",0.5) - pstats.get(p2,{}).get("hard_win_rate",0.5),# hard_wr_diff
        pstats.get(p1,{}).get("gs_win_rate",0.5)   - pstats.get(p2,{}).get("gs_win_rate",0.5),  # gs_wr_diff
        np.log1p(pstats.get(p1,{}).get("total_matches",0)) - np.log1p(pstats.get(p2,{}).get("total_matches",0)), # matches_diff
        0.0,                                       # age_diff unknown for future
        1.0,                                       # best_of_5
        1.0,                                       # is_hard
        H2H[p1][p2] - H2H[p2][p1],                 # h2h_diff
        # serve/return diffs (use hard aggregates)
        (srv_hard.get(p1,{}).get("srv_aces_pg", np.nan)) - (srv_hard.get(p2,{}).get("srv_aces_pg", np.nan)),
        (srv_hard.get(p1,{}).get("srv_df_pg",   np.nan)) - (srv_hard.get(p2,{}).get("srv_df_pg",   np.nan)),
        (srv_hard.get(p1,{}).get("srv_1stin",   np.nan)) - (srv_hard.get(p2,{}).get("srv_1stin",   np.nan)),
        (srv_hard.get(p1,{}).get("srv_1stwon",  np.nan)) - (srv_hard.get(p2,{}).get("srv_1stwon",  np.nan)),
        (srv_hard.get(p1,{}).get("srv_2ndwon",  np.nan)) - (srv_hard.get(p2,{}).get("srv_2ndwon",  np.nan)),
        (srv_hard.get(p1,{}).get("srv_bpsave",  np.nan)) - (srv_hard.get(p2,{}).get("srv_bpsave",  np.nan)),
    ]], dtype=float)

    # NaNs → median 0 assumption (differences)
    feat = np.where(np.isnan(feat), 0.0, feat)

    # Model probs
    p_lr  = float(lr.predict_proba(scaler.transform(feat))[:,1])
    p_hgb = float(hgb.predict_proba(feat)[:,1])
    # Elo prior from overall diff
    elo_diff = feat[0, FEATURE_ORDER.index("elo_diff")]
    p_elo = 1.0/(1.0 + 10.0**(-(elo_diff)/400.0))
    # Blend
    p = W_LR*p_lr + W_HGB*p_hgb + W_ELO*p_elo
    # Add small, zero-mean match-day variance
    eps = np.random.normal(0.0, 0.03)
    p = float(np.clip(p + eps, 0.02, 0.98))
    return p

def simulate_match(p1, p2):
    p = match_prob(p1, p2)
    return p1 if np.random.rand() < p else p2

def simulate_tournament():
    # Round 1
    winners = []
    for _, m in draw_df.iterrows():
        winners.append(simulate_match(m["player1"], m["player2"]))
    # Subsequent rounds
    while len(winners) > 1:
        nxt = []
        for i in range(0, len(winners), 2):
            nxt.append(simulate_match(winners[i], winners[i+1]))
        winners = nxt
    return winners[0] if winners else None

print(f"Running {NUM_SIMULATIONS} simulations...")
champ_counts = Counter()
for i in range(NUM_SIMULATIONS):
    if i % 1000 == 0:
        print(f"  Progress: {i}/{NUM_SIMULATIONS}")
    c = simulate_tournament()
    if c: champ_counts[c] += 1

# =========================
# Results
# =========================
print("\n" + "="*70)
print("US OPEN 2025 PREDICTIONS - ENHANCED MODEL (v2)")
print("="*70)

top = champ_counts.most_common(25)
champion, champ_p = (top[0][0], top[0][1] / NUM_SIMULATIONS * 100.0) if top else ("N/A", 0.0)
print(f"\n🏆 PREDICTED CHAMPION: {champion} ({champ_p:.1f}%)\n")

print("-"*60)
print("TOP 25 CHAMPIONSHIP PROBABILITIES")
print("-"*60)
print(f"{'Rank':<5} {'Player':<25} {'Win %':>8} {'Elo':>8} {'Hard WR':>8}")
print("-"*60)
for r,(p, cnt) in enumerate(top, 1):
    winp = cnt/NUM_SIMULATIONS*100
    elo  = int(elo_overall.get(p,1500))
    hwr  = pstats.get(p,{}).get("hard_win_rate",0)*100
    print(f"{r:<5} {p:<25} {winp:>7.1f}% {elo:>8} {hwr:>7.1f}%")

print("\n-"*60)
print("MODEL INSIGHTS (test-set)")
print("-"*60)
print(f"Players with historical data: {len(players)}")
# Already printed per-model metrics above

print("\n-"*60)
print("TOP SEEDS ANALYSIS")
print("-"*60)
print(f"{'Player':<20} {'Win %':>8} {'Elo':>8} {'Form':>8} {'US Open Elo':>12}")
print("-"*60)
for p in ["Jannik Sinner","Carlos Alcaraz","Novak Djokovic","Alexander Zverev","Taylor Fritz","Daniil Medvedev"]:
    win_pct = next((cnt/NUM_SIMULATIONS*100 for q,cnt in champ_counts.items() if q==p), 0.0)
    elo  = int(elo_overall.get(p,1500))
    form = pstats.get(p,{}).get("recent_form",0.0)*100
    uso  = int(elo_usopen.get(p,1500))
    print(f"{p:<20} {win_pct:>7.1f}% {elo:>8} {form:>7.1f}% {uso:>12}")
print("\n" + "="*70)
