
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US Open 2025 – Visualization Suite (Clean)
------------------------------------------
Consolidated, headless, and robust version of the multi-file code you shared.
- Single entrypoint
- Optional-deps (plotly, networkx, scikit-learn) handled gracefully
- Saves PNGs; no plt.show() calls so it works in headless envs
- CLI: --all / --only / --skip / --outdir / --dpi
- Reproducible randomness via --seed
"""

import argparse
import sys
import warnings
from typing import Callable, Dict, List

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

# Try/guard optional libs
try:
    import seaborn as sns  # optional
    _HAS_SEABORN = True
    sns.set_palette("husl")
except Exception:
    _HAS_SEABORN = False

try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    _HAS_NX = False

try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# --------------------------
# Core shared data
# --------------------------

PLAYERS = ['Djokovic', 'Sinner', 'Alcaraz', 'Medvedev', 'Zverev', 'Fritz', 'Draper', 'Shelton']
CHAMPIONSHIP_PROBS = [17.7, 10.5, 9.5, 9.6, 5.2, 2.6, 2.6, 0.6]

def _save(fig, outdir: str, filename: str, dpi: int = 300):
    out = f"{outdir.rstrip('/')}/{filename}"
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out

# --------------------------
# Viz functions (subset)
# --------------------------

def create_probability_heatmap(outdir: str, dpi: int):
    rounds = ['R1', 'R2', 'R3', 'R4', 'QF', 'SF', 'F']
    n_players = len(PLAYERS)
    prob_matrix = np.zeros((n_players, len(rounds)))
    for i, base_prob in enumerate(CHAMPIONSHIP_PROBS):
        base_survival = (base_prob / 100.0) ** (1/7)
        round_multipliers = [0.95, 0.88, 0.75, 0.65, 0.55, 0.45, 0.35]
        for j in range(len(rounds)):
            adjusted = min(0.98, base_survival + (base_prob/100.0) * round_multipliers[j])
            prob_matrix[i, j] = (adjusted ** (j+1)) * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    im1 = ax1.imshow(prob_matrix, cmap='RdYlBu_r', aspect='auto', interpolation='bilinear')
    ax1.set_xticks(range(len(rounds))); ax1.set_xticklabels(rounds)
    ax1.set_yticks(range(n_players)); ax1.set_yticklabels(PLAYERS)
    ax1.set_title('TOURNAMENT PROGRESSION PROBABILITY HEATMAP\n(% chance of reaching each round)', fontsize=13, fontweight='bold')
    for i in range(n_players):
        for j in range(len(rounds)):
            ax1.text(j, i, f'{prob_matrix[i, j]:.1f}%', ha="center", va="center",
                     color="black" if prob_matrix[i, j] < 50 else "white", fontsize=8, fontweight='bold')
    fig.colorbar(im1, ax=ax1, label='Probability (%)')

    round_avg = np.mean(prob_matrix, axis=0); round_std = np.std(prob_matrix, axis=0)
    ax2.bar(rounds, round_avg, yerr=round_std, capsize=5, alpha=0.8, color='skyblue')
    ax2.set_ylabel('Average Advancement Probability (%)')
    ax2.set_title('ROUND DIFFICULTY ANALYSIS\nAverage probability ± std dev', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, (avg, std) in enumerate(zip(round_avg, round_std)):
        ax2.text(i, avg + std + 2, f'{avg:.1f}±{std:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

    return _save(fig, outdir, "probability_heatmap.png", dpi)

def create_weather_analysis(outdir: str, dpi: int):
    weather_impacts = {
        'Indoor': {'Djokovic': 1.05, 'Sinner': 1.02, 'Alcaraz': 0.97, 'Medvedev': 1.08,
                   'Zverev': 1.03, 'Fritz': 1.01, 'Draper': 1.00, 'Shelton': 0.98},
        'Sunny': {'Djokovic': 1.00, 'Sinner': 1.05, 'Alcaraz': 1.08, 'Medvedev': 0.95,
                  'Zverev': 1.02, 'Fritz': 1.04, 'Draper': 1.03, 'Shelton': 1.05},
        'Windy': {'Djokovic': 1.10, 'Sinner': 1.05, 'Alcaraz': 0.92, 'Medvedev': 1.08,
                  'Zverev': 0.95, 'Fritz': 0.88, 'Draper': 0.93, 'Shelton': 0.85},
        'Hot': {'Djokovic': 1.08, 'Sinner': 1.00, 'Alcaraz': 1.03, 'Medvedev': 0.88,
                'Zverev': 0.92, 'Fritz': 0.95, 'Draper': 0.90, 'Shelton': 0.93},
        'Humid': {'Djokovic': 1.05, 'Sinner': 0.98, 'Alcaraz': 0.95, 'Medvedev': 0.85,
                  'Zverev': 0.90, 'Fritz': 0.92, 'Draper': 0.88, 'Shelton': 0.90}
    }
    conditions = list(weather_impacts.keys())
    impact_matrix = np.array([[weather_impacts[cond][p] for cond in conditions] for p in PLAYERS])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

    im = ax1.imshow(impact_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0.85, vmax=1.10)
    ax1.set_xticks(range(len(conditions))); ax1.set_xticklabels(conditions)
    ax1.set_yticks(range(len(PLAYERS))); ax1.set_yticklabels(PLAYERS)
    ax1.set_title('WEATHER CONDITION IMPACT MATRIX\n(Performance multiplier)', fontweight='bold')
    for i in range(len(PLAYERS)):
        for j in range(len(conditions)):
            color = "white" if impact_matrix[i, j] > 1.0 else "black"
            ax1.text(j, i, f'{impact_matrix[i, j]:.2f}', ha="center", va="center", color=color, fontweight='bold', fontsize=8)
    fig.colorbar(im, ax=ax1, label='Performance Multiplier')

    weather_variance = np.var(impact_matrix, axis=1); weather_mean = np.mean(impact_matrix, axis=1)
    ax2.scatter(weather_variance, weather_mean, s=np.array(CHAMPIONSHIP_PROBS)*12, alpha=0.7, c=range(len(PLAYERS)), cmap='plasma')
    for i, p in enumerate(PLAYERS):
        ax2.annotate(p, (weather_variance[i], weather_mean[i]), xytext=(3, 3), textcoords='offset points', fontsize=9)
    ax2.set_xlabel('Weather Sensitivity (Variance)'); ax2.set_ylabel('Average Weather Performance')
    ax2.set_title('WEATHER ADAPTABILITY\nHigh variance = weather-sensitive', fontweight='bold')
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.7); ax2.grid(True, alpha=0.3)

    us_open_weather = {'Indoor': 0.15, 'Sunny': 0.40, 'Windy': 0.20, 'Hot': 0.35, 'Humid': 0.45}
    expected = []
    for p in PLAYERS:
        expected.append(sum(us_open_weather[c]*weather_impacts[c][p] for c in conditions))
    bars = ax3.bar(PLAYERS, expected, alpha=0.8, color=['red' if v>1.02 else 'orange' if v>1.0 else 'lightblue' for v in expected])
    for b, v in zip(bars, expected):
        ax3.text(b.get_x()+b.get_width()/2., v+0.005, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax3.axhline(1.0, color='black', linestyle='-', alpha=0.5)
    ax3.set_ylabel('Expected Performance Multiplier'); ax3.set_title('EXPECTED US OPEN WEATHER PERFORMANCE', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45); ax3.grid(axis='y', alpha=0.3)

    temps = np.arange(15, 40, 1)
    for player, opt in [('Djokovic',22), ('Medvedev',18), ('Alcaraz',25)]:
        perf = 0.85 + 0.3 * np.exp(-0.005*(temps-opt)**2)
        ax4.plot(temps, perf, linewidth=2, marker='o', markersize=3, label=player, alpha=0.85)
    ax4.set_xlabel('Temperature (°C)'); ax4.set_ylabel('Relative Performance')
    ax4.set_title('TEMPERATURE PERFORMANCE CURVES', fontweight='bold'); ax4.legend(); ax4.grid(True, alpha=0.3)
    ax4.axvspan(25, 32, alpha=0.15, color='red')

    return _save(fig, outdir, "weather_analysis.png", dpi)

def create_time_analysis(outdir: str, dpi: int):
    time_performance = {
        'Djokovic': {11: 0.98, 13: 1.02, 15: 1.05, 17: 1.08, 19: 1.05, 21: 1.00},
        'Sinner': {11: 1.05, 13: 1.08, 15: 1.06, 17: 1.03, 19: 1.00, 21: 0.95},
        'Alcaraz': {11: 1.02, 13: 1.06, 15: 1.08, 17: 1.05, 19: 1.02, 21: 0.98},
        'Medvedev': {11: 0.95, 13: 0.98, 15: 1.02, 17: 1.08, 19: 1.10, 21: 1.05},
        'Zverev': {11: 1.00, 13: 1.03, 15: 1.05, 17: 1.06, 19: 1.03, 21: 0.98},
        'Fritz': {11: 1.08, 13: 1.10, 15: 1.06, 17: 1.02, 19: 0.98, 21: 0.95},
        'Draper': {11: 1.03, 13: 1.05, 15: 1.02, 17: 0.98, 19: 0.95, 21: 0.90},
        'Shelton': {11: 1.10, 13: 1.08, 15: 1.05, 17: 1.00, 19: 0.95, 21: 0.88}
    }
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

    hours = list(range(11, 22))
    for player in PLAYERS:
        perf = []
        available_hours = list(time_performance[player].keys())
        available_perfs = list(time_performance[player].values())
        for h in hours:
            if h in time_performance[player]:
                perf.append(time_performance[player][h])
            else:
                perf.append(np.interp(h, available_hours, available_perfs))
        ax1.plot(hours, perf, marker='o', linewidth=2, label=player, alpha=0.85)
    ax1.axhspan(15, 19, alpha=0.08, color='yellow')
    ax1.set_xlabel('Time of Day'); ax1.set_ylabel('Performance Multiplier')
    ax1.set_title('DAILY PERFORMANCE RHYTHMS'); ax1.legend(bbox_to_anchor=(1.04,1), loc='upper left'); ax1.grid(True, alpha=0.3)

    peak_times, ranges = [], []
    for player in PLAYERS:
        vals = list(time_performance[player].values())
        peak = max(vals); rng = peak - min(vals)
        pt = [t for t, v in time_performance[player].items() if v == peak][0]
        peak_times.append(pt); ranges.append(rng)
    ax2.scatter(peak_times, ranges, s=np.array(CHAMPIONSHIP_PROBS)*15, alpha=0.7, c=range(len(PLAYERS)), cmap='viridis')
    for i, p in enumerate(PLAYERS):
        ax2.annotate(p, (peak_times[i], ranges[i]), xytext=(3, 3), textcoords='offset points', fontsize=9)
    ax2.set_xlabel('Peak Hour'); ax2.set_ylabel('Perf Range'); ax2.set_title('PEAK TIMES vs CONSISTENCY'); ax2.grid(True, alpha=0.3)

    us_open_times = {11:0.05,12:0.10,13:0.15,14:0.15,15:0.20,16:0.15,17:0.10,19:0.08,20:0.02}
    expected = []
    for player in PLAYERS:
        available_hours = list(time_performance[player].keys())
        available_perfs = list(time_performance[player].values())
        total = 0.0
        for t, w in us_open_times.items():
            perf = time_performance[player].get(t, np.interp(t, available_hours, available_perfs))
            total += perf * w
        expected.append(total)
    bars = ax3.bar(PLAYERS, expected, alpha=0.85, color=['darkgreen' if p>1.03 else 'green' if p>1.01 else 'orange' if p>0.99 else 'red' for p in expected])
    for b, v in zip(bars, expected):
        ax3.text(b.get_x()+b.get_width()/2., v+0.002, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax3.axhline(1.0, color='black', linestyle='-', alpha=0.5); ax3.set_title('US OPEN SCHEDULING ADVANTAGE'); ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    early_avgs, late_avgs = [], []
    for player in PLAYERS:
        early_avgs.append(np.mean([time_performance[player].get(t,1.0) for t in [11,12,13,14,15]]))
        late_avgs.append(np.mean([time_performance[player].get(t,1.0) for t in [17,18,19,20,21]]))
    prefs = []
    for e, l in zip(early_avgs, late_avgs):
        if e > l + 0.02: prefs.append('Early Bird')
        elif l > e + 0.02: prefs.append('Night Owl')
        else: prefs.append('Flexible')
    import collections
    counts = collections.Counter(prefs)
    wedges, _ , autotexts = ax4.pie(list(counts.values()), labels=list(counts.keys()), autopct='%1.0f%%', startangle=90)
    ax4.set_title('MATCH TIME PREFERENCES')
    detail = "\n".join([f"{p}: {pref}" for p, pref in zip(PLAYERS, prefs)])
    ax4.text(1.25, 0.5, detail, transform=ax4.transAxes, fontsize=9, va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.6))

    return _save(fig, outdir, "time_analysis.png", dpi)

def create_mental_toughness_analysis(outdir: str, dpi: int):
    data = {
        'Player': PLAYERS,
        'Comeback_Win_Rate': [0.78, 0.65, 0.72, 0.68, 0.58, 0.52, 0.48, 0.45],
        'Tiebreak_Record':    [0.73, 0.69, 0.71, 0.67, 0.62, 0.58, 0.55, 0.52],
        'Deciding_Set_Record':[0.81, 0.72, 0.75, 0.70, 0.65, 0.60, 0.58, 0.55],
        'Break_Point_Conversion':[0.44, 0.38, 0.42, 0.35, 0.32, 0.36, 0.28, 0.30],
        'Save_Match_Points':[0.35, 0.28, 0.32, 0.25, 0.22, 0.18, 0.15, 0.12],
        'Big_Match_Experience':[95,45,55,60,50,35,20,15],
        'Championship_Prob': CHAMPIONSHIP_PROBS
    }
    df = pd.DataFrame(data)
    df['Mental_Toughness_Score'] = (df['Comeback_Win_Rate']*25 + df['Tiebreak_Record']*20 +
                                    df['Deciding_Set_Record']*25 + df['Save_Match_Points']*30)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

    dfs = df.sort_values('Mental_Toughness_Score', ascending=False)
    bars = ax1.barh(dfs['Player'], dfs['Mental_Toughness_Score'], color='gray', alpha=0.8)
    for b, v in zip(bars, dfs['Mental_Toughness_Score']):
        ax1.text(b.get_width()+0.2, b.get_y()+b.get_height()/2, f'{v:.1f}', va='center', fontweight='bold')
    ax1.set_xlabel('Mental Toughness Score'); ax1.set_title('MENTAL TOUGHNESS RANKING'); ax1.grid(axis='x', alpha=0.3)

    ax2.scatter(df['Mental_Toughness_Score'], df['Championship_Prob'],
                s=np.array(df['Big_Match_Experience'])*2, alpha=0.7, c=df['Save_Match_Points']*100, cmap='Reds')
    z = np.polyfit(df['Mental_Toughness_Score'], df['Championship_Prob'], 1); p = np.poly1d(z)
    ax2.plot(df['Mental_Toughness_Score'], p(df['Mental_Toughness_Score']), "g--", alpha=0.8)
    ax2.set_xlabel('Mental Toughness'); ax2.set_ylabel('Championship %'); ax2.set_title('MENTAL TOUGHNESS vs SUCCESS'); ax2.grid(True, alpha=0.3)

    top4 = df.nlargest(4, 'Mental_Toughness_Score')
    metrics = ['Comeback_Win_Rate','Tiebreak_Record','Deciding_Set_Record','Break_Point_Conversion','Save_Match_Points']
    x = np.arange(len(metrics)); width = 0.2; colors = ['red','blue','green','orange']
    for i, (_, row) in enumerate(top4.iterrows()):
        vals = [row[m]*100 for m in metrics]
        ax3.bar(x+i*width, vals, width, label=row['Player'], alpha=0.8, color=colors[i%4])
    ax3.set_xticks(x+width*1.5); ax3.set_xticklabels(['Comeback','Tiebreak','Decider','BP Conv','MP Save'])
    ax3.set_ylabel('Success Rate (%)'); ax3.set_title('PRESSURE PERFORMANCE (Top 4)'); ax3.legend(); ax3.grid(axis='y', alpha=0.3)

    exp_bins = []
    for e in df['Big_Match_Experience']:
        if e >= 80: exp_bins.append('Veteran (80+)')
        elif e >= 50: exp_bins.append('Experienced (50-79)')
        elif e >= 30: exp_bins.append('Developing (30-49)')
        else: exp_bins.append('Newcomer (<30)')
    df['Experience_Category'] = exp_bins
    mental_avg = df.groupby('Experience_Category')['Mental_Toughness_Score'].mean()
    champ_avg = df.groupby('Experience_Category')['Championship_Prob'].mean()
    cats = list(mental_avg.index)
    ax4_t = ax4.twinx()
    b1 = ax4.bar([i-0.2 for i in range(len(cats))], list(mental_avg.values), width=0.4, alpha=0.8, label='Mental Toughness', color='purple')
    b2 = ax4_t.bar([i+0.2 for i in range(len(cats))], list(champ_avg.values), width=0.4, alpha=0.6, label='Championship %', color='gold')
    ax4.set_xticks(range(len(cats))); ax4.set_xticklabels(cats, rotation=30, ha='right')
    ax4.set_ylabel('Mental Toughness'); ax4_t.set_ylabel('Championship %')
    ax4.set_title('EXPERIENCE vs MENTAL STRENGTH')
    lines1, labels1 = ax4.get_legend_handles_labels(); lines2, labels2 = ax4_t.get_legend_handles_labels()
    ax4.legend(lines1+lines2, labels1+labels2, loc='upper left')
    ax4.grid(axis='y', alpha=0.3)

    return _save(fig, outdir, "mental_toughness_analysis.png", dpi)

def create_physical_conditioning_analysis(outdir: str, dpi: int):
    data = {
        'Player': PLAYERS,
        'VO2_Max_Estimate': [58, 62, 65, 55, 57, 60, 59, 63],
        'Match_Length_Tolerance': [4.8, 4.2, 4.5, 3.8, 4.0, 3.9, 3.5, 3.6],
        'Recovery_Rate': [0.92, 0.95, 0.94, 0.88, 0.90, 0.91, 0.89, 0.93],
        'Injury_Frequency': [0.8, 0.3, 0.5, 0.6, 1.2, 0.4, 1.8, 0.2],
        'Training_Load': [85, 88, 90, 82, 78, 86, 75, 89],
        'Age': [37, 23, 21, 28, 27, 27, 22, 21],
        'Championship_Prob': CHAMPIONSHIP_PROBS
    }
    df = pd.DataFrame(data)
    df['Fitness_Score'] = (df['VO2_Max_Estimate']/70*25 + df['Match_Length_Tolerance']/5*20 +
                           df['Recovery_Rate']*25 + (2.0-df['Injury_Frequency'])/2*15 +
                           df['Training_Load']/100*15)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

    dfs = df.sort_values('Fitness_Score', ascending=False)
    bars = ax1.barh(dfs['Player'], dfs['Fitness_Score'], color='green', alpha=0.8)
    for b, v in zip(bars, dfs['Fitness_Score']):
        ax1.text(b.get_width()+0.4, b.get_y()+b.get_height()/2, f'{v:.1f}', va='center', fontweight='bold')
    ax1.set_xlabel('Physical Fitness Composite Score'); ax1.set_title('PHYSICAL FITNESS RANKING'); ax1.grid(axis='x', alpha=0.3)

    ax2.scatter(df['Fitness_Score'], df['Championship_Prob'],
                s=np.array(df['Age'])*8, alpha=0.7, c=df['Injury_Frequency'], cmap='RdYlGn_r')
    z = np.polyfit(df['Fitness_Score'], df['Championship_Prob'], 1); p = np.poly1d(z)
    ax2.plot(df['Fitness_Score'], p(df['Fitness_Score']), "b--", alpha=0.8)
    ax2.set_xlabel('Fitness Score'); ax2.set_ylabel('Championship %'); ax2.set_title('FITNESS vs CHAMPIONSHIP SUCCESS'); ax2.grid(True, alpha=0.3)

    groups = {'Young (≤23)': df[df['Age']<=23], 'Prime (24-28)': df[(df['Age']>23)&(df['Age']<=28)], 'Veteran (29+)': df[df['Age']>28]}
    metrics = ['VO2_Max_Estimate','Match_Length_Tolerance','Recovery_Rate','Injury_Frequency']
    x = np.arange(len(metrics)); width = 0.25; colors = ['lightblue','blue','darkblue']
    for i, (label, g) in enumerate(groups.items()):
        vals = [g[m].mean() if len(g)>0 else 0 for m in metrics]
        vals[3] = (2 - vals[3]) * 2  # invert injury frequency to resistance
        ax3.bar(x+i*width, vals, width, alpha=0.8, label=label, color=colors[i])
    ax3.set_xticks(x+width); ax3.set_xticklabels(['VO2 Max','Endurance\n(h)','Recovery','Injury\nResistance'])
    ax3.set_title('PHYSICAL PERFORMANCE BY AGE GROUP'); ax3.legend(); ax3.grid(axis='y', alpha=0.3)

    durations = np.arange(1.5, 6.0, 0.1)
    top4 = df.nlargest(4, 'Fitness_Score')
    for _, r in top4.iterrows():
        curve = []
        for d in durations:
            base = 100; endu = r['Match_Length_Tolerance']; vo2 = r['VO2_Max_Estimate']/60; age_pen = (r['Age']-20)/20*0.1
            if d <= endu:
                decline = (d/endu)*5 + age_pen*d
            else:
                excess = d - endu
                decline = 5 + excess*15*(1 - vo2 + age_pen)
            curve.append(max(60, base - decline))
        ax4.plot(durations, curve, linewidth=2.5, label=r['Player'], alpha=0.85)
    ax4.axvline(2.5, color='green', linestyle='--', alpha=0.7); ax4.axvline(4.0, color='orange', linestyle='--', alpha=0.7); ax4.axvline(5.0, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Match Duration (h)'); ax4.set_ylabel('Performance Level (%)'); ax4.set_title('STAMINA CURVES (Top 4)'); ax4.legend(); ax4.grid(True, alpha=0.3)
    ax4.set_ylim(55, 105)

    return _save(fig, outdir, "physical_conditioning_analysis.png", dpi)

# --------------------------
# Registry
# --------------------------

VIZ_FUNCS: Dict[str, Callable[[str, int], str]] = {
    "probability_heatmap": create_probability_heatmap,
    "weather_analysis": create_weather_analysis,
    "time_analysis": create_time_analysis,
    "mental_toughness_analysis": create_mental_toughness_analysis,
    "physical_conditioning_analysis": create_physical_conditioning_analysis,
}

# --------------------------
# CLI
# --------------------------

def parse_args(argv: List[str] = None):
    p = argparse.ArgumentParser(description="US Open 2025 – Visualization Suite (Clean)")
    p.add_argument("--all", action="store_true", help="Run all visualizations")
    p.add_argument("--only", nargs="+", choices=sorted(VIZ_FUNCS.keys()), help="Run only these viz keys")
    p.add_argument("--skip", nargs="+", choices=sorted(VIZ_FUNCS.keys()), help="Skip these viz keys")
    p.add_argument("--outdir", default="output", help="Directory for saved PNGs")
    p.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args(argv)

def main(argv: List[str] = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)

    # Choose what to run
    if args.all or (not args.only and not args.skip):
        selected = list(VIZ_FUNCS.keys())
    elif args.only:
        selected = args.only
    else:
        selected = [k for k in VIZ_FUNCS if k not in set(args.skip or [])]

    # Create outdir
    import os
    os.makedirs(args.outdir, exist_ok=True)

    print("="*70)
    print("US OPEN 2025 – VISUALIZATION SUITE (CLEAN)")
    print("Output directory:", args.outdir)
    print("Selected visualizations:", ", ".join(selected))
    print("="*70)

    generated = []
    for key in selected:
        try:
            print(f"→ Generating {key} ...", end=" ")
            path = VIZ_FUNCS[key](args.outdir, args.dpi)
            generated.append(path)
            print("done.")
        except Exception as e:
            print(f"FAILED: {e}")

    print("\nGenerated files:")
    for p in generated:
        print("-", p)
    print("\nTip: run with --only time_analysis probability_heatmap to limit output.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
