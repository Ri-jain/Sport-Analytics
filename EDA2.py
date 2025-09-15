#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 21:33:50 2025

@author: rishabhjain
"""

#!/usr/bin/env python3
"""
Additional Creative Visualizations for US Open 2025 Tennis Predictions
Extends the analysis with new visualization types and perspectives
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta

# =============================================================================
# 1. NETWORK ANALYSIS - Player Relationships & Head-to-Head
# =============================================================================
def create_h2h_network():
    """Create network graph showing head-to-head relationships"""
    
    # Sample H2H data (would come from your actual data)
    h2h_data = [
        ('Novak Djokovic', 'Carlos Alcaraz', 3, 2),
        ('Novak Djokovic', 'Jannik Sinner', 4, 3),
        ('Carlos Alcaraz', 'Daniil Medvedev', 2, 3),
        ('Jannik Sinner', 'Daniil Medvedev', 1, 2),
        ('Alexander Zverev', 'Carlos Alcaraz', 1, 4),
        ('Taylor Fritz', 'Novak Djokovic', 0, 3),
    ]
    
    G = nx.Graph()
    
    # Add nodes (players) with attributes
    players = set()
    for p1, p2, _, _ in h2h_data:
        players.update([p1, p2])
    
    for player in players:
        # Use your actual player data
        prob = {'Novak Djokovic': 17.7, 'Jannik Sinner': 10.5, 'Carlos Alcaraz': 9.5,
                'Daniil Medvedev': 9.6, 'Alexander Zverev': 5.2, 'Taylor Fritz': 2.6}.get(player, 1.0)
        G.add_node(player, championship_prob=prob)
    
    # Add edges with H2H data
    for p1, p2, p1_wins, p2_wins in h2h_data:
        total_matches = p1_wins + p2_wins
        G.add_edge(p1, p2, p1_wins=p1_wins, p2_wins=p2_wins, total=total_matches)
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw nodes with size based on championship probability
    node_sizes = [G.nodes[node]['championship_prob'] * 50 for node in G.nodes()]
    node_colors = [G.nodes[node]['championship_prob'] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          cmap='Reds', alpha=0.8)
    
    # Draw edges with width based on number of matches
    edge_widths = [G[u][v]['total'] * 0.5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')
    
    # Add labels
    nx.draw_networkx_labels(G, pos, {node: node.split()[-1] for node in G.nodes()}, 
                           font_size=10, font_weight='bold')
    
    # Add H2H information as edge labels
    edge_labels = {(u, v): f"{G[u][v]['p1_wins']}-{G[u][v]['p2_wins']}" 
                  for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title('HEAD-TO-HEAD NETWORK ANALYSIS\nNode size = Championship probability, Edge width = Match history',
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('h2h_network.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 2. TIME SERIES ANALYSIS - Elo Rating Evolution
# =============================================================================
def create_elo_evolution():
    """Track Elo rating changes over time for top players"""
    
    # Simulate monthly Elo data (would come from your historical data)
    dates = pd.date_range('2023-01-01', '2024-12-01', freq='M')
    
    players_elo = {
        'Novak Djokovic': np.random.normal(2200, 50, len(dates)) + np.sin(np.arange(len(dates)) * 0.3) * 30,
        'Jannik Sinner': np.random.normal(2100, 60, len(dates)) + np.arange(len(dates)) * 2,
        'Carlos Alcaraz': np.random.normal(2150, 70, len(dates)) + np.sin(np.arange(len(dates)) * 0.5) * 40,
        'Daniil Medvedev': np.random.normal(2000, 40, len(dates)) - np.arange(len(dates)) * 0.5
    }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Elo evolution
    for player, elo_vals in players_elo.items():
        ax1.plot(dates, elo_vals, marker='o', linewidth=2, label=player, alpha=0.8)
    
    ax1.set_title('ELO RATING EVOLUTION (2023-2024)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Elo Rating')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Rating volatility
    volatility = {player: np.std(np.diff(elo_vals)) for player, elo_vals in players_elo.items()}
    current_elo = {player: elo_vals[-1] for player, elo_vals in players_elo.items()}
    
    scatter = ax2.scatter(list(volatility.values()), list(current_elo.values()), 
                         s=200, alpha=0.7, c=range(len(volatility)), cmap='viridis')
    
    for i, player in enumerate(volatility.keys()):
        ax2.annotate(player.split()[-1], 
                    (list(volatility.values())[i], list(current_elo.values())[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Rating Volatility (Standard Deviation)')
    ax2.set_ylabel('Current Elo Rating')
    ax2.set_title('CURRENT RATING vs CONSISTENCY', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('elo_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 3. RISK-REWARD ANALYSIS
# =============================================================================
def create_risk_reward_analysis():
    """Analyze risk vs reward for different betting scenarios"""
    
    players = ['Djokovic', 'Sinner', 'Alcaraz', 'Medvedev', 'Zverev', 'Fritz', 'Draper']
    win_probs = [17.7, 10.5, 9.5, 9.6, 5.2, 2.6, 2.6]
    implied_odds = [100/p for p in win_probs]  # Simplified odds calculation
    
    # Calculate expected value for different bet sizes
    bet_amounts = [10, 25, 50, 100]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Win Probability vs Implied Odds
    scatter = ax1.scatter(win_probs, implied_odds, s=[p*10 for p in win_probs], 
                         alpha=0.7, c=range(len(players)), cmap='plasma')
    
    for i, player in enumerate(players):
        ax1.annotate(player, (win_probs[i], implied_odds[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax1.set_xlabel('Win Probability (%)')
    ax1.set_ylabel('Implied Odds (Payout Multiplier)')
    ax1.set_title('RISK-REWARD MATRIX', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Expected Value Analysis
    expected_values = [(prob/100) * (100/prob - 1) * 100 - (1 - prob/100) * 100 
                      for prob in win_probs]
    
    colors = ['green' if ev > 0 else 'red' for ev in expected_values]
    bars = ax2.bar(players, expected_values, color=colors, alpha=0.7)
    
    for bar, ev in zip(bars, expected_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{ev:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Expected Value (%)')
    ax2.set_title('EXPECTED VALUE ANALYSIS (Fair Odds)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Portfolio Diversification
    # Show different portfolio strategies
    strategies = {
        'Conservative': [0.4, 0.3, 0.2, 0.1, 0, 0, 0],
        'Balanced': [0.25, 0.2, 0.2, 0.15, 0.1, 0.05, 0.05],
        'Aggressive': [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1]
    }
    
    portfolio_ev = {}
    portfolio_risk = {}
    
    for strategy, weights in strategies.items():
        ev = sum(w * ev for w, ev in zip(weights, expected_values))
        risk = np.sqrt(sum(w**2 * (p/100) * (1-p/100) for w, p in zip(weights, win_probs)))
        portfolio_ev[strategy] = ev
        portfolio_risk[strategy] = risk * 100
    
    ax3.scatter(list(portfolio_risk.values()), list(portfolio_ev.values()), 
               s=200, alpha=0.7, c=['blue', 'orange', 'red'])
    
    for strategy in strategies.keys():
        ax3.annotate(strategy, (portfolio_risk[strategy], portfolio_ev[strategy]),
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    ax3.set_xlabel('Portfolio Risk (%)')
    ax3.set_ylabel('Expected Return (%)')
    ax3.set_title('PORTFOLIO STRATEGIES', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Probability Confidence Intervals
    confidence_intervals = [(p-2, p+2) if p > 10 else (max(0, p-1), p+1) for p in win_probs]
    lower_bounds = [ci[0] for ci in confidence_intervals]
    upper_bounds = [ci[1] for ci in confidence_intervals]
    
    ax4.errorbar(range(len(players)), win_probs, 
                yerr=[np.array(win_probs) - np.array(lower_bounds), 
                     np.array(upper_bounds) - np.array(win_probs)],
                fmt='o', capsize=5, capthick=2, alpha=0.8)
    
    ax4.set_xticks(range(len(players)))
    ax4.set_xticklabels(players, rotation=45)
    ax4.set_ylabel('Win Probability (%) with 95% CI')
    ax4.set_title('PREDICTION CONFIDENCE INTERVALS', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('risk_reward_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 4. GEOGRAPHIC/NATIONALITY ANALYSIS
# =============================================================================
def create_nationality_analysis():
    """Analyze performance by player nationality/region"""
    
    player_nationalities = {
        'Novak Djokovic': 'Serbia', 'Jannik Sinner': 'Italy', 'Carlos Alcaraz': 'Spain',
        'Daniil Medvedev': 'Russia', 'Alexander Zverev': 'Germany', 'Taylor Fritz': 'USA',
        'Jack Draper': 'Great Britain', 'Alex De Minaur': 'Australia', 'Casper Ruud': 'Norway',
        'Andrey Rublev': 'Russia', 'Stefanos Tsitsipas': 'Greece', 'Karen Khachanov': 'Russia'
    }
    
    # Group by regions
    regions = {
        'Europe': ['Serbia', 'Italy', 'Spain', 'Germany', 'Great Britain', 'Norway', 'Greece'],
        'Eastern Europe/Russia': ['Russia'],
        'North America': ['USA'],
        'Oceania': ['Australia']
    }
    
    player_probs = {
        'Novak Djokovic': 17.7, 'Jannik Sinner': 10.5, 'Carlos Alcaraz': 9.5,
        'Daniil Medvedev': 9.6, 'Alexander Zverev': 5.2, 'Taylor Fritz': 2.6,
        'Jack Draper': 2.6, 'Alex De Minaur': 2.7, 'Casper Ruud': 3.6,
        'Andrey Rublev': 5.9, 'Stefanos Tsitsipas': 4.8, 'Karen Khachanov': 1.4
    }
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Championship probability by nationality
    countries = list(set(player_nationalities.values()))
    country_probs = {}
    
    for country in countries:
        total_prob = sum(player_probs[player] for player, nat in player_nationalities.items() 
                        if nat == country and player in player_probs)
        country_probs[country] = total_prob
    
    sorted_countries = sorted(country_probs.items(), key=lambda x: x[1], reverse=True)
    countries_sorted, probs_sorted = zip(*sorted_countries)
    
    bars1 = ax1.bar(countries_sorted[:8], probs_sorted[:8], alpha=0.7)
    ax1.set_ylabel('Total Championship Probability (%)')
    ax1.set_title('CHAMPIONSHIP ODDS BY NATIONALITY', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, prob in zip(bars1, probs_sorted[:8]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Regional dominance
    region_probs = {}
    for region, countries_in_region in regions.items():
        total = sum(player_probs.get(player, 0) for player, country in player_nationalities.items() 
                   if country in countries_in_region)
        region_probs[region] = total
    
    # Add Russia separately
    region_probs['Russia'] = sum(player_probs.get(player, 0) for player, country in player_nationalities.items() 
                                if country == 'Russia')
    
    wedges, texts, autotexts = ax2.pie(region_probs.values(), labels=region_probs.keys(), 
                                      autopct='%1.1f%%', startangle=90)
    ax2.set_title('REGIONAL CHAMPIONSHIP DISTRIBUTION', fontweight='bold')
    
    # Plot 3: Age vs Nationality performance
    # Simulate age data
    player_ages = {player: np.random.randint(20, 35) for player in player_probs.keys()}
    
    for player, age in player_ages.items():
        country = player_nationalities.get(player, 'Unknown')
        prob = player_probs.get(player, 0)
        color = {'Serbia': 'red', 'Italy': 'green', 'Spain': 'orange', 'Russia': 'purple',
                'Germany': 'brown', 'USA': 'blue'}.get(country, 'gray')
        
        ax3.scatter(age, prob, s=100, alpha=0.7, c=color, label=country)
    
    # Remove duplicate labels
    handles, labels = ax3.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax3.legend(by_label.values(), by_label.keys(), title='Country')
    
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Championship Probability (%)')
    ax3.set_title('AGE vs PERFORMANCE BY NATIONALITY', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: European dominance timeline (simulated)
    years = list(range(2019, 2025))
    euro_dominance = [65, 70, 62, 68, 72, 75]  # Percentage of top 10 that are European
    
    ax4.plot(years, euro_dominance, marker='o', linewidth=3, markersize=8, color='blue')
    ax4.fill_between(years, euro_dominance, alpha=0.3, color='blue')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('European Players in Top 10 (%)')
    ax4.set_title('EUROPEAN TENNIS DOMINANCE TREND', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(50, 80)
    
    plt.tight_layout()
    plt.savefig('nationality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 5. SERVE ANALYSIS DEEP DIVE
# =============================================================================
def create_serve_analysis():
    """Detailed analysis of serving statistics and their impact"""
    
    # Sample serving data (would come from your actual serve/return aggregates)
    serve_data = {
        'Player': ['Djokovic', 'Sinner', 'Alcaraz', 'Medvedev', 'Zverev', 'Fritz'],
        'Ace_Rate': [8.2, 6.8, 7.9, 5.4, 12.1, 9.3],
        'Double_Fault_Rate': [2.1, 1.8, 2.4, 2.9, 3.2, 2.2],
        'First_Serve_Pct': [64, 68, 62, 61, 59, 65],
        'First_Serve_Won': [77, 75, 79, 73, 81, 76],
        'Second_Serve_Won': [58, 56, 61, 54, 62, 55],
        'Break_Points_Saved': [68, 65, 70, 62, 72, 64],
        'Championship_Prob': [17.7, 10.5, 9.5, 9.6, 5.2, 2.6]
    }
    
    df_serve = pd.DataFrame(serve_data)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Plot 1: Service correlation matrix
    ax1 = fig.add_subplot(gs[0, :2])
    serve_cols = ['Ace_Rate', 'Double_Fault_Rate', 'First_Serve_Pct', 'First_Serve_Won', 
                 'Second_Serve_Won', 'Break_Points_Saved', 'Championship_Prob']
    corr_matrix = df_serve[serve_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax1)
    ax1.set_title('SERVING STATISTICS CORRELATION MATRIX', fontweight='bold')
    
    # Plot 2: Ace rate vs Championship probability
    ax2 = fig.add_subplot(gs[0, 2])
    scatter2 = ax2.scatter(df_serve['Ace_Rate'], df_serve['Championship_Prob'], 
                          s=100, alpha=0.7, c=df_serve['First_Serve_Won'], cmap='viridis')
    
    for i, player in enumerate(df_serve['Player']):
        ax2.annotate(player, (df_serve['Ace_Rate'].iloc[i], df_serve['Championship_Prob'].iloc[i]),
                    xytext=(3, 3), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Aces per Game')
    ax2.set_ylabel('Championship Probability (%)')
    ax2.set_title('POWER vs SUCCESS', fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, label='1st Serve Won %')
    
    # Plot 3: Service efficiency radar
    ax3 = fig.add_subplot(gs[0, 3], projection='polar')
    
    # Normalize data for radar chart
    categories = ['Aces', '1st Serve%', '1st Won', '2nd Won', 'BP Saved']
    
    # Djokovic vs Zverev comparison (power vs consistency)
    djok_data = [8.2/15*100, 64, 77, 58, 68]  # Normalize aces to 0-100 scale
    zver_data = [12.1/15*100, 59, 81, 62, 72]
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    djok_data += djok_data[:1]
    zver_data += zver_data[:1]
    
    ax3.plot(angles, djok_data, 'o-', linewidth=2, label='Djokovic', color='red')
    ax3.fill(angles, djok_data, alpha=0.25, color='red')
    ax3.plot(angles, zver_data, 'o-', linewidth=2, label='Zverev', color='blue')
    ax3.fill(angles, zver_data, alpha=0.25, color='blue')
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_ylim(0, 100)
    ax3.set_title('SERVE STYLE COMPARISON', fontweight='bold')
    ax3.legend()
    
    # Plot 4: Service pressure situations
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Create pressure vs non-pressure serve stats
    pressure_multiplier = np.random.uniform(0.85, 0.95, len(df_serve))
    
    x = np.arange(len(df_serve['Player']))
    width = 0.35
    
    regular_serve = df_serve['First_Serve_Won']
    pressure_serve = regular_serve * pressure_multiplier
    
    bars1 = ax4.bar(x - width/2, regular_serve, width, label='Regular Situations', alpha=0.7)
    bars2 = ax4.bar(x + width/2, pressure_serve, width, label='Pressure Situations', alpha=0.7)
    
    ax4.set_xlabel('Player')
    ax4.set_ylabel('First Serve Won (%)')
    ax4.set_title('SERVING UNDER PRESSURE', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df_serve['Player'], rotation=45)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Service game hold percentage prediction
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Calculate estimated service game hold % from serve stats
    hold_pct = (df_serve['First_Serve_Pct'] * df_serve['First_Serve_Won'] / 100 + 
               (100 - df_serve['First_Serve_Pct']) * df_serve['Second_Serve_Won'] / 100)
    
    bars5 = ax5.bar(df_serve['Player'], hold_pct, 
                   color=[plt.cm.RdYlGn(x/100) for x in hold_pct], alpha=0.8)
    
    for bar, pct in zip(bars5, hold_pct):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax5.set_ylabel('Estimated Hold %')
    ax5.set_title('SERVICE GAME DOMINANCE', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    
    # Plot 6: Double fault impact
    ax6 = fig.add_subplot(gs[1, 3])
    
    # Show relationship between double faults and championship odds
    ax6.scatter(df_serve['Double_Fault_Rate'], df_serve['Championship_Prob'], 
               s=df_serve['Break_Points_Saved']*2, alpha=0.7, c='orange')
    
    for i, player in enumerate(df_serve['Player']):
        ax6.annotate(player, (df_serve['Double_Fault_Rate'].iloc[i], 
                             df_serve['Championship_Prob'].iloc[i]),
                    xytext=(3, 3), textcoords='offset points', fontsize=9)
    
    # Add trendline
    z = np.polyfit(df_serve['Double_Fault_Rate'], df_serve['Championship_Prob'], 1)
    p = np.poly1d(z)
    ax6.plot(df_serve['Double_Fault_Rate'], p(df_serve['Double_Fault_Rate']), "r--", alpha=0.8)
    
    ax6.set_xlabel('Double Faults per Game')
    ax6.set_ylabel('Championship Probability (%)')
    ax6.set_title('UNFORCED ERRORS COST', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Bottom row: Combined serving efficiency score
    ax7 = fig.add_subplot(gs[2, :])
    
    # Create composite serving score
    serve_score = (df_serve['Ace_Rate'] * 2 + 
                  df_serve['First_Serve_Won'] + 
                  df_serve['Second_Serve_Won'] + 
                  df_serve['Break_Points_Saved'] - 
                  df_serve['Double_Fault_Rate'] * 5)
    
    # Sort by serve score
    sorted_indices = serve_score.argsort()[::-1]
    
    bars7 = ax7.barh(range(len(df_serve)), serve_score.iloc[sorted_indices], 
                    color=plt.cm.plasma(np.arange(len(df_serve)) / len(df_serve)))
    
    ax7.set_yticks(range(len(df_serve)))
    ax7.set_yticklabels(df_serve['Player'].iloc[sorted_indices])
    ax7.set_xlabel('Composite Serving Score')
    ax7.set_title('OVERALL SERVING EXCELLENCE RANKING', fontweight='bold')
    ax7.grid(axis='x', alpha=0.3)
    
    # Add championship probability as secondary info
    for i, idx in enumerate(sorted_indices):
        prob = df_serve['Championship_Prob'].iloc[idx]
        ax7.text(serve_score.iloc[idx] + 2, i, f'({prob:.1f}%)', 
                va='center', fontsize=10, alpha=0.7)
    
    plt.savefig('serve_analysis_deep_dive.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 6. INTERACTIVE TOURNAMENT SIMULATOR
# =============================================================================
def create_interactive_simulator():
    """Create interactive tournament outcome simulator"""
    
    # This would integrate with your actual simulation function
    fig = go.Figure()
    
    # Simulation results data
    simulation_runs = range(1, 101)  # 100 simulation runs
    djokovic_wins = np.cumsum(np.random.binomial(1, 0.177, 100))
    sinner_wins = np.cumsum(np.random.binomial(1, 0.105, 100))
    alcaraz_wins = np.cumsum(np.random.binomial(1, 0.095, 100))
    
    fig.add_trace(go.Scatter(x=list(simulation_runs), y=djokovic_wins,
                            mode='lines', name='Djokovic Wins',
                            line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=list(simulation_runs), y=sinner_wins,
                            mode='lines', name='Sinner Wins',
                            line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=list(simulation_runs), y=alcaraz_wins,
                            mode='lines', name='Alcaraz Wins',
                            line=dict(color='green', width=3)))
    
    fig.update_layout(title='TOURNAMENT SIMULATION CONVERGENCE<br>Cumulative Wins Over 100 Simulations',
                     xaxis_title='Simulation Number',
                     yaxis_title='Cumulative Championships Won',
                     hovermode='x unified')
    
    fig.show()
    
    return fig

# =============================================================================
# 7. PLAYING STYLE CLUSTER ANALYSIS
# =============================================================================
def create_style_clusters():
    """Cluster players by playing style and analyze success patterns"""
    
    # Simulate playing style data
    players = ['Djokovic', 'Sinner', 'Alcaraz', 'Medvedev', 'Zverev', 'Fritz', 
               'Draper', 'De Minaur', 'Ruud', 'Rublev', 'Tsitsipas']
    
    # Style dimensions (0-100 scale)
    style_data = {
        'Aggression': [70, 85, 90, 60, 80, 75, 85, 80, 50, 90, 75],
        'Defense': [95, 70, 70, 90, 60, 55, 60, 85, 85, 65, 70],
        'Net_Play': [60, 40, 70, 30, 45, 50, 55, 35, 40, 35, 80],
        'Power': [80, 85, 90, 70, 95, 90, 80, 65, 70, 85, 75],
        'Consistency': [95, 80, 75, 85, 70, 70, 65, 90, 90, 70, 80],
        'Championship_Prob': [17.7, 10.5, 9.5, 9.6, 5.2, 2.6, 2.6, 2.7, 3.6, 5.9, 4.8]
    }
    
    df_style = pd.DataFrame(style_data, index=players)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Cluster analysis using style dimensions
    from sklearn.cluster import KMeans
    
    X = df_style[['Aggression', 'Defense', 'Power', 'Consistency']].values
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Plot 1: Aggression vs Defense with clusters
    scatter1 = axes[0,0].scatter(df_style['Aggression'], df_style['Defense'], 
                                c=clusters, s=df_style['Championship_Prob']*10, 
                                cmap='viridis', alpha=0.7)
    
    for i, player in enumerate(players):
        axes[0,0].annotate(player, (df_style['Aggression'][i], df_style['Defense'][i]),
                          xytext=(3, 3), textcoords='offset points', fontsize=9)
    
    axes[0,0].set_xlabel('Aggression Level')
    axes[0,0].set_ylabel('Defensive Ability')
    axes[0,0].set_title('PLAYING STYLE CLUSTERS\n(Size = Championship Probability)')
    
    # Plot 2: Power vs Consistency
    scatter2 = axes[0,1].scatter(df_style['Power'], df_style['Consistency'],
                                c=df_style['Championship_Prob'], s=100,
                                cmap='Reds', alpha=0.7)
    
    for i, player in enumerate(players):
        axes[0,1].annotate(player, (df_style['Power'][i], df_style['Consistency'][i]),
                          xytext=(3, 3), textcoords='offset points', fontsize=9)
    
    axes[0,1].set_xlabel('Power Level')
    axes[0,1].set_ylabel('Consistency Level')
    axes[0,1].set_title('POWER vs CONSISTENCY')
    plt.colorbar(scatter2, ax=axes[0,1], label='Championship %')
    
    # Plot 3: Style profile radar for top 3
    ax3 = axes[1,0]
    ax3.remove()
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')
    
    categories = ['Aggression', 'Defense', 'Net Play', 'Power', 'Consistency']
    top_3 = ['Djokovic', 'Sinner', 'Alcaraz']
    colors = ['red', 'blue', 'green']
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    for player, color in zip(top_3, colors):
        values = [df_style.loc[player, cat] for cat in categories]
        values += values[:1]
        
        ax3.plot(angles, values, 'o-', linewidth=2, label=player, color=color)
        ax3.fill(angles, values, alpha=0.25, color=color)
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_ylim(0, 100)
    ax3.set_title('TOP 3 STYLE PROFILES', fontweight='bold')
    ax3.legend()
    
    # Plot 4: Success by playing style
    style_success = {}
    cluster_names = {0: 'Aggressive Baseliners', 1: 'Defensive Specialists', 2: 'Power Players'}
    
    for cluster_id in range(3):
        cluster_players = [players[i] for i in range(len(players)) if clusters[i] == cluster_id]
        avg_prob = np.mean([df_style.loc[p, 'Championship_Prob'] for p in cluster_players])
        style_success[cluster_names[cluster_id]] = avg_prob
    
    bars4 = axes[1,1].bar(style_success.keys(), style_success.values(), 
                         color=['orange', 'blue', 'red'], alpha=0.7)
    
    for bar, prob in zip(bars4, style_success.values()):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    axes[1,1].set_ylabel('Average Championship Probability (%)')
    axes[1,1].set_title('SUCCESS BY PLAYING STYLE')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('playing_style_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("ADDITIONAL TENNIS PREDICTION VISUALIZATIONS")
    print("="*70)
    
    visualizations = [
        ("Head-to-Head Network", create_h2h_network),
        ("Elo Evolution", create_elo_evolution), 
        ("Risk-Reward Analysis", create_risk_reward_analysis),
        ("Nationality Analysis", create_nationality_analysis),
        ("Serve Analysis Deep Dive", create_serve_analysis),
        ("Playing Style Clusters", create_style_clusters)
    ]
    
    for name, func in visualizations:
        print(f"\nCreating {name}...")
        try:
            func()
            print(f"✓ {name} completed")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    print("\n" + "="*70)
    print("ADDITIONAL VISUALIZATION IDEAS TO IMPLEMENT:")
    print("="*70)
    print("• Match-by-match probability heatmap throughout tournament")
    print("• Player fatigue simulation (performance decline over rounds)")
    print("• Weather impact analysis (indoor/outdoor conditions)")
    print("• Injury risk assessment based on recent matches")
    print("• Court speed impact on different playing styles") 
    print("• Time of day performance variations")
    print("• Crowd support factor (home vs away)")
    print("• Surface transition analysis (clay → hard court)")
    print("• Mental toughness indicators (comeback statistics)")
    print("• Physical conditioning metrics (stamina in long matches)")
    print("="*70)
    
    
    #!/usr/bin/env python3
"""
Advanced Tennis Analytics Suite
10 specialized visualizations for tournament prediction enhancement
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Core player data
PLAYERS = ['Djokovic', 'Sinner', 'Alcaraz', 'Medvedev', 'Zverev', 'Fritz', 'Draper', 'Shelton']
CHAMPIONSHIP_PROBS = [17.7, 10.5, 9.5, 9.6, 5.2, 2.6, 2.6, 0.6]

print("="*80)
print("ADVANCED TENNIS ANALYTICS SUITE")
print("10 Specialized Tournament Prediction Visualizations")
print("="*80)

# =============================================================================
# 1. MATCH-BY-MATCH PROBABILITY HEATMAP
# =============================================================================
def create_probability_heatmap():
    """Tournament progression probability heatmap"""
    print("\n1. Creating Match-by-Match Probability Heatmap...")
    
    # Simulate tournament progression probabilities
    rounds = ['R1', 'R2', 'R3', 'R4', 'QF', 'SF', 'F']
    n_players = len(PLAYERS)
    
    # Create probability matrix (players × rounds)
    prob_matrix = np.zeros((n_players, len(rounds)))
    
    for i, base_prob in enumerate(CHAMPIONSHIP_PROBS):
        # Calculate round-by-round survival probabilities
        base_survival = (base_prob / 100) ** (1/7)  # 7th root for 7 rounds
        
        for j, round_name in enumerate(rounds):
            # Adjust probability based on round difficulty
            round_multipliers = [0.95, 0.88, 0.75, 0.65, 0.55, 0.45, 0.35]
            adjusted_prob = min(0.98, base_survival + (base_prob/100) * round_multipliers[j])
            
            # Cumulative probability through this round
            cumulative_prob = adjusted_prob ** (j + 1)
            prob_matrix[i, j] = cumulative_prob * 100
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Main heatmap
    im1 = ax1.imshow(prob_matrix, cmap='RdYlBu_r', aspect='auto', interpolation='bilinear')
    ax1.set_xticks(range(len(rounds)))
    ax1.set_xticklabels(rounds)
    ax1.set_yticks(range(n_players))
    ax1.set_yticklabels(PLAYERS)
    ax1.set_title('TOURNAMENT PROGRESSION PROBABILITY HEATMAP\n(% chance of reaching each round)', 
                  fontweight='bold', fontsize=14)
    
    # Add probability values
    for i in range(n_players):
        for j in range(len(rounds)):
            text = ax1.text(j, i, f'{prob_matrix[i, j]:.1f}%', 
                           ha="center", va="center", color="black" if prob_matrix[i, j] < 50 else "white",
                           fontweight='bold', fontsize=9)
    
    plt.colorbar(im1, ax=ax1, label='Probability (%)')
    
    # Round difficulty analysis
    round_avg_prob = np.mean(prob_matrix, axis=0)
    round_std = np.std(prob_matrix, axis=0)
    
    ax2.bar(rounds, round_avg_prob, yerr=round_std, capsize=5, alpha=0.7, color='skyblue')
    ax2.set_ylabel('Average Advancement Probability (%)')
    ax2.set_title('ROUND DIFFICULTY ANALYSIS\nAverage probability ± standard deviation', 
                  fontweight='bold', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (avg, std) in enumerate(zip(round_avg_prob, round_std)):
        ax2.text(i, avg + std + 2, f'{avg:.1f}±{std:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('probability_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 2. PLAYER FATIGUE SIMULATION
# =============================================================================
def create_fatigue_simulation():
    """Simulate performance decline due to fatigue"""
    print("\n2. Creating Player Fatigue Simulation...")
    
    # Fatigue parameters by player type
    fatigue_profiles = {
        'Djokovic': {'base_stamina': 95, 'recovery': 0.92, 'age_factor': 0.88},  # Older but experienced
        'Sinner': {'base_stamina': 88, 'recovery': 0.95, 'age_factor': 0.98},   # Young, good recovery
        'Alcaraz': {'base_stamina': 92, 'recovery': 0.94, 'age_factor': 0.97},  # Young, aggressive style
        'Medvedev': {'base_stamina': 85, 'recovery': 0.90, 'age_factor': 0.93}, # Tall, more fatigue
        'Zverev': {'base_stamina': 87, 'recovery': 0.91, 'age_factor': 0.94},   # Tall, recent injuries
        'Fritz': {'base_stamina': 82, 'recovery': 0.89, 'age_factor': 0.95},    # Power player
        'Draper': {'base_stamina': 80, 'recovery': 0.93, 'age_factor': 0.99},   # Young but injury-prone
        'Shelton': {'base_stamina': 84, 'recovery': 0.92, 'age_factor': 0.99}   # Young power player
    }
    
    rounds = ['R1', 'R2', 'R3', 'R4', 'QF', 'SF', 'F']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot 1: Performance decline through tournament
    for i, player in enumerate(PLAYERS):
        profile = fatigue_profiles[player]
        performance = [100]  # Start at 100%
        
        for round_num in range(1, len(rounds)):
            # Simulate match difficulty and duration
            match_difficulty = np.random.uniform(0.7, 1.3)
            match_duration = np.random.uniform(1.5, 4.0)  # Hours
            
            # Calculate fatigue impact
            fatigue_impact = (match_difficulty * match_duration) / profile['base_stamina'] * 100
            recovery_factor = profile['recovery'] ** round_num
            age_impact = profile['age_factor']
            
            # New performance level
            new_performance = performance[-1] * recovery_factor * age_impact - fatigue_impact
            performance.append(max(60, new_performance))  # Floor at 60%
        
        ax1.plot(rounds, performance, marker='o', linewidth=2, label=player, alpha=0.8)
    
    ax1.set_ylabel('Performance Level (%)')
    ax1.set_title('FATIGUE SIMULATION: Performance Decline Through Tournament', 
                  fontweight='bold', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(60, 105)
    
    # Plot 2: Recovery ability comparison
    recovery_rates = [fatigue_profiles[p]['recovery'] for p in PLAYERS]
    stamina_levels = [fatigue_profiles[p]['base_stamina'] for p in PLAYERS]
    
    scatter = ax2.scatter(recovery_rates, stamina_levels, s=np.array(CHAMPIONSHIP_PROBS)*10, 
                         alpha=0.7, c=range(len(PLAYERS)), cmap='viridis')
    
    for i, player in enumerate(PLAYERS):
        ax2.annotate(player, (recovery_rates[i], stamina_levels[i]),
                    xytext=(3, 3), textcoords='offset points', fontsize=10)
    
    ax2.set_xlabel('Recovery Rate (per round)')
    ax2.set_ylabel('Base Stamina Level')
    ax2.set_title('RECOVERY vs STAMINA\n(Bubble size = Championship probability)', 
                  fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Match duration impact
    match_durations = np.linspace(1.5, 5.0, 50)
    
    for player in ['Djokovic', 'Alcaraz', 'Medvedev']:
        profile = fatigue_profiles[player]
        fatigue_impact = []
        
        for duration in match_durations:
            impact = (duration / profile['base_stamina']) * 100 * (1 - profile['recovery'])
            fatigue_impact.append(impact)
        
        ax3.plot(match_durations, fatigue_impact, linewidth=2, label=player)
    
    ax3.set_xlabel('Match Duration (hours)')
    ax3.set_ylabel('Fatigue Impact (%)')
    ax3.set_title('MATCH DURATION IMPACT ON FATIGUE\nTop 3 Contenders', 
                  fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Age factor analysis
    ages = [37, 23, 21, 28, 27, 27, 22, 21]  # Approximate current ages
    age_factors = [fatigue_profiles[p]['age_factor'] for p in PLAYERS]
    
    ax4.scatter(ages, age_factors, s=np.array(CHAMPIONSHIP_PROBS)*15, 
               alpha=0.7, c='orange', edgecolors='black')
    
    for i, player in enumerate(PLAYERS):
        ax4.annotate(player, (ages[i], age_factors[i]),
                    xytext=(3, 3), textcoords='offset points', fontsize=10)
    
    # Add trendline
    z = np.polyfit(ages, age_factors, 1)
    p = np.poly1d(z)
    ax4.plot(ages, p(ages), "r--", alpha=0.8, linewidth=2)
    
    ax4.set_xlabel('Player Age')
    ax4.set_ylabel('Age Impact Factor')
    ax4.set_title('AGE vs FATIGUE RESISTANCE\n(Size = Championship probability)', 
                  fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fatigue_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 3. WEATHER IMPACT ANALYSIS
# =============================================================================
def create_weather_analysis():
    """Analyze weather and playing condition impacts"""
    print("\n3. Creating Weather Impact Analysis...")
    
    # Weather condition performance adjustments
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
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Weather impact heatmap
    impact_matrix = np.array([[weather_impacts[cond][player] for cond in conditions] 
                             for player in PLAYERS])
    
    im = ax1.imshow(impact_matrix, cmap='RdYlBu_r', aspect='auto')
    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels(conditions)
    ax1.set_yticks(range(len(PLAYERS)))
    ax1.set_yticklabels(PLAYERS)
    ax1.set_title('WEATHER CONDITION IMPACT MATRIX\n(Performance multiplier)', 
                  fontweight='bold')
    
    # Add values to heatmap
    for i in range(len(PLAYERS)):
        for j in range(len(conditions)):
            color = "white" if impact_matrix[i, j] > 1.0 else "black"
            ax1.text(j, i, f'{impact_matrix[i, j]:.2f}', 
                    ha="center", va="center", color=color, fontweight='bold')
    
    plt.colorbar(im, ax=ax1, label='Performance Multiplier')
    
    # Plot 2: Best/worst weather performers
    weather_variance = np.var(impact_matrix, axis=1)
    weather_mean = np.mean(impact_matrix, axis=1)
    
    scatter = ax2.scatter(weather_variance, weather_mean, s=np.array(CHAMPIONSHIP_PROBS)*12,
                         alpha=0.7, c=range(len(PLAYERS)), cmap='plasma')
    
    for i, player in enumerate(PLAYERS):
        ax2.annotate(player, (weather_variance[i], weather_mean[i]),
                    xytext=(3, 3), textcoords='offset points')
    
    ax2.set_xlabel('Weather Sensitivity (Variance)')
    ax2.set_ylabel('Average Weather Performance')
    ax2.set_title('WEATHER ADAPTABILITY\nHigh variance = weather-sensitive', 
                  fontweight='bold')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Neutral')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: US Open weather probability simulation
    # Simulate weather probabilities for US Open dates
    us_open_weather = {
        'Indoor': 0.15,  # Rain delays, roof closed
        'Sunny': 0.40,   # Ideal conditions
        'Windy': 0.20,   # Flushing Meadows is windy
        'Hot': 0.35,     # Late August/early September heat
        'Humid': 0.45    # NYC humidity
    }
    
    # Calculate expected performance under US Open conditions
    expected_performance = []
    for player in PLAYERS:
        exp_perf = sum(us_open_weather[cond] * weather_impacts[cond][player] 
                      for cond in conditions)
        expected_performance.append(exp_perf)
    
    bars = ax3.bar(PLAYERS, expected_performance, alpha=0.7, 
                  color=['red' if p > 1.02 else 'orange' if p > 1.0 else 'lightblue' 
                        for p in expected_performance])
    
    # Add value labels
    for bar, perf in zip(bars, expected_performance):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{perf:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.axhline(y=1.0, color='black', linestyle='-', alpha=0.5)
    ax3.set_ylabel('Expected Performance Multiplier')
    ax3.set_title('EXPECTED US OPEN WEATHER PERFORMANCE\nBased on historical weather patterns', 
                  fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Temperature performance curves
    temperatures = np.arange(15, 40, 1)  # Celsius
    
    for player in ['Djokovic', 'Medvedev', 'Alcaraz']:
        base_hot = weather_impacts['Hot'][player]
        
        # Create temperature response curve
        optimal_temp = 22 if player == 'Djokovic' else 18 if player == 'Medvedev' else 25
        temp_performance = []
        
        for temp in temperatures:
            # Gaussian-like response around optimal temperature
            temp_factor = np.exp(-0.005 * (temp - optimal_temp)**2)
            adjusted_perf = 0.85 + 0.3 * temp_factor
            temp_performance.append(adjusted_perf)
        
        ax4.plot(temperatures, temp_performance, linewidth=2, marker='o', 
                markersize=4, label=player, alpha=0.8)
    
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Relative Performance')
    ax4.set_title('TEMPERATURE PERFORMANCE CURVES\nTop 3 Contenders', 
                  fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axvspan(25, 32, alpha=0.2, color='red', label='Typical US Open range')
    
    plt.tight_layout()
    plt.savefig('weather_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 4. INJURY RISK ASSESSMENT
# =============================================================================
def create_injury_risk_assessment():
    """Assess injury risk based on recent match load and history"""
    print("\n4. Creating Injury Risk Assessment...")
    
    # Injury risk factors (simulated based on known player histories)
    injury_data = {
        'Player': PLAYERS,
        'Recent_Matches': [28, 35, 40, 30, 22, 25, 18, 20],  # Matches in last 6 months
        'Injury_History': [6, 2, 3, 4, 8, 3, 9, 1],  # Career significant injuries
        'Age_Factor': [37, 23, 21, 28, 27, 27, 22, 21],
        'Playing_Style_Risk': [3, 6, 8, 4, 5, 7, 6, 8],  # 1-10 scale, aggressive = higher
        'Recovery_Time': [48, 36, 40, 44, 52, 42, 60, 38],  # Hours between matches needed
        'Championship_Prob': CHAMPIONSHIP_PROBS
    }
    
    df_injury = pd.DataFrame(injury_data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Injury risk composite score
    # Calculate composite risk score
    df_injury['Age_Risk'] = (df_injury['Age_Factor'] - 20) / 20 * 100
    df_injury['Workload_Risk'] = (df_injury['Recent_Matches'] / 50) * 100
    df_injury['History_Risk'] = (df_injury['Injury_History'] / 10) * 100
    df_injury['Style_Risk'] = df_injury['Playing_Style_Risk'] * 10
    
    df_injury['Composite_Risk'] = (df_injury['Age_Risk'] * 0.25 + 
                                  df_injury['Workload_Risk'] * 0.3 + 
                                  df_injury['History_Risk'] * 0.25 + 
                                  df_injury['Style_Risk'] * 0.2)
    
    # Sort by risk
    df_sorted = df_injury.sort_values('Composite_Risk', ascending=False)
    
    colors = ['red' if risk > 60 else 'orange' if risk > 40 else 'green' 
             for risk in df_sorted['Composite_Risk']]
    
    bars = ax1.barh(df_sorted['Player'], df_sorted['Composite_Risk'], 
                   color=colors, alpha=0.7)
    
    for bar, risk in zip(bars, df_sorted['Composite_Risk']):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{risk:.1f}', va='center', fontweight='bold')
    
    ax1.set_xlabel('Composite Injury Risk Score')
    ax1.set_title('TOURNAMENT INJURY RISK ASSESSMENT\nRed=High, Orange=Medium, Green=Low', 
                  fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Workload vs Performance
    scatter = ax2.scatter(df_injury['Recent_Matches'], df_injury['Championship_Prob'],
                         s=df_injury['Composite_Risk']*3, alpha=0.7,
                         c=df_injury['Age_Factor'], cmap='viridis')
    
    for i, player in enumerate(df_injury['Player']):
        ax2.annotate(player, (df_injury['Recent_Matches'].iloc[i], 
                             df_injury['Championship_Prob'].iloc[i]),
                    xytext=(3, 3), textcoords='offset points')
    
    ax2.set_xlabel('Recent Matches Played (6 months)')
    ax2.set_ylabel('Championship Probability (%)')
    ax2.set_title('WORKLOAD vs PERFORMANCE\nBubble size = Injury risk, Color = Age', 
                  fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Age')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Recovery time analysis
    ax3.scatter(df_injury['Recovery_Time'], df_injury['Championship_Prob'],
               s=150, alpha=0.7, c=df_injury['Composite_Risk'], cmap='Reds')
    
    for i, player in enumerate(df_injury['Player']):
        ax3.annotate(player, (df_injury['Recovery_Time'].iloc[i],
                             df_injury['Championship_Prob'].iloc[i]),
                    xytext=(3, 3), textcoords='offset points')
    
    # Add trendline
    z = np.polyfit(df_injury['Recovery_Time'], df_injury['Championship_Prob'], 1)
    p = np.poly1d(z)
    ax3.plot(df_injury['Recovery_Time'], p(df_injury['Recovery_Time']), 
            "r--", alpha=0.8, linewidth=2)
    
    ax3.set_xlabel('Required Recovery Time (hours)')
    ax3.set_ylabel('Championship Probability (%)')
    ax3.set_title('RECOVERY NEEDS vs SUCCESS\nColor intensity = Injury risk', 
                  fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Risk mitigation strategies
    risk_categories = ['Age\nFactor', 'Match\nLoad', 'Injury\nHistory', 'Playing\nStyle']
    risk_weights = [0.25, 0.30, 0.25, 0.20]
    
    # Show risk breakdown for top 3 players by championship probability
    top_3 = df_injury.nlargest(3, 'Championship_Prob')
    
    x = np.arange(len(risk_categories))
    width = 0.25
    
    for i, (_, player_row) in enumerate(top_3.iterrows()):
        player_risks = [player_row['Age_Risk'], player_row['Workload_Risk'],
                       player_row['History_Risk'], player_row['Style_Risk']]
        
        ax4.bar(x + i*width, player_risks, width, label=player_row['Player'],
               alpha=0.7)
    
    ax4.set_xlabel('Risk Categories')
    ax4.set_ylabel('Risk Score')
    ax4.set_title('RISK BREAKDOWN: TOP 3 CONTENDERS\nHigher bars = higher risk', 
                  fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(risk_categories)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('injury_risk_assessment.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 5. COURT SPEED IMPACT
# =============================================================================
def create_court_speed_analysis():
    """Analyze how court speed affects different playing styles"""
    print("\n5. Creating Court Speed Impact Analysis...")
    
    # Court speed ratings (1-10 scale, 10 = fastest)
    court_speeds = {
        'US Open': 6.5,  # Medium-fast hard court
        'Australian Open': 6.0,  # Medium hard court
        'Wimbledon': 8.5,  # Fast grass
        'French Open': 3.0,  # Slow clay
        'Indian Wells': 5.5,  # Slower hard court
        'Miami': 7.0,  # Faster hard court
        'Cincinnati': 7.5   # Fast hard court
    }
    
    # Player performance by court speed (performance multiplier)
    speed_performance = {
        'Djokovic': {1: 0.95, 3: 1.10, 5: 1.08, 6.5: 1.05, 8: 1.00, 10: 0.98},
        'Sinner': {1: 0.85, 3: 0.95, 5: 1.05, 6.5: 1.08, 8: 1.12, 10: 1.15},
        'Alcaraz': {1: 0.90, 3: 1.15, 5: 1.10, 6.5: 1.05, 8: 1.02, 10: 1.00},
        'Medvedev': {1: 0.80, 3: 0.85, 5: 0.98, 6.5: 1.05, 8: 1.15, 10: 1.20},
        'Zverev': {1: 0.88, 3: 0.92, 5: 1.02, 6.5: 1.08, 8: 1.12, 10: 1.18},
        'Fritz': {1: 0.82, 3: 0.88, 5: 1.00, 6.5: 1.10, 8: 1.15, 10: 1.20},
        'Draper': {1: 0.85, 3: 0.90, 5: 1.02, 6.5: 1.08, 8: 1.15, 10: 1.18},
        'Shelton': {1: 0.78, 3: 0.82, 5: 0.95, 6.5: 1.05, 8: 1.18, 10: 1.25}
    }
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Performance curves by court speed
    speeds = np.linspace(1, 10, 50)
    
    for player in PLAYERS[:4]:  # Top 4 players
        performance_curve = []
        for speed in speeds:
            # Interpolate performance based on known points
            perf_points = list(speed_performance[player].keys())
            perf_values = list(speed_performance[player].values())
            
            # Linear interpolation
            interp_perf = np.interp(speed, perf_points, perf_values)
            performance_curve.append(interp_perf)
        
        ax1.plot(speeds, performance_curve, linewidth=3, marker='o', 
                markersize=4, label=player, alpha=0.8)
    
    # Mark US Open speed
    ax1.axvline(x=court_speeds['US Open'], color='red', linestyle='--', 
               alpha=0.7, linewidth=2, label='US Open Speed')
    
    ax1.set_xlabel('Court Speed Rating (1=Very Slow, 10=Very Fast)')
    ax1.set_ylabel('Performance Multiplier')
    ax1.set_title('COURT SPEED PERFORMANCE CURVES\nTop 4 Championship Contenders', 
                  fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.75, 1.3)
    
    # Plot 2: US Open speed optimization
    us_open_speed = court_speeds['US Open']
    us_open_performance = []
    
    for player in PLAYERS:
        # Get performance at US Open speed
        perf_points = list(speed_performance[player].keys())
        perf_values = list(speed_performance[player].values())
        us_perf = np.interp(us_open_speed, perf_points, perf_values)
        us_open_performance.append(us_perf)
    
    # Adjust championship probabilities by court speed
    adjusted_probs = np.array(CHAMPIONSHIP_PROBS) * np.array(us_open_performance)
    
    # Create comparison
    x = np.arange(len(PLAYERS))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, CHAMPIONSHIP_PROBS, width, 
                   label='Base Probability', alpha=0.7, color='lightblue')
    bars2 = ax2.bar(x + width/2, adjusted_probs, width, 
                   label='Speed-Adjusted', alpha=0.7, color='darkblue')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', 
                    fontsize=9)
    
    ax2.set_xlabel('Player')
    ax2.set_ylabel('Championship Probability (%)')
    ax2.set_title('US OPEN COURT SPEED ADJUSTMENT\nHow surface affects odds', 
                  fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(PLAYERS, rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Playing style vs optimal court speed
    playing_styles = {
        'Djokovic': {'baseline': 9, 'power': 7, 'defense': 10, 'net': 6},
        'Sinner': {'baseline': 8, 'power': 9, 'defense': 7, 'net': 5},
        'Alcaraz': {'baseline': 8, 'power': 9, 'defense': 7, 'net': 8},
        'Medvedev': {'baseline': 10, 'power': 6, 'defense': 9, 'net': 3},
        'Zverev': {'baseline': 8, 'power': 10, 'defense': 6, 'net': 4},
        'Fritz': {'baseline': 7, 'power': 10, 'defense': 5, 'net': 5},
        'Draper': {'baseline': 7, 'power': 8, 'defense': 6, 'net': 6},
        'Shelton': {'baseline': 6, 'power': 10, 'defense': 4, 'net': 7}
    }
    
    # Calculate optimal court speed for each player
    optimal_speeds = []
    power_ratings = []
    
    for player in PLAYERS:
        # Players with more power prefer faster courts
        power = playing_styles[player]['power']
        defense = playing_styles[player]['defense']
        
        # Optimal speed calculation
        optimal_speed = 3 + (power / 10) * 6 - (defense / 10) * 2
        optimal_speeds.append(optimal_speed)
        power_ratings.append(power)
    
    scatter = ax3.scatter(optimal_speeds, power_ratings, 
                         s=np.array(CHAMPIONSHIP_PROBS)*15, alpha=0.7,
                         c=range(len(PLAYERS)), cmap='plasma')
    
    for i, player in enumerate(PLAYERS):
        ax3.annotate(player, (optimal_speeds[i], power_ratings[i]),
                    xytext=(3, 3), textcoords='offset points')
    
    ax3.axvline(x=us_open_speed, color='red', linestyle='--', 
               alpha=0.7, linewidth=2, label='US Open Speed')
    
    ax3.set_xlabel('Optimal Court Speed')
    ax3.set_ylabel('Power Rating (1-10)')
    ax3.set_title('PLAYING STYLE vs PREFERRED COURT SPEED\nBubble size = Championship probability', 
                  fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Tournament comparison
    tournaments = list(court_speeds.keys())
    tournament_speeds = list(court_speeds.values())
    
    # Calculate average performance across all players for each tournament
    avg_performance = []
    for tournament, speed in court_speeds.items():
        total_perf = 0
        for player in PLAYERS:
            perf_points = list(speed_performance[player].keys())
            perf_values = list(speed_performance[player].values())
            player_perf = np.interp(speed, perf_points, perf_values)
            total_perf += player_perf
        avg_performance.append(total_perf / len(PLAYERS))
    
    colors = ['red' if t == 'US Open' else 'lightblue' for t in tournaments]
    bars = ax4.bar(tournaments, avg_performance, color=colors, alpha=0.7)
    
    for bar, perf in zip(bars, avg_performance):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{perf:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax4.axhline(y=1.0, color='black', linestyle='-', alpha=0.5)
    ax4.set_ylabel('Average Performance Multiplier')
    ax4.set_title('TOURNAMENT SURFACE COMPARISON\nAverage performance across all players', 
                  fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('court_speed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 6. TIME OF DAY PERFORMANCE
# =============================================================================
def create_time_analysis():
    """Analyze performance variations by time of day"""
    print("\n6. Creating Time of Day Performance Analysis...")
    
    # Time-based performance data (multipliers by hour)
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
    
    # Plot 1: Time performance curves
    hours = list(range(11, 22))
    
    for player in PLAYERS:
        performance_by_hour = []
        for hour in hours:
            if hour in time_performance[player]:
                performance_by_hour.append(time_performance[player][hour])
            else:
                # Interpolate missing hours
                available_hours = list(time_performance[player].keys())
                available_perfs = list(time_performance[player].values())
                interp_perf = np.interp(hour, available_hours, available_perfs)
                performance_by_hour.append(interp_perf)
        
        ax1.plot(hours, performance_by_hour, marker='o', linewidth=2, 
                label=player, alpha=0.8)
    
    ax1.axhspan(15, 19, alpha=0.1, color='yellow', label='Prime TV Hours')
    ax1.set_xlabel('Time of Day (24-hour format)')
    ax1.set_ylabel('Performance Multiplier')
    ax1.set_title('DAILY PERFORMANCE RHYTHMS\nAll Championship Contenders', 
                  fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(10.5, 21.5)
    
    # Plot 2: Peak performance times
    peak_times = []
    performance_ranges = []
    
    for player in PLAYERS:
        player_perfs = list(time_performance[player].values())
        peak_perf = max(player_perfs)
        min_perf = min(player_perfs)
        performance_range = peak_perf - min_perf
        
        # Find peak time
        for time, perf in time_performance[player].items():
            if perf == peak_perf:
                peak_times.append(time)
                break
        
        performance_ranges.append(performance_range)
    
    scatter = ax2.scatter(peak_times, performance_ranges, 
                         s=np.array(CHAMPIONSHIP_PROBS)*15, alpha=0.7,
                         c=range(len(PLAYERS)), cmap='viridis')
    
    for i, player in enumerate(PLAYERS):
        ax2.annotate(player, (peak_times[i], performance_ranges[i]),
                    xytext=(3, 3), textcoords='offset points')
    
    ax2.set_xlabel('Peak Performance Time (hour)')
    ax2.set_ylabel('Performance Range (max - min)')
    ax2.set_title('PEAK TIMES vs CONSISTENCY\nBubble size = Championship probability', 
                  fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(10, 22)
    
    # Plot 3: US Open scheduling impact
    # Typical US Open match times and their frequency
    us_open_times = {11: 0.05, 12: 0.10, 13: 0.15, 14: 0.15, 15: 0.20, 
                    16: 0.15, 17: 0.10, 19: 0.08, 20: 0.02}
    
    # Calculate expected performance for US Open scheduling
    expected_us_open_perf = []
    for player in PLAYERS:
        total_weighted_perf = 0
        for time, frequency in us_open_times.items():
            if time in time_performance[player]:
                player_perf = time_performance[player][time]
            else:
                # Interpolate
                available_hours = list(time_performance[player].keys())
                available_perfs = list(time_performance[player].values())
                player_perf = np.interp(time, available_hours, available_perfs)
            
            total_weighted_perf += player_perf * frequency
        
        expected_us_open_perf.append(total_weighted_perf)
    
    bars = ax3.bar(PLAYERS, expected_us_open_perf, alpha=0.7,
                  color=['darkgreen' if p > 1.03 else 'green' if p > 1.01 
                        else 'orange' if p > 0.99 else 'red' 
                        for p in expected_us_open_perf])
    
    for bar, perf in zip(bars, expected_us_open_perf):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{perf:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.axhline(y=1.0, color='black', linestyle='-', alpha=0.5)
    ax3.set_ylabel('Expected Performance Multiplier')
    ax3.set_title('US OPEN SCHEDULING ADVANTAGE\nBased on typical match times', 
                  fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Early vs Late match preference
    early_performance = []
    late_performance = []
    
    for player in PLAYERS:
        # Average of 11-15 hours (early)
        early_times = [11, 12, 13, 14, 15]
        early_avg = np.mean([time_performance[player].get(t, 1.0) for t in early_times])
        early_performance.append(early_avg)
        
        # Average of 17-21 hours (late)
        late_times = [17, 18, 19, 20, 21]
        late_avg = np.mean([time_performance[player].get(t, 1.0) for t in late_times])
        late_performance.append(late_avg)
    
    # Create preference categories
    preferences = []
    for early, late in zip(early_performance, late_performance):
        if early > late + 0.02:
            preferences.append('Early Bird')
        elif late > early + 0.02:
            preferences.append('Night Owl')
        else:
            preferences.append('Flexible')
    
    # Count preferences
    pref_counts = pd.Series(preferences).value_counts()
    
    # Pie chart
    colors = ['gold', 'navy', 'gray']
    wedges, texts, autotexts = ax4.pie(pref_counts.values, labels=pref_counts.index,
                                      autopct='%1.0f%%', colors=colors[:len(pref_counts)],
                                      startangle=90)
    
    ax4.set_title('MATCH TIME PREFERENCES\nAmong Top Championship Contenders', 
                  fontweight='bold')
    
    # Add player details
    detail_text = []
    for i, player in enumerate(PLAYERS):
        detail_text.append(f"{player}: {preferences[i]}")
    
    ax4.text(1.3, 0.5, '\n'.join(detail_text), transform=ax4.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('time_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 7. CROWD SUPPORT ANALYSIS
# =============================================================================
def create_crowd_analysis():
    """Analyze home crowd support and pressure effects"""
    print("\n7. Creating Crowd Support Analysis...")
    
    # Crowd support factors
    crowd_data = {
        'Player': PLAYERS,
        'Home_Support': [0.2, 0.1, 0.1, 0.1, 0.1, 0.8, 0.7, 0.9],  # US players get more
        'Pressure_Handling': [9.5, 7.5, 8.0, 8.5, 7.0, 6.5, 6.0, 5.5],  # 1-10 scale
        'Experience_Years': [20, 8, 6, 11, 12, 9, 4, 3],
        'Grand_Slam_Titles': [24, 2, 4, 1, 0, 0, 0, 0],
        'Media_Attention': [9.5, 8.0, 9.0, 7.5, 7.0, 6.5, 6.0, 5.5],  # 1-10 scale
        'Championship_Prob': CHAMPIONSHIP_PROBS
    }
    
    df_crowd = pd.DataFrame(crowd_data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Home support vs Pressure handling
    scatter = ax1.scatter(df_crowd['Home_Support'], df_crowd['Pressure_Handling'],
                         s=df_crowd['Championship_Prob']*15, alpha=0.7,
                         c=df_crowd['Experience_Years'], cmap='viridis')
    
    for i, player in enumerate(df_crowd['Player']):
        ax1.annotate(player, (df_crowd['Home_Support'].iloc[i], 
                             df_crowd['Pressure_Handling'].iloc[i]),
                    xytext=(3, 3), textcoords='offset points')
    
    ax1.set_xlabel('Home Crowd Support Level (0-1)')
    ax1.set_ylabel('Pressure Handling Ability (1-10)')
    ax1.set_title('HOME SUPPORT vs PRESSURE RESISTANCE\nBubble size = Championship %, Color = Experience', 
                  fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='Years of Experience')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Crowd impact on performance
    # Simulate different crowd scenarios
    crowd_scenarios = ['Hostile', 'Neutral', 'Supportive', 'Very Supportive']
    crowd_multipliers = [-0.05, 0.0, 0.03, 0.06]
    
    performance_adjustments = {}
    for scenario, multiplier in zip(crowd_scenarios, crowd_multipliers):
        scenario_performance = []
        for _, player_data in df_crowd.iterrows():
            # Base adjustment from crowd
            base_adjustment = multiplier
            
            # Modify by pressure handling ability
            pressure_factor = (player_data['Pressure_Handling'] - 5) / 5 * 0.02
            
            # Modify by experience
            exp_factor = min(0.02, player_data['Experience_Years'] / 100)
            
            # Home support bonus/penalty
            if scenario == 'Supportive' or scenario == 'Very Supportive':
                home_bonus = player_data['Home_Support'] * multiplier * 2
            else:
                home_bonus = -player_data['Home_Support'] * abs(multiplier) * 0.5
            
            total_adjustment = base_adjustment + pressure_factor + exp_factor + home_bonus
            scenario_performance.append(1 + total_adjustment)
        
        performance_adjustments[scenario] = scenario_performance
    
    # Plot performance adjustments
    x = np.arange(len(PLAYERS))
    width = 0.2
    
    colors = ['red', 'gray', 'lightgreen', 'darkgreen']
    for i, (scenario, performances) in enumerate(performance_adjustments.items()):
        ax2.bar(x + i*width, performances, width, label=scenario, 
               color=colors[i], alpha=0.7)
    
    ax2.axhline(y=1.0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Player')
    ax2.set_ylabel('Performance Multiplier')
    ax2.set_title('CROWD SCENARIO IMPACT ON PERFORMANCE', fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(PLAYERS, rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Experience vs Media pressure
    ax3.scatter(df_crowd['Experience_Years'], df_crowd['Media_Attention'],
               s=df_crowd['Grand_Slam_Titles']*20 + 50, alpha=0.7,
               c=df_crowd['Championship_Prob'], cmap='Reds')
    
    for i, player in enumerate(df_crowd['Player']):
        ax3.annotate(player, (df_crowd['Experience_Years'].iloc[i],
                             df_crowd['Media_Attention'].iloc[i]),
                    xytext=(3, 3), textcoords='offset points')
    
    # Add trendline
    z = np.polyfit(df_crowd['Experience_Years'], df_crowd['Media_Attention'], 1)
    p = np.poly1d(z)
    ax3.plot(df_crowd['Experience_Years'], p(df_crowd['Experience_Years']), 
            "b--", alpha=0.8, linewidth=2)
    
    ax3.set_xlabel('Professional Experience (years)')
    ax3.set_ylabel('Media Attention Level (1-10)')
    ax3.set_title('EXPERIENCE vs MEDIA SPOTLIGHT\nBubble size = Grand Slam titles', 
                  fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Pressure moments performance
    # Simulate performance in high-pressure situations
    pressure_situations = ['Break Point\nDefense', 'Serving for\nthe Match', 'Tiebreaks', 
                          'Deciding Set', 'Match Point\nDown']
    
    # Performance in pressure situations (relative to normal)
    pressure_performance = {}
    for situation in pressure_situations:
        situation_perf = []
        for _, player_data in df_crowd.iterrows():
            base_perf = player_data['Pressure_Handling'] / 10  # Convert to 0-1
            experience_bonus = min(0.1, player_data['Experience_Years'] / 200)
            slam_bonus = min(0.1, player_data['Grand_Slam_Titles'] / 50)
            
            # Different situations have different difficulty
            situation_difficulty = {'Break Point\nDefense': 0.9, 'Serving for\nthe Match': 0.8,
                                  'Tiebreaks': 0.85, 'Deciding Set': 0.82, 'Match Point\nDown': 0.75}
            
            final_perf = (base_perf + experience_bonus + slam_bonus) * \
                        situation_difficulty.get(situation.replace('\n', ' '), 0.8)
            situation_perf.append(final_perf)
        
        pressure_performance[situation] = situation_perf
    
    # Create heatmap
    pressure_matrix = np.array([pressure_performance[sit] for sit in pressure_situations]).T
    
    im = ax4.imshow(pressure_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
    ax4.set_xticks(range(len(pressure_situations)))
    ax4.set_xticklabels(pressure_situations, rotation=45, ha='right')
    ax4.set_yticks(range(len(PLAYERS)))
    ax4.set_yticklabels(PLAYERS)
    ax4.set_title('PRESSURE SITUATION PERFORMANCE\n(1.0 = Perfect, 0.5 = Major struggles)', 
                  fontweight='bold')
    
    # Add values
    for i in range(len(PLAYERS)):
        for j in range(len(pressure_situations)):
            color = "white" if pressure_matrix[i, j] > 0.75 else "black"
            ax4.text(j, i, f'{pressure_matrix[i, j]:.2f}', 
                    ha="center", va="center", color=color, fontweight='bold', fontsize=9)
    
    plt.colorbar(im, ax=ax4, label='Performance Level')
    
    plt.tight_layout()
    plt.savefig('crowd_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 8. SURFACE TRANSITION ANALYSIS
# =============================================================================
def create_surface_transition_analysis():
    """Analyze clay to hard court transition effects"""
    print("\n8. Creating Surface Transition Analysis...")
    
    # Surface transition data
    transition_data = {
        'Player': PLAYERS,
        'Clay_Matches_2024': [8, 12, 15, 6, 4, 3, 2, 1],  # Recent clay court matches
        'Hard_Adaptation_Speed': [8.5, 9.0, 7.5, 9.5, 8.0, 9.0, 8.5, 8.0],  # 1-10 scale
        'Movement_Adjustment': [9.0, 8.5, 8.0, 9.5, 7.5, 8.5, 7.0, 7.5],  # Footwork adaptation
        'Shot_Selection_Change': [8.0, 7.5, 9.0, 8.5, 8.0, 8.5, 7.5, 7.0],  # Tactical adaptation
        'Days_Since_Clay': [45, 35, 30, 50, 60, 65, 70, 75],  # Days since last clay match
        'Championship_Prob': CHAMPIONSHIP_PROBS
    }
    
    df_transition = pd.DataFrame(transition_data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Clay exposure vs Hard court adaptation
    scatter = ax1.scatter(df_transition['Clay_Matches_2024'], 
                         df_transition['Hard_Adaptation_Speed'],
                         s=df_transition['Championship_Prob']*12, alpha=0.7,
                         c=df_transition['Days_Since_Clay'], cmap='viridis')
    
    for i, player in enumerate(df_transition['Player']):
        ax1.annotate(player, (df_transition['Clay_Matches_2024'].iloc[i],
                             df_transition['Hard_Adaptation_Speed'].iloc[i]),
                    xytext=(3, 3), textcoords='offset points')
    
    ax1.set_xlabel('Clay Court Matches in 2024')
    ax1.set_ylabel('Hard Court Adaptation Speed (1-10)')
    ax1.set_title('CLAY EXPOSURE vs HARD COURT ADAPTATION\nBubble size = Championship %, Color = Days since clay', 
                  fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='Days Since Last Clay Match')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Transition timeline impact
    days_since_clay = np.arange(0, 91, 5)  # 0-90 days
    transition_performance = []
    
    for days in days_since_clay:
        if days <= 14:
            # Fresh off clay - still adjusting
            performance = 0.85 + (days / 14) * 0.10
        elif days <= 30:
            # Optimal transition period
            performance = 0.95 + ((30 - days) / 16) * 0.08
        elif days <= 60:
            # Good hard court rhythm
            performance = 1.0 + ((days - 30) / 30) * 0.05
        else:
            # Potentially rusty if too long
            performance = max(0.98, 1.05 - ((days - 60) / 30) * 0.07)
        
        transition_performance.append(performance)
    
    ax2.plot(days_since_clay, transition_performance, linewidth=3, color='blue')
    ax2.fill_between(days_since_clay, transition_performance, alpha=0.3, color='blue')
    
    # Mark current players
    for _, player_data in df_transition.iterrows():
        days = player_data['Days_Since_Clay']
        if days <= 90:
            perf_index = int(days / 5)
            if perf_index < len(transition_performance):
                ax2.scatter(days, transition_performance[perf_index], 
                           s=100, color='red', alpha=0.7, zorder=5)
                ax2.annotate(player_data['Player'], (days, transition_performance[perf_index]),
                           xytext=(3, 3), textcoords='offset points', fontsize=9)
    
    ax2.axvspan(20, 45, alpha=0.2, color='green', label='Optimal Transition Window')
    ax2.set_xlabel('Days Since Last Clay Match')
    ax2.set_ylabel('Hard Court Performance Multiplier')
    ax2.set_title('SURFACE TRANSITION TIMELINE\nOptimal timing for clay-to-hard adaptation', 
                  fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.8, 1.1)
    
    # Plot 3: Adaptation component analysis
    adaptation_components = ['Movement\nAdjustment', 'Shot Selection\nChange', 'Hard Adaptation\nSpeed']
    
    # Calculate composite adaptation score
    df_transition['Composite_Adaptation'] = (
        df_transition['Movement_Adjustment'] * 0.4 + 
        df_transition['Shot_Selection_Change'] * 0.3 +
        df_transition['Hard_Adaptation_Speed'] * 0.3
    )
    
    # Create radar chart for top 4 players
    from math import pi
    
    top_4_indices = df_transition.nlargest(4, 'Championship_Prob').index
    top_4_players = df_transition.loc[top_4_indices, 'Player'].tolist()
    
    # Number of variables
    N = 3
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, idx in enumerate(top_4_indices):
        values = [
            df_transition.loc[idx, 'Movement_Adjustment'],
            df_transition.loc[idx, 'Shot_Selection_Change'],
            df_transition.loc[idx, 'Hard_Adaptation_Speed']
        ]
        values += values[:1]
        
        ax3.plot(angles, values, 'o-', linewidth=2, 
                label=df_transition.loc[idx, 'Player'], color=colors[i])
        ax3.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(adaptation_components)
    ax3.set_ylim(0, 10)
    ax3.set_title('ADAPTATION SKILLS COMPARISON\nTop 4 Championship Contenders', 
                  fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Plot 4: Clay specialists vs Hard court specialists
    # Categorize players by surface preference
    surface_preferences = []
    adaptation_scores = df_transition['Composite_Adaptation'].tolist()
    clay_matches = df_transition['Clay_Matches_2024'].tolist()
    
    for i, player in enumerate(PLAYERS):
        clay_exposure = clay_matches[i]
        adaptation = adaptation_scores[i]
        
        if clay_exposure > 10 and adaptation < 8.0:
            category = 'Clay Specialist'
        elif clay_exposure < 5 and adaptation > 8.5:
            category = 'Hard Court Specialist'
        else:
            category = 'All-Surface Player'
        
        surface_preferences.append(category)
    
    # Create performance comparison
    categories = ['Clay Specialist', 'All-Surface Player', 'Hard Court Specialist']
    category_performance = []
    category_counts = []
    
    for category in categories:
        players_in_category = [i for i, pref in enumerate(surface_preferences) if pref == category]
        if players_in_category:
            avg_prob = np.mean([CHAMPIONSHIP_PROBS[i] for i in players_in_category])
            category_performance.append(avg_prob)
            category_counts.append(len(players_in_category))
        else:
            category_performance.append(0)
            category_counts.append(0)
    
    # Dual axis plot
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar([i - 0.2 for i in range(len(categories))], category_performance, 
                   width=0.4, label='Avg Championship %', alpha=0.7, color='skyblue')
    bars2 = ax4_twin.bar([i + 0.2 for i in range(len(categories))], category_counts, 
                        width=0.4, label='Player Count', alpha=0.7, color='lightcoral')
    
    # Add value labels
    for bar, val in zip(bars1, category_performance):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    for bar, val in zip(bars2, category_counts):
        height = bar.get_height()
        ax4_twin.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                     f'{val}', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_xlabel('Surface Specialization')
    ax4.set_ylabel('Average Championship Probability (%)', color='blue')
    ax4_twin.set_ylabel('Number of Players', color='red')
    ax4.set_title('SURFACE SPECIALIZATION vs SUCCESS\nUS Open hard court advantage', 
                  fontweight='bold')
    ax4.set_xticks(range(len(categories)))
    ax4.set_xticklabels(categories, rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # Legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('surface_transition_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 9. MENTAL TOUGHNESS INDICATORS
# =============================================================================
def create_mental_toughness_analysis():
    """Analyze comeback statistics and mental resilience"""
    print("\n9. Creating Mental Toughness Analysis...")
    
    # Mental toughness metrics
    mental_data = {
        'Player': PLAYERS,
        'Comeback_Win_Rate': [0.78, 0.65, 0.72, 0.68, 0.58, 0.52, 0.48, 0.45],  # Win rate when down
        'Tiebreak_Record': [0.73, 0.69, 0.71, 0.67, 0.62, 0.58, 0.55, 0.52],  # Tiebreak win %
        'Deciding_Set_Record': [0.81, 0.72, 0.75, 0.70, 0.65, 0.60, 0.58, 0.55],  # 5th set win %
        'Break_Point_Conversion': [0.44, 0.38, 0.42, 0.35, 0.32, 0.36, 0.28, 0.30],  # Convert BP %
        'Save_Match_Points': [0.35, 0.28, 0.32, 0.25, 0.22, 0.18, 0.15, 0.12],  # Save MP %
        'Big_Match_Experience': [95, 45, 55, 60, 50, 35, 20, 15],  # Big matches played
        'Championship_Prob': CHAMPIONSHIP_PROBS
    }
    
    df_mental = pd.DataFrame(mental_data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Mental toughness composite score
    # Calculate weighted mental toughness score
    df_mental['Mental_Toughness_Score'] = (
        df_mental['Comeback_Win_Rate'] * 25 +
        df_mental['Tiebreak_Record'] * 20 +
        df_mental['Deciding_Set_Record'] * 25 +
        df_mental['Save_Match_Points'] * 30
    )
    
    # Sort by mental toughness
    df_sorted = df_mental.sort_values('Mental_Toughness_Score', ascending=False)
    
    colors = ['gold' if score > 18 else 'silver' if score > 15 else 'bronze' if score > 12 else 'gray'
             for score in df_sorted['Mental_Toughness_Score']]
    
    bars = ax1.barh(df_sorted['Player'], df_sorted['Mental_Toughness_Score'],
                   color=colors, alpha=0.7)
    
    for bar, score in zip(bars, df_sorted['Mental_Toughness_Score']):
        ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontweight='bold')
    
    ax1.set_xlabel('Mental Toughness Composite Score')
    ax1.set_title('MENTAL TOUGHNESS RANKING\nBased on comeback stats, tiebreaks, and clutch performance', 
                  fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Clutch performance correlation
    scatter = ax2.scatter(df_mental['Mental_Toughness_Score'], df_mental['Championship_Prob'],
                         s=df_mental['Big_Match_Experience']*2, alpha=0.7,
                         c=df_mental['Save_Match_Points']*100, cmap='Reds')
    
    for i, player in enumerate(df_mental['Player']):
        ax2.annotate(player, (df_mental['Mental_Toughness_Score'].iloc[i],
                             df_mental['Championship_Prob'].iloc[i]),
                    xytext=(3, 3), textcoords='offset points')
    
    # Add trendline
    z = np.polyfit(df_mental['Mental_Toughness_Score'], df_mental['Championship_Prob'], 1)
    p = np.poly1d(z)
    ax2.plot(df_mental['Mental_Toughness_Score'], p(df_mental['Mental_Toughness_Score']), 
            "g--", alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Mental Toughness Score')
    ax2.set_ylabel('Championship Probability (%)')
    ax2.set_title('MENTAL TOUGHNESS vs SUCCESS\nBubble size = Big match experience, Color = MP save rate', 
                  fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Match Point Save Rate (%)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Pressure situations breakdown
    pressure_metrics = ['Comeback\nWin Rate', 'Tiebreak\nRecord', 'Deciding Set\nRecord', 
                        'Break Point\nConversion', 'Save Match\nPoints']
    
    # Normalize metrics to 0-100 scale for comparison
    normalized_data = {}
    for metric in ['Comeback_Win_Rate', 'Tiebreak_Record', 'Deciding_Set_Record', 
                   'Break_Point_Conversion', 'Save_Match_Points']:
        normalized_data[metric] = df_mental[metric] * 100
    
    # Show top 4 players
    top_4_mental = df_mental.nlargest(4, 'Mental_Toughness_Score')
    
    x = np.arange(len(pressure_metrics))
    width = 0.2
    colors_radar = ['red', 'blue', 'green', 'orange']
    
    for i, (_, player_data) in enumerate(top_4_mental.iterrows()):
        values = [
            player_data['Comeback_Win_Rate'] * 100,
            player_data['Tiebreak_Record'] * 100,
            player_data['Deciding_Set_Record'] * 100,
            player_data['Break_Point_Conversion'] * 100,
            player_data['Save_Match_Points'] * 100
        ]
        
        ax3.bar(x + i*width, values, width, label=player_data['Player'], 
               alpha=0.7, color=colors_radar[i])
    
    ax3.set_xlabel('Pressure Situations')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('PRESSURE PERFORMANCE BREAKDOWN\nTop 4 Mental Toughness Leaders', 
                  fontweight='bold')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(pressure_metrics)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Experience vs Mental strength relationship
    # Create experience categories
    experience_categories = []
    for exp in df_mental['Big_Match_Experience']:
        if exp >= 80:
            experience_categories.append('Veteran (80+)')
        elif exp >= 50:
            experience_categories.append('Experienced (50-79)')
        elif exp >= 30:
            experience_categories.append('Developing (30-49)')
        else:
            experience_categories.append('Newcomer (<30)')
    
    df_mental['Experience_Category'] = experience_categories
    
    # Calculate average mental toughness by experience category
    exp_mental_avg = df_mental.groupby('Experience_Category')['Mental_Toughness_Score'].mean()
    exp_champ_avg = df_mental.groupby('Experience_Category')['Championship_Prob'].mean()
    
    categories = list(exp_mental_avg.index)
    mental_avgs = list(exp_mental_avg.values)
    champ_avgs = list(exp_champ_avg.values)
    
    # Create dual-axis comparison
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar([i - 0.2 for i in range(len(categories))], mental_avgs, 
                   width=0.4, label='Mental Toughness', alpha=0.7, color='purple')
    bars2 = ax4_twin.bar([i + 0.2 for i in range(len(categories))], champ_avgs, 
                        width=0.4, label='Championship %', alpha=0.7, color='gold')
    
    # Add value labels
    for bar, val in zip(bars1, mental_avgs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    for bar, val in zip(bars2, champ_avgs):
        height = bar.get_height()
        ax4_twin.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_xlabel('Experience Level')
    ax4.set_ylabel('Average Mental Toughness Score', color='purple')
    ax4_twin.set_ylabel('Average Championship Probability (%)', color='gold')
    ax4.set_title('EXPERIENCE vs MENTAL STRENGTH\nDoes big-match experience build mental toughness?', 
                  fontweight='bold')
    ax4.set_xticks(range(len(categories)))
    ax4.set_xticklabels(categories, rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # Legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('mental_toughness_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 10. PHYSICAL CONDITIONING METRICS
# =============================================================================
def create_physical_conditioning_analysis():
    "Analyze stamina and physical conditioning metrics"
    print("\n10. Creating Physical Conditioning Analysis...")
    
    # Physical conditioning data
    physical_data = {
        'Player': PLAYERS,
        'VO2_Max_Estimate': [58, 62, 65, 55, 57, 60, 59, 63],  # Estimated aerobic capacity
        'Match_Length_Tolerance': [4.8, 4.2, 4.5, 3.8, 4.0, 3.9, 3.5, 3.6],  # Hours can maintain peak
        'Recovery_Rate': [0.92, 0.95, 0.94, 0.88, 0.90, 0.91, 0.89, 0.93],  # Between-match recovery
        'Injury_Frequency': [0.8, 0.3, 0.5, 0.6, 1.2, 0.4, 1.8, 0.2],  # Injuries per year
        'Training_Load': [85, 88, 90, 82, 78, 86, 75, 89],  # Training intensity (1-100)
        'Age': [37, 23, 21, 28, 27, 27, 22, 21],
        'Body_Mass_Index': [23.1, 22.8, 22.5, 23.8, 24.2, 23.4, 24.0, 22.9],
        'Championship_Prob': CHAMPIONSHIP_PROBS
    }
    
    df_physical = pd.DataFrame(physical_data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot 1: Physical fitness composite score
    # Calculate fitness score (higher is better)
    df_physical['Fitness_Score'] = (
        df_physical['VO2_Max_Estimate'] / 70 * 25 +  # Aerobic capacity
        df_physical['Match_Length_Tolerance'] / 5 * 20 +  # Endurance
        df_physical['Recovery_Rate'] * 25 +  # Recovery ability
        (2.0 - df_physical['Injury_Frequency']) / 2 * 15 +  # Injury resistance
        df_physical['Training_Load'] / 100 * 15  # Training capacity
    )
    
    # Sort by fitness score
    df_sorted = df_physical.sort_values('Fitness_Score', ascending=False)
    
    colors = ['darkgreen' if score > 85 else 'green' if score > 80 else 'orange' if score > 75 else 'red'
             for score in df_sorted['Fitness_Score']]
    
    bars = ax1.barh(df_sorted['Player'], df_sorted['Fitness_Score'],
                   color=colors, alpha=0.7)
    
    for bar, score in zip(bars, df_sorted['Fitness_Score']):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontweight='bold')
    
    ax1.set_xlabel('Physical Fitness Composite Score')
    ax1.set_title('PHYSICAL FITNESS RANKING\nBased on endurance, recovery, and injury resistance', 
                  fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Fitness vs Performance correlation
    scatter = ax2.scatter(df_physical['Fitness_Score'], df_physical['Championship_Prob'],
                         s=df_physical['Age']*8, alpha=0.7,
                         c=df_physical['Injury_Frequency'], cmap='RdYlGn_r')
    
    for i, player in enumerate(df_physical['Player']):
        ax2.annotate(player, (df_physical['Fitness_Score'].iloc[i],
                             df_physical['Championship_Prob'].iloc[i]),
                    xytext=(3, 3), textcoords='offset points')
    
    # Add trendline
    z = np.polyfit(df_physical['Fitness_Score'], df_physical['Championship_Prob'], 1)
    p = np.poly1d(z)
    ax2.plot(df_physical['Fitness_Score'], p(df_physical['Fitness_Score']), 
            "b--", alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Physical Fitness Score')
    ax2.set_ylabel('Championship Probability (%)')
    ax2.set_title('FITNESS vs CHAMPIONSHIP SUCCESS\nBubble size = Age, Color = Injury frequency', 
                  fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Injury Frequency (per year)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Age vs Physical metrics
    # Show how different physical attributes change with age
    age_groups = ['Young (≤23)', 'Prime (24-28)', 'Veteran (29+)']
    
    young_players = df_physical[df_physical['Age'] <= 23]
    prime_players = df_physical[(df_physical['Age'] > 23) & (df_physical['Age'] <= 28)]
    veteran_players = df_physical[df_physical['Age'] > 28]
    
    groups = [young_players, prime_players, veteran_players]
    
    # Calculate averages for each age group
    metrics = ['VO2_Max_Estimate', 'Match_Length_Tolerance', 'Recovery_Rate', 'Injury_Frequency']
    group_averages = {}
    
    for i, group in enumerate(groups):
        group_averages[age_groups[i]] = {}
        for metric in metrics:
            if len(group) > 0:
                group_averages[age_groups[i]][metric] = group[metric].mean()
            else:
                group_averages[age_groups[i]][metric] = 0
    
    # Create grouped bar chart
    x = np.arange(len(metrics))
    width = 0.25
    colors_age = ['lightblue', 'blue', 'darkblue']
    
    for i, age_group in enumerate(age_groups):
        values = [group_averages[age_group][metric] for metric in metrics]
        # Normalize injury frequency (invert and scale)
        values[3] = (2 - values[3]) * 2  # Convert to "injury resistance"
        
        ax3.bar(x + i*width, values, width, label=age_group, 
               alpha=0.7, color=colors_age[i])
    
    # Adjust metric labels
    metric_labels = ['VO2 Max', 'Endurance\n(hours)', 'Recovery\nRate', 'Injury\nResistance']
    ax3.set_xlabel('Physical Metrics')
    ax3.set_ylabel('Average Score')
    ax3.set_title('PHYSICAL PERFORMANCE BY AGE GROUP\n(Higher is better for all metrics)', 
                  fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(metric_labels)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Long match performance prediction
    # Simulate performance decline in long matches
    match_durations = np.arange(1.5, 6.0, 0.1)
    
    # Show performance curves for top 4 fittest players
    top_4_fit = df_physical.nlargest(4, 'Fitness_Score')
    
    for _, player_data in top_4_fit.iterrows():
        performance_curve = []
        
        for duration in match_durations:
            # Base performance starts at 100%
            base_performance = 100
            
            # Fitness factors
            endurance_factor = player_data['Match_Length_Tolerance']
            vo2_factor = player_data['VO2_Max_Estimate'] / 60
            age_penalty = (player_data['Age'] - 20) / 20 * 0.1
            
            # Performance decline calculation
            if duration <= endurance_factor:
                # Within tolerance - minimal decline
                decline = (duration / endurance_factor) * 5 + age_penalty * duration
            else:
                # Beyond tolerance - steeper decline
                excess_duration = duration - endurance_factor
                decline = 5 + excess_duration * 15 * (1 - vo2_factor + age_penalty)
            
            final_performance = max(60, base_performance - decline)
            performance_curve.append(final_performance)
        
        ax4.plot(match_durations, performance_curve, linewidth=2.5, 
                label=player_data['Player'], alpha=0.8)
    
    # Add typical match duration markers
    ax4.axvline(x=2.5, color='green', linestyle='--', alpha=0.7, label='Typical match (2.5h)')
    ax4.axvline(x=4.0, color='orange', linestyle='--', alpha=0.7, label='Long match (4h)')
    ax4.axvline(x=5.0, color='red', linestyle='--', alpha=0.7, label='Epic match (5h)')
    
    ax4.set_xlabel('Match Duration (hours)')
    ax4.set_ylabel('Performance Level (%)')
    ax4.set_title('STAMINA CURVES: PERFORMANCE IN LONG MATCHES\nTop 4 Physically Fittest Players', 
                  fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(55, 105)
    
    # Add shaded regions for performance zones
    ax4.axhspan(90, 100, alpha=0.1, color='green', label='Peak Performance')
    ax4.axhspan(80, 90, alpha=0.1, color='yellow', label='Good Performance')  
    ax4.axhspan(70, 80, alpha=0.1, color='orange', label='Declining Performance')
    ax4.axhspan(60, 70, alpha=0.1, color='red', label='Struggling')
    
    plt.tight_layout()
    plt.savefig('physical_conditioning_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    
    # List of all visualization functions
    visualizations = [
        ("Match-by-Match Probability Heatmap", create_probability_heatmap),
        ("Player Fatigue Simulation", create_fatigue_simulation),
        ("Weather Impact Analysis", create_weather_analysis),
        ("Injury Risk Assessment", create_injury_risk_assessment),
        ("Court Speed Impact", create_court_speed_analysis),
        ("Time of Day Performance", create_time_analysis),
        ("Crowd Support Analysis", create_crowd_analysis),
        ("Surface Transition Analysis", create_surface_transition_analysis),
        ("Mental Toughness Indicators", create_mental_toughness_analysis),
        ("Physical Conditioning Metrics", create_physical_conditioning_analysis)
    ]
    
    # Execute all visualizations
    for name, func in visualizations:
        try:
            func()
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    print("\n" + "="*80)
    print("ADVANCED TENNIS ANALYTICS SUITE - SUMMARY")
    print("="*80)
    print("Generated 10 specialized visualization files:")
    print("1. probability_heatmap.png - Tournament progression probabilities")
    print("2. fatigue_simulation.png - Performance decline through rounds")
    print("3. weather_analysis.png - Climate and condition impacts")
    print("4. injury_risk_assessment.png - Tournament injury risk factors")
    print("5. court_speed_analysis.png - Surface speed optimization")
    print("6. time_analysis.png - Daily performance rhythms")
    print("7. crowd_analysis.png - Home support and pressure effects")
    print("8. surface_transition_analysis.png - Clay to hard court adaptation")
    print("9. mental_toughness_analysis.png - Clutch performance metrics")
    print("10. physical_conditioning_analysis.png - Stamina and fitness analysis")
    print("\nThese advanced analytics provide comprehensive insights into:")
    print("• Tournament-specific performance factors")
    print("• Player adaptation and resilience")
    print("• Environmental and psychological influences")
    print("• Physical and mental conditioning impacts")
    print("="*80)
    
    
    
    
    