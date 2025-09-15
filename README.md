# Sport-Analytics

# 🎾 ATP Tennis Match Prediction & US Open 2025 Forecast  

**Duration:** Jan 2025 – Apr 2025  
**Institution:** Northeastern University  
**Role:** Data Analyst | Sports Strategist  
**Tools Used:** Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn  

---

##  Project Objective  
This project leverages six years of ATP men’s tennis match data (2019–2024) to predict potential winners of the 2025 US Open. By integrating advanced feature engineering (Elo ratings, mental toughness, fitness scores) with predictive modeling and tournament simulations, the project uncovers key performance drivers behind Grand Slam success.  

---

##  Business Questions Addressed  
- **Tournament Forecasting:** Who are the most likely champions for the 2025 US Open?  
- **Performance Drivers:** Do serve %, break-point conversion, and stamina correlate with higher win probability?  
- **Player Momentum:** How do Elo rating trends reflect form, consistency, and volatility?  
- **Match Conditions:** Do scheduling and weather adaptability significantly impact outcomes?  
- **Stamina vs. Mental Toughness:** How do younger players compare to veterans in pressure scenarios?  

---

## Data Sources  
- **ATP Match Data (2019–2024)** covering:  
  - Player names, rankings, and outcomes  
  - Serve/return statistics, break-point conversion, match durations  
  - Surface information (hard, clay, grass)  
  - Championship progression records  

---

##  Key Tasks Performed  
- Cleaned and standardized six years of ATP match data across multiple datasets  
- Engineered advanced features: **Elo ratings, mental toughness, stamina curves, weather adaptability, scheduling advantage**  
- Built classification models (Logistic Regression & Random Forest) with **~87% accuracy** in predicting match outcomes  
- Ran **10,000 Monte Carlo tournament simulations** to project US Open 2025 outcomes  
- Designed **visual dashboards** showing probability heatmaps, player progression, and round-by-round advancement  

---

## Key Insights  
- **Djokovic** showed unmatched consistency in Elo stability and adaptability across conditions  
- **Sinner** displayed the sharpest Elo climb, with strong momentum in 2024 form trends  
- **Alcaraz** led in physical conditioning and long-match stamina, reinforcing his next-gen contender status  
- Younger players excelled in **fitness metrics**, while veterans dominated in **mental toughness & pressure performance**  
- Simulations consistently highlighted **Djokovic, Sinner, and Alcaraz** as top contenders, with **Sinner** emerging as the projected **2025 US Open Champion**  

---

##  Tools & Libraries (Key Elements Used)  
- **Pandas** → data cleaning, merging multi-season datasets, feature engineering  
- **NumPy** → Elo updates, probability computations, simulation math  
- **Scikit-Learn** → `LogisticRegression`, `RandomForestClassifier`, `train_test_split`, `StandardScaler`, `accuracy_score`  
- **Matplotlib & Seaborn** → probability heatmaps, stamina curves, weather adaptability plots, tournament progression visuals  

---

