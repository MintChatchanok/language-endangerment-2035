# Project Description
This project investigates the global risk of language extinction using machine learning. I analyze endangered languages worldwide to:
- Classify languages by their endangerment level
- Predict at-risk regions based on language loss trends
- Forecast how many languages will be lost by 2035 in each country
- My models uncover patterns that may guide future preservation strategies.

# Key Results
| Task           | Best Model            | F1 / R²   | Notes                                                                  |
| -------------- | --------------------- | --------- | ---------------------------------------------------------------------- |
| Classification | XGBoost               | F1 = 0.66 | Best at detecting critical categories (Extinct, Critically Endangered) |
| Regression     | Blended CatBoost + RF | R² = 0.04 | Generalization was limited due to sparse features                      |
| Forecasting    | Bootstrap simulation  | 📉        | Brazil, India, and Mexico may lose 15–17 languages each                |

# Objectives
- Classify languages based on endangerment severity
- Predict endangered language counts per country
- Forecast global risk zones from 2025–2035

# Methodology Overview
- Data Source: Kaggle’s Endangered Languages Dataset
- Cleaning & Feature Engineering: Label simplification, continent tags
- Models: Random Forest, XGBoost, Neural Network, CatBoost
- Visualization: Choropleth mapping, regional risk summaries
- Forecasting: Bootstrapped time-proxy simulation (3 generations ≈ 75 years)

# Quick Start
# Clone the repository
git clone https://github.com/MintChatchanok/language-endangerment-2035.git
cd language-endangerment-2035

# Set up your environment
pip install -r requirements.txt

# Run training
python src/train_model.py

# Run forecasting
python src/forecast.py

# Repository Structure

<pre> 
language-endangerment-2035/
│
├─ src/               # Scripts for preprocessing, training, forecasting
│  ├─ dataset.py      # Load and clean dataset
│  ├─ features.py     # Feature engineering and encoding
│  ├─ train_model.py  # Classification and regression training
│  └─ forecast.py     # Bootstrapped forecasting logic
│
├─ data/              # (Optional) Input data or data loading references
├─ README.md          # This file
├─ requirements.txt   # Python package dependencies </pre>

# Insights
- Asia and Africa have the highest total endangered languages
- Oceania and North America are critically vulnerable
- Brazil, India, and Mexico are projected hotspots for language extinction
