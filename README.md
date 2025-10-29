 Video Game Sales Forecasting — Supervised Learning Project

## Project Overview
- **Goal:** Predict global video game sales (millions of units) using a supervised regression pipeline.
- **Dataset:** [Kaggle · Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) (`vgsales.csv`), ~16k titles across platforms, genres, publishers, and regional sales figures.
- **Motivation:** Help publishers/retailers anticipate demand, inform inventory and marketing decisions, and demonstrate end-to-end supervised learning skills.

## Repository Structure
- `result.ipynb` — executed notebook with saved outputs for review.
- `data/` — place the downloaded `vgsales.csv` file (not tracked in git)

## Reproduction Steps
1. Install dependencies (Python 3.9+):
   ```bash
   pip install pandas seaborn scikit-learn matplotlib
   ```
2. Download the dataset:
   ```bash
   kaggle datasets download -d gregorut/videogamesales -p data/ --unzip
   ```
3. Open `video_game_sales_project.ipynb` and run cells sequentially.

## Exploratory Data Analysis
- Global sales are extremely right-skewed; log view reveals a long tail driven by a handful of blockbuster titles.
- Wii/DS/PS2 (Nintendo/Sony) dominate platform share; Action, Sports, and Shooter genres lead total sales.
- Release activity peaks between 2005–2010; coverage declines for newer years.
- Publisher frequency follows a power-law distribution — a few mega-publishers (Nintendo, EA, Activision, Ubisoft) account for most releases and revenue.

## Feature Engineering & Modeling
- **Features used:** `platform`, `genre`, `publisher`, `release_decade`, `year`, `publisher_freq`.
- **Preprocessing:** median imputation, numeric scaling, and dense one-hot encoding for categoricals (supports tree models).
- **Models evaluated:** Dummy baseline, LinearRegression, Ridge, Lasso, RandomForest, HistGradientBoosting (with lightweight GridSearchCV).
- **Metrics:** MAE, RMSE, R² on train/test splits (80/20) with fixed random seed.

## Key Results
- **Best model:** HistGradientBoostingRegressor  
  - MAE ≈ 0.52M units (≈16% improvement over the mean baseline)  
  - RMSE ≈ 1.92M units, R² ≈ 0.12
- RandomForest performs comparably (MAE ≈ 0.53M) but slightly worse overall.
- Ridge/Lasso regressions stabilize around MAE ≈ 0.55M; vanilla LinearRegression collapses under multicollinearity, providing a cautionary example.
- Permutation importance highlights publisher identity, platform, and publisher release volume as the strongest drivers; year contributes modestly, while release decade adds little once year is present.

## Discussion & Insights
- Heavy-tailed sales remain difficult to model; residual analysis shows under-prediction of mega-hits (>5M units) and over-prediction of niche releases.
- Publisher/platform signals dominate because metadata lacks consumer sentiment or marketing context.
- Regularization or tree-based methods are essential for categorical-heavy data; naive linear models fail without it.

## Lessons Learned & Future Work
- **What didn’t work:** Unregularized linear regression (negative R²), inclusion of regional sales features due to target leakage risk, and expansive grid searches without runtime guards.
- **Next steps:**  
  1. Enrich the feature set with review sentiment, franchise membership, or install-base estimates.  
  2. Explore alternative objectives (hit classification, quantile regression) to better capture heavy-tailed outcomes.  
  3. Use time-aware splits (train on pre-2013, test on newer releases) to evaluate temporal generalization.
