# Predicting Smartphone Addiction

A machine learning classification project to predict smartphone addiction based on daily usage metrics, behavioral habits, and demographic features.

## Project Overview
The objective of this project is to determine whether an individual exhibits smartphone addiction (`addicted_label`: `1` for addicted, `0` for not addicted).

The model analyzes behavioral indicators such as screen time, sleep duration, notification frequency, gaming and social media usage, as well as demographic factors.

## Dataset Description
The dataset contains information across multiple behavioral and lifestyle dimensions:
- `daily_screen_time_hours`: Average daily screen time.
- `social_media_hours`: Time spent on social media applications.
- `gaming_hours`: Time spent gaming on the phone.
- `work_study_hours`: Productive screen time for work or study.
- `sleep_hours`: Daily sleep duration.
- `notifications_per_day`: Number of phone notifications received.
- `app_opens_per_day`: Number of times apps are launched daily.
- `weekend_screen_time`: Screen time during weekends.
- `stress_level`: Self-reported stress level (Low, Medium, High).
- `academic_work_impact`: Impact on academic or work productivity (Yes, No).
- `age` and `gender`: Demographic variables.
- `addicted_label`: Target binary classification label (0 = Not Addicted, 1 = Addicted).

## Methodology and Pipeline

### 1. Exploratory Data Analysis (EDA)
- Analyzed feature distributions and missing values across both training (691,369 samples) and test (296,302 samples) sets.
- Evaluated target class balance (~68.3% addicted vs 31.7% non-addicted).
- Discovered synthetic generator mathematical constraints where screen time components do not sum to total screen time, creating strong budget residuals.

### 2. Feature Engineering & Synthetic Generator Footprints
1. **Extended Decimal Lattice Features**:
   - `frac_col`: Continuous sub-unit remainder (`value - floor(value)`) capturing generator decimal distributions.
   - `d1_col`: First decimal digit (`floor(value * 10) % 10`) exposing discretization artifacts.
   - `is_int_col`: Binary indicator for exact integer measurements (`frac == 0`).
   - `is_half_col`: Binary indicator for half-unit increments (`frac == 0.5`).
   - Applied across continuous columns: `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `weekend_screen_time`.
2. **Transductive Population Frequency Encodings**:
   - Empirical frequency statistics computed across the combined pool of training and test sets to capture generator discretization density without target leakage.
3. **Generator Budget Constraints & Discrepancies**:
   - `accounted_screen_time`: Sum of social media, gaming, and work/study hours.
   - `unaccounted_screen_time`: Latent budget discrepancy (`daily_screen_time_hours - accounted_screen_time`).
   - `unaccounted_ratio`: Ratio of unaccounted screen time to total screen time.
   - `weekend_budget_residual`: Latent budget residual for weekend duration (`weekend_screen_time - 2 * (social + gaming + work)`).
   - `non_study_screen_hours`: Non-productive screen time (`daily_screen_time_hours - work_study_hours`).
   - `entertainment_hours`: Total recreational screen time (`social_media_hours + gaming_hours`).
   - `entertainment_ratio`: Proportion of daily screen time spent on entertainment.
   - `unproductive_to_productive`: Ratio of entertainment hours to work and study hours.
4. **Circadian Dynamics & Micro-Interactions**:
   - `awake_hours`: `24.0 - sleep_hours`.
   - `free_awake_hours`: Active waking hours excluding work and study.
   - `screen_time_to_awake_ratio`: Proportion of waking hours spent on screens.
   - `screen_to_sleep_ratio`: Screen time relative to sleep duration.
   - `sleep_deficit`: Sleep deprivation relative to standard baseline.
   - `notifications_per_awake_hour` & `app_opens_per_awake_hour`: Usage density during waking periods.
   - `minutes_per_app_open`: Average duration spent per app open.
   - `compulsive_check_rate`: Frequency of app opens per minute of active screen time.
   - `weekend_vs_daily_diff`: Increase in screen time during weekends.
   - `total_weekly_screen_time`: Combined 7-day screen time exposure.
   - `addiction_index_v2`: Weighted composite indicator combining non-study screen hours, recreational use, notifications, and sleep deprivation.
   - `high_screen_low_sleep`, `high_notif_high_open`, `severe_impact_stress`: Risk flags.

### 3. Leak-Free All-Column Nested Target Encoding
- Applied nested 5-fold out-of-fold Bayesian smoothed target encoding across all 12 raw continuous and categorical features.
- Explicit string level `__missing__` assigned to missing values to preserve missingness distribution.
- Prior smoothing factor `m = 10.0` with simultaneous frequency encodings.

### 4. Titan Dual-Seed 10-Fold 4-Model Diverse Deep Ensemble (80 Models)
- Implemented a Dual-Seed (Seeds 42 & 2024) 10-Fold Stratified Cross-Validation pipeline combining 4 diverse gradient boosting architectures (80 models total):
  1. `LightGBM Classifier` (20 models): Leaf-wise booster (`num_leaves=190`, `max_depth=12`, `feature_fraction=0.60`, `learning_rate=0.025`, `n_estimators=1700`).
  2. `XGBoost Classifier` (20 models): Histogram-based tree booster (`max_depth=8`, `colsample_bytree=0.60`, `subsample=0.80`, `learning_rate=0.030`, `n_estimators=1200`).
  3. `CatBoost Classifier` (20 models): Symmetric oblivious decision trees (`depth=7`, `learning_rate=0.035`, `l2_leaf_reg=3.0`, `iterations=1400`).
  4. `HistGradientBoostingClassifier` (20 models): Binned gradient booster (`max_leaf_nodes=190`, `l2_regularization=0.5`, `learning_rate=0.035`, `max_iter=280`).
- Meta-optimized non-negative blending weights via Nelder-Mead on dual-seed out-of-fold probability distributions.

## Results and Benchmark

| Metric | Standalone / OOF CV | Titan Dual-Seed 80-Model Ensemble |
| :--- | :--- | :--- |
| Bagged Dual-Seed LightGBM OOF AUC | 0.96840 | - |
| Bagged Dual-Seed XGBoost OOF AUC | 0.96844 | - |
| Bagged Dual-Seed CatBoost OOF AUC | 0.96791 | - |
| Bagged Dual-Seed HistGBM OOF AUC | 0.96756 | - |
| **Dual-Seed 10-Fold 4-Model Ensemble OOF ROC-AUC** | - | **0.96853** |
| **Overall Classification Accuracy** | - | **90.94%** |
| **F1-Score** | - | **0.93608** |
| **Precision** | - | **0.93698** |
| **Recall** | - | **0.93518** |
| **Peak Individual Fold AUC** | - | **0.96942** |
| **Previous Verified Public Leaderboard ROC-AUC** | - | **0.96936** |

## Progression History Across Iterations

| Iteration | Pipeline Architecture | Features | OOF ROC-AUC | Public Leaderboard |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Baseline HistGBM | 11 | 0.92310 | 0.92374 |
| 2 | 5-Fold HistGBM with Basic FE | 18 | 0.96380 | 0.96393 |
| 3 | 5-Fold Dual Ensemble (LGB + XGB) | 27 | 0.96440 | 0.96459 |
| 4 | 10-Fold Dual Ensemble | 38 | 0.96482 | 0.96494 |
| 5 | 10-Fold Triple Ensemble | 43 | 0.96525 | 0.96539 |
| 6 | Ultra 10-Fold Triple Ensemble | 49 | 0.96550 | 0.96567 |
| 7 | Grandmaster 10-Fold Triple Ensemble | 60 | 0.96568 | 0.96578 |
| 8 | Ultra Grandmaster 10-Fold Triple Ensemble | 73 | 0.96574 | 0.96584 |
| 9 | SOTA 10-Fold Ensemble + Budget Residuals | 77 | 0.96605 | 0.96719 |
| 10 | Dual-Seed 60-Model Ensemble + Bayesian TE | 78 | 0.96616 | 0.96745 |
| 11 | Titan Tri-Seed 90-Model Deep Ensemble | 78 | 0.96617 | 0.96734 |
| 12 | Breakthrough 10-Fold 4-Model Ensemble | 71 | 0.96822 | **0.96936** |
| 13 | **Titan Dual-Seed 80-Model SOTA Ensemble** | **95** | **0.96853** | **Ready for Submission** |

## Project Structure
```text
Predicting_Smartphone_Addiction/
|
+-- Data/
|   +-- train.csv              # Training dataset with labels
|   +-- test.csv               # Test dataset for evaluation
|   +-- sample_submission.csv  # Submission format template
|
+-- main.ipynb                 # End-to-end data processing and model training notebook
+-- submission.csv             # Final prediction output file
+-- README.md                  # Project documentation
+-- .gitignore                 # Git ignore configuration
```

## Requirements
- Python 3.10+
- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- catboost
- scipy
- matplotlib
- seaborn
