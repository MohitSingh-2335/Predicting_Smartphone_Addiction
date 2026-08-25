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

### 2. Feature Engineering & Generator Budget Residuals
Created 78 domain-specific features capturing generator budget residuals, circadian dynamics, compulsive micro-interaction metrics, and interaction flags:
- `accounted_screen_time`: Sum of social media, gaming, and work/study hours.
- `unaccounted_screen_time`: Latent budget discrepancy (`daily_screen_time_hours - accounted_screen_time`).
- `unaccounted_ratio`: Ratio of unaccounted screen time to total screen time.
- `weekend_budget_residual`: Latent budget residual for weekend duration (`weekend_screen_time - 2 * (social + gaming + work)`).
- `non_study_screen_hours`: Non-productive screen time (`daily_screen_time_hours - work_study_hours`).
- `non_study_to_sleep_ratio`: Ratio of non-study screen time to sleep hours.
- `entertainment_hours`: Total recreational screen time (`social_media_hours + gaming_hours`).
- `entertainment_ratio`: Proportion of daily screen time spent on entertainment.
- `unproductive_to_productive`: Ratio of entertainment hours to work and study hours.
- `screen_time_to_awake_ratio`: Proportion of waking hours spent looking at a screen (`daily_screen_time_hours / (24 - sleep_hours)`).
- `free_awake_hours`: Free waking hours excluding work and study (`(24 - sleep_hours) - work_study_hours`).
- `screen_to_free_awake_ratio`: Proportion of free awake hours spent on entertainment screen activities.
- `high_screen_low_sleep`: High risk flag for >= 8h screen time and <= 5h sleep.
- `high_notif_high_open`: High risk interaction flag for notifications and app opens.
- `severe_impact_stress`: Compound flag for severe academic work impact combined with high stress.
- `notifications_per_awake_hour`: Notification density during active waking hours.
- `minutes_per_app_open`: Average duration spent per app open.
- `compulsive_check_rate`: Frequency of app opens per minute of active screen time.
- `weekend_vs_daily_diff`: Increase in screen time during weekends.
- `total_weekly_screen_time`: Combined 7-day screen time exposure.
- `addiction_index_v2`: Weighted composite indicator combining non-study screen hours, recreational use, notifications, and sleep deprivation.
- `extreme_screen_flag` and `severe_sleep_debt`: Binary flags for extreme behavioral habits (>12h screen or <4.5h sleep).
- `num_missing`: Count of missing fields per record.

### 3. Bayesian Smoothed Out-of-Fold Target Encoding
- Computed leak-free Bayesian smoothed target encoding across discrete lookup features (`notifications_per_day`, `app_opens_per_day`, `age`, and `stress_academic_combo_code`).
- Smoothing formulation: `(count * mean + 20 * global_mean) / (count + 20)`.
- Mappings derived strictly on training splits and mapped to validation and test folds.

### 4. Titan Tri-Seed 10-Fold Stratified Ensemble (90 Models)
- Implemented a Tri-Seed 10-Fold Stratified Cross-Validation pipeline (Seeds 42, 2024, and 777) training 90 total deep models:
  1. `LightGBM Classifier` (30 models): Deep leaf-wise gradient booster (`num_leaves=200`, `max_depth=12`, `feature_fraction=0.60`, `learning_rate=0.020`, `n_estimators=1800`).
  2. `XGBoost Classifier` (30 models): Histogram-based tree booster (`max_depth=9`, `colsample_bytree=0.60`, `subsample=0.80`, `learning_rate=0.025`, `n_estimators=1200`).
  3. `HistGradientBoostingClassifier` (30 models): Bin-based gradient booster (`max_leaf_nodes=180`, `l2_regularization=0.5`, `learning_rate=0.035`, `max_iter=280`).
- Averaging across all three random seeds minimizes partition variance, eliminates fold boundaries, and produces smooth prediction distributions.

## Results and Benchmark

| Metric | Standalone / OOF CV | Tri-Seed 90-Model Titan Ensemble |
| :--- | :--- | :--- |
| Tri-Seed Bagged LightGBM OOF ROC-AUC | 0.96611 | - |
| Tri-Seed Bagged XGBoost OOF ROC-AUC | 0.96603 | - |
| Tri-Seed Bagged HistGBM OOF ROC-AUC | 0.96479 | - |
| **Tri-Seed 10-Fold Ensemble OOF ROC-AUC** | - | **0.96617** |
| **Overall Classification Accuracy** | - | **90.58%** |
| **F1-Score** | - | **0.93387** |
| **Precision** | - | **0.93031** |
| **Recall** | - | **0.93747** |
| **Peak Individual Fold AUC** | - | **0.96712** |
| **Verified Public Leaderboard ROC-AUC** | - | **0.96745** |

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
| 10 | Dual-Seed 60-Model Ensemble + Bayesian TE | 78 | 0.96616 | **0.96745** |
| 11 | **Titan Tri-Seed 90-Model Deep Ensemble** | **78** | **0.96617** | **Ready for Submission** |

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
- scipy
- matplotlib
- seaborn
