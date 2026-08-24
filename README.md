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
Created 77 domain-specific features capturing generator budget residuals, circadian dynamics, compulsive micro-interaction metrics, and interaction flags:
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

### 3. Out-of-Fold Target Encoding
- Computed strict out-of-fold target encoding on high-variance discrete lookup features (`notifications_per_day` and `app_opens_per_day`).
- Mappings were derived strictly on training splits and mapped to validation and test folds, eliminating any data leakage while providing strong conditional target risk signals to tree split algorithms.

### 4. 10-Fold Stratified Tri-Model Ensemble
- Implemented a 10-Fold Stratified Cross-Validation pipeline combining:
  1. `LightGBM Classifier` (50% weight): Deep leaf-wise gradient booster (`num_leaves=180`, `max_depth=12`, `feature_fraction=0.60`, `learning_rate=0.025`).
  2. `XGBoost Classifier` (35% weight): Histogram-based tree booster (`max_depth=8`, `colsample_bytree=0.60`, `subsample=0.80`, `learning_rate=0.030`).
  3. `HistGradientBoostingClassifier` (15% weight): Bin-based gradient booster (`max_leaf_nodes=160`, `l2_regularization=0.5`, `learning_rate=0.040`).
- Probability blending preserved confidence margins for confident predictions.

## Results and Benchmark

| Metric | Standalone / OOF CV | SOTA 10-Fold Ensemble |
| :--- | :--- | :--- |
| LightGBM Full OOF ROC-AUC | 0.96580 | - |
| XGBoost Full OOF ROC-AUC | 0.96575 | - |
| HistGBM Full OOF ROC-AUC | 0.96454 | - |
| 10-Fold Triple Ensemble OOF ROC-AUC | - | 0.96589 |
| Overall Classification Accuracy | - | 90.54% |
| F1-Score | - | 0.93359 |
| Precision | - | 0.93000 |
| Recall | - | 0.93721 |
| Peak Individual Fold AUC | - | 0.96696 |

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
- matplotlib
- seaborn
