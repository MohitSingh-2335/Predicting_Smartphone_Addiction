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
- Analyzed feature distributions and missing values across both training and test sets.
- Examined target class balance (~68.3% addicted vs 31.7% non-addicted).

### 2. Feature Engineering
Created 78 domain-specific interaction, generator budget residual, circadian ratio, temporal, micro-interaction, and categorical interaction features:
- `accounted_screen_time`: Sum of social media, gaming, and work/study hours.
- `unaccounted_screen_time`: Latent budget discrepancy (`daily_screen_time_hours - accounted_screen_time`).
- `unaccounted_ratio`: Ratio of unaccounted screen time to total screen time.
- `weekend_budget_residual`: Latent budget residual for weekend duration.
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
- `stress_academic_combo_code`, `gender_stress_combo_code`, `triple_combo_code`: High-cardinality multi-way interaction codes.

### 3. Data Preprocessing
- Encoded categorical variables (`gender`, `academic_work_impact`, `stress_level`) into numerical formats and interaction codes.
- Retained the complete dataset (691,369 samples) without discarding rows with missing values.

### 4. Model Training and Cross-Validation
- Implemented a 4-Family Diverse 10-Fold Stratified Ensemble combining `LightGBM` (leaf-wise gradient boosting with 190 leaves, depth 12), `XGBoost` (histogram tree method with depth 9), `CatBoost` (symmetric oblivious trees), and `HistGradientBoostingClassifier` (scikit-learn monotonic booster) across 78 engineered features.
- Total of 40 models trained across 10 folds.
- Predictions converted to uniform percentile ranks (`rankdata(p) / len(p)`) and meta-optimized using Nelder-Mead optimization on honest out-of-fold predictions.

## Results and Benchmark

| Metric | Standalone / OOF CV | Ensemble Result |
| :--- | :--- | :--- |
| LightGBM Full OOF ROC-AUC | 0.96417 | - |
| HistGBM Full OOF ROC-AUC | 0.96371 | - |
| XGBoost Full OOF ROC-AUC | 0.96358 | - |
| CatBoost Full OOF ROC-AUC | 0.95811 | - |
| 4-Family Rank Ensemble OOF ROC-AUC | - | 0.96432 |
| Previous Best Public Leaderboard | - | 0.96584 |

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
