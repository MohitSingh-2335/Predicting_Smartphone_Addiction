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

### 2. Feature Engineering & Multi-Frequency Encodings
1. **Multi-Frequency Trigonometric Lookups**:
   - High-frequency periodic transformations (`sin` and `cos` with periods 10.0, 20.0, 50.0) applied to discrete generator lookup keys (`notifications_per_day`, `app_opens_per_day`, `age`).
   - Enables tree partition algorithms to capture complex disjoint interval unions and recurring synthetic table patterns.
2. **Extended Decimal Lattice & Remainder Trigonometry**:
   - `frac_col`: Continuous sub-unit remainder (`value - floor(value)`) capturing generator decimal distributions.
   - `d1_col`: First decimal digit (`floor(frac * 10)`) exposing discretization boundaries.
   - `is_int_col` & `is_half_col`: Binary indicator flags for exact integer and half-unit measurements.
   - `sin_frac` & `cos_frac`: Trigonometric coordinates on remainder circles (`2 * pi * frac`).
   - Applied across continuous columns: `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `weekend_screen_time`.
3. **Transductive Population Frequency Encodings**:
   - Empirical frequency statistics computed across the combined pool of training and test sets to capture generator discretization density without target leakage.
4. **Generator Budget Constraints & Discrepancies**:
   - `accounted_screen_time`: Sum of social media, gaming, and work/study hours.
   - `unaccounted_screen_time`: Latent budget discrepancy (`daily_screen_time_hours - accounted_screen_time`).
   - `unaccounted_ratio`: Ratio of unaccounted screen time to total screen time.
   - `weekend_budget_residual`: Latent budget residual for weekend duration (`weekend_screen_time - 2 * (social + gaming + work)`).
   - `non_study_screen_hours`: Non-productive screen time (`daily_screen_time_hours - work_study_hours`).
   - `entertainment_hours`: Total recreational screen time (`social_media_hours + gaming_hours`).
   - `entertainment_ratio`: Proportion of daily screen time spent on entertainment.
   - `unproductive_to_productive`: Ratio of entertainment hours to work and study hours.
5. **Circadian Dynamics & Micro-Interactions**:
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

### 3. Leak-Free 10-Fold Nested Bayesian Target Encoding
- Applied nested 10-fold out-of-fold Bayesian smoothed target encoding across all 12 raw continuous and categorical features.
- Explicit string level `__missing__` assigned to missing values to preserve missingness distribution.
- Prior smoothing factor `m = 10.0` with simultaneous in-fold frequency encodings.

### 4. Ultra SOTA Top-100 Rank-01 Multi-Stream Ensemble Pipeline
- Implemented a multi-stream ensembling architecture combining dual 50-model cross-fitted streams with Rank-01 Percentile Normalization:
  - **Stream 1 (In-House SOTA 0.97024 Anchor)**: 50-model logit-stack ensemble across 10 stratified folds on Seed 42 combining XGBoost TE+Lattice, CatBoost Native Ordered TE, LightGBM TE+Lattice, Deep XGBoost Hist, and CatBoost Aug Ratios.
  - **Stream 2 (Multi-Scale Engine on Seed 2026)**: 50-model multi-scale gradient boosted tree ensemble on Seed 2026 combining XGBoost Depth-6 Hist, XGBoost Depth-5 Regularized, LightGBM NumLeaves=63, Deep XGBoost Depth-8 Hist, and CatBoost Native Ordered TE.
  - **Feature Enhancements**: Integer and half-integer arithmetic flags (`is_int`, `is_half`), exact decimal digits (`d1_col`), sub-unit position (`frac_col`), and transductive composition ratios.
  - **Rank-01 Normalization & Dual Master Fusion**: Convex percentile rank combination ($R_{\text{final}} = 0.55 \cdot R_{\text{Stream1}} + 0.45 \cdot R_{\text{Stream2}}$) mapped uniformly onto $[0, 1]$ to optimize global ROC-AUC rank discrimination.

## Results and Benchmark

| Metric | Stream 1 Champion OOF | Stream 2 Multi-Scale OOF | Ultra SOTA Rank-01 Ensemble |
| :--- | :--- | :--- | :--- |
| XGBoost Hist OOF AUC | 0.96883 | 0.96880 | - |
| XGBoost Regularized OOF AUC | - | 0.96881 | - |
| CatBoost Native Ordered TE OOF AUC | 0.96870 | 0.96836 | - |
| LightGBM OOF AUC | 0.96866 | 0.96862 | - |
| Deep XGBoost Hist OOF AUC | 0.96883 | 0.96882 | - |
| **Stacked OOF ROC-AUC** | **0.96915** | **0.96903** | **Dual Master Fusion** |
| **Overall Classification Accuracy** | 91.05% | 91.04% | **91.05%** |
| **F1-Score** | 0.93710 | 0.93699 | **0.93710** |
| **Precision** | 0.93478 | 0.93457 | **0.93478** |
| **Recall** | 0.93943 | 0.93942 | **0.93943** |
| **Peak Individual Fold AUC** | 0.96956 | 0.96979 | **0.96979** |
| **Verified Public Leaderboard ROC-AUC** | **0.97024** | - | **Ready for Submission (Target 0.9715+)** |

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
| 12 | Breakthrough 10-Fold 4-Model Ensemble | 71 | 0.96822 | 0.96936 |
| 13 | Titan Dual-Seed 80-Model SOTA Ensemble | 95 | 0.96853 | 0.96956 |
| 14 | Apex Dual-Seed 80-Model SOTA Ensemble | 125 | 0.96870 | 0.96968 |
| 15 | Zenith Tri-Seed 120-Model GPU Pipeline | 146 | 0.96840 | 0.96944 |
| 16 | Pure 0.970+ SOTA Logit-Stack Ensemble | 35 | 0.96915 | 0.97024 |
| 17 | **Ultra SOTA Top-100 Rank-01 Multi-Stream Ensemble** | **37** | **0.96915** | **Ready for Submission** |

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
