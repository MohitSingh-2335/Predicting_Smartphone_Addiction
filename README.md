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

### 4. Regime-Gated Mixture of Experts (MoE) 60-Model GPU Pipeline
- Implemented a 10-Fold Stratified Cross-Validation pipeline combining 6 distinct core deep model architectures with dynamic regime-gating meta-learning:
  1. `Deep CatBoost Native Ordered TE` (10 models): GPU-accelerated symmetric decision trees using internal permutation-based ordered target statistics on raw string levels (`depth=7`, `learning_rate=0.035`, `iterations=2600`, `task_type='GPU'`).
  2. `Deep XGBoost Hist` (10 models): Deep histogram tree booster on Bayesian target encodings and decimal lattice (`max_depth=8`, `learning_rate=0.022`, `subsample=0.80`, `colsample_bytree=0.70`, `device='cuda'`).
  3. `XGBoost TE + Lattice` (10 models): Balanced histogram-based booster on Bayesian target encodings and decimal lattice (`max_depth=6`, `learning_rate=0.028`, `subsample=0.80`, `colsample_bytree=0.80`, `device='cuda'`).
  4. `Regularized XGBoost` (10 models): Regularized shallow tree booster for smooth gradients (`max_depth=5`, `learning_rate=0.032`, `reg_alpha=1.5`, `reg_lambda=6.0`, `device='cuda'`).
  5. `High-Capacity LightGBM` (10 models): Deep leaf-wise booster on Bayesian target encodings and decimal lattice (`num_leaves=127`, `max_depth=9`, `learning_rate=0.025`, `colsample_bytree=0.80`, `subsample=0.80`).
  6. `CatBoost on Augmented Ratios` (10 models): Decorrelated ratio booster for subtractive error correction (`depth=6`, `learning_rate=0.040`, `iterations=2000`, `task_type='GPU'`).
- **Regime-Gated Mixture of Experts (MoE) Meta-Layer**:
  - Extracts 22 mathematical regime attributes (`is_int`, `is_half`, `frac`, `nan_count`, raw categorical levels).
  - Dynamically routes samples combining **Multi-Seed Linear Logit Stacking** and a **Regime-Gated Decision Tree Stacker** with uncompressed continuous probability calibration.

## Results and Benchmark

| Metric | Standalone OOF AUC | Linear Meta-Stacker | Regime-Gated MoE Pipeline |
| :--- | :--- | :--- | :--- |
| Deep CatBoost Depth-7 OOF AUC | 0.96867 | - | - |
| Deep XGBoost Depth-8 OOF AUC | 0.96885 | - | - |
| XGBoost Depth-6 Hist OOF AUC | 0.96885 | - | - |
| XGBoost Depth-5 Reg OOF AUC | 0.96883 | - | - |
| LightGBM Num127 OOF AUC | 0.96864 | - | - |
| CatBoost Augmented Ratios OOF AUC | 0.96077 | - | - |
| **Final Stacked OOF ROC-AUC** | - | **0.96915** | **0.96920** |
| **Overall Classification Accuracy** | - | 91.04% | **91.05%** |
| **F1-Score** | - | 0.93703 | **0.93710** |
| **Precision** | - | 0.93484 | **0.93482** |
| **Recall** | - | 0.93922 | **0.93940** |
| **Peak Individual Fold AUC** | - | **0.96957** | **0.96957** |
| **Public Leaderboard ROC-AUC** | - | **0.97024** | **Ready for Submission (Target 0.9715+)** |

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
| 17 | Rank-01 Multi-Stream Normalization | 37 | 0.96915 | 0.97019 |
| 18 | Deep 60-Model GPU Dual-Engine Meta-Stack | 37 | 0.96920 | 0.97024 |
| 19 | **Regime-Gated Mixture of Experts (MoE) 60-Model GPU Pipeline** | **37** | **0.96920** | **Ready for Submission** |

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
