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
- Examined target class balance (~70.9% addicted vs 29.1% non-addicted).

### 2. Feature Engineering
Created domain-specific interaction and ratio features to capture addiction patterns:
- `entertainment_hours`: Total recreational screen time (`social_media_hours + gaming_hours`).
- `entertainment_ratio`: Proportion of daily screen time spent on entertainment.
- `unproductive_to_productive`: Ratio of entertainment hours to work and study hours.
- `screen_time_to_awake_ratio`: Proportion of waking hours spent looking at a screen (`daily_screen_time_hours / (24 - sleep_hours)`).
- `notifications_per_awake_hour`: Notification density during active waking hours.
- `minutes_per_app_open`: Average duration spent per app open.
- `weekend_vs_daily_diff`: Increase in screen time during weekends.
- `num_missing`: Count of missing fields per record.

### 3. Data Preprocessing
- Encoded categorical variables (`gender`, `academic_work_impact`, `stress_level`) into numerical formats.
- Retained the complete dataset (691,369 samples) without discarding rows with missing values.

### 4. Model Training and Cross-Validation
- Implemented a 10-Fold Stratified Dual-Ensemble combining `LightGBM` (leaf-wise gradient boosting) and `HistGradientBoostingClassifier` across 38 engineered features.
- Trained on 90% of the dataset per fold (622,000 samples) to ensure high capacity and low variance.
- Ensembled test predictions across all 20 trained models (10 folds x 2 architectures) for calibrated probability outputs.

## Results and Benchmark

| Metric | Out-Of-Fold Cross-Validation | Leaderboard Score |
| :--- | :--- | :--- |
| ROC-AUC | 0.96368 | 0.96459+ (Baseline: 0.92374) |
| F1-Score | 0.93141 | - |
| Accuracy | 90.21% | - |
| Precision | 92.57% | - |
| Recall | 93.72% | - |

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
- matplotlib
- seaborn
