# Data Drift and Model Degradation Monitoring in E-commerce

## Recommendation System Drift

### 1. Project Overview

This project focuses on monitoring changes in user and product
interaction behavior in an e-commerce environment.

Recommendation systems depend on historical user-product interaction
patterns. Changes in these patterns over time can introduce data drift
and potentially affect recommendation-system performance.

The project aims to analyze historical e-commerce interaction data,
identify distribution changes over time, and establish a foundation
for monitoring recommendation-system drift.

---

## 2. Dataset

Dataset used:

Retailrocket Recommender System Dataset

The Phase 1 analysis uses:

- `events.csv`

The dataset contains user-product interaction events.

### Event Types

- `view`
- `addtocart`
- `transaction`

---

## 3. Dataset Statistics

- Total interactions: 2,756,101
- Unique users: 1,407,580
- Unique products: 235,061
- Unique transactions: 17,672
- Duplicate rows removed: 460

### Data Period

May 3, 2015 to September 18, 2015

---

## 4. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the raw event dataset.
2. Checked dataset structure and data types.
3. Checked missing values.
4. Checked invalid visitor and product IDs.
5. Converted timestamps into datetime format.
6. Identified duplicate records.
7. Removed 460 duplicate records.
8. Saved the cleaned dataset separately.

The original dataset is preserved in:

`data/raw/events.csv`

The cleaned dataset is stored in:

`data/processed/events_cleaned.csv`

---

## 5. Exploratory Data Analysis

The following analyses were performed:

- Event distribution
- User activity
- Product popularity
- Daily interaction activity
- Monthly interaction behavior
- User behavior across time periods
- Product behavior across time periods

---

## 6. Time-Based Analysis

The dataset was divided into three periods:

| Period | Months |
|---|---|
| Baseline | May–June 2015 |
| Comparison | July–August 2015 |
| Recent | September 2015 |

These periods are used to compare changes in user and product behavior.

---

## 7. Drift Analysis

Jensen-Shannon Divergence (JSD) was used to measure distribution
differences between the baseline and later periods.

### Event Distribution Drift

| Comparison | JSD |
|---|---:|
| Baseline vs Comparison | 0.00281 |
| Baseline vs Recent | 0.00189 |

The overall event distribution remained relatively stable.

### Product Distribution Drift

| Comparison | JSD |
|---|---:|
| Baseline vs Comparison | 0.39279 |
| Baseline vs Recent | 0.52029 |

A substantially larger shift was observed in product interaction
distribution.

### User Behavioral Drift

| Comparison | JSD |
|---|---:|
| Baseline vs Comparison | 0.79883 |
| Baseline vs Recent | 0.81930 |

Average interactions per active user also decreased:

| Period | Average interactions/user |
|---|---:|
| Baseline | 1.979 |
| Comparison | 1.856 |
| Recent | 1.748 |

---

## 8. Preliminary Findings

The analysis indicates that overall event-type proportions remained
relatively stable over time.

However, substantial changes were observed in:

- Product interaction distribution
- User interaction behavior
- Average interactions per active user

The Top-10 product overlap was:

- Baseline vs Comparison: 1/10
- Baseline vs Recent: 0/10

These findings indicate that e-commerce interaction behavior changed
over time and provide evidence of data drift relevant to a
recommendation system.

Model degradation cannot yet be concluded because recommendation-model
performance has not been evaluated in Phase 1.

---

## 9. Project Structure

```text
Recommendation-Drift-Monitoring/
│
├── data/
│   ├── raw/
│   │   └── events.csv
│   │
│   └── processed/
│       ├── events_cleaned.csv
│       ├── period_summary.csv
│       ├── drift_results.csv
│       └── phase1_summary.csv
│
├── notebooks/
│   └── 01_data_analysis.ipynb
│
├── README.md
│
└── ...



### Save it

Press:

**Ctrl + S**

---

## Important: don't upload the large CSV files to GitHub

Before our first push, we'll create `.gitignore` so:

```text
events.csv
events_cleaned.csv