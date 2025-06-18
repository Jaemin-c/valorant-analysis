# valorant-analysis
Jaemin Chung - jaeminc@umich.edu

## Step 1: Introduction

Valorant is a competitive tactical shooter where ranking up is the core goal for many players. At the very top of the ladder, **Radiant** players represent the top 0.1%, while **Immortal** ranks (Immortal 1–3) make up the highly skilled tier just below. Understanding the differences between these two groups is useful for both competitive players and analysts: it sheds light on what performance truly separates the best from the nearly best.

Our project aims to answer the following question:

> **Among Radiant and Immortal players, what differentiates Radiants from Immortals in terms of in-game performance stats?**

### Why This Question Matters

This question is relevant because it addresses what it really takes to become one of the best players in Valorant. While many players might assume that K/D ratio is everything, our analysis explores whether other factors like team coordination, consistency, or early-round impact matter more. Competitive gamers, coaches, and game designers alike can benefit from a clearer understanding of what defines top-tier performance.

---

### About the Dataset

- **Dataset size**: 85,574 rows  
- **Source**: A scraped dataset of Radiant and Immortal-ranked players and their match statistics.(Kaggle)

### 🔍 Relevant Columns Used:

| Column Name     | Description |
|-----------------|-------------|
| `kd_ratio`      | Kill-to-death ratio (K/D), a core performance indicator. |
| `win_percent`   | Overall win percentage across matches. |
| `damage_round`  | Average damage dealt per round. |
| `first_bloods`  | Number of rounds where the player secured the first kill. |
| `kills_round`   | Average number of kills per round. |
| `clutches`      | Number of rounds won as the last surviving player. |
| `flawless`      | Number of flawless rounds won by the team. |
| `rating`        | Player's actual rank (Radiant, Immortal 1/2/3). |
| `agent_1`       | The agent most commonly played by the user. |

We focus our analysis on these columns to explore which gameplay stats most strongly correlate with Radiant-level success, and whether some Immortals show Radiant-like characteristics.


## Step 2: Data Cleaning

To ensure consistent and reliable analysis, we cleaned the raw `val_stats.csv` dataset through the following steps:

### 🔧 1. Parsing Numerical Columns

Several numeric features were stored as strings or had missing/invalid values. We used `pd.to_numeric(..., errors='coerce')` to convert these columns safely to numeric values, coercing invalid entries to `NaN`. The columns processed include:

- `kd_ratio`
- `win_percent`
- `wins`
- `damage_round`
- `first_bloods`
- `kills_round`
- `clutches`
- `flawless`

This step was necessary to prepare the features for aggregation, visualization, and modeling.

### 2. Dropping Incomplete Rows

We removed rows with missing values in critical performance columns to ensure accuracy in later analysis.

```python
df.dropna(subset=['kd_ratio', 'win_percent', 'damage_round', 'wins', 'clutches', 'flawless', 'first_bloods'], inplace=True)
```

This improved data quality and reliability across all statistical comparisons.

### 🧍 3. Filling Missing Categorical Fields

Some non-numeric columns like player `name`, `tag`, and agent selections had blanks or missing values. We replaced these with the string `"Unknown"` to avoid errors in grouping and filtering.

```python
df['name'].replace(to_replace=['', np.nan], value='Unknown', inplace=True)
df['tag'].replace(to_replace=['', np.nan], value='Unknown', inplace=True)
df['agent_2'].replace(to_replace=['', np.nan], value='Unknown', inplace=True)
df['agent_3'].replace(to_replace=['', np.nan], value='Unknown', inplace=True)
```

### 4. Filtering for Relevant Ratings

To focus our project on the difference between **Radiant** and **Immortal** players, we filtered the dataset to only include the following ratings:

- Radiant
- Immortal 1
- Immortal 2
- Immortal 3

We then grouped all Immortal tiers together:

```python
df = df[df['rating'].isin(['Radiant', 'Immortal 1', 'Immortal 2', 'Immortal 3'])].copy()
df['rating_group'] = df['rating'].replace({
    'Immortal 1': 'Immortal',
    'Immortal 2': 'Immortal',
    'Immortal 3': 'Immortal',
    'Radiant': 'Radiant'
})
```

This allowed us to simplify comparisons and focus the model on the key question: what distinguishes Radiants from Immortals?

| region   | name          | tag    | rating   |   damage_round |   headshots |   headshot_percent |   aces |   clutches |   flawless |   first_bloods | kills   | deaths   |   assists |   kd_ratio |   kills_round |   most_kills |   score_round |   wins |   win_percent | agent_1   | agent_2   | agent_3   | gun1_name   |   gun1_head |   gun1_body |   gun1_legs |   gun1_kills | gun2_name   |   gun2_head |   gun2_body |   gun2_legs |   gun2_kills | gun3_name   |   gun3_head |   gun3_body |   gun3_legs |   gun3_kills | rating_group   |
|:---------|:--------------|:-------|:---------|---------------:|------------:|-------------------:|-------:|-----------:|-----------:|---------------:|:--------|:---------|----------:|-----------:|--------------:|-------------:|--------------:|-------:|--------------:|:----------|:----------|:----------|:------------|------------:|------------:|------------:|-------------:|:------------|------------:|------------:|------------:|-------------:|:------------|------------:|------------:|------------:|-------------:|:---------------|
| NA       | ShimmyXD      | #NA1   | Radiant  |          135.8 |         992 |               24.9 |      0 |        140 |         80 |            161 | 1,506   | 1,408    |       703 |       1.07 |           0.7 |           29 |         208.8 |     59 |          59.6 | Fade      | Viper     | Omen      | Vandal      |          35 |          59 |           5 |          802 | Phantom     |          33 |          62 |           5 |          220 | Classic     |          36 |          60 |           3 |          147 | Radiant        |
| NA       | XSET Cryo     | #cells | Radiant  |          170.3 |         879 |               28.3 |      2 |        122 |         94 |            316 | 1,608   | 1,187    |       206 |       1.35 |           1   |           32 |         270.6 |     52 |          65.8 | Chamber   | Jett      | Raze      | Vandal      |          41 |          56 |           3 |          689 | Operator    |           8 |          91 |           0 |          226 | Phantom     |          32 |          63 |           5 |          137 | Radiant        |
| NA       | PuRelittleone | #yoruW | Radiant  |          147.5 |         720 |               24   |      3 |        117 |         59 |            216 | 1,115   | 1,064    |       267 |       1.05 |           0.8 |           39 |         227.8 |     42 |          65.6 | Yoru      | Jett      | Chamber   | Vandal      |          38 |          57 |           4 |          444 | Phantom     |          36 |          61 |           3 |          231 | Operator    |           8 |          91 |           1 |          102 | Radiant        |
| NA       | Boba          | #0068  | Radiant  |          178.2 |         856 |               37.3 |      3 |         83 |         49 |            235 | 1,134   | 812      |       157 |       1.4  |           1   |           37 |         277   |     32 |          62.8 | Jett      | Chamber   | KAY/O     | Vandal      |          51 |          47 |           2 |          754 | Sheriff     |          48 |          51 |           1 |           48 | Phantom     |          44 |          56 |           0 |           36 | Radiant        |
| NA       | i love mina   | #kelly | Radiant  |          149.8 |         534 |               24.4 |      2 |         71 |         38 |            137 | 869     | 781      |       213 |       1.11 |           0.8 |           29 |         230.9 |     32 |          62.8 | Jett      | Raze      | Chamber   | Vandal      |          36 |          60 |           4 |          419 | Spectre     |          21 |          71 |           8 |           65 | Operator    |           8 |          92 |           0 |           64 | Radiant        |


## Step 3: Problem Identification

For this project, we aim to build a **predictive classification model** that distinguishes between **Radiant** and **Immortal** players in *Valorant* based on their in-game performance statistics.

### Prediction Problem  
**Can we predict whether a player is Radiant or Immortal based on their in-game stats such as KD ratio, win percentage, and damage per round?**

This is a **binary classification** task, where the response variable is:

- `rating` — a categorical variable with two classes: `"Radiant"` and `"Immortal"` (merged from Immortal 1, 2, and 3 for simplicity).

We chose this problem because reaching Radiant rank is extremely competitive, and identifying what makes Radiant players stand out can offer insights to aspiring high-rank players or coaches.

### Response Variable  
- **Response**: `rating`  
- **Type**: Categorical (Binary: Radiant vs. Immortal)

This was chosen because our broader research question is:  
**“What differentiates Radiants from Immortals in terms of in-game performance stats?”**

By modeling this as a classification problem, we can both **predict rank tier** and **gain insight into feature importance**.

### Model Type: Predictive or Inferential  
This is a **predictive model**, not inferential. Our goal is to **accurately predict a player's rank tier**, not necessarily to interpret causal relationships between the features and the rank. The input features we use (e.g., win percentage, clutch counts, KD ratio) are **realistically known before** a rank is assigned or changed — thus they are suitable for prediction.

### Evaluation Metric  
We will evaluate model performance using **F1-Score**, as our classes may be imbalanced (e.g., fewer Radiants than Immortals). F1-score is preferred over accuracy in this context because it balances both **precision** and **recall**, giving a better sense of model performance when one class is harder to predict correctly.
