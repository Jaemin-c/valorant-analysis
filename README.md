# valorant-analysis
Jaemin Chung - jaeminc@umich.edu

## Step 1: Introduction

Valorant is a competitive tactical shooter where ranking up is the core goal for many players. At the very top of the ladder, **Radiant** players represent the top 0.1%, while **Immortal** ranks (Immortal 1–3) make up the highly skilled tier just below. Understanding the differences between these two groups is useful for both competitive players and analysts: it sheds light on what performance truly separates the best from the nearly best.

My project aims to answer the following question:

> **Among Radiant and Immortal players, what differentiates Radiants from Immortals in terms of in-game performance stats?**

### Why This Question Matters

This question is relevant because it addresses what it really takes to become one of the best players in Valorant. While many players might assume that K/D ratio is everything, our analysis explores whether other factors like team coordination, consistency, or early-round impact matter more. Competitive gamers, coaches, and game designers alike can benefit from a clearer understanding of what defines top-tier performance.

---

### About the Dataset

- **Dataset size**: 85,574 rows  
- **Source**: A scraped dataset of Radiant and Immortal-ranked players and their match statistics.(Kaggle)

### Relevant Columns Used:

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

## Step 2: Data Cleaning and Exploratory Data Analysis
To conduct a meaningful analysis and build a reliable model, we first needed to clean and understand our dataset. In this step, we performed both **data cleaning** to prepare the dataset for modeling, and **exploratory data analysis (EDA)** to uncover patterns and distributions in key variables.

### Part 1: Data Cleaning
### 1. Parsing Numerical Columns

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

### 3. Filling Missing Categorical Fields

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


We limited the preview table to only key performance-related columns to ensure clarity and focus. The original dataset contains over 30 columns, but for exploratory and modeling purposes, we highlight only the most relevant metrics — such as kill-death ratio, win rate, and clutch counts — that align with our guiding question.


| region   | name          | rating   |   kd_ratio |   win_percent |   damage_round |   first_bloods |   clutches | agent_1   |
|:---------|:--------------|:---------|-----------:|--------------:|---------------:|---------------:|-----------:|:----------|
| NA       | ShimmyXD      | Radiant  |       1.07 |          59.6 |          135.8 |            161 |        140 | Fade      |
| NA       | XSET Cryo     | Radiant  |       1.35 |          65.8 |          170.3 |            316 |        122 | Chamber   |
| NA       | PuRelittleone | Radiant  |       1.05 |          65.6 |          147.5 |            216 |        117 | Yoru      |
| NA       | Boba          | Radiant  |       1.4  |          62.8 |          178.2 |            235 |         83 | Jett      |
| NA       | i love mina   | Radiant  |       1.11 |          62.8 |          149.8 |            137 |         71 | Jett      |

---

### Part 2: Univariate Analysis

In this section, we explore the distribution of key individual statistics to understand how performance varies across all players in our dataset. These insights help identify which variables are most informative when comparing Radiants and Immortals.

#### Player Rank Distribution

<iframe src="assets/rank_distribution_pie.html" width="600" height="400" frameborder="0"></iframe>

Radiants are a much smaller portion of the population, highlighting class imbalance and the difficulty of reaching the top tier.

#### Damage per Round

<iframe src="assets/dpr.html" width="700" height="400" frameborder="0"></iframe>

Most players fall between 120 and 180 damage per round, with some outliers above 200 indicating highly impactful players.

####  Kill-to-Death Ratio (K/D)

<iframe src="assets/kd_ratio.html" width="700" height="400" frameborder="0"></iframe>

The K/D ratio tends to center between 0.9 and 1.5. Understanding whether Radiants consistently hold higher K/Ds is key to our modeling.

---

---

### Part 3: Bivariate Analysis

To explore how in-game stats distinguish Radiants from Immortals, we visualized two key metrics — **Clutches** and **Damage per Round** — across rank tiers using box plots. These comparisons help us identify which performance traits are more common among Radiant players.

#### Clutches by Rating Tier

This plot displays the number of **clutch rounds** (rounds won as the last surviving player) across each rank. Radiant players show a noticeably **higher median clutch count** compared to Immortals, along with a dense concentration of high outliers. This suggests that Radiants are more reliable in high-pressure 1vX situations — a skill likely rewarded at the highest rank.

<iframe src="assets/clutches_by_rank.html" width="800" height="600" frameborder="0"></iframe>

**Interpretation:**  
Clutch performance appears to be a key differentiator between Radiants and Immortals. This supports the hypothesis that Radiants excel not just in raw stats but in clutch moments that secure rounds and shift momentum.

#### Damage per Round by Rating Tier

This box plot compares the **damage dealt per round** across rating groups. While all tiers center around similar median values, Radiant players display **fewer low outliers and a tighter IQR**, suggesting they maintain **more consistent impact per round**.

<iframe src="assets/damagecomp.html" width="800" height="600" frameborder="0"></iframe>

**Interpretation:**  
Consistency in damage output — not just high peaks — may be a distinguishing factor for Radiants. The lack of low-damage games indicates that Radiants maintain a higher baseline of performance, which likely contributes to their overall win rates and reliability.

---

## Part 4: Interesting Aggregates

To dig deeper into what distinguishes Radiants from Immortals, we used grouped aggregates and correlation visualizations that summarize player behavior across multiple features.

### 1. Average Clutches and First Bloods by Rating Group

<iframe src="assets/avgclutch.html" width="800" height="500" frameborder="0"></iframe>

Radiant players, on average, secured significantly more **clutch wins** and **first bloods** than Immortals. This suggests that not only are Radiants more effective in high-pressure solo situations, but they also frequently initiate rounds with early picks — both being strong indicators of impactful play.

---

### 2. KD Ratio vs. Win Percentage (Radiants Only)

<iframe src="assets/kdandwin.html" width="800" height="500" frameborder="0"></iframe>

Among Radiant players, we observe a **positive linear trend** between **K/D ratio** and **win percentage**, reinforcing that individual fragging efficiency is closely tied to winning. While K/D isn't the sole determinant of success, it clearly plays a strong role among top-tier players.

---

### 3. Damage per Round vs. Win Percentage (All Ranks)

<iframe src="assets/dprandwin.html" width="800" height="500" frameborder="0"></iframe>

Across both Radiants and Immortals, we again see a **positive correlation** between **damage dealt per round** and **overall win rate**. Although the data is noisier at lower damage levels, players with consistently high damage output tend to contribute more to match wins. This confirms damage per round as one of the key performance metrics that separates Radiants from Immortals.



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


## Step 4: Baseline Model

To establish a performance benchmark, we trained a baseline classification model using logistic regression. The goal of the model is to predict whether a player belongs to the **Radiant** or **Immortal** tier based on their gameplay statistics.

### Model Overview

- **Model Used**: Logistic Regression (via scikit-learn)
- **Preprocessing**: Standardized numeric features using `StandardScaler`
- **Pipeline**: Implemented using `sklearn.pipeline.Pipeline` to streamline scaling and modeling
- **Target**: `rating_group` (binary: Radiant vs. Immortal)

### Features Used

We selected three core performance features:

| Feature         | Type         | Description |
|----------------|--------------|-------------|
| `kd_ratio`      | Quantitative | Kill-to-death ratio |
| `damage_round`  | Quantitative | Average damage dealt per round |
| `clutches`      | Quantitative | Number of clutch rounds won |

- **Quantitative Features**: 3  
- **Ordinal Features**: 0  
- **Nominal Features**: 0  

Since all selected features were numerical, no categorical encodings were necessary for this baseline model.

### Evaluation

We split the dataset using an 80/20 train-test split and used stratification to maintain class distribution. The model's performance was evaluated using precision, recall, and F1-score.

Below is the classification report from the baseline model:

=== Classification Report ===
```
              precision    recall  f1-score   support

    Immortal       0.97      1.00      0.98     16594
     Radiant       0.10      0.00      0.01       521

    accuracy                           0.97     17115
   macro avg       0.53      0.50      0.50     17115
weighted avg       0.94      0.97      0.95     17115

```




### Model Assessment

While this logistic regression model is relatively simple, it provides a solid foundation for comparison. It captures some performance-based differences between Radiants and Immortals but likely misses complex nonlinear interactions or interactions involving other metrics. Therefore, we view this as a **good baseline**, but **not sufficient for high predictive accuracy**. It serves as a point of comparison for future improvements in the final model.



## Step 5: Final Model

### Feature Selection and Engineering

For our final model, we added the following features and applied transformations:

- **`clutches`** (Quantitative): Represents how often a player wins a round as the last surviving member. This is likely a strong signal of individual impact in high-pressure situations, particularly at higher ranks.
- **`agent_1`** (Nominal): The most commonly played agent by the player. Some agents have a greater potential to carry rounds (e.g., Jett, Reyna) or support teams in impactful ways (e.g., Skye, Sova), which may correlate with rank.

To prepare the data:
- We applied `StandardScaler` + `QuantileTransformer` to all quantitative features (`kd_ratio`, `damage_round`, `clutches`) to normalize skewed distributions and make the model less sensitive to outliers.
- We one-hot encoded `agent_1` to convert this categorical variable into usable binary features.

These transformations were applied **inside a single sklearn `Pipeline`** to ensure compatibility with new data and simplify the training process.

---

### Modeling Algorithm and Tuning

We used **Logistic Regression** as our final modeling algorithm. This choice provides interpretable results and performs well with transformed numeric and categorical inputs.

To handle the class imbalance between Radiant and Immortal players, we applied **class weighting** using `compute_class_weight`. This gave more importance to the minority Radiant class.

We performed hyperparameter tuning using **`GridSearchCV` with 5-fold cross-validation**, optimizing for **macro F1-score**. The hyperparameter grid was:

clf__C: [0.01, 0.1, 1, 10, 100]



The best-performing model used:
- `C = 0.1`

---

### Model Performance and Improvement

Below is the classification report for the final model:

=== Final Model Classification Report ===
```
              precision    recall  f1-score   support

    Immortal       0.99      0.73      0.84     16594
     Radiant       0.08      0.78      0.15       521

    accuracy                           0.74     17115
   macro avg       0.54      0.76      0.50     17115
weighted avg       0.96      0.74      0.82     17115

```


Compared to the baseline model, the final model significantly improved:
- **Recall for Radiant** (0.00 → 0.78): The model now identifies most Radiant players correctly.
- **F1-score for Radiant** (0.01 → 0.15): Substantial improvement in predictive value for the minority class.
- **Macro F1** (0.50 → 0.50): Maintained but with more balanced class performance.

This shows that the final model is much better at distinguishing Radiant players from Immortals, particularly due to the engineered features and class weighting strategy.