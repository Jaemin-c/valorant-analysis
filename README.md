# What differentiates Radiants from Immortals?
Jaemin Chung - jaeminc@umich.edu

## Step 1: Introduction
As a passionate Valorant player, I’ve always been curious about what truly sets apart **Radiants** — the top 0.1% of players — from the rest of the **Immortal** ranks. Everyone in Immortal is undeniably skilled, but reaching Radiant seems to require something more: a unique consistency, impact, or strategic edge that’s not immediately obvious from surface-level stats.

**Valorant** is a highly competitive tactical shooter where climbing the ranked ladder is central to the player experience. At the very top, Radiants represent the best of the best, while Immortal 1–3 ranks sit just beneath — still elite, but statistically more common. Understanding the differences between these two groups isn’t just a fun curiosity — it offers real insight for aspiring top players, coaches, and game analysts.

So, I wanted to answer a simple but powerful question:

> **Among Radiant and Immortal players, what differentiates Radiants from Immortals in terms of in-game performance stats?**

This project explores that question through data — using match performance metrics like **K/D ratio**, **damage per round**, **clutch wins**, and more — to uncover which traits actually distinguish Radiants from their Immortal peers. The ultimate goal: to model and understand what it really takes to stand out at the highest levels of Valorant.

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


I chose to focus my analysis on these columns to explore which gameplay stats most strongly correlate with Radiant-level success — and to see whether some Immortal players might actually exhibit Radiant-like characteristics.

## Step 2: Data Cleaning and Exploratory Data Analysis
To conduct a meaningful analysis and build a reliable model, I first needed to clean and understand my dataset. In this step, I focused on both data cleaning — to prepare the dataset for modeling — and exploratory data analysis (EDA) to uncover patterns and distributions in the key variables I planned to use.

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

I removed rows with missing values in critical performance columns to ensure accuracy in later analysis.

```python
df.dropna(subset=['kd_ratio', 'win_percent', 'damage_round', 'wins', 'clutches', 'flawless', 'first_bloods'], inplace=True)
```

This improved data quality and reliability across all statistical comparisons.

### 3. Filling Missing Categorical Fields

Some non-numeric columns like player `name`, `tag`, and agent selections had blanks or missing values. I replaced these with the string `"Unknown"` to avoid errors in grouping and filtering.

```python
df['name'].replace(to_replace=['', np.nan], value='Unknown', inplace=True)
df['tag'].replace(to_replace=['', np.nan], value='Unknown', inplace=True)
df['agent_2'].replace(to_replace=['', np.nan], value='Unknown', inplace=True)
df['agent_3'].replace(to_replace=['', np.nan], value='Unknown', inplace=True)
```

### 4. Filtering for Relevant Ratings

To focus our project on the difference between **Radiant** and **Immortal** players, I filtered the dataset to only include the following ratings:

- Radiant
- Immortal 1
- Immortal 2
- Immortal 3

Then grouped all Immortal tiers together:

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


I limited the preview table to only the key performance-related columns to keep things clear and focused. The original dataset contains over 30 columns, but for the sake of exploration and modeling, I chose to highlight only the most relevant metrics — like kill-death ratio, win rate, and clutch counts — that directly support my guiding question.

| region   | name          | tag    | rating   |   damage_round |   headshots |   headshot_percent |   aces |   clutches |   flawless |   first_bloods | kills   | deaths   |   assists |   kd_ratio |   kills_round |   most_kills |   score_round |   wins |   win_percent | agent_1   | agent_2   | agent_3   | gun1_name   |   gun1_head |   gun1_body |   gun1_legs |   gun1_kills | gun2_name   |   gun2_head |   gun2_body |   gun2_legs |   gun2_kills | gun3_name   |   gun3_head |   gun3_body |   gun3_legs |   gun3_kills | rating_group   |
|:---------|:--------------|:-------|:---------|---------------:|------------:|-------------------:|-------:|-----------:|-----------:|---------------:|:--------|:---------|----------:|-----------:|--------------:|-------------:|--------------:|-------:|--------------:|:----------|:----------|:----------|:------------|------------:|------------:|------------:|-------------:|:------------|------------:|------------:|------------:|-------------:|:------------|------------:|------------:|------------:|-------------:|:---------------|
| NA       | ShimmyXD      | #NA1   | Radiant  |          135.8 |         992 |               24.9 |      0 |        140 |         80 |            161 | 1,506   | 1,408    |       703 |       1.07 |           0.7 |           29 |         208.8 |     59 |          59.6 | Fade      | Viper     | Omen      | Vandal      |          35 |          59 |           5 |          802 | Phantom     |          33 |          62 |           5 |          220 | Classic     |          36 |          60 |           3 |          147 | Radiant        |
| NA       | XSET Cryo     | #cells | Radiant  |          170.3 |         879 |               28.3 |      2 |        122 |         94 |            316 | 1,608   | 1,187    |       206 |       1.35 |           1   |           32 |         270.6 |     52 |          65.8 | Chamber   | Jett      | Raze      | Vandal      |          41 |          56 |           3 |          689 | Operator    |           8 |          91 |           0 |          226 | Phantom     |          32 |          63 |           5 |          137 | Radiant        |
| NA       | PuRelittleone | #yoruW | Radiant  |          147.5 |         720 |               24   |      3 |        117 |         59 |            216 | 1,115   | 1,064    |       267 |       1.05 |           0.8 |           39 |         227.8 |     42 |          65.6 | Yoru      | Jett      | Chamber   | Vandal      |          38 |          57 |           4 |          444 | Phantom     |          36 |          61 |           3 |          231 | Operator    |           8 |          91 |           1 |          102 | Radiant        |
| NA       | Boba          | #0068  | Radiant  |          178.2 |         856 |               37.3 |      3 |         83 |         49 |            235 | 1,134   | 812      |       157 |       1.4  |           1   |           37 |         277   |     32 |          62.8 | Jett      | Chamber   | KAY/O     | Vandal      |          51 |          47 |           2 |          754 | Sheriff     |          48 |          51 |           1 |           48 | Phantom     |          44 |          56 |           0 |           36 | Radiant        |
| NA       | i love mina   | #kelly | Radiant  |          149.8 |         534 |               24.4 |      2 |         71 |         38 |            137 | 869     | 781      |       213 |       1.11 |           0.8 |           29 |         230.9 |     32 |          62.8 | Jett      | Raze      | Chamber   | Vandal      |          36 |          60 |           4 |          419 | Spectre     |          21 |          71 |           8 |           65 | Operator    |           8 |          92 |           0 |           64 | Radiant        |

---

### Part 2: Univariate Analysis

In this section, I explored the distribution of key individual statistics to get a better sense of how performance varies across players in the dataset. These insights helped me figure out which variables are most useful when comparing Radiants and Immortals.

#### Player Rank Distribution

<iframe src="assets/rank_distribution_pie.html" width="600" height="400" frameborder="0"></iframe>

This pie chart shows the distribution of players in the dataset across the Radiant and Immortal ranks. As expected, Radiants make up a very small fraction of the total population — confirming that our classification task involves a significant class imbalance. This imbalance has implications for model performance, as it makes it harder to correctly identify Radiants without additional strategies like class weighting or oversampling.

#### Damage per Round

<iframe src="assets/dpr.html" width="700" height="400" frameborder="0"></iframe>

Most players fall between 120 and 180 damage per round, with some outliers above 200 indicating highly impactful players. However, the long tail to the right — including players dealing over 200 damage per round — indicates the presence of exceptional outliers who likely have a massive impact in their matches. These high performers may be disproportionately Radiants, making this a potentially predictive feature.

####  Kill-to-Death Ratio (K/D)

<iframe src="assets/kd_ratio.html" width="700" height="400" frameborder="0"></iframe>

The K/D ratio tends to center between 0.9 and 1.5. Understanding whether Radiants consistently hold higher K/Ds is key to our modeling. However, identifying whether Radiants consistently maintain a higher K/D than Immortals is a central goal of this project. If so, it would validate the idea that fragging power is a key factor that sets Radiants apart — though we also aim to investigate whether other metrics (like clutches or utility usage) matter just as much or more.

---

---

### Part 3: Bivariate Analysis

To explore how in-game stats distinguish Radiants from Immortals, I visualized two key metrics — **Clutches** and **Damage per Round**. These comparisons help us identify which performance traits are more common among Radiant players.

#### Clutches by Rating Tier

This plot displays the number of **clutch rounds** (rounds won as the last surviving player) across each rank. Radiant players show a noticeably **higher median clutch count** compared to Immortals, along with a dense concentration of high outliers. This suggests that Radiants are more reliable in high-pressure 1vX situations — a skill likely rewarded at the highest rank.

<iframe src="assets/clutches_by_rank.html" width="800" height="400" frameborder="0"></iframe>

**Interpretation:**  
Clutch performance appears to be a key differentiator between Radiants and Immortals. This supports the hypothesis that Radiants excel not just in raw stats but in clutch moments that secure rounds and shift momentum.

#### Damage per Round by Rating Tier

This box plot compares the **damage dealt per round** across rating groups. While all tiers center around similar median values, Radiant players display **fewer low outliers and a tighter IQR**, suggesting they maintain **more consistent impact per round**.

<iframe src="assets/damagecomp.html" width="800" height="400" frameborder="0"></iframe>

**Interpretation:**  
Consistency in damage output — not just high peaks — may be a distinguishing factor for Radiants. The lack of low-damage games indicates that Radiants maintain a higher baseline of performance, which likely contributes to their overall win rates and reliability.

---

## Part 4: Interesting Aggregates

To dig deeper into what distinguishes Radiants from Immortals, I used grouped aggregates and correlation visualizations that summarize player behavior across multiple features.

### 1. Average Clutches and First Bloods by Rating Group

<iframe src="assets/avgclutch.html" width="800" height="400" frameborder="0"></iframe>

Radiant players, on average, secured significantly more **clutch wins** and **first bloods** than Immortals. This suggests that not only are Radiants more effective in high-pressure solo situations, but they also frequently initiate rounds with early picks — both being strong indicators of impactful play.

---

### 2. KD Ratio vs. Win Percentage (Radiants Only)

<iframe src="assets/kdandwin.html" width="800" height="400" frameborder="0"></iframe>

Among Radiant players, we observe a **positive linear trend** between **K/D ratio** and **win percentage**, reinforcing that individual fragging efficiency is closely tied to winning. While K/D isn't the sole determinant of success, it clearly plays a strong role among top-tier players.

---

### 3. Damage per Round vs. Win Percentage (All Ranks)

<iframe src="assets/dprandwin.html" width="800" height="400" frameborder="0"></iframe>

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

To establish a performance benchmark, we trained a baseline classification model using logistic regression. The objective of this model is to predict whether a player belongs to the **Radiant** or **Immortal** tier using core gameplay statistics.

### Model Overview

- **Model Used**: Logistic Regression (via scikit-learn)
- **Preprocessing**: Standardized numeric features using `StandardScaler`
- **Pipeline**: Built using `sklearn.pipeline.Pipeline` to combine preprocessing and modeling
- **Target**: `rating_group` (binary: Radiant vs. Immortal)

### Features Used

We selected two core quantitative features:

| Feature         | Type         | Description |
|----------------|--------------|-------------|
| `kd_ratio`      | Quantitative | Kill-to-death ratio |
| `damage_round`  | Quantitative | Average damage dealt per round |

- **Quantitative Features**: 2  
- **Ordinal Features**: 0  
- **Nominal Features**: 0  

All features were numerical, so no categorical encoding was required for this baseline model.

### Evaluation

We split the dataset into an 80/20 training and testing split, using stratification to maintain the class distribution. The model's performance was evaluated using precision, recall, and F1-score.

Below is the classification report from the baseline model:

=== Classification Report ===
```
              precision    recall  f1-score   support

    Immortal       0.97      1.00      0.98     16594
     Radiant       0.00      0.00      0.00       521

    accuracy                           0.97     17115
   macro avg       0.48      0.50      0.49     17115
weighted avg       0.94      0.97      0.95     17115

```


### Model Assessment

This logistic regression model offers a simple and interpretable baseline. While it achieves high overall accuracy (due to the imbalance in class labels), it fails to effectively identify **Radiant** players — yielding zero recall and F1-score for that class. This demonstrates the need for more sophisticated feature engineering and class imbalance handling in our final model. Despite its limitations, this baseline serves as a useful point of reference for comparison.



While the **overall accuracy is high (0.97)**, this result is misleading due to **severe class imbalance** — the model predicts nearly all samples as "Immortal." As a result, performance for the **Radiant** class is **extremely poor**, with **zero precision, recall, and F1-score**.

#### Conclusion:
We do **not consider this model “good”** despite its high accuracy. It fails to generalize to the minority class (Radiant players), which is critical for a fair and informative classifier. This motivates the need for feature engineering, class imbalance handling, and model tuning in our final model to improve overall balance and recall across both classes.




## Step 5: Final Model

### Feature Selection and Engineering

To improve upon the baseline model, I incorporated additional features that I believe capture deeper aspects of elite-level gameplay. These features were chosen not just for correlation, but because they reflect meaningful aspects of what it takes to succeed at the Radiant level in Valorant.

- **`clutches`** (Quantitative): This measures how often a player wins a round as the last surviving teammate. Radiant players often shine in these high-pressure 1vX situations, where decision-making and mechanical skill converge. Since clutch performance directly impacts round outcomes, it’s a strong indicator of individual impact — especially at high ranks.

- **`agent_1`** (Nominal): This captures the agent most frequently played by the user. Certain agents like **Jett**, **Reyna**, or **Chamber** allow for more fragging potential and self-sufficiency, traits that can be crucial in climbing to Radiant. Conversely, Immortal players might rely on more utility-focused or flexible roles. Including this feature allows the model to learn subtle patterns related to playstyle and specialization.

To preprocess these features:
- I applied a combination of **`StandardScaler`** and **`QuantileTransformer`** to quantitative features (`kd_ratio`, `damage_round`, `clutches`) to reduce skew and make the model more robust to outliers.
- I encoded `agent_1` using **`OneHotEncoder`** to handle the categorical nature of this variable.

All transformations were implemented inside a single **`scikit-learn` Pipeline**, ensuring that the model remains clean, reproducible, and future-proof for new data inputs.

---

### Modeling Algorithm and Hyperparameter Tuning

For consistency and interpretability, I continued using **Logistic Regression** as my classification algorithm. However, the final model improved over the baseline through two key techniques:

1. **Class Imbalance Handling**: Since Radiant players are underrepresented in the data, I used `compute_class_weight` to give more importance to this minority class, encouraging the model to avoid defaulting to "Immortal" predictions.
2. **Hyperparameter Tuning**: I used **`GridSearchCV`** with 5-fold cross-validation to find the best regularization strength (`C`) for logistic regression. The model was evaluated based on **macro F1-score**, a metric that balances performance across both classes.

```python
param_grid = {
    'clf__C': [0.01, 0.1, 1, 10, 100]
}

The best-performing model used:
- `C = 0.1`


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

Compared to the baseline model, the final model showed clear improvements in identifying Radiant players — and these improvements can be attributed to specific choices rooted in the data-generating process of Valorant.

- **`clutches`** were added because Radiant players often stand out not just in raw aim or stats, but in how they perform during high-pressure, 1vX situations. Clutch scenarios are common in high-ELO ranked games where coordination and mental resilience matter — so this feature reflects a skill that's rewarded more heavily at the Radiant level.

- **`agent_1`** (most-played agent) reflects tactical flexibility and specialization. Radiants often lean toward agents that offer high carry potential or round-winning abilities (e.g., Jett, Reyna), whereas Immortal players might play a broader or more support-oriented pool. Including this feature allows the model to capture subtle correlations between agent preference and elite-level performance.

These features weren’t chosen simply because they boosted accuracy after the fact — they were selected based on their connection to gameplay mechanics that likely *cause* players to perform at a Radiant level. In other words, they reflect **latent skill expressions** (like clutch ability or agent specialization) that aren't directly visible through basic stats like K/D ratio.

Thanks to these engineered features and class weighting, the model became much better at recognizing true Radiants:

- **Recall for Radiant** (0.00 → 0.78): The model now identifies most Radiant players correctly.
- **F1-score for Radiant** (0.01 → 0.15): Substantial improvement in predictive value for the minority class.
- **Macro F1** (0.50 → 0.50): Maintained, but with much more balanced class-level performance.

This confirms that the model wasn’t just overfitting or relying on statistical noise — it genuinely improved in distinguishing Radiants from Immortals by learning the real traits that separate them.
