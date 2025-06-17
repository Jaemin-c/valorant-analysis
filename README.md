# valorant-analysis
Welcome! This project compares Radiant and Immortal players in Valorant using data analysis.

## Step 1: Introduction

Valorant is a competitive tactical shooter where ranking up is the core goal for many players. At the very top of the ladder, **Radiant** players represent the top 0.1%, while **Immortal** ranks (Immortal 1–3) make up the highly skilled tier just below. Understanding the differences between these two groups is useful for both competitive players and analysts: it sheds light on what performance truly separates the best from the nearly best.

Our project aims to answer the following question:

> **Among Radiant and Immortal players, what differentiates Radiants from Immortals in terms of in-game performance stats?**

### Why This Question Matters

This question is relevant because it addresses what it really takes to become one of the best players in Valorant. While many players might assume that K/D ratio is everything, our analysis explores whether other factors like team coordination, consistency, or early-round impact matter more. Competitive gamers, coaches, and game designers alike can benefit from a clearer understanding of what defines top-tier performance.

---

### 📊 About the Dataset

- **Dataset size**: 85,574 rows  
- **Source**: A scraped dataset of Radiant and Immortal-ranked players and their match statistics.

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
