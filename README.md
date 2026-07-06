# Galaxy Zoo: Morphology Classification with Conditional Neural Networks

This repository contains a PyTorch-based Deep Learning pipeline designed to classify the morphologies of distant galaxies. It was developed for the [Galaxy Zoo Kaggle Competition](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge).

## Performance
* **Private Leaderboard Score (RMSE):** `0.15116`
* **Public Leaderboard Score (RMSE):** `0.15107`
* **Overfitting Margin:** `< 0.0001`

---

## Architectural Highlights: The "True Tree" Model
Instead of treating this as a standard multi-label classification problem, the architecture explicitly models the dataset's domain logic. The Galaxy Zoo decision tree is highly conditional. 

This model utilizes an **EfficientNet-B3** backbone but completely replaces the final classifier with 11 distinct, sequentially gated output heads. The forward pass mathematically enforces the conditional probabilities of the physical decision tree:

```python
prob_not_edge = out_q2[:, 1:2]
out_q3 = self.q3(features) * prob_not_edge