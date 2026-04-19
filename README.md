# Cats vs Dogs Classification using Classical Machine Learning

This project solves a binary image classification problem: classify an image as **Cat (0)** or **Dog (1)** using classical machine learning methods instead of CNNs.

## Project Overview

The pipeline is based on:
- HOG (Histogram of Oriented Gradients)
- LBP (Local Binary Patterns)
- Color Histograms
- StandardScaler
- PCA
- SVM

## Experiments

Two experiments were conducted with different image sizes:

### Experiment 1 — 64×64
- Accuracy: **0.7808**
- Precision: **0.7897**
- Recall: **0.7657**
- F1-score: **0.7775**

### Experiment 2 — 128×128
- Accuracy: **0.8029**
- Precision: **0.8086**
- Recall: **0.7939**
- F1-score: **0.8012**

The **128×128 configuration** gave the best results and was selected as the final model.

## Files

- `notebook.ipynb` — implementation of the full pipeline
- `report.pdf` — project report

## Final Prediction Function

The final prediction function is:

```python
cats_dogs_classification(image)
