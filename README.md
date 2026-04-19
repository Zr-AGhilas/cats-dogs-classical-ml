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

- **64x64**
  - Accuracy: 0.7808
  - Precision: 0.7897
  - Recall: 0.7657
  - F1-score: 0.7775

- **128x128**
  - Accuracy: 0.8029
  - Precision: 0.8086
  - Recall: 0.7939
  - F1-score: 0.8012

The **128x128 configuration** gave the best results and was selected as the final model.

## Files
- `notebook.ipynb` — implementation of the full pipeline
- `report.pdf` — project report

## Final Function
The final prediction function is:

```python
cats_dogs_classification(image)


## Quick Test on a Validation Image

The notebook also includes a quick visual test on one validation image to verify that the final function works correctly on a single sample.

```python
idx = 0
sample_path = val_paths[idx]
true_label = y_val[idx]
pred_label = cats_dogs_classification(sample_path)

label_to_name = {0: "Cat", 1: "Dog"}
img = load_rgb_image(sample_path)

plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.title(
    f"True: {label_to_name[true_label]} ({true_label}) | "
    f"Predicted: {label_to_name[pred_label]} ({pred_label})"
)
plt.axis("off")
plt.show()
