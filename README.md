# Cats vs Dogs Classification using Classical Machine Learning

This project solves a binary image classification problem: classify an image as **Cat (0)** or **Dog (1)** using classical machine learning methods instead of CNNs.

## Project Aim

The main objective of this project was not only to build a cat-vs-dog classifier, but also to better understand how image classification works **before** using deep learning methods. More specifically, the goal was to explore how visual information can be extracted manually from images and how a machine learning model can use these extracted features to perform classification.

This project also helped provide intuition about what CNNs do \emph{under the hood}. While CNNs automatically learn useful visual patterns such as edges, textures, and shapes, in this work these characteristics were described explicitly using handcrafted feature extraction methods. In this way, the project offers a more interpretable and educational view of image classification.

## Project Overview

A complete classical machine learning pipeline was designed for this task. The system is based on the following main stages:

- image loading and preprocessing,
- handcrafted feature extraction,
- feature normalization,
- dimensionality reduction,
- classification.

The following tools were used:

- **HOG (Histogram of Oriented Gradients)**  
  Used to capture edge, contour, and shape information. This is useful because cats and dogs often differ in face structure, ear shape, and body outline.

- **LBP (Local Binary Patterns)**  
  Used to describe local texture patterns. This helps represent texture details such as fur appearance.

- **Color Histograms**  
  Used to capture the distribution of colors in the image. This provides complementary visual information in addition to shape and texture.

- **StandardScaler**  
  Used to normalize the feature values so that all extracted descriptors contribute more fairly to the model.

- **PCA (Principal Component Analysis)**  
  Used to reduce the dimensionality of the combined feature vector, remove redundancy, and keep the most informative variance.

- **SVM (Support Vector Machine)**  
  Used as the final classifier because it is well suited to binary classification and performs well with high-dimensional handcrafted features.

The goal was therefore to build a complete image classification system using manually extracted visual descriptors and a classical machine learning model, while also gaining a deeper understanding of the principles behind image recognition.

## Dataset

The dataset contains images from two classes:
- **Cat** → label `0`
- **Dog** → label `1`

Total number of images: **18,000**

The dataset was split into training and validation sets for model development and evaluation.

## Experiments

Two experiments were conducted using different image sizes.

### Experiment 1 — 64×64 images
- Accuracy: **0.7808**
- Precision: **0.7897**
- Recall: **0.7657**
- F1-score: **0.7775**

### Experiment 2 — 128×128 images
- Accuracy: **0.8029**
- Precision: **0.8086**
- Recall: **0.7939**
- F1-score: **0.8012**

The **128×128 configuration** achieved the best performance and was selected as the final model.

## Files

- `notebook.ipynb` — implementation of the full machine learning pipeline
- `report.pdf` — project report
- `images/quick_test.png` — example of a quick prediction test

## Quick Test on a Validation Image

The notebook includes a quick visual test on one validation image to verify that the final prediction function works correctly on a single sample.

### Example Output

![Quick Test](images/quick_test.png)

## Final Prediction Function

The final prediction function is:

```python
cats_dogs_classification(image)
