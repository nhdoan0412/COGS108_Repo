# Product Description–Based Price Prediction Using NLP and Machine Learning

**UC San Diego COGS 108 · Spring 2025 · Four-person team**

A machine-learning project that predicts **Amazon electronics prices from product descriptions** using natural-language processing and ensemble regression.

## Problem

Online marketplaces contain large numbers of products whose prices vary substantially even when their descriptions appear similar. This project explores whether textual product information alone can provide enough signal to estimate price.

The core question is:

> **How accurately can we predict an electronics product's price from its written description?**

A model like this could support product analytics, pricing research, catalog quality checks, or price-estimation tools when structured product attributes are incomplete.

## Project Snapshot

- **Dataset:** 1,400+ Amazon electronics listings
- **Input:** product-description text
- **Target:** product price
- **Text representation:** TF-IDF
- **Model:** Random Forest Regressor
- **Evaluation:** R² and Mean Absolute Error
- **Result:** **R² = 0.72**
- **Mean Absolute Error:** **under $150**

## ML Pipeline

```text
Raw Product Listings
        ↓
Text Cleaning / Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Sparse Feature Matrix
        ↓
Random Forest Regressor
        ↓
Price Prediction
        ↓
R² / MAE Evaluation
```

## Technical Approach

### 1. Text preprocessing

Product descriptions were cleaned and transformed into machine-readable text features.

### 2. TF-IDF feature extraction

TF-IDF converts product-description terms into weighted numerical features, emphasizing words that are informative for a specific listing while reducing the influence of terms that appear broadly throughout the dataset.

### 3. Sparse representation

Because TF-IDF creates high-dimensional text vectors, the pipeline uses sparse matrix representations to avoid unnecessary memory usage.

### 4. Random Forest regression

A Random Forest Regressor was trained on the text-derived features to model nonlinear relationships between product language and price.

### 5. Evaluation

The model was evaluated with:

- **R²** — how much of the variation in price is explained by the model
- **Mean Absolute Error (MAE)** — average absolute difference between predicted and actual prices

The final model achieved **R² = 0.72** and **MAE below $150**.

## Tech Stack

- Python
- pandas
- scikit-learn
- TF-IDF
- Random Forest Regression
- NLP preprocessing
- sparse matrices
- Jupyter Notebook

## Team

This project was completed by a **four-person team** for UC San Diego's COGS 108 course.

The original graded submission was maintained in the private course repository `COGS108/Group041_SP25`. This repository is the public portfolio-facing mirror for the project.

## Repository Status

The project documentation is public here for portfolio use. The final course notebook can be mirrored into this repository after exporting it from the private course repository.
