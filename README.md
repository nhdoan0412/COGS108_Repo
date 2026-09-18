# Predicting Amazon Electronics Prices with NLP

**UC San Diego COGS 108 · Spring 2025 · 5-person team**

A machine-learning project investigating whether the **language used in Amazon product descriptions can predict electronics prices**.

## Quick Links

- **Final analysis:** [FinalProject_Group041_SP25.ipynb](./FinalProject_Group041_SP25.ipynb)
- **Dataset:** [datasets/amazon.csv](./datasets/amazon.csv)
- **Project video:** https://www.youtube.com/watch?v=-ZV_pPy97YE
- **Original course README:** [COURSE_README.md](./COURSE_README.md)

## Project Snapshot

- **Dataset:** 1,465 Amazon product listings, 16 original variables
- **Core fields:** product name, product description, actual price
- **NLP representation:** TF-IDF with unigrams + bigrams
- **Model:** Random Forest Regressor
- **Tuning:** GridSearchCV with 5-fold cross-validation
- **Best CV R²:** **0.752**
- **Final test R²:** **0.63**
- **MAE:** **$28.13**
- **RMSE:** **$80.27**

## Research Question

> To what extent can natural-language features extracted from Amazon product descriptions predict electronics pricing, and which linguistic elements are the strongest predictors?

We hypothesized that higher-priced products would use more technical, detailed, and specialized language, while lower-priced items would tend to use shorter and more generic descriptions.

## Dataset & Preprocessing

The project uses a public Amazon sales dataset with **1,465 rows × 16 columns**.

We:

- converted `actual_price` from Indian rupees to USD,
- removed missing values in key fields,
- lowercased and cleaned description text,
- tokenized descriptions,
- removed stopwords and non-informative tokens,
- engineered description length, average word length, and long-word-count features,
- applied TF-IDF using unigrams and bigrams.

## Modeling Pipeline

```text
Amazon product listings
        ↓
Text cleaning + tokenization
        ↓
Stopword / token filtering
        ↓
TF-IDF vectorization
(max 300 features, 1–2 grams)
        ↓
80/20 train-test split
        ↓
Random Forest Regressor
        ↓
GridSearchCV
        ↓
R² / MAE / RMSE evaluation
```

The best hyperparameters were:

```text
max_depth = 50
min_samples_split = 5
n_estimators = 300
```

## Results

On the held-out test set:

| Metric | Result |
| --- | ---: |
| R² | **0.63** |
| MAE | **28.13** |
| RMSE | **80.27** |

The tuned model explained about **63% of the variation in product price using description text alone**.

High-importance terms included technical phrases such as **“port connect,” “4k,” “refresh rate,” “ultra,” and “camera.”** This supported the hypothesis that technical specificity in product descriptions is associated with higher-priced products.

The model performed better on lower-priced items and was less reliable for premium products, where the dataset had fewer examples.

## Exploratory Analysis

The analysis also found:

- description length had a weak positive correlation with price,
- long-word count had a weak positive correlation with price,
- average word length had almost no linear correlation with price,
- most products were concentrated in the lower price range,
- technical terms and specifications were prominent in product descriptions.

## My Contributions

**Nhan Doan (Nick)**

- refined the research question,
- sourced the dataset from Kaggle,
- cleaned and preprocessed the dataset,
- conducted TF-IDF feature extraction,
- performed exploratory data analysis,
- contributed to the prediction model,
- contributed to the final presentation video,
- refined the final report.

## Tech Stack

- Python
- pandas / NumPy
- scikit-learn
- TF-IDF
- Random Forest Regression
- GridSearchCV
- NLTK
- Matplotlib / Seaborn
- Jupyter Notebook

## Repository Structure

| File | Description |
| --- | --- |
| `FinalProject_Group041_SP25.ipynb` | Final submitted analysis |
| `EDACheckpoint_Group041_SP25.ipynb` | Exploratory-data-analysis checkpoint |
| `DataCheckpoint_Group041_SP25.ipynb` | Data-cleaning checkpoint |
| `ProjectProposal_Group041_SP25.ipynb` | Initial project proposal |
| `Project.ipynb` | Earlier project notebook |
| `datasets/amazon.csv` | Public Amazon dataset used in the analysis |
| `COURSE_README.md` | Original course-repository README |

## Project Provenance

This repository is the **public portfolio mirror** of our UC San Diego COGS 108 Group 041 project.

The original course repository, `COGS108/Group041_SP25`, is private. I also retained a true GitHub fork at `nhdoan0412/Group041_SP25` so the original team commit history and repository provenance are preserved.

The public mirror exists so recruiters and collaborators can review the project while the private fork remains the source-of-truth copy for the original course history.

The final course notebook explicitly selected **YES — make available** for public release.

## Team

The project was completed by:

- Nhan Doan
- Karan Derebail
- Hansel Puthenparambil
- Zachary Elian
- Joshua McDevitt

## Limitations & Future Work

The TF-IDF representation captures word importance but not deeper semantic context. For example, it may not distinguish the same technical term used in very different product categories.

Potential improvements include:

- semantic embeddings such as Word2Vec or BERT,
- brand/category features,
- larger and more balanced price distributions,
- additional product metadata,
- stronger treatment of high-price outliers.
