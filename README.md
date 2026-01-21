# Email Spam Detection using NLP & Machine Learning

An end-to-end **email spam/phishing detection** project that uses Natural Language Processing (NLP) and supervised machine learning to classify emails as **Spam/Phishing (1)** or **Legitimate/Ham (0)**.

This repo includes a full pipeline: dataset loading/standardization, preprocessing, feature engineering (TF‑IDF + metadata features), model training and comparison, evaluation, and saving a production-ready model artifact (`.pkl`).

## Project overview

- **Goal:** Build and evaluate models that accurately distinguish spam/phishing emails from legitimate emails.
- **Dataset:** “Phishing Email Dataset” from Kaggle, combining multiple corpora (CEAS08, Enron, Ling-Spam, Nazario, Nigerian Fraud, SpamAssassin), totaling ~82k emails with a near-balanced label split.
- **Models compared:** Naive Bayes, Linear SVM, Logistic Regression, Random Forest.

## Results (high level)

All tested models achieved strong performance, with **Random Forest** selected as the best overall model in the report based on accuracy/F1/ROC‑AUC tradeoffs.  
The report includes confusion-matrix analysis and a full metric comparison (accuracy, precision, recall, F1, ROC‑AUC).

## Repository contents

- `Email_Spam_Detection.ipynb` — Main notebook implementing the full workflow (EDA → preprocessing → modeling → evaluation → saving).
- `Email-Spam-Detection-using-NLP-Report.pdf` — Final write-up with methodology, evaluation, and discussion.
- `bestspamdetector.pkl` — Saved model bundle (generated when you run the notebook).

## Methodology

### 1) Data loading & standardization
Loads multiple CSV files, standardizes common column names (e.g., `body`, `subject`, `label`), and merges into a single dataset.

### 2) Text preprocessing
The notebook builds a reusable preprocessing pipeline that includes: lowercase conversion, HTML/URL/email removal, special-character cleanup, tokenization, stopword removal, and lemmatization.

### 3) Feature engineering
- **TF‑IDF vectorization** (with uni-grams and bi-grams).
- Additional engineered features such as length/word count, capitalization ratio, punctuation counts, URL/email counts, and “spam trigger word” counts.

### 4) Model training & evaluation
Trains and compares Naive Bayes, SVM, Logistic Regression, and Random Forest using metrics including accuracy, precision, recall, F1, and ROC‑AUC, plus confusion-matrix analysis.  
Hyperparameter tuning is performed via `GridSearchCV` for selected models.

### 5) Model saving
The notebook saves the final model + vectorizer + preprocessor into a single pickle file and includes helper functions to load it back for inference.

## How to run

### Option A: Run the notebook
1. Open `Email_Spam_Detection.ipynb`.
2. Ensure the dataset CSVs are available in the expected `data/` paths used in the notebook (e.g., `data/CEAS08.csv`, `data/Enron.csv`, etc.).
3. Run all cells to reproduce EDA, training, evaluation, and generate `bestspamdetector.pkl`.

### Option B: Load the saved model for inference
After generating `bestspamdetector.pkl`, use the notebook’s provided `load_model(...)` and `predict_email(...)` utilities to classify new email text.

## Requirements

Key Python libraries used:
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `nltk` (tokenization, stopwords, lemmatization)
- Optional visualization libs: `wordcloud`

## Notes / limitations

- The dataset covers a historical range (late 1990s–2008), so production use should include periodic retraining with newer samples to capture evolving spam/phishing tactics.
- The project is binary classification (spam/phishing vs legitimate); future work in the report suggests multi-class categorization and deeper models.

## Credits

- Authors: Dennis Sharon Cheruvathoor, Rutwik Ganagi, Northeastern University.
- Dataset: “Phishing Email Dataset” on Kaggle (see report references).
