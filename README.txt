README - AI4Cyber Project

Project Overview
This project applies machine learning to two cybersecurity-related text classification tasks:
1. Spam Detection – classify emails as spam or ham.
2. Misinformation Detection – classify social media posts as fake or real.

The project demonstrates how to preprocess raw data, explore datasets, train machine learning models, and evaluate their performance.

Environment Setup
1. Open Anaconda Prompt and navigate to the project directory:
   cd <copy and paste your file directory>

2. Create and activate a new environment:
   conda create -n ai4cyber python=3.10 -y
   conda activate ai4cyber

3. Install required dependencies:
   pip install pandas scikit-learn matplotlib wordcloud openpyxl joblib

Project Structure
ai4cyber/
data/
    raw/          Original datasets (not included in submission)
    interim/      Processed datasets (merged.csv lives here)
models/         Trained ML models (.pkl)
reports/        Plots (EDA, wordclouds, etc.)
src/            Source code
    d01_load_and_clean.py
    d02_explore.py
    d03_model.py
  README.txt

Steps to Run

1. Preprocess Data
Cleans raw datasets, merges into merged.csv.
python src/d01_load_and_clean.py
Output: data/interim/merged.csv

2. Exploratory Data Analysis (EDA)
Generates plots such as class distribution and word clouds.
python src/d02_explore.py
Output: reports/class_distribution.png and related plots.

3. Train Models
Trains Logistic Regression and Naive Bayes for both tasks.
python src/d03_model.py
Output: Precision, Recall, F1-scores, Confusion Matrices, and trained models saved in models/.

Using the Models for Prediction
Example code to load a saved model and run predictions:

import joblib
vectorizer = joblib.load("models/spam_detection_tfidf.pkl")
model = joblib.load("models/spam_detection_logreg.pkl")
sample = ["You won $1000, click here to claim!"]
sample_vec = vectorizer.transform(sample)
print(model.predict(sample_vec))  # Output: ['spam']

