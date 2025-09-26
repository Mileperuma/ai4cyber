# File: d03_model.py
# Purpose: this script handles the real "training" stage of our project.
# In simple terms, we take the cleaned datasets, split them into training/testing groups,
# feed them into machine learning models, and then evaluate how well the models learn.
# The idea is similar to practicing before an exam: the training set is the practice work,
# and the test set is like the real exam where we see how well the model performs.

import pandas as pd
import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# 1. SPAM DETECTION
# ==============================
print("\n=== Training Spam Detection Models ===")

# load the cleaned spam dataset which has "text" and "label"
# here "label" is either spam or ham (ham just means normal emails)
spam = pd.read_csv("data/interim/spam_clean.csv")

X_spam = spam["text"]   # the actual email content
y_spam = spam["label"]  # the ground-truth label we want the model to learn

# split data into training and testing
# 80% is used for training, 20% held back to test the model later
X_train, X_test, y_train, y_test = train_test_split(
    X_spam, y_spam, test_size=0.2, random_state=42
)

# TF-IDF vectorization:
# emails are just plain words. Machines cannot directly understand them,
# so we transform the text into numbers. TF-IDF basically gives weight to words
# depending on how frequent and how unique they are. For example, the word "free"
# might carry higher importance in spam compared to normal mail.
tfidf = TfidfVectorizer(max_features=5000)  # limit to 5000 most important words
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Logistic Regression:
# this is like drawing a boundary between spam and ham in a high-dimensional space.
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train_tfidf, y_train)
y_pred_log = logreg.predict(X_test_tfidf)

print("\n[Spam Detection | LOGREG]")
print(classification_report(y_test, y_pred_log))  # shows precision, recall, f1-score
print(confusion_matrix(y_test, y_pred_log))      # shows how many were correctly/incorrectly classified

# save the model and vectorizer so we can reuse them for predictions later
joblib.dump(logreg, "models/spam_logreg.pkl")
joblib.dump(tfidf, "models/spam_tfidf.pkl")

# Naive Bayes:
# this model assumes word occurrences are independent.
# Surprisingly, this simple assumption works extremely well in text problems.
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)

print("\n[Spam Detection | NAIVE BAYES]")
print(classification_report(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))

joblib.dump(nb, "models/spam_nb.pkl")


# --- SVM ---
svm = LinearSVC(max_iter=2000)
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)
print("\n[Spam Detection | SVM]")
print(classification_report(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
joblib.dump(svm, "models/spam_svm.pkl")

# ==============================
# 2. MALWARE DETECTION
# ==============================
print("\n=== Training Malware Detection Models ===")

# load the malware dataset (numeric features, not text-based like spam)
malware = pd.read_csv("data/interim/malware_clean.csv")

# drop the "hash" column because it is just an identifier,
# it doesn’t give meaningful information to the model
if "hash" in malware.columns:
    malware = malware.drop(columns=["hash"])

X_malware = malware.drop(columns=["label"])  # features like file size, entropy, etc.
y_malware = malware["label"].map({"malware": 1, "benign": 0})
# we encode labels into numbers because ML models usually need numeric labels

# again split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_malware, y_malware, test_size=0.2, random_state=42
)

# Logistic Regression:
# even though malware data is numeric, logistic regression can still find
# decision boundaries to separate benign from malicious files.
logreg_m = LogisticRegression(max_iter=200)
logreg_m.fit(X_train, y_train)
y_pred_m_log = logreg_m.predict(X_test)

print("\n[Malware Detection | LOGREG]")
print(classification_report(y_test, y_pred_m_log))
print(confusion_matrix(y_test, y_pred_m_log))

joblib.dump(logreg_m, "models/malware_logreg.pkl")

# Random Forest:
# this is an ensemble model, meaning it builds many decision trees and
# takes the majority vote. It is quite powerful for structured numeric data.
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_m_rf = rf.predict(X_test)

print("\n[Malware Detection | RANDOM FOREST]")
print(classification_report(y_test, y_pred_m_rf))
print(confusion_matrix(y_test, y_pred_m_rf))

joblib.dump(rf, "models/malware_rf.pkl")
