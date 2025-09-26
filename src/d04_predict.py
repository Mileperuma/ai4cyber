# File: d04_predict.py
# Purpose: take the models we trained earlier (spam + malware)
# and actually put them to the test with new, unseen inputs.
# This is the "demo" stage where the project feels alive.

import joblib
import pandas as pd

# --- Load trained models ---
# spam needs both the vectorizer (to turn text into numbers) and the classifier itself.
# malware only needs the RandomForest since its features are already numeric.
spam_tfidf   = joblib.load("models/spam_tfidf.pkl")
spam_model   = joblib.load("models/spam_logreg.pkl")
malware_model = joblib.load("models/malware_rf.pkl")

# --- Spam Prediction Helper ---
def predict_spam(text):
    """Take in a raw email/message string → transform into tf-idf features → predict.
    Logistic Regression spits out 'spam' or 'ham' depending on patterns it saw in training."""
    X_tfidf = spam_tfidf.transform([text])
    pred = spam_model.predict(X_tfidf)[0]
    return pred   # model was trained with labels 'spam'/'ham', so no need to remap


# --- Malware Prediction Helper ---
def predict_malware(features_dict):
    """ Take in one sample's feature dictionary (like file size, entropy, opcode counts),
    wrap it in a DataFrame so sklearn understands it, and ask the RandomForest to decide.
    Output is either 'malware' or 'benign'."""


    X = pd.DataFrame([features_dict])
    pred = malware_model.predict(X)[0]
    return "malware" if pred == 1 else "benign"

if __name__ == "__main__":

    # --- Quick sanity checks ---
    # 1. Spam detection examples
    print("Spam test:", predict_spam("Win money now, click this link"))  # obvious spam
    print("Spam test:", predict_spam("Hey John, are we still on for tomorrow?"))  # casual ham

    # 2. Malware detection example
    # Use the raw dataset → strip out columns we didn’t train on (hash + label) → keep features only.
    df = pd.read_csv("data/raw/2malware/Malware dataset.csv")
    sample_row = df.drop(columns=["hash", "classification"]).iloc[0].to_dict()

    print("Malware test:", predict_malware(sample_row))
