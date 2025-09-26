# File: d01_load_and_clean.py
# Purpose: In this stage we prepare the raw datasets into a proper cleaned format.
# also needs some adjustments before we can train any models. If we skip this step,
# the later parts of the project will not perform well. So here we handle spam data
# and malware data separately, and then we save them into the interim folder which
# is like a “staging area” before real modelling begins.

import pandas as pd   # pandas helps us to load csv files and manage them like tables
import os             # for making sure folders exist to save our files

# helper function: clean up email text
def clean_text(text):
    """In raw datasets, not every entry is nice and tidy. Sometimes there are null values,
    sometimes the text comes with capital letters or unnecessary spaces.
    If we directly use them, then the machine will see the same word as two different things
    (for example, "FREE" vs "free"). So here we unify the style by:
      - turning everything lowercase,
      - trimming spaces,
      - and for non-string values we just turn them into empty strings.
    This way, the data is consistent and ready for further processing."""
    if not isinstance(text, str):
        return ""
    return text.strip().lower()

def main():
    # first we create the interim folder if it doesn’t exist.
    # Think of this as building the shelf before we put the cleaned data onto it.
    os.makedirs("data/interim", exist_ok=True)

    # --- 1. SPAM DATASET ---
    # load the raw spam dataset which has two columns: text + spam(0 or 1)
    spam = pd.read_csv("data/raw/2spam/emails.csv")

    # rename spam column to label, so that both datasets (spam + malware)
    # follow the same convention. It is like setting the same language in two teams.
    spam = spam.rename(columns={"text": "text", "spam": "label"})

    # map the numbers into words: 0 means ham, 1 means spam.
    # It is easier to read and also avoids confusion later.
    spam["label"] = spam["label"].map({0: "ham", 1: "spam"})

    # clean all the text using the helper function we wrote above
    spam["text"] = spam["text"].apply(clean_text)

    # save the cleaned dataset into interim folder.
    # Now the file is in a better condition compared to the raw one.
    spam.to_csv("data/interim/spam_clean.csv", index=False)
    print("✅ Saved spam_clean.csv")
    print(spam["label"].value_counts())  # quick check to see how many spam/ham emails we have

    # --- 2. MALWARE DATASET ---
    # load the malware dataset, this one is heavier with numeric features
    # and has a classification column which says benign or malware
    malware = pd.read_csv("data/raw/2malware/Malware dataset.csv")

    # rename classification column to label, again for consistency
    malware = malware.rename(columns={"classification": "label"})

    # here we don’t really need mapping because the dataset already uses words
    # but we still normalize the values just in case
    malware["label"] = malware["label"].map({"benign": "benign", "malware": "malware"})

    # save the cleaned version
    malware.to_csv("data/interim/malware_clean.csv", index=False)
    print("✅ Saved malware_clean.csv")
    print(malware["label"].value_counts())  # should usually show close to balanced data

# this ensures the script runs only when we directly execute it
# and not when imported somewhere else
if __name__ == "__main__":
    main()
