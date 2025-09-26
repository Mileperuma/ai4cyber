# File: d02_explore_spam.py
# Purpose: this script is all about exploring the spam dataset before we jump
# into training. Think of it like getting to know your data first —
# what labels it has, how long the messages are, and which words dominate.
# These insights not only confirm that our data is clean but also give us
# intuition about why a model might work the way it does.

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

# make sure the reports folder exists so we can safely drop our plots there
os.makedirs("reports", exist_ok=True)

def main():
    # --- load the cleaned spam dataset ---
    spam = pd.read_csv("data/interim/spam_clean.csv")

    # shape tells us how many rows and columns there are
    print("Dataset shape:", spam.shape)
    print(spam.head())  # quick peek at the first few rows

    # --- label distribution ---
    # here we want to know how many spam vs ham emails we have.
    # imbalance matters a lot in ML — if spam is too rare, the model might just
    # learn to always say "ham".
    counts = spam["label"].value_counts()
    print("\nLabel distribution:\n", counts)

    counts.plot(kind="bar", title="Spam vs Ham")
    plt.savefig("reports/spam_label_distribution.png")
    plt.close()

    # --- average text length ---
    # some spam emails are short, like “win $$$ now”, while ham (normal) emails
    # tend to be longer with more context. This can be a strong signal for classification.
    spam["text_length"] = spam["text"].apply(len)
    spam.groupby("label")["text_length"].mean().plot(
        kind="bar", title="Average Text Length"
    )
    plt.savefig("reports/spam_text_length.png")
    plt.close()

    # --- word clouds ---
    # a word cloud is a quick way to see the most common words in each category.
    # for spam, we usually expect words like “win”, “free”, “click”.
    # for ham, more natural conversation words show up.
    for label in ["ham", "spam"]:
        text = " ".join(spam[spam["label"] == label]["text"].astype(str).tolist())
        if len(text.strip()) > 0:  # only build if text isn’t empty
            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            wc.to_file(f"reports/wordcloud_{label}.png")
            print(f"✅ Wordcloud saved for {label}")

if __name__ == "__main__":
    main()
