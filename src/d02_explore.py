# d02_explore.py
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

# make sure reports/ exists
os.makedirs("reports", exist_ok=True)

def plot_wordcloud(df, labels, title_prefix):
    """Generate wordclouds for specific labels in a dataset"""
    for label in labels:
        text = " ".join(df[df["label"] == label]["text"].dropna().astype(str))
        if len(text.strip()) == 0:
            print(f"⚠️ Skipping word cloud for {label} (no text found)")
            continue
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"{title_prefix} Word Cloud - {label}")
        plt.savefig(f"reports/{title_prefix.lower()}_wordcloud_{label}.png")
        plt.close()

def main():
    # Load dataset
    df = pd.read_csv("data/interim/merged.csv")
    print("Dataset shape:", df.shape)

    # --- 1. Class Distribution (all labels) ---
    class_counts = df["label"].value_counts()
    print("\nOverall class distribution:\n", class_counts)

    class_counts.plot(kind="bar", color=["skyblue", "salmon", "lightgreen", "orange"])
    plt.title("Overall Class Distribution")
    plt.ylabel("Count")
    plt.savefig("reports/overall_class_distribution.png")
    plt.close()

    # --- 2. Spam vs Ham ---
    spam_df = df[df["label"].isin(["spam", "ham"])]
    spam_counts = spam_df["label"].value_counts()
    spam_counts.plot(kind="bar", color=["salmon", "skyblue"])
    plt.title("Spam vs Ham Distribution")
    plt.ylabel("Count")
    plt.savefig("reports/spam_vs_ham_distribution.png")
    plt.close()

    spam_df["text_length"] = spam_df["text"].apply(lambda x: len(str(x).split()))
    avg_length_spam = spam_df.groupby("label")["text_length"].mean()
    avg_length_spam.plot(kind="bar", color=["salmon", "skyblue"])
    plt.title("Average Text Length (Spam vs Ham)")
    plt.ylabel("Avg #words")
    plt.savefig("reports/spam_vs_ham_text_length.png")
    plt.close()

    plot_wordcloud(spam_df, ["spam", "ham"], "SpamHam")

    # --- 3. Real vs Fake ---
    misinfo_df = df[df["label"].isin(["real", "fake"])]
    misinfo_counts = misinfo_df["label"].value_counts()
    misinfo_counts.plot(kind="bar", color=["green", "orange"])
    plt.title("Real vs Fake Distribution")
    plt.ylabel("Count")
    plt.savefig("reports/real_vs_fake_distribution.png")
    plt.close()

    misinfo_df["text_length"] = misinfo_df["text"].apply(lambda x: len(str(x).split()))
    avg_length_misinfo = misinfo_df.groupby("label")["text_length"].mean()
    avg_length_misinfo.plot(kind="bar", color=["green", "orange"])
    plt.title("Average Text Length (Real vs Fake)")
    plt.ylabel("Avg #words")
    plt.savefig("reports/real_vs_fake_text_length.png")
    plt.close()

    plot_wordcloud(misinfo_df, ["real", "fake"], "Misinfo")

    print("\n✅ EDA completed. Plots saved in reports/")

if __name__ == "__main__":
    main()
