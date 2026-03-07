import pandas as pd
import re


def map_sentiment(rating):
    return rating - 1


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_prepare_data(train_path="data/train.csv", test_path="data/test.csv"):
    print("Loading training data...")
    train_df = pd.read_csv(train_path)

    print("Loading test data...")
    test_df = pd.read_csv(test_path)

    train_df = train_df[["review_text", "class_index"]]
    test_df = test_df[["review_text", "class_index"]]

    train_df["label"] = train_df["class_index"].apply(map_sentiment)
    test_df["label"] = test_df["class_index"].apply(map_sentiment)

    train_df["clean_text"] = train_df["review_text"].apply(clean_text)
    test_df["clean_text"] = test_df["review_text"].apply(clean_text)

    print("Data preparation complete.")

    return (
        train_df["clean_text"].tolist(),
        test_df["clean_text"].tolist(),
        train_df["label"].tolist(),
        test_df["label"].tolist()
    )