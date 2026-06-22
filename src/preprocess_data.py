import re
import pandas as pd


TARGET_PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Savings account",
    "Money transfer"
]


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    boilerplate_patterns = [
        r"i am writing to file a complaint",
        r"i am writing this complaint",
        r"to whom it may concern"
    ]

    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text)

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def filter_dataset(df):

    narrative_col = "Consumer complaint narrative"
    product_col = "Product"

    df = df[df[product_col].isin(TARGET_PRODUCTS)]

    df = df[
        df[narrative_col].notna()
        & (df[narrative_col].str.strip() != "")
    ]

    df["cleaned_narrative"] = (
        df[narrative_col]
        .apply(clean_text)
    )

    return df


def save_filtered_dataset(input_path, output_path):

    df = pd.read_csv(input_path)

    filtered_df = filter_dataset(df)

    filtered_df.to_csv(output_path, index=False)

    return filtered_df


if __name__ == "__main__":

    save_filtered_dataset(
        "data/raw/complaints.csv",
        "data/filtered_complaints.csv"
    )
    