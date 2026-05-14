import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"journalist", "publication_id", "date", "title", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["journalist", "text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 100]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df.reset_index(drop=True)
