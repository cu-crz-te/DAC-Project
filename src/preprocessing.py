"""
Preprocessing module for Garena community NLP project.

This script loads raw comment data, performs basic text cleaning,
and saves a cleaned version for later modeling steps.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """
    Load raw comment data.

    Parameters
    ----------
    path : str or Path
        Path to the raw data file (for example: data/raw_comments.csv).

    Returns
    -------
    pd.DataFrame
        Raw data as a DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    return pd.read_csv(path)


def basic_clean(text: str) -> str:
    """
    Apply simple text cleaning to a single comment string.

    Operations include:
    - Remove URLs
    - Remove non word characters
    - Lowercase

    Parameters
    ----------
    text : str
        Original text.

    Returns
    -------
    str
        Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip().lower()
    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "comment",
    output_column: str = "clean_text",
) -> pd.DataFrame:
    """
    Apply text cleaning to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw comments.
    text_column : str
        Column name of the original text.
    output_column : str
        Column name for cleaned text.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added cleaned text column.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")
    df = df.copy()
    df[output_column] = df[text_column].apply(basic_clean)
    # Optional: drop rows with empty cleaned text
    df = df[df[output_column].str.len() > 0]
    df = df.reset_index(drop=True)
    return df


def save_clean_data(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save the cleaned DataFrame to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    path : str or Path
        Target path (for example: data/clean_comments.csv).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_preprocessing(
    raw_path: str | Path = "data/raw_comments.csv",
    output_path: str | Path = "data/clean_comments.csv",
    text_column: str = "comment",
    clean_column: str = "clean_text",
) -> None:
    """
    High level function that performs the full preprocessing pipeline.

    This can be called from command line or from a notebook.

    Parameters
    ----------
    raw_path : str or Path
        Path to the raw input file.
    output_path : str or Path
        Path where the cleaned data will be written.
    text_column : str
        Name of the original text column.
    clean_column : str
        Name given to the cleaned text column.
    """
    df_raw = load_raw_data(raw_path)
    df_clean = preprocess_dataframe(df_raw, text_column=text_column, output_column=clean_column)
    save_clean_data(df_clean, output_path)
    print(f"Saved cleaned data to: {output_path}")


if __name__ == "__main__":
    # Adjust column names or paths here to match your actual dataset
    run_preprocessing(
        raw_path="data/raw_comments.csv",
        output_path="data/clean_comments.csv",
        text_column="comment",     # change to your actual text column
        clean_column="clean_text",
    )
