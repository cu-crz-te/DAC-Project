"""
Modeling module for Garena community NLP project.

This script trains a BERTopic model on cleaned comments and
optionally prepares outputs for later analysis or visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


def load_clean_data(path: str | Path = "data/clean_comments.csv") -> pd.DataFrame:
    """
    Load cleaned comment data.

    Parameters
    ----------
    path : str or Path
        Path to the cleaned data.

    Returns
    -------
    pd.DataFrame
        Cleaned data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clean data file not found at: {path}")
    return pd.read_csv(path)


def build_bertopic_model() -> BERTopic:
    """
    Create a BERTopic model with a basic configuration.

    You can adjust vectorizer settings or pass in custom embeddings
    if your notebook uses a more advanced setup.
    """
    vectorizer_model = CountVectorizer(stop_words="english", max_features=50000)
    topic_model = BERTopic(vectorizer_model=vectorizer_model)
    return topic_model


def train_topics(
    df: pd.DataFrame,
    text_column: str = "clean_text",
) -> Tuple[BERTopic, list[int], list[float]]:
    """
    Train BERTopic using the provided DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing cleaned comments.
    text_column : str
        Column name of the cleaned text.

    Returns
    -------
    topic_model : BERTopic
        Trained topic model.
    topics : list[int]
        Topic assignment for each document.
    probs : list[float]
        Topic probability for each document.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    texts = df[text_column].tolist()
    topic_model = build_bertopic_model()
    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics, probs


def save_model_and_topics(
    model: BERTopic,
    topics: list[int],
    probs: list[float],
    df: pd.DataFrame,
    model_path: str | Path = "models/bertopic_model",
    topic_assign_path: str | Path = "results/topic_assignments.csv",
) -> None:
    """
    Save trained model and topic assignments.

    Parameters
    ----------
    model : BERTopic
        Trained BERTopic model.
    topics : list[int]
        Topic assignment for each document.
    probs : list[float]
        Topic probabilities.
    df : pd.DataFrame
        DataFrame used for training, will be joined with topics and probabilities.
    model_path : str or Path
        Path where the model will be saved.
    topic_assign_path : str or Path
        Path for the topic assignment file.
    """
    model_path = Path(model_path)
    topic_assign_path = Path(topic_assign_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    topic_assign_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_path)

    df_out = df.copy()
    df_out["topic"] = topics
    df_out["topic_prob"] = probs
    df_out.to_csv(topic_assign_path, index=False)

    print(f"Saved BERTopic model to: {model_path}")
    print(f"Saved topic assignments to: {topic_assign_path}")


def run_modeling(
    clean_data_path: str | Path = "data/clean_comments.csv",
    text_column: str = "clean_text",
    model_path: str | Path = "models/bertopic_model",
    topic_assign_path: str | Path = "results/topic_assignments.csv",
) -> None:
    """
    High level function to perform the full modeling pipeline.

    Parameters
    ----------
    clean_data_path : str or Path
        Path to the cleaned comment data.
    text_column : str
        Column containing cleaned text.
    model_path : str or Path
        Path where the model object will be stored.
    topic_assign_path : str or Path
        Path for topic assignment results.
    """
    df_clean = load_clean_data(clean_data_path)
    topic_model, topics, probs = train_topics(df_clean, text_column=text_column)
    save_model_and_topics(topic_model, topics, probs, df_clean, model_path, topic_assign_path)


if __name__ == "__main__":
    run_modeling(
        clean_data_path="data/clean_comments.csv",
        text_column="clean_text",
        model_path="models/bertopic_model",
        topic_assign_path="results/topic_assignments.csv",
    )
