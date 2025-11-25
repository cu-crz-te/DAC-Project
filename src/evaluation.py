"""
Evaluation and visualization module for Garena community NLP project.

This script loads a trained BERTopic model and topic assignments,
then produces simple summaries and plots for interpretation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from bertopic import BERTopic


def load_trained_model(path: str | Path = "models/bertopic_model") -> BERTopic:
    """
    Load a trained BERTopic model.

    Parameters
    ----------
    path : str or Path
        Path to the saved model.

    Returns
    -------
    BERTopic
        Trained topic model.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {path}")
    model = BERTopic.load(path)
    return model


def load_topic_assignments(path: str | Path = "results/topic_assignments.csv") -> pd.DataFrame:
    """
    Load topic assignment results.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file containing topic assignments.

    Returns
    -------
    pd.DataFrame
        Data with assigned topics and probabilities.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Topic assignment file not found at: {path}")
    return pd.read_csv(path)


def export_top_words(
    model: BERTopic,
    n_words: int = 10,
    output_path: str | Path = "results/topic_top_words.csv",
) -> None:
    """
    Export top words per topic to a CSV file.

    Parameters
    ----------
    model : BERTopic
        Trained topic model.
    n_words : int
        Number of top words per topic.
    output_path : str or Path
        Path where the CSV will be saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    topics_info = model.get_topic_info()
    rows = []
    for topic_id in topics_info["Topic"]:
        if topic_id == -1:
            continue
        words = model.get_topic(topic_id)
        top_words = [w for w, _ in words[:n_words]]
        rows.append(
            {
                "topic_id": topic_id,
                "top_words": ", ".join(top_words),
            }
        )

    df_topics = pd.DataFrame(rows)
    df_topics.to_csv(output_path, index=False)
    print(f"Saved topic top words to: {output_path}")


def plot_topic_distribution(
    df: pd.DataFrame,
    output_path: str | Path = "results/topic_distribution.png",
) -> None:
    """
    Plot the count of each topic.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'topic' column.
    output_path : str or Path
        Path to save the figure.
    """
    if "topic" not in df.columns:
        raise ValueError("DataFrame must contain a 'topic' column.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    topic_counts = df["topic"].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    topic_counts.plot(kind="bar")
    plt.xlabel("Topic")
    plt.ylabel("Count")
    plt.title("Topic Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved topic distribution plot to: {output_path}")


def run_evaluation(
    model_path: str | Path = "models/bertopic_model",
    topic_assign_path: str | Path = "results/topic_assignments.csv",
) -> None:
    """
    High level function to run evaluation steps.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved BERTopic model.
    topic_assign_path : str or Path
        Path to the topic assignment CSV.
    """
    model = load_trained_model(model_path)
    df_topics = load_topic_assignments(topic_assign_path)

    export_top_words(model, n_words=10, output_path="results/topic_top_words.csv")
    plot_topic_distribution(df_topics, output_path="results/topic_distribution.png")


if __name__ == "__main__":
    run_evaluation(
        model_path="models/bertopic_model",
        topic_assign_path="results/topic_assignments.csv",
    )
