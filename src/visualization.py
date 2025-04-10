```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter
import warnings

# Suppress warnings for cleaner output if needed
# warnings.filterwarnings("ignore")


def plot_utterance_length_distribution(transcript_df: pd.DataFrame, title: str = "Distribution of Utterance Lengths"):
    """
    Plots the distribution of utterance lengths using Plotly.

    Args:
        transcript_df (pd.DataFrame): DataFrame with an 'utterance' column.
        title (str): Title for the plot.
    """
    if 'utterance' not in transcript_df.columns:
        print("Error: 'utterance' column not found in DataFrame.")
        return
    if transcript_df['utterance'].isnull().all():
        print("Error: 'utterance' column contains only null values.")
        return

    # Calculate length in words, handling potential NaN/None values
    utterance_lengths = transcript_df['utterance'].dropna().astype(str).str.split().str.len()

    if utterance_lengths.empty:
        print("Error: No valid utterances found to calculate lengths.")
        return

    fig = px.histogram(
        x=utterance_lengths,
        nbins=50,
        title=title,
        labels={'x': 'Utterance Length (Number of Words)', 'count': 'Frequency'} # Plotly uses 'count' for y-axis label in histograms
    )
    fig.update_layout(bargap=0.1, yaxis_title="Frequency")
    fig.show()


def plot_speaker_contribution(transcript_df: pd.DataFrame, title: str = "Speaker Contribution (Number of Utterances)"):
    """
    Plots the contribution of each speaker based on the number of utterances using Plotly.

    Args:
        transcript_df (pd.DataFrame): DataFrame with a 'speaker' column.
        title (str): Title for the plot.
    """
    if 'speaker' not in transcript_df.columns:
        print("Error: 'speaker' column not found in DataFrame.")
        return
    if transcript_df['speaker'].isnull().all():
        print("Error: 'speaker' column contains only null values.")
        return

    speaker_counts = transcript_df['speaker'].dropna().value_counts()

    if speaker_counts.empty:
        print("Error: No valid speakers found to plot contribution.")
        return

    fig = px.pie(
        values=speaker_counts.values,
        names=speaker_counts.index,
        title=title,
        hole=0.3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.show()


def plot_sentiment_trend(sentiment_trends: pd.DataFrame, time_col: str = 'segment_id', sentiment_col: str = 'sentiment_score', title: str = "Sentiment Trend Over Conversation Segments"):
    """
    Plots the sentiment trend over time segments using an interactive Plotly line chart.

    Args:
        sentiment_trends (pd.DataFrame): DataFrame with time/segment and sentiment score columns.
        time_col (str): Name of the column representing time or segments.
        sentiment_col (str): Name of the column representing sentiment score.
        title (str): Title for the plot.
    """
    if time_col not in sentiment_trends.columns or sentiment_col not in sentiment_trends.columns:
        print(f"Error: Required columns '{time_col}' or '{sentiment_col}' not found.")
        return
    if sentiment_trends[[time_col, sentiment_col]].isnull().all().all():
         print(f"Error: Columns '{time_col}' and '{sentiment_col}' contain only null values.")
         return

    # Ensure data is not empty after dropping NaNs
    plot_data = sentiment_trends[[time_col, sentiment_col]].dropna()
    if plot_data.empty:
        print(f"Error: No valid data found in columns '{time_col}' and '{sentiment_col}' after dropping NaNs.")
        return

    # Ensure time_col is suitable for plotting (numeric or sortable)
    try:
        plot_data = plot_data.sort_values(by=time_col)
    except Exception as e:
        print(f"Error sorting by '{time_col}': {e}. Ensure it's a sortable type.")
        return

    fig = px.line(
        plot_data,
        x=time_col,
        y=sentiment_col,
        title=title,
        markers=True,
        labels={time_col: "Time / Segment", sentiment_col: "Sentiment Score"}
    )
    fig.update_layout(xaxis_title="Time / Segment", yaxis_title="Sentiment Score")
    fig.show()


def plot_sentiment_distribution(transcript_df: pd.DataFrame, sentiment_col: str = 'sentiment_score', title: str = "Distribution of Sentiment Scores"):
    """
    Plots the distribution of sentiment scores across utterances using Plotly.

    Args:
        transcript_df (pd.DataFrame): DataFrame with a sentiment score column.
        sentiment_col (str): Name of the column representing sentiment score.
        title (str): Title for the plot.
    """
    if sentiment_col not in transcript_df.columns:
        print(f"Error: Sentiment column '{sentiment_col}' not found.")
        return
    if transcript_df[sentiment_col].isnull().all():
        print(f"Error: Sentiment column '{sentiment_col}' contains only null values.")
        return

    sentiment_scores = transcript_df[sentiment_col].dropna()
    if sentiment_