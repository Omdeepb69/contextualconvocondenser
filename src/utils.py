import json
import os
import re
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Management ---

DEFAULT_CONFIG = {
    'model': {
        'summarizer': 'facebook/bart-large-cnn',
        'sentiment': 'distilbert-base-uncased-finetuned-sst-2-english',
        'ner': 'dbmdz/bert-large-cased-finetuned-conll03-english', # Example NER model
    },
    'processing': {
        'segment_duration_minutes': 5,
        'min_segment_length': 50, # Minimum characters to process a segment
        'summary_max_length': 150,
        'summary_min_length': 30,
    },
    'output': {
        'summary_dir': 'output/summaries',
        'visualization_dir': 'output/visualizations',
        'metrics_dir': 'output/metrics',
    },
    'visualization': {
        'sentiment_plot_title': 'Conversation Sentiment Trend',
        'sentiment_xlabel': 'Time Segment',
        'sentiment_ylabel': 'Sentiment Score',
        'entity_plot_title': 'Key Entity Frequency',
        'entity_xlabel': 'Entity',
        'entity_ylabel': 'Frequency',
        'plot_format': 'png',
        'dpi': 300,
    }
}

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    config_path = Path(config_path)
    if not config_path.is_file():
        logger.warning(f"Configuration file not found at {config_path}. Using default configuration.")
        return DEFAULT_CONFIG
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Simple merge strategy: update default config with loaded values
        # More sophisticated merging might be needed for nested dicts if partial configs are allowed
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        return merged_config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}. Using default configuration.")
        return DEFAULT_CONFIG

# --- File Operations ---

def ensure_dir_exists(dir_path: Union[str, Path]) -> Path:
    """Ensures a directory exists, creating it if necessary."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_transcript(file_path: Union[str, Path]) -> Optional[str]:
    """Reads transcript text from a file."""
    file_path = Path(file_path)
    if not file_path.is_file():
        logger.error(f"Transcript file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading transcript file {file_path}: {e}")
        return None

def save_summary(summary_data: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Saves the structured summary data to a JSON file."""
    output_path = Path(output_path)
    ensure_dir_exists(output_path.parent)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Summary saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving summary to {output_path}: {e}")

def save_plot(fig: plt.Figure, output_path: Union[str, Path], dpi: int = 300) -> None:
    """Saves a matplotlib figure to a file."""
    output_path = Path(output_path)
    ensure_dir_exists(output_path.parent)
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        logger.info(f"Plot saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving plot to {output_path}: {e}")

# --- Text Processing ---

def clean_text(text: str) -> str:
    """Performs basic text cleaning."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    # Add more specific cleaning rules if needed (e.g., removing timestamps if not parsed)
    return text

def split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences using basic punctuation."""
    # A more robust sentence tokenizer (like NLTK or SpaCy) might be better
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def split_by_time_or_token(
    transcript_lines: List[str],
    segment_duration_minutes: Optional[int] = 5,
    tokens_per_segment: int = 500
) -> List[Tuple[Optional[datetime], Optional[datetime], str]]:
    """
    Splits transcript lines into segments based on timestamps or token count.
    Assumes lines might start with a timestamp like '[HH:MM:SS] Speaker: Message'.
    If timestamps are unreliable or absent, falls back to token count.
    Returns list of tuples: (start_time, end_time, segment_text)
    """
    segments = []
    current_segment_text = ""
    current_segment_start_time = None
    last_timestamp = None
    first_timestamp = None
    time_based = False

    # Try to detect timestamps and parse the first one
    for i, line in enumerate(transcript_lines):
        match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]', line)
        if match:
            try:
                # Use a dummy date, only time matters for duration calculation
                timestamp = datetime.strptime(match.group(1), '%H:%M:%S')
                if first_timestamp is None:
                    first_timestamp = timestamp
                    current_segment_start_time = timestamp
                    last_timestamp = timestamp
                    time_based = True # Found at least one valid timestamp
                last_timestamp = timestamp
                break # Found the first timestamp, proceed
            except ValueError:
                continue # Ignore lines with malformed timestamps

    if not time_based or segment_duration_minutes is None:
        logger.info("Using token-based segmentation.")
        current_token_count = 0
        segment_text_list = []
        for line in transcript_lines:
            cleaned_line = re.sub(r'^\[\d{2}:\d{2}:\d{2}\]\s*[^:]*:\s*', '', line).strip() # Remove potential timestamp/speaker
            line_tokens = len(cleaned_line.split())
            if current_token_count + line_tokens > tokens_per_segment and current_token_count > 0:
                segments.append((None, None, " ".join(segment_text_list)))
                segment_text_list = [cleaned_line]
                current_token_count = line_tokens
            else:
                segment_text_list.append(cleaned_line)
                current_token_count += line_tokens
        if segment_text_list: # Add the last segment
             segments.append((None, None, " ".join(segment_text_list)))
        return segments

    # Time-based segmentation
    logger.info("Using time-based segmentation.")
    segment_delta = timedelta(minutes=segment_duration_minutes)
    current_segment_lines = []

    for line in transcript_lines:
        match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]', line)
        timestamp = None
        line_content = line
        if match:
            try:
                timestamp = datetime.strptime(match.group(1), '%H:%M:%S')
                # Handle potential day rollover if conversation spans midnight (simplistic)
                if last_timestamp and timestamp < last_timestamp:
                     # Assuming it rolled over, add a day. This is fragile.
                     # A better approach needs absolute timestamps or date info.
                     timestamp += timedelta(days=1)
                last_timestamp = timestamp
                line_content = re.sub(r'^\[\d{2}:\d{2}:\d{2}\]\s*[^:]*:\s*', '', line).strip() # Remove timestamp/speaker
            except ValueError:
                timestamp = last_timestamp # Use last known timestamp if parsing fails

        if timestamp is None: # Handle lines without timestamps
            timestamp = last_timestamp if last_timestamp else first_timestamp

        if current_segment_start_time is None:
             current_segment_start_time = timestamp if timestamp else first_timestamp

        # Check if the current line belongs to the next segment
        if timestamp and current_segment_start_time and (timestamp >= current_segment_start_time + segment_delta):
            if current_segment_lines:
                segment_end_time = timestamp # End time is start of the line that triggered the split
                segments.append((current_segment_start_time, segment_end_time, "\n".join(current_segment_lines)))
            current_segment_lines = [line_content]
            current_segment_start_time = timestamp
        else:
            current_segment_lines.append(line_content)

    # Add the last segment
    if current_segment_lines:
        # Use the very last known timestamp as the end time for the final segment
        final_end_time = last_timestamp if last_timestamp else (current_segment_start_time + segment_delta if current_segment_start_time else None)
        segments.append((current_segment_start_time, final_end_time, "\n".join(current_segment_lines)))

    # Format timestamps for output if they exist
    formatted_segments = []
    for start, end, text in segments:
        start_str = start.strftime('%H:%M:%S') if start else None
        end_str = end.strftime('%H:%M:%S') if end else None
        formatted_segments.append((start_str, end_str, text))

    return formatted_segments


# --- Data Visualization ---

def plot_sentiment_trend(
    sentiment_scores: List[float],
    segment_labels: List[str],
    output_path: Union[str, Path],
    config: Dict[str, Any]
) -> None:
    """Plots sentiment scores over time segments."""
    if not sentiment_scores or not segment_labels or len(sentiment_scores) != len(segment_labels):
        logger.warning("Insufficient or mismatched data for sentiment plotting.")
        return

    vis_config = config.get('visualization', {})
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['red' if score < 0 else 'green' if score > 0 else 'grey' for score in sentiment_scores]
    
    # Use segment labels directly on x-axis
    x_ticks = range(len(segment_labels))
    ax.bar(x_ticks, sentiment_scores, color=colors, alpha=0.7, width=0.6)
    ax.plot(x_ticks, sentiment_scores, marker='o', linestyle='-', color='darkblue', alpha=0.8) # Trend line

    # Add horizontal line at y=0
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    ax.set_title(vis_config.get('sentiment_plot_title', 'Conversation Sentiment Trend'), fontsize=16)
    ax.set_xlabel(vis_config.get('sentiment_xlabel', 'Time Segment'), fontsize=12)
    ax.set_ylabel(vis_config.get('sentiment_ylabel', 'Sentiment Score (e.g., -1 to 1)'), fontsize=12)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(segment_labels, rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_plot(fig, output_path, dpi=vis_config.get('dpi', 300))


def plot_entity_counts(
    entity_counts: Dict[str, int],
    output_path: Union[str, Path],
    config: Dict[str, Any],
    top_n: int = 15
) -> None:
    """Plots the frequency of the top N named entities."""
    if not entity_counts:
        logger.warning("No entity data provided for plotting.")
        return

    vis_config = config.get('visualization', {})
    
    # Sort entities by frequency and take top N
    sorted_entities = sorted(entity_counts.items(), key=lambda item: item[1], reverse=True)
    top_entities = dict(sorted_entities[:top_n])

    if not top_entities:
        logger.warning("No entities left after filtering for plotting.")
        return

    labels = list(top_entities.keys())
    counts = list(top_entities.values())

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.4))) # Adjust height based on number of labels

    sns.barplot(x=counts, y=labels, palette='viridis', ax=ax, orient='h')

    ax.set_title(vis_config.get('entity_plot_title', 'Key Entity Frequency'), fontsize=16)
    ax.set_xlabel(vis_config.get('entity_ylabel', 'Frequency'), fontsize=12) # Swapped labels for horizontal
    ax.set_ylabel(vis_config.get('entity_xlabel', 'Entity'), fontsize=12) # Swapped labels for horizontal
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Add count labels on bars
    for i, count in enumerate(counts):
        ax.text(count + 0.1, i, str(count), va='center', fontsize=9)

    plt.tight_layout()
    save_plot(fig, output_path, dpi=vis_config.get('dpi', 300))


# --- Metrics Calculation ---

def calculate_rouge(reference_summary: str, generated_summary: str) -> Dict[str, Dict[str, float]]:
    """Calculates ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)."""
    if not reference_summary or not generated_summary:
        logger.warning("Cannot calculate ROUGE score with empty reference or generated summary.")
        return {}
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_summary, generated_summary)
        # Convert Score objects to dictionaries for easier serialization/logging
        return {key: {'precision': score.precision, 'recall': score.recall, 'fmeasure': score.fmeasure}
                for key, score in scores.items()}
    except Exception as e:
        logger.error(f"Error calculating ROUGE scores: {e}")
        return {}

def calculate_classification_metrics(
    true_labels: List[Any],
    predicted_labels: List[Any],
    average: str = 'weighted'
) -> Dict[str, float]:
    """Calculates precision, recall, F1-score, and accuracy for classification tasks."""
    if not true_labels or not predicted_labels or len(true_labels) != len(predicted_labels):
        logger.warning("Cannot calculate classification metrics with invalid or mismatched labels.")
        return {}
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average=average, zero_division=0
        )
        accuracy = accuracy_score(true_labels, predicted_labels)
        return {
            'accuracy': accuracy,
            f'{average}_precision': precision,
            f'{average}_recall': recall,
            f'{average}_f1_score': f1,
        }
    except Exception as e:
        logger.error(f"Error calculating classification metrics: {e}")
        return {}

# --- Formatting ---

def format_summary_output(
    decisions: List[str],
    action_items: List[Dict[str, str]], # e.g., [{'assignee': 'Alice', 'task': 'Update report'}]
    unanswered_questions: List[str],
    sentiment_trends: List[Dict[str, Any]], # e.g., [{'segment': '0-5 min', 'score': 0.8, 'label': 'Positive'}]
    overall_summary: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Formats the extracted information into a structured dictionary."""
    output = {
        "metadata": metadata if metadata else {},
        "overall_summary": overall_summary,
        "key_decisions": decisions,
        "action_items": action_items,
        "unanswered_questions": unanswered_questions,
        "sentiment_analysis": {
            "trend": sentiment_trends,
            # Could add overall sentiment score here if calculated
        }
        # Add sections for topics or entities if generated
    }
    return output

def format_metrics(metrics: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Saves calculated metrics to a JSON file."""
    output_path = Path(output_path)
    ensure_dir_exists(output_path.parent)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {output_path}: {e}")


# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # This block is for demonstration/testing of utils functions
    # It won't run when imported as a module

    # --- Config Example ---
    print("--- Config Example ---")
    # Create a dummy config file
    dummy_config_path = Path("temp_config.yaml")
    with open(dummy_config_path, 'w') as f:
        yaml.dump({'model': {'summarizer': 'google/pegasus-xsum'}}, f)
    config = load_config(dummy_config_path)
    print(f"Loaded config: {config['model']['summarizer']}") # Should be pegasus
    config_default = load_config("non_existent_config.yaml")
    print(f"Default config: {config_default['model']['summarizer']}") # Should be BART
    dummy_config_path.unlink() # Clean up dummy file

    # --- File Ops Example ---
    print("\n--- File Ops Example ---")
    ensure_dir_exists("temp_output/summaries")
    ensure_dir_exists("temp_output/visualizations")
    print("Created temp_output directories.")
    # Create dummy transcript
    dummy_transcript_path = Path("temp_transcript.txt")
    dummy_transcript_content = """[00:00:05] Alice: Hi Bob, let's discuss the project plan.
[00:00:15] Bob: Sure Alice. I think we should finalize the requirements by Friday.
[00:00:30] Alice: Agreed. Can you take the action item to draft the initial list?
[00:00:45] Bob: Yes, I can do that. What about the budget? Is it approved?
[00:01:00] Alice: Not yet, that's still pending. I'll follow up.
[00:05:10] Charlie: Joining late, sorry. What did I miss?
[00:05:20] Alice: We decided Bob will draft requirements by Friday. Budget is pending.
[00:05:35] Bob: Correct. Any questions, Charlie?
[00:05:45] Charlie: No, sounds good for now. I feel positive about this plan.
[00:06:00] Alice: Great. Let's wrap up then. This was productive but short.
"""
    with open(dummy_transcript_path, 'w') as f:
        f.write(dummy_transcript_content)
    transcript_text = read_transcript(dummy_transcript_path)
    print(f"Read transcript (first 50 chars): {transcript_text[:50]}...")

    # --- Text Processing Example ---
    print("\n--- Text Processing Example ---")
    cleaned = clean_text("  [00:00:15] Bob: Sure Alice.   ")
    print(f"Cleaned text: '{cleaned}'")
    sentences = split_into_sentences("First sentence. Second one! Third?")
    print(f"Split sentences: {sentences}")

    # Segmentation example
    transcript_lines = dummy_transcript_content.strip().split('\n')
    time_segments = split_by_time_or_token(transcript_lines, segment_duration_minutes=5)
    print(f"Time-based segments ({len(time_segments)}):")
    for i, (start, end, text) in enumerate(time_segments):
        print(f"  Segment {i+1} ({start}-{end}): {text[:30]}...")

    token_segments = split_by_time_or_token(transcript_lines, segment_duration_minutes=None, tokens_per_segment=30)
    print(f"\nToken-based segments ({len(token_segments)}):")
    for i, (start, end, text) in enumerate(token_segments):
        print(f"  Segment {i+1}: {text[:30]}...")


    # --- Visualization Example ---
    print("\n--- Visualization Example ---")
    # Dummy data
    sent_scores = [-0.5, 0.8, 0.1, 0.9, -0.2]
    seg_labels = ['0-5m', '5-10m', '10-15m', '15-20m', '20-25m']
    sent_plot_path = Path(config_default['output']['visualization_dir']) / "sentiment_trend.png"
    ensure_dir_exists(sent_plot_path.parent)
    plot_sentiment_trend(sent_scores, seg_labels, sent_plot_path, config_default)
    print(f"Sentiment plot saved to {sent_plot_path}")

    entities = {'project plan': 5, 'Alice': 4, 'Bob': 3, 'requirements': 2, 'Friday': 2, 'budget': 1}
    ent_plot_path = Path(config_default['output']['visualization_dir']) / "entity_counts.png"
    plot_entity_counts(entities, ent_plot_path, config_default)
    print(f"Entity plot saved to {ent_plot_path}")

    # --- Metrics Example ---
    print("\n--- Metrics Example ---")
    ref = "The team decided to finalize requirements by Friday. Bob will draft them. Budget is pending."
    gen = "Bob is drafting requirements by Friday. Alice will check the budget status."
    rouge_scores = calculate_rouge(ref, gen)
    print(f"ROUGE Scores: {json.dumps(rouge_scores, indent=2)}")

    true_sent = ['positive', 'negative', 'positive', 'neutral']
    pred_sent = ['positive', 'positive', 'positive', 'negative']
    class_metrics = calculate_classification_metrics(true_sent, pred_sent)
    print(f"Classification Metrics: {class_metrics}")

    # --- Formatting Example ---
    print("\n--- Formatting Example ---")
    summary_data = format_summary_output(
        decisions=["Finalize requirements by Friday"],
        action_items=[{'assignee': 'Bob', 'task': 'Draft initial requirements list'}, {'assignee': 'Alice', 'task': 'Follow up on budget approval'}],
        unanswered_questions=["Is the budget approved?"],
        sentiment_trends=[{'segment': '0-5 min', 'score': 0.6, 'label': 'Positive'}, {'segment': '5-10 min', 'score': 0.2, 'label': 'Slightly Positive'}],
        overall_summary="The team agreed on the requirement deadline, assigned drafting to Bob, and noted the budget needs follow-up.",
        metadata={'filename': 'meeting_log.txt', 'duration_minutes': 10}
    )
    summary_path = Path(config_default['output']['summary_dir']) / "example_summary.json"
    save_summary(summary_data, summary_path)
    print(f"Formatted summary saved to {summary_path}")

    # --- Clean up ---
    print("\n--- Cleaning up ---")
    dummy_transcript_path.unlink()
    import shutil
    if Path("temp_output").exists():
         shutil.rmtree("temp_output")
    if Path("output").exists(): # Clean up generated plots/summaries
         shutil.rmtree("output")
    print("Temporary files and directories removed.")