```python
import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import nltk
import logging

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_TOKENIZER = "facebook/bart-large-cnn"

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def parse_transcript_simple(filepath: str) -> pd.DataFrame:
    """
    Parses a simple transcript format (e.g., 'Speaker A: Text').
    Assumes each line is an utterance. Handles potential errors.
    """
    utterances = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {e}")
        raise

    pattern = re.compile(r"^(?:([\w\s]+):\s)?(.*)$")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if match:
            speaker, text = match.groups()
            speaker = speaker.strip() if speaker else f"Unknown Speaker {i+1}"
            text = clean_text(text)
            if text:
                 utterances.append({"speaker": speaker, "utterance": text})
        else:
            cleaned_line = clean_text(line)
            if cleaned_line:
                utterances.append({"speaker": f"Unknown Speaker {i+1}", "utterance": cleaned_line})

    if not utterances:
        logging.warning(f"No utterances parsed from file: {filepath}")
        return pd.DataFrame(columns=["speaker", "utterance"])

    return pd.DataFrame(utterances)


def segment_conversation(df: pd.DataFrame, segment_length: Optional[int] = None, overlap: int = 0) -> List[pd.DataFrame]:
    """
    Segments the conversation DataFrame into smaller chunks.
    Useful for processing long conversations or for time-based analysis.
    Returns a list of DataFrames, each representing a segment.
    """
    if segment_length is None or segment_length <= 0:
        return [df]

    if overlap >= segment_length:
        raise ValueError("Overlap must be smaller than segment_length")

    num_utterances = len(df)
    segments = []
    start = 0
    while start < num_utterances:
        end = min(start + segment_length, num_utterances)
        segment_df = df.iloc[start:end].copy()
        segments.append(segment_df)
        start += segment_length - overlap
        if end == num_utterances:
             break
    return segments


def tokenize_dataframe(df: pd.DataFrame, tokenizer_name: str = DEFAULT_TOKENIZER, text_column: str = "utterance", max_length: Optional[int] = None) -> pd.DataFrame:
    """
    Adds tokenized columns (input_ids, attention_mask) to the DataFrame.
    Operates on individual utterances.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        logging.error(f"Failed to load tokenizer '{tokenizer_name}': {e}")
        raise

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    df[text_column] = df[text_column].fillna("")

    logging.info(f"Tokenizing column '{text_column}' using {tokenizer_name}...")

    effective_max_length = max_length if max_length is not None else tokenizer.model_max_length
    if effective_max_length is None or effective_max_length > 1024 : # Set a practical limit if needed
        default_max_len = 512
        logging.warning(f"Tokenizer max length is {effective_max_length}. Using {default_max_len} as default if not specified via max_length argument.")
        effective_max_length = max_length if max_length is not None else default_max_len


    tokenized_outputs = tokenizer(
        df[text_column].tolist(),
        padding="max_length" if max_length is not None else True,
        truncation=True,
        max_length=effective_max_length,
        return_tensors=None,
        add_special_tokens=True
    )

    df['input_ids'] = tokenized_outputs['input_ids']
    df