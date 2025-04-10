```python
import argparse
import os
import sys
import re
import warnings
from collections import defaultdict
import numpy as np
import spacy
from transformers import pipeline, logging as hf_logging
import torch

# Suppress warnings and logs for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
DEFAULT_SUMMARIZER_MODEL = "facebook/bart-large-cnn"
DEFAULT_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_NER_MODEL = "en_core_web_sm"
DEFAULT_SEGMENT_LINES = 50 # Number of lines per sentiment segment

# Helper Functions
def download_spacy_model(model_name):
    """Checks if a spaCy model is installed and downloads it if not."""
    try:
        spacy.load(model_name)
        print(f"Spacy model '{model_name}' found.", file=sys.stderr)
    except OSError:
        print(f"Spacy model '{model_name}' not found. Attempting download...", file=sys.stderr)
        try:
            spacy.cli.download(model_name)
            spacy.load(model_name) # Verify download
            print(f"Model '{model_name}' downloaded successfully.", file=sys.stderr)
        except SystemExit: # spacy.cli.download raises SystemExit on failure
             print(f"Error: Failed to download spacy model '{model_name}'.", file=sys.stderr)
             print(f"Please install it manually: python -m spacy download {model_name}", file=sys.stderr)
             sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during spaCy model download: {e}", file=sys.stderr)
            print(f"Please try installing it manually: python -m spacy download {model_name}", file=sys.stderr)
            sys.exit(1)

def read_transcript(filepath):
    """Reads the transcript text from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{filepath}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)

def write_summary(summary_data, filepath=None):
    """Formats and writes the summary data to console or file."""
    output_lines = [
        "Contextual Conversation Condenser Summary",
        "=========================================",
        "",
        "Overall Summary",
        "---------------",
        summary_data.get('overall_summary', 'N/A'),
        "",
        "Key Decisions",
        "-------------",
    ]
    output_lines.extend([f"- {item}" for item in summary_data.get('decisions', ['N/A'])])
    output_lines.extend([
        "",
        "Action Items",
        "------------",
    ])
    output_lines.extend([f"- {item}" for item in summary_data.get('action_items', ['N/A'])])
    output_lines.extend([
        "",
        "Unanswered Questions",
        "--------------------",
    ])
    output_lines.extend([f"- {item}" for item in summary_data.get('questions', ['N/A'])])
    output_lines.extend([
        "",
        "Sentiment Trend ('Vibe Check')",
        "------------------------------",
    ])

    sentiments = summary_data.get('sentiment_trend', [])
    if sentiments:
        for i, sent in enumerate(sentiments):
             output_lines.append(f"Segment {i+1}: Label={sent['label']}, Score={sent['score']:.4f}")
    else:
        output_lines.append("N/A")

    output_str = "\n".join(output_lines)

    if filepath:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(output_str)
            print(f"Summary successfully written to '{filepath}'", file=sys.stderr)
        except Exception as e:
            print(f"Error writing summary to file '{filepath}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output_str)


# Core NLP Functions
def get_summarizer(model_name, use_gpu):
    """Loads the Hugging Face summarization pipeline."""
    try:
        device = 0 if torch.cuda.is_available() and use_gpu else -1
        summarizer = pipeline("summarization", model=model_name, device=device)
        device_name = 'GPU' if device == 0 else 'CPU'
        print(f"Summarizer pipeline loaded successfully using model '{model_name}' on device {device_name}.", file=sys.stderr)
        return summarizer
    except Exception as e:
        print(f"Error loading summarization model '{model_name}': {e}", file=sys.stderr)
        print("Please ensure you have PyTorch installed and the model name is correct.", file=sys.stderr)
        sys.exit(1)

def get_sentiment_analyzer(model_name, use_gpu):
    """Loads the Hugging Face sentiment analysis pipeline."""
    try:
        device = 0 if torch.cuda.is_available() and use_gpu else -1
        sentiment_analyzer = pipeline("sentiment-analysis", model=model_name, device=device)
        device_name = 'GPU' if device == 0 else 'CPU'
        print(f"Sentiment analysis pipeline loaded successfully using model '{model_name}' on device {device_name}.", file=sys.stderr)
        return sentiment_analyzer
    except Exception as e:
        print(f"Error loading sentiment analysis model '{model_name}': {e}", file=sys.stderr)
        sys.exit(1)

def get_ner_model(model_name):
    """Loads the spaCy NER model."""
    try:
        # Disable unnecessary pipes for efficiency if only using NER
        nlp = spacy.load(model_name, disable=["tagger", "parser", "lemmatizer"])
        print(f"NER model '{model_name}' loaded successfully.", file=sys.stderr)
        return nlp
    except Exception as e:
        print(f"Error loading spaCy NER model '{model_name}': {e}", file=sys.stderr)
        sys.exit(1)

def summarize_text(text, summarizer):
    """Generates a summary of the input text using the loaded pipeline."""
    try:
        # Estimate token count roughly (split by space) for length calculation
        word_count = len(text.split())
        # Set min/max length relative to input size, within model limits
        model_max_length = summarizer.tokenizer.model_max_length
        # Use a fraction of word count, clamped to reasonable bounds
        min_len = max(30, min(int(word_count * 0.1), 100))
        max_len = max(min_len + 50, min(int(word_count * 0.3), model_max_length - 20)) # Ensure max_len > min_len and within limits

        # Summarize (truncation is handled by the pipeline by default if text > model_max_length)
        summary = summarizer(text, min_length=min_len, max_length=max_len, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}", file=sys.stderr)
        # Attempt summarization without length constraints as fallback
        try:
            print("Retrying summarization with default lengths...", file=sys.stderr)
            summary = summarizer(text, do_sample=False)[0]['summary_text']
            return summary
        except Exception as e_fallback:
            print(f"Fallback summarization also failed: {e_fallback}", file=sys.stderr)
            return "Summarization failed."


def analyze_sentiment_segments(text, sentiment_analyzer, segment_lines):
    """Analyzes sentiment over segments of the transcript."""
    lines = [line for line in text.splitlines() if line.strip()] # Ignore empty lines
    if not lines:
        return []

    segments = ['\n'.join(lines[i:i + segment_lines]) for i in range(0, len(lines), segment_lines)]
    segment_sentiments = []
    try:
        # Process segments, handling potential truncation needed for the model
        results = sentiment_analyzer(segments, truncation=True)
        segment_sentiments = results if isinstance(results, list) else [results] # Ensure list output
        return segment_sentiments
    except Exception as e:
        print(f"Error during sentiment analysis: {e}", file=sys.stderr)
        return []

def extract_key_elements(text, nlp):
    """Extracts decisions, action items, and questions using NER and heuristics."""
    decisions = set()
    action_items = set()
    questions = set()
    persons = set()

    try:
        # Process the entire text at once for context
        doc = nlp(text)

        # Collect all PERSON entities first
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                persons.add(ent.text.strip())

        # Process sentence by sentence
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        for i, sent in enumerate(sentences):
            lower_sent = sent.lower()
            sent_doc = nlp(sent) # Re-process sentence for local entities if needed

            # Identify Decisions (keywords + structure)
            decision_keywords = ["decided", "agreed", "confirmed", "decision is", "plan is", "will proceed", "conclusion is", "we should", "let's go with"]
            if any(keyword in lower_sent for keyword in decision_keywords) and len(sent.split()) > 4:
                decisions.add(sent)

            # Identify Action Items (keywords, verbs, presence of PERSON)
            action_keywords = ["action item", "task", "to-do", "next step", "assign", "need to", "will", "responsible for"]
            action_verbs = ["follow up", "investigate", "research", "send", "create", "update", "complete", "prepare", "schedule", "contact", "circle back", "look into", "handle"]
            has_action_keyword = any(keyword in lower_sent for keyword in action_keywords)
            has_action_verb = any(verb in lower_sent for verb in action_verbs)
            has_person_entity = any(ent.label_ == "PERSON" for ent in sent_doc.ents)
            mentions_known_person = any(person in sent for person in persons if len(person.split()) > 1) # Avoid single names matching common words

            if (has_action_keyword or has_action_verb) and (has_person_entity or mentions_known_person) and len(sent.split()) > 3:
                 action_items.add(sent)
            # Catch simpler "I will / Person will" cases
            elif re.search(r"\b(i|we|he|she|they|[A-Z][a-z]+)\s+(will|gonna|going to)\s+([a-z]+)", sent):
                 if any(verb in lower_sent for verb in action_verbs) and len(sent.split()) > 3:
                     action_items.add(sent)


            # Identify Questions
            if sent.endswith("?"):
                # Basic check: is the *next* sentence potentially an answer? (Very naive)
                is_answered = False
                if i + 1 < len(sentences):
                    next_sent_lower = sentences[i+1].lower()
                    # Simple check if next sentence contains keywords or doesn't start like a new question/topic
                    if not next_sent_lower.endswith("?") and not next_sent_lower.startswith(("so", "well", "and", "but")):
                        is_answered = True # Assume answered for simplicity here

                # For this version, let's list all questions found
                # if not is_answered: # Keep this logic commented out for now to list all Qs
                questions.add(sent)


        # Format results
        final_decisions = sorted(list(decisions)) if decisions else ["No specific decisions identified."]
        final_actions = sorted(list(action_items)) if action_items else ["No specific action items identified."]
        final_questions = sorted(list(questions)) if questions else ["No specific questions identified."]

        return {
            "decisions": final_decisions,
            "action_items": final_actions,
            "questions": final_questions
        }
    except Exception as e:
        print(f"Error during key element extraction: {e}", file=sys.stderr)
        # Print traceback for debugging if needed
        # import traceback
        # traceback.print_exc()
        return {
            "decisions": ["Extraction failed due to error."],
            "action_items": ["Extraction failed due to error."],
            "questions": ["Extraction failed due to error."]
        }


# Main Function
def main():
    parser = argparse.ArgumentParser(
        description="ContextualConvoCondenser: Summarize conversation transcripts, extracting key info and sentiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file", help="Path to the conversation transcript text file.")
    parser.add_argument("-o", "--output_file", help="Path to save the structured summary (optional, prints to console if omitted).", default=None)
    parser.add_argument("--summarizer_model", help="Hugging Face model for summarization.", default=DEFAULT_SUMMARIZER_MODEL)
    parser.add_argument("--sentiment_model", help="Hugging Face model for sentiment analysis.", default=DEFAULT_SENTIMENT_MODEL)
    parser.add_argument("--ner_model", help="Spacy model for Named Entity Recognition.", default=DEFAULT_NER_MODEL)
    parser.add_argument("--segment_lines", type=int, help="Number of non-empty lines per sentiment analysis segment.", default=DEFAULT_SEGMENT_LINES)
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU usage (force CPU).")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    use_gpu = not args.no_gpu
    if use_gpu and not torch.cuda.is_available():
        print("Warning: --no_