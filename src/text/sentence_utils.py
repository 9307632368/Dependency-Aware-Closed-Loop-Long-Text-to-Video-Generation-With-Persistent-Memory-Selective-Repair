# src/text/sentence_utils.py
import re
from typing import List, Dict, Any


def split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter using punctuation.
    Good enough for prompts and narrative text.
    """
    text = text.strip()
    if not text:
        return []

    # split on punctuation followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def get_sentence_spans(text: str) -> List[Dict[str, Any]]:
    """
    Returns sentence text with start/end char spans in original text.
    """
    sentences = split_sentences(text)
    spans = []

    search_start = 0
    for sent in sentences:
        idx = text.find(sent, search_start)
        if idx == -1:
            idx = text.find(sent)
        if idx == -1:
            start = -1
            end = -1
        else:
            start = idx
            end = idx + len(sent)
            search_start = end

        spans.append({
            "sentence": sent,
            "start_char_index": start,
            "end_char_index": end
        })

    return spans