# text_preprocessor.py
"""
General text preprocessing utilities for BabyLM datasets.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import ftfy
import pandas as pd


class BasePreprocessor(ABC):
    """
    Base class for text preprocessors.
    Subclass this for source-specific preprocessing.
    """

    def __init__(self):
        super().__init__()

    def process_text(self, text: str) -> str:
        """
        General preprocessing: fix unicode and normalize whitespace.
        """
        text = fix_unicode(text)
        text = normalize_whitespace(text)
        return text


class TranscriptPreprocessor(BasePreprocessor):
    """Preprocessor for transcript-style content (e.g., CHILDES)."""

    def __init__(
        self,
        remove_speaker_labels: bool = True,
        remove_annotations: bool = True,
    ):
        self.remove_speaker_labels = remove_speaker_labels
        self.remove_annotations = remove_annotations
        self.speaker_pattern = re.compile(r"^[A-Z]{3}:\s*")  # "MOT: Hello"
        self.annotation_pattern = re.compile(r"%[a-z]+:.*$", re.MULTILINE)

    def preprocess_line(self, line: str) -> str:
        # super().process_text will already apply fix_unicode and normalize_whitespace
        if self.remove_annotations and line.strip().startswith("%"):
            return ""
        if self.remove_speaker_labels:
            line = self.speaker_pattern.sub("", line)
        line = normalize_punctuation(line)
        return line

    def process_text(self, text: str) -> str:
        text = super().process_text(text)
        lines = text.splitlines()
        processed_lines = [self.preprocess_line(line) for line in lines]
        processed_lines = [l for l in processed_lines if l.strip()]
        return "\n".join(processed_lines)


class SubtitlePreprocessor(BasePreprocessor):
    """Preprocessor for subtitle-like content."""

    def __init__(self):
        self.speaker_pattern = re.compile(r"^[A-Z\s]+:\s*")
        self.music_note_pattern = re.compile(r"[♪♫]+")

    def preprocess_line(self, line: str) -> str:
        # super().process_text will already apply fix_unicode and normalize_whitespace
        line = self.speaker_pattern.sub("", line)
        line = self.music_note_pattern.sub("", line)
        line = remove_stage_directions(line)
        line = remove_timestamps(line)
        line = normalize_punctuation(line)
        return line

    def process_text(self, text: str) -> str:
        text = super().process_text(text)
        lines = text.splitlines()
        processed_lines = [self.preprocess_line(line) for line in lines]
        processed_lines = [l for l in processed_lines if l.strip()]
        return "\n".join(processed_lines)


class BookPreprocessor(BasePreprocessor):
    """
    Preprocessor for book-like formats (e.g. textbooks, child books, child wikis, etc.).
    """

    def process_text(self, text: str) -> str:
        text = super().process_text(text)
        text = remove_xml_tags(text)
        text = normalize_punctuation(text)
        text = remove_urls(text)
        return text


class QEDPreprocessor(BasePreprocessor):
    """
    Preprocessor for QED dataset formats.
    """

    def process_text(self, text: str) -> str:
        text = super().process_text(text)
        text = remove_xml_tags(text)
        text = normalize_punctuation(text)
        text = remove_urls(text)
        return text


def normalize_punctuation(text: str) -> str:
    """Normalize various punctuation marks, quotes, ellipsis, underscores, and dashes."""
    import re

    # Normalize all double quotes to "
    text = re.sub(r"[“”‟❝❞〝〞＂«»]", '"', text)
    # Normalize all single quotes to '
    text = re.sub(r"[‘’‛`´′‵']", "'", text)

    # Normalize ellipsis ...
    # look for sequences of dots and optional;uy spaces, then replaces with "..."
    # Ensure these specific ellipsis chars are handled
    text = re.sub(r"[…‥]", "...", text)
    # moved here to handle a variety of ellipsis forms
    text = re.sub(r"(\.[\s\.]*){3,}", "...", text)

    # Normalize underscores _
    text = re.sub(r"[_]+", "_", text)

    # Normalize dashes -
    text = re.sub(r"[–—-]", "-", text)

    return text


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def remove_xml_tags(text: str) -> str:
    """Remove any remaining XML/HTML tags."""
    tag_pattern = re.compile(r"<[^>]+>")
    return tag_pattern.sub("", text)


def fix_unicode(text: str) -> str:
    """Fix unicode issues using ftfy."""
    return ftfy.fix_text(text)


def remove_annotations(text: str) -> str:
    # Common transcript patterns
    text = re.sub(r"%[a-z]+:.*$", "", text, flags=re.MULTILINE)  # %act: points
    return text


def remove_speaker_labels(text: str) -> str:
    text = re.sub(r"^[A-Z\s]+:\s*", "", text)  # "JOHN: Hello"
    return text


def remove_music_note_symbols(text: str) -> str:
    # Remove music note symbols
    return re.sub(r"[♪♫]+", "", text)


def remove_timestamps(text: str) -> str:
    """Remove timestamp patterns from text."""
    # Timestamps: [00:00:00], (00:00:00), 00:00:00
    timestamp_patterns = [
        re.compile(r"\[\d{1,2}:\d{2}:\d{2}\]"),
        re.compile(r"\(\d{1,2}:\d{2}:\d{2}\)"),
        re.compile(r"^\d{1,2}:\d{2}:\d{2}\s*"),
        re.compile(r"\s*\d{1,2}:\d{2}:\d{2}\s*$"),
    ]
    for pattern in timestamp_patterns:
        text = pattern.sub("", text)
    return text


def remove_stage_directions(text: str) -> str:
    """Remove [bracketed] stage directions."""
    # Stage directions: [Music], [Applause], etc.
    stage_direction_pattern = re.compile(r"\[([^\]]+)\]")
    return stage_direction_pattern.sub("", text)


def remove_subtitle_artifacts(text: str) -> str:
    """Remove common subtitle artifacts like leading dashes, line numbers, etc."""
    # Common subtitle artifacts
    subtitle_artifacts = [
        re.compile(r"^(-\s*)+"),  # Leading dashes
        re.compile(r"^\d+\s*$"),  # Line numbers
        re.compile(r"^\s*\.\.\.\s*$"),  # Ellipsis-only lines
    ]
    for pattern in subtitle_artifacts:
        text = pattern.sub("", text)
    return text


def normalize_whitespace(
    text: str,
    preserve_paragraphs: bool = True,
    remove_newlines: bool = False,
) -> str:
    # normalize paragraphs
    text = re.sub(
        r"\n\n+", "\n\n", text
    )  # Normalize multiple newlines to double newlines

    # preserve paragraph structure
    if preserve_paragraphs:
        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")

        processed_paragraphs = []

        for para in paragraphs:
            # Replace multiple spaces with single space within paragraph
            para = re.sub(r"[ \t]+", " ", para)

            # Replace single newlines with space within paragraph
            if remove_newlines:
                para = re.sub(r"\n", " ", para)

            para = para.strip()
            # Only keep non-empty paragraphs
            if para:
                processed_paragraphs.append(para)

        # Join paragraphs back with double newlines
        return "\n\n".join(processed_paragraphs)

    if remove_newlines:
        # Collapse all whitespace
        text = re.sub(r"\s+", " ", text)
    else:
        # Collapse multiple spaces to single space
        text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def create_preprocessor(category: str) -> BasePreprocessor:
    """
    Factory function to create a preprocessor based on the document category.
    """
    preprocessors = {
        # group "book" formats together
        "educational": BookPreprocessor,
        "child-books": BookPreprocessor,
        "child-wiki": BookPreprocessor,
        "child-news": BookPreprocessor,
        "simplified-text": BookPreprocessor,
        # transcript-like, speech
        "subtitles": SubtitlePreprocessor,
        "child-directed-speech": TranscriptPreprocessor,
        "child-available-speech": TranscriptPreprocessor,
        # special
        "qed": QEDPreprocessor,
        # padding-wikipedia padding-xxx?
    }
    return preprocessors[category]()


def preprocess_dataset(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset based on document category.
    """
    processed_groups = []
    for category, group in dataset_df.groupby("category"):
        group = group.copy()
        processor = create_preprocessor(str(category))
        group["text"] = group["text"].apply(processor.process_text)
        processed_groups.append(group)

    return pd.concat(processed_groups)
