# text_preprocessor.py
"""
General text preprocessing utilities for BabyLM datasets.
Can be extended for source-specific preprocessing needs.
"""

import re
from typing import List, Optional, Callable, Dict, Any
import ftfy
from abc import ABC, abstractmethod
from language_scripts import SCRIPT_NAMES


class BasePreprocessor(ABC):
    """
    Base class for text preprocessors.
    Subclass this for source-specific preprocessing.
    """

    def __init__(
        self,
        lowercase: bool = False,
        normalize_whitespace: bool = True,
        fix_unicode: bool = True,
        remove_timestamps: bool = False,
        remove_stage_directions: bool = False,
        preserve_paragraphs: bool = True,
        replace_newline_within_paragraph: bool = False,
        custom_steps: Optional[List[Callable]] = None,
    ):
        """
        Initialize preprocessor with configuration.

        Args:
            lowercase: Whether to lowercase text (default: False to preserve capitalization)
            normalize_whitespace: Whether to normalize whitespace
            fix_unicode: Whether to fix unicode issues with ftfy
            remove_timestamps: Whether to remove timestamp patterns
            remove_stage_directions: Whether to remove [bracketed] stage directions
            preserve_paragraphs: Whether to preserve paragraph breaks (double newlines)
            replace_newline_within_paragraph: Whether to replace single newlines with space within paragraphs
            custom_steps: List of additional preprocessing functions
        """
        self.lowercase = lowercase
        self.normalize_whitespace = normalize_whitespace
        self.fix_unicode = fix_unicode
        self.remove_timestamps = remove_timestamps
        self.remove_stage_directions = remove_stage_directions
        self.preserve_paragraphs = preserve_paragraphs
        self.replace_newline_within_paragraph = replace_newline_within_paragraph
        self.custom_steps = custom_steps or []

        # Compile regex patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns used in preprocessing."""
        # Stage directions: [Music], [Applause], etc.
        self.stage_direction_pattern = re.compile(r"\[([^\]]+)\]")

        # Timestamps: [00:00:00], (00:00:00), 00:00:00
        self.timestamp_patterns = [
            re.compile(r"\[\d{1,2}:\d{2}:\d{2}\]"),
            re.compile(r"\(\d{1,2}:\d{2}:\d{2}\)"),
            re.compile(r"^\d{1,2}:\d{2}:\d{2}\s*"),
            re.compile(r"\s*\d{1,2}:\d{2}:\d{2}\s*$"),
        ]

        # Multiple whitespace
        self.whitespace_pattern = re.compile(r"\s+")

        # Common subtitle artifacts
        self.subtitle_artifacts = [
            re.compile(r"^(-\s*)+"),  # Leading dashes
            re.compile(r"^\d+\s*$"),  # Line numbers
            re.compile(r"^\s*\.\.\.\s*$"),  # Ellipsis-only lines
        ]

    def _fix_unicode(self, text: str) -> str:
        return ftfy.fix_text(text)

    def _normalize_whitespace(self, text: str) -> str:
        if self.preserve_paragraphs:
            paragraphs = text.split("\n\n")
            processed_paragraphs = []
            for para in paragraphs:
                para = re.sub(r"[ \t]+", " ", para)
                if self.replace_newline_within_paragraph:
                    para = re.sub(r"\n", " ", para)
                para = para.strip()
                if para:
                    processed_paragraphs.append(para)
            return "\n\n".join(processed_paragraphs)
        else:
            text = self.whitespace_pattern.sub(" ", text)
            return text.strip()

    def _remove_timestamps(self, text: str) -> str:
        for pattern in self.timestamp_patterns:
            text = pattern.sub("", text)
        return text

    def _remove_stage_directions(self, text: str) -> str:
        return self.stage_direction_pattern.sub("", text)

    def preprocess_text(self, text: str) -> str:
        """
        Apply all general and category-specific preprocessing steps to text.
        For line-based preprocessors, override this method to split and process lines.
        """
        if self.fix_unicode:
            text = self._fix_unicode(text)
        if self.remove_timestamps:
            text = self._remove_timestamps(text)
        if self.remove_stage_directions:
            text = self._remove_stage_directions(text)
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        if self.lowercase:
            text = text.lower()
        for step in self.custom_steps:
            text = step(text)
        text = text.strip()
        return text


# --- Category-specific preprocessors ---


class TranscriptPreprocessor(BasePreprocessor):
    """Preprocessor for transcript-style content (e.g., CHILDES)."""

    def __init__(
        self,
        remove_speaker_labels: bool = True,
        remove_annotations: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.remove_speaker_labels = remove_speaker_labels
        self.remove_annotations = remove_annotations
        self.speaker_pattern = re.compile(r"^[A-Z]{3}:\s*")  # "MOT: Hello"
        self.annotation_pattern = re.compile(r"%[a-z]+:.*$", re.MULTILINE)

    def preprocess_line(self, line: str) -> str:
        if self.remove_annotations and line.strip().startswith("%"):
            return ""
        if self.remove_speaker_labels:
            line = self.speaker_pattern.sub("", line)
        return super().preprocess_text(line)

    def preprocess_text(self, text: str) -> str:
        lines = text.splitlines()
        processed_lines = [self.preprocess_line(line) for line in lines]
        # Remove empty lines
        processed_lines = [l for l in processed_lines if l.strip()]
        return "\n".join(processed_lines)


class SubtitlePreprocessor(BasePreprocessor):
    """Preprocessor for subtitle-like content."""

    def __init__(self, **kwargs):
        kwargs.setdefault("remove_stage_directions", True)
        kwargs.setdefault("remove_timestamps", True)
        kwargs.setdefault("lowercase", False)
        super().__init__(**kwargs)
        self.speaker_pattern = re.compile(r"^[A-Z\s]+:\s*")
        self.music_note_pattern = re.compile(r"[♪♫]+")

    def preprocess_line(self, line: str) -> str:
        line = self.speaker_pattern.sub("", line)
        line = self.music_note_pattern.sub("", line)
        return super().preprocess_text(line)

    def preprocess_text(self, text: str) -> str:
        lines = text.splitlines()
        processed_lines = [self.preprocess_line(line) for line in lines]
        processed_lines = [l for l in processed_lines if l.strip()]
        return "\n".join(processed_lines)


class BookPreprocessor(BasePreprocessor):
    def preprocess_text(self, text: str) -> str:
        text = remove_xml_tags(text)
        text = normalize_punctuation(text)
        text = remove_urls(text)
        return super().preprocess_text(text)


class WikiPreprocessor(BasePreprocessor):
    def preprocess_text(self, text: str) -> str:
        text = remove_xml_tags(text)
        text = normalize_punctuation(text)
        text = remove_urls(text)
        return super().preprocess_text(text)


class NewsPreprocessor(BasePreprocessor):
    def preprocess_text(self, text: str) -> str:
        text = remove_xml_tags(text)
        text = normalize_punctuation(text)
        text = remove_urls(text)
        return super().preprocess_text(text)


class QEDPreprocessor(BasePreprocessor):
    def preprocess_text(self, text: str) -> str:
        text = remove_xml_tags(text)
        text = normalize_punctuation(text)
        text = remove_urls(text)
        return super().preprocess_text(text)


class SimplifiedTextPreprocessor(BasePreprocessor):
    def preprocess_text(self, text: str) -> str:
        text = remove_xml_tags(text)
        text = normalize_punctuation(text)
        text = remove_urls(text)
        return super().preprocess_text(text)


class EducationalPreprocessor(BasePreprocessor):
    def preprocess_text(self, text: str) -> str:
        text = remove_xml_tags(text)
        text = normalize_punctuation(text)
        text = remove_urls(text)
        return super().preprocess_text(text)


# Category-to-preprocessor mapping
CATEGORY_PREPROCESSOR_MAP = {
    "child-directed-speech": TranscriptPreprocessor,
    "child-available-speech": TranscriptPreprocessor,
    "subtitles": SubtitlePreprocessor,
    "child-books": BookPreprocessor,
    "child-wiki": WikiPreprocessor,
    "child-news": NewsPreprocessor,
    "qed": QEDPreprocessor,
    "simplified-text": SimplifiedTextPreprocessor,
    "educational": EducationalPreprocessor,
}


def get_preprocessor_for_category(category: str, **kwargs) -> BasePreprocessor:
    if category not in CATEGORY_PREPROCESSOR_MAP:
        raise ValueError(f"Unknown category: {category}")
    return CATEGORY_PREPROCESSOR_MAP[category](**kwargs)


# Example custom preprocessing functions that can be added
def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def normalize_punctuation(text: str) -> str:
    """Normalize various punctuation marks, quotes, ellipsis, underscores, and dashes."""
    import re

    # Normalize all double quotes to "
    text = re.sub(r"[“”‟❝❞〝〞＂«»]", '"', text)
    # Normalize all single quotes to '
    text = re.sub(r"[‘’‛`´′‵']", "'", text)

    # Normalize ellipsis ...
    # It looks for sequences of dots and optional spaces, then replaces with "..."
    # This pattern should be applied before normalizing other types of multiple dots to avoid
    # interference.
    text = re.sub(r"(\.[\s\.]*){3,}", "...", text)
    text = re.sub(
        r"[…‥]", "...", text
    )  # Ensure these specific ellipsis chars are handled

    # Normalize underscores _
    text = re.sub(r"[_]+", "_", text)

    # Normalize dashes -
    text = re.sub(r"[–—-]", "-", text)

    return text


def remove_xml_tags(text: str) -> str:
    """Remove any remaining XML/HTML tags."""
    tag_pattern = re.compile(r"<[^>]+>")
    return tag_pattern.sub("", text)
