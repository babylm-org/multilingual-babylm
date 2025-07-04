# text_preprocessor.py
"""
General text preprocessing utilities for BabyLM datasets.
"""

import pandas as pd

from abc import ABC
from preprocessor_utils import (
    fix_unicode,
    normalize_whitespace,
    normalize_punctuation,
    remove_urls,
    remove_xml_tags,
    remove_extra_spaces,
    remove_stage_directions,
    remove_timestamps,
    remove_annotations,
    remove_speaker_labels,
    remove_music_note_symbols,
)


class BasePreprocessor(ABC):
    """
    Base class for text preprocessors.
    Subclass this for source-specific preprocessing.
    """

    def __init__(self):
        super().__init__()

    def process_text(self, text: str, preserve_tab: bool = False) -> str:
        """
        General preprocessing: fix unicode and normalize whitespace.
        """
        text = fix_unicode(text)
        text = normalize_whitespace(text, preserve_tab=preserve_tab)
        return text


class TranscriptPreprocessor(BasePreprocessor):
    """Preprocessor for transcript-style content (e.g., CHILDES)."""

    def __init__(
        self,
    ):
        super().__init__()

    def preprocess_line(self, line: str) -> str:
        # super().process_text will already apply fix_unicode and normalize_whitespace
        line = remove_annotations(line)
        if line == "":
            return ""
        line = normalize_punctuation(line)
        line = remove_extra_spaces(line, preserve_tab=True)
        return line

    def process_text(self, text: str) -> str:
        # transcript expected format is: *SPEAKER_LABEL + \t + utterance + \t + [PLACEHOLDER] (optionally)
        # for now we just preserve the tab characters
        text = super().process_text(text, preserve_tab=True)
        lines = text.splitlines()
        processed_lines = [self.preprocess_line(line) for line in lines]
        processed_lines = [l for l in processed_lines if l.strip()]
        return "\n".join(processed_lines)


class SubtitlePreprocessor(BasePreprocessor):
    """Preprocessor for subtitle-like content."""

    def __init__(self):
        super().__init__()

    def preprocess_line(self, line: str) -> str:
        # super().process_text will already apply fix_unicode and normalize_whitespace
        line = remove_speaker_labels(line)
        line = remove_music_note_symbols(line)
        line = remove_stage_directions(line)
        line = remove_timestamps(line)
        line = normalize_punctuation(line)
        line = remove_extra_spaces(line)
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
        text = remove_extra_spaces(text)
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
        text = remove_extra_spaces(text)
        return text


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
