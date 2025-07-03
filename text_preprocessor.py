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

    @abstractmethod
    def process_text(self, text: str) -> str:
        """
        Process raw text input.
        Should be implemented by subclasses.
        """

    def base_processing(self, text: str) -> str:
        """
        Base processing steps common to all preprocessors.
        """
        text = fix_unicode(text)
        text = normalize_punctuation(text)
        text = normalize_whitespace(text)
        return text


def preprocess_dataset(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset based on document category.
    """
    processed_groups = []
    for category, group in dataset_df.groupby("category"):
        group = group.copy()
        processor = create_preprocessor(category)
        group["text"] = group["text"].apply(processor.process_text)
        processed_groups.append(group)

    return pd.concat(processed_groups)


def apply_llm_preprocessing(
    dataset_df: pd.DataFrame,
    llm_preprocessing_args,
) -> pd.DataFrame:
    """
    Apply LLM-based preprocessing to the dataset.
    Uses Ollama for filtering and processing text.
    """
    llm_processor = LLMPreprocessor(**llm_preprocessing_args)
    # Apply LLM preprocessing to each text entry
    dataset_df["text"] = dataset_df["text"].apply(llm_processor.process_text)
    return dataset_df


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
        "child-directed-speech": ChildDirectedPreprocessor,
        "child-available-speech": ChildAvailablePreprocessor,
        # special
        "qed": QEDPreprocessor,
        # padding-wikipedia padding-xxx?
    }
    return preprocessors[category]()


class BookPreprocessor(BasePreprocessor):
    """
    Preprocessor for book-like formats (e.g. textbooks, child books, child wikis, etc.).
    """

    def process_text(self, text: str) -> str:
        text = super().base_processing(text)
        return text
        return text


class SubtitlePreprocessor(BasePreprocessor):
    """
    Preprocessor for subtitle-like formats.
    """

    def process_text(self, text: str) -> str:
        text = super().base_processing(text)
        return text


class ChildAvailablePreprocessor(BasePreprocessor):
    """
    Preprocessor for child-available speech.
    """

    def process_text(self, text: str) -> str:
        text = super().base_processing(text)
        return text


class ChildDirectedPreprocessor(BasePreprocessor):
    """
    Preprocessor for child-directed and child-produced speech.
    """

    def process_text(self, text: str) -> str:
        text = super().base_processing(text)
        return text


class QEDPreprocessor(BasePreprocessor):
    """
    Preprocessor for QED dataset formats.
    """

    def process_text(self, text: str) -> str:
        text = super().base_processing(text)
        return text


class LLMPreprocessor(BasePreprocessor):
    """
    Preprocessor that uses LLMs via Ollama for filtering and processing.
    Can be used for quality filtering, content appropriateness, etc.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        prompt_path: Path = "llm_prompt.txt",
        filter_threshold: float = 0.7,
        process_lines: bool = False,
        ollama_base_url: str = "http://localhost:11434",
        **kwargs,
    ):
        """
        Initialize LLM preprocessor.

        Args:
            model: Ollama model name to use
            prompt: Custom prompt for filtering/processing
            filter_threshold: Score threshold for filtering (0-1)
            ollama_base_url: Ollama API base URL
            **kwargs: Additional arguments for BasePreprocessor

        """
        super().__init__(**kwargs)
        self.model = model
        self.filter_threshold = filter_threshold
        self.ollama_base_url = ollama_base_url
        self.prompt_path = prompt_path
        self.process_lines = process_lines
        # Default prompt if none provided
        with open(self.prompt_path, encoding="utf-8") as f:
            self.prompt = f.read().strip()

        # Check if Ollama is available
        self._check_ollama_connection()

    def _check_ollama_connection(self):
        """Check if Ollama is running and accessible."""
        try:
            import requests

            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code != 200:
                print(f"Warning: Ollama may not be running at {self.ollama_base_url}")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            print("Make sure Ollama is running with: ollama serve")

    def _call_llm(self, text: str) -> dict[str, Any]:
        """Call Ollama API with the text."""
        import json

        import requests

        # Format the prompt with the text
        formatted_prompt = self.prompt.format(text=text[:2000])  # Limit text length

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": formatted_prompt,
                    "stream": False,
                    "temperature": 0.1,  # Low temperature for consistent results
                    "format": "json",
                },
            )

            if response.status_code == 200:
                result = response.json()
                # Parse the JSON response from the model
                try:
                    evaluation = json.loads(result["response"])
                    return evaluation
                except json.JSONDecodeError:
                    # Fallback if model doesn't return valid JSON
                    return {"score": 0.5, "reason": "Invalid response format"}
            else:
                return {"score": 0.5, "reason": f"API error: {response.status_code}"}

        except Exception as e:
            print(f"Error calling LLM: {e}")
            return {"score": 0.5, "reason": str(e)}

    def _process_text_with_llm(self, text: str) -> str:
        # Get LLM evaluation
        evaluation = self._call_llm(text)

        # Check if text passes threshold
        score = evaluation.get("score", 0.5)
        if score < self.filter_threshold:
            if hasattr(self, "log_filtered"):
                self.log_filtered(text, evaluation.get("reason", "Below threshold"))
            return ""

        # Return processed text if provided by LLM, otherwise original
        return evaluation.get("processed_text", text)

    def process_text(self, text: str) -> str:
        """
        Preprocess text using LLM evaluation.

        Returns empty string if text doesn't meet threshold.
        """
        # check if  process_lines options is set
        if self.process_lines:
            lines = text.split("\n")
            processed_lines = []

            # Process in chunks for efficiency
            chunk_size = 10
            for i in range(0, len(lines), chunk_size):
                chunk = lines[i : i + chunk_size]
                chunk_text = "\n".join(chunk)

                processed_chunk = self._process_text_with_llm(chunk_text)
                if processed_chunk:
                    # Split back into lines
                    processed_lines.extend(processed_chunk.split("\n"))

            return "\n".join(processed_lines)

        return self._process_text_with_llm(text)


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
    text = re.sub(r"[♪♫]+", "", text)


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
