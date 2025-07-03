# text_preprocessor.py
"""
General text preprocessing utilities for BabyLM datasets.
Can be extended for source-specific preprocessing needs.
"""

import re
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path
import ftfy
from abc import ABC, abstractmethod
import csv
import json
from datasets import load_from_disk
from language_scripts import SCRIPT_NAMES


class BasePreprocessor(ABC):
    """
    Base class for text preprocessors.
    Subclass this for source-specific preprocessing.
    """

    def __init__(
        self,
        lowercase: bool = False,  # Changed default to False
        normalize_whitespace: bool = True,
        fix_unicode: bool = True,
        remove_timestamps: bool = True,
        remove_stage_directions: bool = True,
        preserve_paragraphs: bool = True,  # New option
        replace_newline_within_paragraph: bool = True,  # NEW OPTION
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

    @abstractmethod
    def read_source(self, source_path: Path) -> str:
        """
        Read and extract text from source file.
        Must be implemented by subclasses.
        """
        pass

    def preprocess_text(self, text: str) -> str:
        """
        Apply all preprocessing steps to text.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        # Apply preprocessing steps in order
        if self.fix_unicode:
            text = self._fix_unicode(text)

        if self.remove_timestamps:
            text = self._remove_timestamps(text)

        if self.remove_stage_directions:
            text = self._remove_stage_directions(text)

        # Apply any custom preprocessing steps
        for step in self.custom_steps:
            text = step(text)

        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        if self.lowercase:
            text = text.lower()

        # Final cleanup
        text = text.strip()

        return text

    def preprocess_lines(self, lines: List[str]) -> List[str]:
        """
        Preprocess a list of lines (e.g., sentences).

        Args:
            lines: List of text lines

        Returns:
            List of preprocessed lines
        """
        processed_lines = []

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Check for subtitle artifacts to skip
            if any(pattern.match(line) for pattern in self.subtitle_artifacts):
                continue

            # Preprocess the line
            processed = self.preprocess_text(line)

            # Only keep non-empty results
            if processed:
                processed_lines.append(processed)

        return processed_lines

    def _fix_unicode(self, text: str) -> str:
        """Fix unicode issues using ftfy."""
        return ftfy.fix_text(text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text while preserving paragraph structure."""
        if self.preserve_paragraphs:
            # Split by double newlines (paragraphs)
            paragraphs = text.split("\n\n")

            # Process each paragraph
            processed_paragraphs = []
            for para in paragraphs:
                # Replace multiple spaces with single space within paragraph
                para = re.sub(r"[ \t]+", " ", para)
                if self.replace_newline_within_paragraph:
                    # Replace single newlines with space within paragraph
                    para = re.sub(r"\n", " ", para)
                # Strip leading/trailing whitespace
                para = para.strip()
                if para:  # Only keep non-empty paragraphs
                    processed_paragraphs.append(para)

            # Join paragraphs back with double newlines
            return "\n\n".join(processed_paragraphs)
        else:
            # Original behavior - collapse all whitespace
            text = self.whitespace_pattern.sub(" ", text)
            return text.strip()

    def _remove_timestamps(self, text: str) -> str:
        """Remove timestamp patterns from text."""
        for pattern in self.timestamp_patterns:
            text = pattern.sub("", text)
        return text

    def _remove_stage_directions(self, text: str) -> str:
        """Remove [bracketed] stage directions."""
        # Keep track of what we're removing for logging
        removed = self.stage_direction_pattern.findall(text)
        if removed and hasattr(self, "log_removed"):
            self.log_removed("stage_directions", removed)

        return self.stage_direction_pattern.sub("", text)

    def process_file(
        self,
        source_path: Path,
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single file from source to output.

        Args:
            source_path: Path to source file
            output_path: Path to write processed text
            metadata: Optional metadata to include in result

        Returns:
            Dictionary with processing results and statistics
        """
        # Read source
        text = self.read_source(source_path)

        # Get original stats
        original_length = len(text)
        original_lines = text.count("\n") + 1

        # Preprocess
        if isinstance(text, list):
            # Handle line-based input
            processed_lines = self.preprocess_lines(text)
            processed_text = "\n".join(processed_lines)
            processed_line_count = len(processed_lines)
        else:
            # Handle full text input
            processed_text = self.preprocess_text(text)
            processed_line_count = processed_text.count("\n") + 1

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(processed_text)

        # Validate script field in metadata if present
        if metadata and "script" in metadata and metadata["script"] not in SCRIPT_NAMES:
            raise ValueError(
                f"Invalid script code '{metadata['script']}' in metadata for file {source_path}. Must be ISO 15924 code."
            )

        # Return statistics
        result = {
            "source_file": str(source_path),
            "output_file": str(output_path),
            "original_length": original_length,
            "processed_length": len(processed_text),
            "original_lines": original_lines,
            "processed_lines": processed_line_count,
            "compression_ratio": (
                len(processed_text) / original_length if original_length > 0 else 0
            ),
        }

        if metadata:
            result.update(metadata)

        return result


class TextFilePreprocessor(BasePreprocessor):
    """Simple preprocessor for plain text files."""

    def read_source(self, source_path: Path) -> str:
        """Read plain text file."""
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()


class SubtitlePreprocessor(BasePreprocessor):
    """Preprocessor specifically for subtitle-like content."""

    def __init__(self, **kwargs):
        # Set subtitle-specific defaults
        kwargs.setdefault("remove_stage_directions", True)
        kwargs.setdefault("remove_timestamps", True)
        kwargs.setdefault("lowercase", False)  # Preserve capitalization by default
        super().__init__(**kwargs)

        # Add subtitle-specific patterns
        self.speaker_pattern = re.compile(r"^[A-Z\s]+:\s*")  # "JOHN: Hello"
        self.music_note_pattern = re.compile(r"[♪♫]+")

    def read_source(self, source_path: Path) -> str:
        """Read subtitle file."""
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def preprocess_text(self, text: str) -> str:
        """Apply subtitle-specific preprocessing."""
        # Remove speaker labels if present
        text = self.speaker_pattern.sub("", text)

        # Remove music note symbols
        text = self.music_note_pattern.sub("", text)

        # Continue with base preprocessing
        return super().preprocess_text(text)


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

        # Common transcript patterns
        self.speaker_pattern = re.compile(r"^[A-Z]{3}:\s*")  # "MOT: Hello"
        self.annotation_pattern = re.compile(
            r"%[a-z]+:.*$", re.MULTILINE
        )  # %act: points

    def read_source(self, source_path: Path) -> List[str]:
        """Read transcript file as lines."""
        with open(source_path, "r", encoding="utf-8") as f:
            return f.readlines()

    def preprocess_lines(self, lines: List[str]) -> List[str]:
        """Process transcript lines."""
        processed_lines = []

        for line in lines:
            # Skip annotation lines
            if self.remove_annotations and line.strip().startswith("%"):
                continue

            # Remove speaker labels
            if self.remove_speaker_labels:
                line = self.speaker_pattern.sub("", line)

            # Continue with base preprocessing
            processed = self.preprocess_text(line)
            if processed:
                processed_lines.append(processed)

        return processed_lines


class CSVPreprocessor(TextFilePreprocessor):
    """Preprocessor for CSV files with text and metadata fields."""

    def __init__(self, text_field="text", **kwargs):
        super().__init__(**kwargs)
        self.text_field = text_field

    def process_csv(self, csv_path: Path, output_dir: Path) -> Dict[str, Any]:
        metadata_mapping = {}
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                text = row.get(self.text_field, "")
                if not text:
                    continue
                processed_text = self.preprocess_text(text)
                doc_id = str(i)
                out_path = output_dir / f"{doc_id}.txt"
                meta = {k: v for k, v in row.items() if k != self.text_field}
                # Validate script field if present
                if "script" in meta and meta["script"] not in SCRIPT_NAMES:
                    raise ValueError(
                        f"Invalid script code '{meta['script']}' in CSV row {i}. Must be ISO 15924 code."
                    )
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(processed_text)
                metadata_mapping[doc_id] = meta
        return metadata_mapping


class HFDatasetPreprocessor(TextFilePreprocessor):
    """Preprocessor for HuggingFace datasets with text and metadata fields."""

    def __init__(self, dataset_id, split=None, text_field="text", **kwargs):
        super().__init__(**kwargs)
        self.dataset_id = dataset_id
        self.split = split
        self.text_field = text_field

    def process_hf_dataset(self, output_dir: Path) -> Dict[str, Any]:
        metadata_mapping = {}
        ds = load_from_disk(self.dataset_id)
        # If no split, ds is a dict of splits, use the first split
        if self.split and hasattr(ds, "split"):
            ds = ds[self.split]
        elif isinstance(ds, dict):
            ds = next(iter(ds.values()))
        for i, row in enumerate(ds):
            if isinstance(row, dict):
                text = row.get(self.text_field, "")
                meta = {k: v for k, v in row.items() if k != self.text_field}
            else:
                text = getattr(row, self.text_field, "")
                meta = {k: v for k, v in row.__dict__.items() if k != self.text_field}
            if not text:
                continue
            processed_text = self.preprocess_text(text)
            doc_id = str(i)
            out_path = output_dir / f"{doc_id}.txt"
            # Validate script field if present
            if "script" in meta and meta["script"] not in SCRIPT_NAMES:
                raise ValueError(
                    f"Invalid script code '{meta['script']}' in HF dataset row {i}. Must be ISO 15924 code."
                )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(processed_text)
            metadata_mapping[doc_id] = meta
        return metadata_mapping


class JSONPreprocessor(TextFilePreprocessor):
    """Preprocessor for JSON files with text and metadata fields."""

    def __init__(self, text_field="text", **kwargs):
        super().__init__(**kwargs)
        self.text_field = text_field

    def process_json(self, json_path: Path, output_dir: Path) -> Dict[str, Any]:

        metadata_mapping = {}
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Support both list of dicts and dict of dicts
        if isinstance(data, dict):
            items = list(data.values())
        else:
            items = data
        for i, row in enumerate(items):
            if isinstance(row, dict):
                text = row.get(self.text_field, "")
                meta = {k: v for k, v in row.items() if k != self.text_field}
            else:
                continue
            if not text:
                continue
            processed_text = self.preprocess_text(text)
            doc_id = str(i)
            out_path = output_dir / f"{doc_id}.txt"
            # Validate script field if present
            if "script" in meta and meta["script"] not in SCRIPT_NAMES:
                raise ValueError(
                    f"Invalid script code '{meta['script']}' in JSON row {i}. Must be ISO 15924 code."
                )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(processed_text)
            metadata_mapping[doc_id] = meta
        return metadata_mapping


# Update factory
def create_preprocessor(source_type: str, **kwargs) -> BasePreprocessor:
    preprocessors = {
        "text": TextFilePreprocessor,
        "subtitle": SubtitlePreprocessor,
        "transcript": TranscriptPreprocessor,
        "csv": CSVPreprocessor,
        "hf": HFDatasetPreprocessor,
        "json": JSONPreprocessor,
    }
    if source_type not in preprocessors:
        raise ValueError(
            f"Unknown source type: {source_type}. "
            f"Available: {list(preprocessors.keys())}"
        )
    return preprocessors[source_type](**kwargs)


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
