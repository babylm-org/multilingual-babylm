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
            custom_steps: List of additional preprocessing functions
        """
        self.lowercase = lowercase
        self.normalize_whitespace = normalize_whitespace
        self.fix_unicode = fix_unicode
        self.remove_timestamps = remove_timestamps
        self.remove_stage_directions = remove_stage_directions
        self.preserve_paragraphs = preserve_paragraphs
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


class LLMPreprocessor(BasePreprocessor):
    """
    Preprocessor that uses LLMs via Ollama for filtering and processing.
    Can be used for quality filtering, content appropriateness, etc.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        prompt: Optional[str] = None,
        filter_threshold: float = 0.7,
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

        # Default prompt if none provided
        self.prompt = (
            prompt
            or """
        Evaluate if this text is appropriate for children's language learning.
        Consider factors like:
        - Age-appropriate content
        - Educational value
        - Language quality
        - Absence of inappropriate content
        
        Respond with ONLY a JSON object in this format:
        {"score": 0.0-1.0, "reason": "brief explanation", "processed_text": "cleaned version if needed"}
        
        Text to evaluate:
        {text}
        """
        )

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

    def read_source(self, source_path: Path) -> str:
        """Read text file."""
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def _call_llm(self, text: str) -> Dict[str, Any]:
        """Call Ollama API with the text."""
        import requests
        import json

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

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text using LLM evaluation.

        Returns empty string if text doesn't meet threshold.
        """
        # First apply base preprocessing if configured
        text = super().preprocess_text(text)

        if not text:
            return ""

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

    def preprocess_lines(self, lines: List[str]) -> List[str]:
        """Process lines with LLM filtering."""
        # For efficiency, you might want to batch lines together
        # and evaluate chunks rather than individual lines
        processed_lines = []

        # Process in chunks for efficiency
        chunk_size = 10
        for i in range(0, len(lines), chunk_size):
            chunk = lines[i : i + chunk_size]
            chunk_text = "\n".join(chunk)

            processed_chunk = self.preprocess_text(chunk_text)
            if processed_chunk:
                # Split back into lines
                processed_lines.extend(processed_chunk.split("\n"))

        return processed_lines


def create_preprocessor(source_type: str, **kwargs) -> BasePreprocessor:
    """
    Factory function to create appropriate preprocessor.

    Args:
        source_type: Type of source ('text', 'subtitle', 'transcript', 'llm', etc.)
        **kwargs: Configuration parameters for the preprocessor

    Returns:
        Configured preprocessor instance
    """
    preprocessors = {
        "text": TextFilePreprocessor,
        "subtitle": SubtitlePreprocessor,
        "transcript": TranscriptPreprocessor,
        "llm": LLMPreprocessor,
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
    text = re.sub(r"[“”‟❝❞〝〞＂]", '"', text)
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
