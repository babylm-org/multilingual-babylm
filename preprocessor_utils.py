# text_preprocessor.py
"""
General text preprocessing utilities for BabyLM datasets.
"""

import re
import ftfy


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
    """Remove transcript annotation lines starting with % and followed by a speaker label (any case, possibly with spaces) and colon (e.g., '%act:', '%MOTHER:', '%John Smith:')."""
    # Remove the line if it starts with % and then a word/words (any case, possibly with spaces) and colon
    return "" if re.match(r"^\s*%[A-Za-z][A-Za-z\s]{0,40}:", text) else text


def remove_speaker_labels(text: str) -> str:
    """Remove speaker labels from the beginning of lines (e.g., 'John:', 'MOT:', 'MOTHER:', 'John Smith:').
    Matches any word or words (with spaces), followed by a colon and optional whitespace.
    """
    return re.sub(r"^[^:]{1,40}:\s*", "", text)


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


def remove_extra_spaces(text: str, preserve_tab: bool = False) -> str:
    """Remove extra spaces from text, collapsing multiple spaces into a single space."""
    # Collapse multiple spaces to a single space
    space_re = r"[ ]+" if preserve_tab else r"[ \t]+"
    text = re.sub(space_re, " ", text).strip()
    return text

def normalize_whitespace(
    text: str,
    preserve_paragraphs: bool = True,
    remove_newlines: bool = False,
    preserve_tab: bool = False
) -> str:

    space_re = r"[ ]+" if preserve_tab else r"[ \t]+"

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
            para = re.sub(space_re, " ", para)

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
        # Collapse newlines to spaces
        text = re.sub(r"\n", " ", text)

    # Collapse multiple spaces to single space
    text = re.sub(space_re, " ", text)

    return text.strip()

