# language_filter.py
"""
Language and script filtering using GlotLID v3.
Simple segmentation and majority vote for document-level language/script prediction.
"""

import os
from collections import defaultdict
from typing import Dict, Optional, Any
import fasttext
import pandas as pd
from huggingface_hub import hf_hub_download


class LanguageFilter:
    """Language and script filter using GlotLID v3."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the language filter.

        Args:
            model_path: Path to local model.bin file, if None will download from HF

        """
        self.model = self._load_model(model_path)

    def _load_model(
        self, model_path: Optional[str] = None
    ) -> fasttext.FastText._FastText:
        """Load GlotLID model."""
        if model_path and os.path.exists(model_path):
            glotlid_model_path = model_path
        else:
            # Check if model.bin exists locally before downloading
            model_filename = "model.bin"
            local_path = os.path.join(os.path.dirname(__file__), model_filename)
            if os.path.exists(local_path):
                glotlid_model_path = local_path
            else:
                glotlid_model_path = hf_hub_download(
                    repo_id="cis-lmu/glotlid", filename=model_filename, cache_dir=None
                )

        return fasttext.load_model(glotlid_model_path)

    def predict_language_script(
        self, text: str, top_k: int = 3
    ) -> list[tuple[str, str, float]]:
        """
        Predict language and script for text.

        Args:
            text: Input text
            top_k: Number of top predictions to return

        Returns:
            List of (language_code, script, probability) tuples

        """
        # GlotLID v3 cannot process text with newlines; replace with spaces
        if "\n" in text:
            text = text.replace("\n", " ")

        labels, probs = self.model.predict(text, k=top_k)

        results = []
        for label, prob in zip(labels, probs):
            # Remove '__label__' prefix
            label_clean = label.replace("__label__", "")

            # Extract language and script
            if "_" in label_clean:
                lang_code, script = label_clean.split("_", 1)
            else:
                lang_code = label_clean
                script = "unknown"

            results.append((lang_code, script, float(prob)))

        return results

    def segment_text(
        self,
        text: str,
        min_words: int = 10,
        min_chars: int = 50,
        max_words: int = 200,
        max_chars: int = 1000,
    ) -> list[str]:
        """
        Simple segmentation by newlines, merging short segments, and splitting long ones.

        Args:
            text: Input text
            min_words: Minimum words per segment
            min_chars: Minimum characters per segment
            max_words: Maximum words per segment
            max_chars: Maximum characters per segment

        Returns:
            List of text segments

        """
        # Split by newlines
        segments = [s.strip() for s in text.split("\n") if s.strip()]

        # Merge consecutive short segments
        merged_segments = []
        buffer = ""
        for segment in segments:
            candidate = (buffer + " " + segment).strip() if buffer else segment
            words = candidate.split()
            if len(words) < min_words or len(candidate) < min_chars:
                buffer = candidate
                continue
            if buffer:
                merged_segments.append(buffer)
                buffer = ""
            # Now check if candidate is too long
            while len(words) > max_words or len(candidate) > max_chars:
                # Try to split at sentence boundary if possible
                split_idx = None
                for sep in [". ", "! ", "? "]:
                    idx = candidate.find(sep, max_chars // 2)
                    if idx != -1:
                        split_idx = idx + len(sep)
                        break
                if split_idx:
                    merged_segments.append(candidate[:split_idx].strip())
                    candidate = candidate[split_idx:].strip()
                    words = candidate.split()
                else:
                    # Fallback: hard split
                    merged_segments.append(" ".join(words[:max_words]))
                    candidate = " ".join(words[max_words:]).strip()
                    words = candidate.split()
            if candidate:
                merged_segments.append(candidate)
        if buffer:
            merged_segments.append(buffer)
        # Filter out any remaining too-short segments
        final_segments = [
            s
            for s in merged_segments
            if len(s.split()) >= min_words and len(s) >= min_chars
        ]
        return final_segments

    def merge_short_segments(
        self,
        segments: list[str],
        min_words: int = 10,
        min_chars: int = 50,
        max_words: int = 200,
        max_chars: int = 1000,
    ) -> list[str]:
        """Merge consecutive short segments to meet minimum requirements, and split long ones."""
        if not segments:
            return []
        merged = []
        buffer = ""
        for segment in segments:
            candidate = (buffer + " " + segment).strip() if buffer else segment
            words = candidate.split()
            if len(words) < min_words or len(candidate) < min_chars:
                buffer = candidate
                continue
            if buffer:
                merged.append(buffer)
                buffer = ""
            # Split long segments
            while len(words) > max_words or len(candidate) > max_chars:
                split_idx = None
                for sep in [". ", "! ", "? "]:
                    idx = candidate.find(sep, max_chars // 2)
                    if idx != -1:
                        split_idx = idx + len(sep)
                        break
                if split_idx:
                    merged.append(candidate[:split_idx].strip())
                    candidate = candidate[split_idx:].strip()
                    words = candidate.split()
                else:
                    merged.append(" ".join(words[:max_words]))
                    candidate = " ".join(words[max_words:]).strip()
                    words = candidate.split()
            if candidate:
                merged.append(candidate)
        if buffer:
            merged.append(buffer)
        final_segments = [
            s for s in merged if len(s.split()) >= min_words and len(s) >= min_chars
        ]
        return final_segments

    def predict_document_language_script(
        self,
        text: str,
        min_words: int = 10,
        min_chars: int = 50,
        min_confidence: float = 0.3,
    ) -> tuple[str, str, float, dict]:
        """
        Predict language and script for entire document using majority vote by word count.

        Args:
            text: Document text
            min_words: Minimum words per segment
            min_chars: Minimum characters per segment
            min_confidence: Minimum confidence threshold for predictions

        Returns:
            Tuple of (language_code, script, confidence, metadata)
            metadata contains segment details and voting information

        """
        # Segment text
        segments = self.segment_text(text, min_words, min_chars)
        segments = self.merge_short_segments(segments, min_words, min_chars)

        if not segments:
            return (
                "unknown",
                "unknown",
                0.0,
                {"num_segments": 0, "valid_segments": 0, "predictions": []},
            )
        # Predict for each segment
        lang_word_votes = defaultdict(int)
        script_word_votes = defaultdict(int)
        all_predictions = []
        valid_predictions = 0
        total_words = 0

        for segment in segments:
            predictions = self.predict_language_script(segment, top_k=1)
            words_in_segment = len(segment.split())
            if predictions:
                lang_code, script, confidence = predictions[0]
                if confidence >= min_confidence:
                    lang_word_votes[lang_code] += words_in_segment
                    script_word_votes[script] += words_in_segment
                    valid_predictions += 1
                    total_words += words_in_segment
                all_predictions.append(
                    {
                        "segment": (
                            segment[:100] + "..." if len(segment) > 100 else segment
                        ),
                        "predictions": [(lang_code, script, confidence)],
                        "words": words_in_segment,
                    }
                )
        # Get majority by word count
        if lang_word_votes:
            best_lang = max(lang_word_votes.items(), key=lambda x: x[1])[0]
            lang_word_count = lang_word_votes[best_lang]
            lang_confidence = lang_word_count / total_words if total_words > 0 else 0.0
        else:
            best_lang = "unknown"
            lang_confidence = 0.0

        if script_word_votes:
            best_script = max(script_word_votes.items(), key=lambda x: x[1])[0]
            script_word_count = script_word_votes[best_script]
            script_confidence = (
                script_word_count / total_words if total_words > 0 else 0.0
            )
        else:
            best_script = "unknown"
            script_confidence = 0.0

        # Overall confidence is average of language and script confidence
        overall_confidence = (lang_confidence + script_confidence) / 2.0

        metadata = {
            "num_segments": len(segments),
            "valid_segments": valid_predictions,
            "predictions": all_predictions,
            "language_word_votes": dict(lang_word_votes),
            "script_word_votes": dict(script_word_votes),
            "language_confidence": lang_confidence,
            "script_confidence": script_confidence,
            "total_words": total_words,
        }

        return best_lang, best_script, overall_confidence, metadata

    def filter_document(
        self,
        text: str,
        expected_language: str,
        expected_script: str,
        min_confidence: float = 0.5,
        min_words: int = 10,
        min_chars: int = 50,
    ) -> Dict[str, Any]:
        """Filter a single document by language and script.

        Args:
            text: Document text
            expected_language: Expected language code (ISO 639-3)
            expected_script: Expected script (e.g., 'Latn', 'Arab', etc.)
            min_confidence: Minimum confidence for predictions
            min_words: Minimum words per segment
            min_chars: Minimum characters per segment

        Returns:
            Dictionary with filtering result and prediction details
        """
        if not text.strip():
            return {
                "match": False,
                "reason": "empty",
                "predicted_language": None,
                "predicted_script": None,
                "confidence": 0.0,
                "metadata": {},
            }
        pred_lang, pred_script, confidence, metadata = (
            self.predict_document_language_script(
                text, min_words, min_chars, min_confidence
            )
        )
        lang_match = pred_lang.lower() == expected_language.lower()
        script_match = pred_script.lower() == expected_script.lower()
        match = lang_match and script_match and confidence >= min_confidence
        return {
            "match": match,
            "predicted_language": pred_lang,
            "predicted_script": pred_script,
            "confidence": confidence,
            "metadata": metadata,
            "language_match": lang_match,
            "script_match": script_match,
            "reason": None if match else "mismatch",
        }

    def filter_documents(
        self,
        documents_df: pd.DataFrame,
        expected_language: str,
        expected_script: str,
        min_confidence: float = 0.5,
        min_words: int = 10,
        min_chars: int = 50,
    ) -> dict[str, Any]:
        """
        Filter documents by language and script.

        Args:
            documents_df: DataFrame containing document metadata
            expected_language: Expected language code (ISO 639-3)
            expected_script: Expected script (e.g., 'Latn', 'Arab', etc.)
            min_confidence: Minimum confidence for predictions
            min_words: Minimum words per segment
            min_chars: Minimum characters per segment

        Returns:
            Dictionary with statistics about filtered documents

        """
        results = {
            "matching": [],
            "mismatched": [],
            "errors": [],
            "statistics": defaultdict(int),
        }

        match_ids = []
        mismatch_ids = []
        for _, row in documents_df.iterrows():
            try:
                # Predict language and script
                pred_lang, pred_script, confidence, metadata = (
                    self.predict_document_language_script(
                        row["text"], min_words, min_chars, min_confidence
                    )
                )
                # Check if it matches expected language and script
                lang_match = pred_lang.lower() == expected_language.lower()
                script_match = pred_script.lower() == expected_script.lower()
                document_id = row["doc_id"]

                file_info = {
                    "filename": document_id + ".txt",
                    "predicted_language": pred_lang,
                    "predicted_script": pred_script,
                    "confidence": confidence,
                    "metadata": metadata,
                    "language_match": lang_match,
                    "script_match": script_match,
                }
                match = lang_match and script_match and confidence >= min_confidence
                # Decide where to place the file
                if match:
                    match_ids.append(document_id)
                    results["matching"].append(file_info)
                    results["statistics"]["matching"] += 1

                else:
                    mismatch_ids.append(document_id)
                    results["mismatched"].append(file_info)
                    results["statistics"]["mismatched"] += 1
                    results["statistics"][f"mismatched_{pred_lang}_{pred_script}"] += 1

                results["statistics"]["total_processed"] += 1
            except Exception as e:
                error_msg = f"{file_info['filename']}: {e!s}"
                results["errors"].append(error_msg)
                results["statistics"]["errors"] += 1

        results["match_ids"] = match_ids
        results["mismatch_ids"] = mismatch_ids

        return results


def filter_dataset_for_lang_and_script(
    dataset_table: pd.DataFrame,
    language_code: str,
    script_code: str,
    language_filter_threshold: float,
):
    lang_filter = LanguageFilter()

    filter_results = lang_filter.filter_documents(
        dataset_table,
        expected_language=language_code,
        expected_script=script_code,
        min_confidence=language_filter_threshold,
    )

    print_filtering_results(filter_results, language_code, script_code)
    # Only keep matching documents

    matching_ids = set(filter_results["match_ids"])
    dataset_table = dataset_table[dataset_table["doc_id"].isin(matching_ids)]

    return dataset_table


def print_filtering_results(
    results: dict, expected_language: str, expected_script: str
):
    """Print filtering results summary."""
    stats = results["statistics"]

    print(f"\n{'=' * 60}")
    print("LANGUAGE FILTERING RESULTS")
    print(f"{'=' * 60}")
    print(f"Expected Language: {expected_language}")
    print(f"Expected Script: {expected_script}")
    print()
    print(f"Total files processed: {stats['total_processed']}")
    print(f"Matching files: {stats['matching']}")
    print(f"Mismatched files: {stats['mismatched']}")
    print(f"Errors: {stats['errors']}")

    if stats["total_processed"] > 0:
        match_rate = (stats["matching"] / stats["total_processed"]) * 100
        print(f"Match rate: {match_rate:.1f}%")

    # Show breakdown of mismatched languages
    print("\nMismatched files breakdown:")
    for key, count in stats.items():
        if key.startswith("mismatched_") and key != "mismatched":
            lang_script = key.replace("mismatched_", "")
            print(f"  {lang_script}: {count} files")

    # Show some examples of mismatched files
    if results["mismatched"]:
        print("\nExamples of mismatched files:")
        for i, file_info in enumerate(results["mismatched"][:5]):
            print(
                f"  {file_info['filename']}: {file_info['predicted_language']}_{file_info['predicted_script']} "
                f"(confidence: {file_info['confidence']:.3f})"
            )

    if results["errors"]:
        print("\nErrors:")
        for error in results["errors"][:5]:
            print(f"  {error}")
        if len(results["errors"]) > 5:
            print(f"  ... and {len(results['errors']) - 5} more errors")

    print(f"{'=' * 60}")
