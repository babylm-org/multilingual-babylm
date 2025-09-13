import pandas as pd
from typing import Any
import hashlib

from loguru import logger

BYTE_PREMIUMS_PATH = "resources/byte_coefs_20240233.tsv"


eng_sizes_per_tier = {
    "tier_1M": 5.430,
    "tier_10M": 54.30,
    "tier_100M": 543.00,
}


def dataframe_to_docs(dataset_df: pd.DataFrame) -> list[dict[str, Any]]:
    docs = []
    for i, row in dataset_df.iterrows():
        text = row.get("text", "")
        if not text:
            continue
        meta = {k: v for k, v in row.items() if k != "text"}
        doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        docs.append(
            {
                "text": text,
                "doc-id": doc_id,
                "metadata": meta,
            }
        )
    return docs


def get_byte_premium_factor(lang: str, script: str):
    all_data_df = pd.read_csv(BYTE_PREMIUMS_PATH, sep="\t", header=0)
    all_data_df = all_data_df[all_data_df["byte_coef"].notnull()]
    script = script.lower()

    exceptions = {
        "ara": "arb",  # map macro-language code "ara" tp MSA code "arb"
        "fas": "pes",  # map macro-languae Persian "fas" to Iranian persian "pes"
    }

    if lang in exceptions:
        lang = exceptions[lang]

    # hardcoded values, not present in the original .tsv
    hardcoded = {
        "mak": 1.250697369  # calculated using byte-premium tool
    }
    if lang in hardcoded:
        return hardcoded[lang]

    lang = lang.lower()
    lang_entries = all_data_df[all_data_df["lang"] == lang]
    num_lang_entries = len(lang_entries)
    if num_lang_entries == 0:
        logger.info(
            "No pre-calculated byte-premium factor, see instructions on README on how to calculate it."
        )
        return None
    elif num_lang_entries == 1:
        return lang_entries["byte_coef"].values[0]
    else:
        lang_script_entries = all_data_df[
            all_data_df["lang_script"] == f"{lang}_{script}"
        ]
        assert len(lang_script_entries) == 1, (
            f"Double entrie for {lang}_{script} value. Should be impossible."
        )
        return lang_script_entries["byte_coef"].values[0]


def bytes_in_text(text: str) -> float:
    return len(text.encode("utf-8")) / 1_000_000  # Convert to MB


def normalize_script(script: str) -> str:
    """
    Normalize script names to a consistent format.
    """
    if script == "Latin":
        return "Latn"
    elif script == "Cyrillic":
        return "Cyrl"
    elif script == "Arabic":
        return "Arab"
    elif script == "Chinese":
        return "Hani"
    # Add more normalizations as needed
    return script


def get_dataset_tier(dataset_size, factor, percent_tolerance):
    """
    Determine the tier based on dataset size.
    Allow for `percent_tolerance` difference from the target tier value
    """

    def is_within_percentage(x, target, percent_tolerance):
        return abs(x - target) <= (percent_tolerance) * abs(target)

    tiers = eng_sizes_per_tier.copy()
    target_sizes = [
        (name.split("_")[-1], size * factor) for name, size in tiers.items()
    ]
    sorted_pairs = sorted(target_sizes, key=lambda x: x[1])

    for name, target in target_sizes:
        if is_within_percentage(dataset_size, target, percent_tolerance):
            return name, target, tiers["tier_" + name]

    # dataset_size is below the tier threshold
    for name, target in target_sizes:
        if dataset_size < target:
            return "< " + name, target, target, tiers["tier_" + name]

    # dataset_size exceeds largest tier threshold
    name = sorted_pairs[-1][0]
    target = sorted_pairs[-1][-1]
    return "> " + name, target, tiers["tier_" + name]


def get_dataset_tier_to_pad(dataset_size, factor):
    """
    Determine the tier to pad to based on dataset size.
    """
    if dataset_size < eng_sizes_per_tier["tier_1M"] * factor:
        dataset_tier = "tier_1M"
    elif dataset_size < eng_sizes_per_tier["tier_10M"] * factor:
        dataset_tier = "tier_10M"
    elif dataset_size < eng_sizes_per_tier["tier_100M"] * factor:
        dataset_tier = "tier_100M"
    else:
        dataset_tier = None
        logger.info(
            "Dataset size exceeds the largest tier of 100M MB, no need for padding"
        )
    return dataset_tier


def get_dataset_size(dataset_df: pd.DataFrame, text_field: str = "text") -> float:
    return dataset_df[text_field].apply(bytes_in_text).sum()


def deduplicate_rows(
    selected_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """
    Deduplicate rows based on the 'text' field.
    """
    df = pd.DataFrame(selected_rows)
    deduped_df = df.drop_duplicates(subset=["text"], keep="first")
    deduped = [
        {str(k): v for k, v in row.items()}
        for row in deduped_df.to_dict(orient="records")
    ]
    data_count = sum(bytes_in_text(r["text"]) for r in deduped)
    return list(deduped), data_count


def check_if_required_padding_met(
    rows: list[dict[str, Any]], data_count: float, required_padding: float
) -> bool:
    """
    Check if the required padding has been met.
    """
    if data_count >= required_padding:
        rows, data_count = deduplicate_rows(rows)
        if data_count >= required_padding:
            return True
    return False
