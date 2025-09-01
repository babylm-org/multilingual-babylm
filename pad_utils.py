import pandas as pd
from typing import Any
import hashlib

byte_premium_factors = {
    "eng": 1.0,
    "nld": 1.0516739,
    "ukr": 1.7514786,
    "zho": 0.9893825,
    "bul": 1.8123562,
    "ind": 1.1788023,
    "fra": 1.1742064,
    "deu": 1.0537171,
    "jpn": 1.322025,
    "ita": 1.066923,
    "spa": 1.0838621,
    "ell": 1.9673049,
    "pol": 1.0774161,
    "eus": 1.0595837,
    "ara": 1.4651134,
    "srp": 1.4249495,
    "por": 1.097927,
    "heb": 1.3555346,
    "est": 0.9677856,
    "cym": 1.0265667,
    "hrv": 0.9897218,
    "swe": 1.0210256,
    "ron": 1.1151666,
    "kor": 1.2933602,
    "isl": 1.1543925,
    "afr": 1.0373004,
    "xho": 1.198886,
    "zul": 1.1639372,
    "sot": 1.1661078,
    "nso": 1.1156964,
    "hun": 1.0199851,
    "ces": 1.0358867,
    "yue": 0.8624614,
    "cat": 1.0926706,
    "jav": 1.1468458,
    "dan": 1.0210658,
    "tha": 2.7416472,
    "nor": 1.125316,
    "tur": 1.0444815,
    "fas": 1.5973263,
    "rus": 1.8228284,
    "gle": 1.9749562,
    "crl": 2.6007383,
    "tsn": 1.1739403,
    "yuw": 1.605417,
    "tam": 2.7290997,
    "slv": 0.97215,
    "mop": 1.6077918,
    "mar": 2.4793565,
    "ltz": 1.225349,
    "bug": 1.2279017,
    "ace": 1.2419926,
    "ban": 1.2695436,
    "mak": 1.250697369,
}

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


def get_byte_premium_factor(lang: str):
    return byte_premium_factors.get(lang)


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
            return name

    # dataset_size is below the tier threshold
    for name, target in target_sizes:
        if dataset_size < target:
            return "< " + name

    # dataset_size exceeds largest tier threshold
    return "> " + sorted_pairs[-1]


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
        print("Dataset size exceeds the largest tier of 100M MB, no need for padding")
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
