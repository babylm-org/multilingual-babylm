import pandas as pd
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from iso639 import Lang
from typing import Any

from babylm_dataset_builder import BabyLMDatasetBuilder, DatasetConfig

import os
from dotenv import load_dotenv

byte_premium_factors = {
    "eng" : 1.0000000,
    "nld" : 1.0516739,
    "ukr" : 1.7514786,
    "zho" : 0.9893825,
    "bul" : 1.8123562,
    "ind" : 1.1788023,
    "fra" : 1.1742064,
    "deu" : 1.0537171,
    "jpn" : 1.3220250,
    "ita" : 1.0669230,
    "spa" : 1.0838621,
    "ell" : 1.9673049,
    "pol" : 1.0774161,
    "eus" : 1.0595837,
    "ara" : 1.4651134,
    "srp" : 1.4249495,
    "por" : 1.0979270,
    "heb" : 1.3555346,
    "est" : 0.9677856,
    "cym" : 1.0265667,
    "hrv" : 0.9897218,
    "swe" : 1.0210256,
    "ron" : 1.1151666,
    "kor" : 1.2933602,
    "isl" : 1.1543925,
    "afr" : 1.0373004,
    "xho" : 1.1988860,
    "zul" : 1.1639372,
    "sot" : 1.1661078,
    "nso" : 1.1156964
}

eng_sizes_per_tier = {
    "tier_1M" : 5.430,
    "tier_10M" : 54.30,
    "tier_100M" : 543.00,
}


def dataframe_to_docs(dataset_df : pd.DataFrame) -> list[dict[str, Any]]:
    docs = []
    for i, row in dataset_df.iterrows():
        text = row.get("text", "")
        if not text:
            continue
        meta = {k: v for k, v in row.items() if k != "text"}
        doc_id = row.get("doc_id") or row.get("id") or None
        docs.append(
            {
                "text": text,
                "doc_id": doc_id if doc_id is not None else str(i),
                "metadata": meta,
            }
        )
    return docs


def bytes_in_text(text: str) -> int:
    return len(text.encode('utf-8')) / 1_000_000  # Convert to MB


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


def pad_dataset_to_next_tier(
        dataset_df: pd.DataFrame,
        language_code: str,
        padding_resource: str = 'open-subtitles'
    ) -> dict[str, Any]:

    factor = byte_premium_factors.get(language_code)
    # calculate dataset size in Byte Premium space in MB
    dataset_size = dataset_df['text'].apply(bytes_in_text).sum()

    if factor is None:
        print('Byte premium for language code:', language_code, 'not found. Available codes:', list(byte_premium_factors.keys()))
        return {"dataset": dataset_df, "byte_premium_factor": factor, "dataset_size": dataset_size}

    # get tier to pad to
    if dataset_size < eng_sizes_per_tier["tier_1M"] * factor:
        dataset_tier = "tier_1M"
    elif dataset_size < eng_sizes_per_tier["tier_10M"] * factor:
        dataset_tier = "tier_10M"
    elif dataset_size < eng_sizes_per_tier["tier_100M"] * factor:
        dataset_tier = "tier_100M"
    else:
        print('Dataset size exceeds the largest tier of 100M equivalent English words, no need for padding')
        return {"dataset": dataset_df, "byte_premium_factor": factor, "dataset_size": dataset_size}

    required_padding_in_mb = eng_sizes_per_tier[dataset_tier] * factor - dataset_size
    if padding_resource == 'open-subtitles':
        iso_639_1_code = Lang(language_code).pt1
        repo_id = f'BabyLM-community/babylm-{iso_639_1_code}-subtitles'
        print(f"Loading OpenSubtitles data for language: {language_code} using repo: {repo_id}")
        load_dotenv()
        HF_token = os.getenv("HF_TOKEN")
        try:
            padding_dataset = load_dataset(repo_id, streaming=True, split="train", token=HF_token)
        except DatasetNotFoundError as e:
            print(f"OpenSubtitles dataset not found for {repo_id}: {e}")
            return {"dataset": dataset_df, "byte_premium_factor": factor, "dataset_size": dataset_size}

        # dataset might be huge so we stream it
        data_count = 0
        selected_rows = []
        for row in padding_dataset:
            num_bytes = bytes_in_text(row['text'])
            data_count += num_bytes
            selected_rows.append(row)
            if data_count >= required_padding_in_mb:
                break

        padding_df = pd.DataFrame(selected_rows)
        padding_df['script'] = padding_df['script'].apply(normalize_script)

        # pass through builder to validate documents
        docs_padding = dataframe_to_docs(padding_df)


        dataset_padding_config = DatasetConfig(language_code=language_code)
        builder_padding = BabyLMDatasetBuilder(dataset_padding_config, merge_existing=False)


        builder_padding.add_documents_from_iterable(docs_padding, {})
        dataset_padding_df = builder_padding.create_dataset_table()


        # concatenate the original dataset with the padding dataset
        dataset_df = pd.concat([dataset_df, dataset_padding_df], ignore_index=True)
        dataset_df.reset_index(drop=True, inplace=True)

    else:
        print(f"Padding resource '{padding_resource}' is not supported.")
        return {"dataset": dataset_df, "byte_premium_factor": factor, "dataset_size": dataset_size}

    final_dataset_size = dataset_df['text'].apply(bytes_in_text).sum()

    tier_words = dataset_tier.split('_')[-1]
    print(f"\n{'=' * 60}")
    print("PADDING RESULTS")
    print(f"{'=' * 60}")
    print(f"Padding language: {language_code} with data from {padding_resource} to tier {tier_words} words")
    print(f"Downloaded data from repo: {repo_id}")
    print(f'Initial dataset size: {dataset_size:.3f} MB')
    print(f"Byte Premium factor for {language_code}: {factor}")
    print(f'Required dataset size to match {tier_words} words of English ({eng_sizes_per_tier[dataset_tier]} MB) is {eng_sizes_per_tier[dataset_tier] * factor:.3f} MB')
    print(f"Padding data size: {data_count:.3f} MB")
    print(f"Final dataset size after padding: {final_dataset_size:.3f} MB")
    print(f"{'=' * 60}\n")


    return {"dataset": dataset_df, "byte_premium_factor": factor, "dataset_size": final_dataset_size}