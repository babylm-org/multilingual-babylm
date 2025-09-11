import pandas as pd
from datasets import load_dataset, get_dataset_config_names
from iso639 import Lang
from typing import Any, Optional

from babylm_dataset_builder import BabyLMDatasetBuilder, DatasetConfig
from preprocessor import preprocess_dataset

import os
from dotenv import load_dotenv

import warnings
from copy import deepcopy

from pad_utils import (
    bytes_in_text,
    check_if_required_padding_met,
    deduplicate_rows,
    normalize_script,
    dataframe_to_docs,
    eng_sizes_per_tier,
    get_byte_premium_factor,
    get_dataset_tier_to_pad,
    get_dataset_size,
)

from pad_language_specific import pad_language_specific, language_specific_pads

from tqdm import tqdm, TqdmWarning

warnings.filterwarnings("ignore", category=TqdmWarning)


def remove_padding_data(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove padding from the dataset.
    """

    filtered_docs = []
    categories_removed = set()
    categories_kept = set()
    sources_kept_subtitles = set()
    for doc in docs:
        metadata = doc.get("metadata", {})
        category = metadata.get("category", "n/a")
        data_source = metadata.get("data-source", "n/a")

        if category.startswith("padding-") or data_source == "OpenSubtitles":
            categories_removed.add(category)
            continue

        categories_kept.add(category)
        if category == "subtitles":
            sources_kept_subtitles.add(data_source)

        filtered_docs.append(doc)

    print(f"Removed {len(docs) - len(filtered_docs)} padding documents from dataset.")
    print(f"Document categories removed: {categories_removed}")
    print(f"Document categories kept: {categories_kept}")
    print(f'Data-sources kept for category "subtitles" : {sources_kept_subtitles}')

    return filtered_docs


def pad_with_opensubtitles(
    language_code: str,
    required_padding: float,
    HF_token: str,
):
    data_count = 0
    selected_rows = []
    iso_639_1_code = Lang(language_code).pt1
    repo_id = f"BabyLM-community/babylm-{iso_639_1_code}-subtitles"
    print(
        f"Loading OpenSubtitles data for language: {language_code} using repo: {repo_id}"
    )
    try:
        padding_dataset = load_dataset(
            repo_id, streaming=True, split="train", token=HF_token
        )
        pbar = tqdm(total=required_padding, bar_format="{l_bar}{bar}")
        for row in padding_dataset:
            if not isinstance(row, dict):
                continue
            row["category"] = "padding-opensubtitles"
            text = row.get("text", "")
            num = bytes_in_text(text)
            data_count += num
            selected_rows.append(deepcopy(row))
            pbar.update(num)
            if check_if_required_padding_met(
                selected_rows, data_count, required_padding
            ):
                break
        pbar.close()
        selected_rows, data_count = deduplicate_rows(selected_rows)
        return selected_rows, data_count, repo_id
    except Exception as e:
        print(f"OpenSubtitles dataset not found or error for {repo_id}: {e}")
        return [], 0, repo_id


def pad_with_wikipedia(
    language_code: str,
    script_code: str,
    required_padding: float,
    HF_token: str,
):
    data_count = 0
    selected_rows = []
    wiki_repo = "omarkamali/wikipedia-monthly"
    all_subsets = get_dataset_config_names(wiki_repo)
    lang_code_lower = language_code.lower()
    matching_subsets = [s for s in all_subsets if s.split(".")[-1] == lang_code_lower]
    if not matching_subsets:
        # Try iso639-1
        try:
            iso1 = Lang(language_code).pt1.lower()
        except Exception:
            iso1 = None
        if iso1:
            matching_subsets = [s for s in all_subsets if s.split(".")[-1] == iso1]
    if not matching_subsets:
        wiki_repo = "wikimedia/wikipedia"
        all_subsets = get_dataset_config_names(wiki_repo)
        lang_code_lower = language_code.lower()
        matching_subsets = [
            s for s in all_subsets if s.split(".")[-1] == lang_code_lower
        ]
        if not matching_subsets:
            print(f"Wikipedia: No matching subset found for language {language_code}")
            return [], 0, wiki_repo
    last_subset = None
    try:
        for wiki_subset in matching_subsets:
            last_subset = wiki_subset
            print(
                f"Loading Wikipedia data for language: {language_code}, subset: {wiki_subset}"
            )
            wiki_dataset = load_dataset(
                wiki_repo,
                name=wiki_subset,
                split="train",
                streaming=True,
                token=HF_token,
            )
            pbar = tqdm(total=required_padding, bar_format="{l_bar}{bar}")
            for row in wiki_dataset:
                if not isinstance(row, dict):
                    continue
                row.pop("id", None)
                row["script"] = script_code
                row["category"] = "padding-wikipedia"
                row["data-source"] = "Wikipedia"
                row["age-estimate"] = "n/a"
                row["license"] = "cc-by-sa-4.0"
                num = bytes_in_text(row.get("text", ""))
                data_count += num
                selected_rows.append(deepcopy(row))
                pbar.update(num)
                if check_if_required_padding_met(
                    selected_rows, data_count, required_padding
                ):
                    break
            pbar.close()
            selected_rows, data_count = deduplicate_rows(selected_rows)
            if data_count >= required_padding:
                break
        if last_subset is not None:
            return selected_rows, data_count, f"{wiki_repo}/{last_subset}"
        else:
            return [], 0, wiki_repo
    except Exception as e:
        msg = f"Wikipedia dataset not found or error for {wiki_repo}/{last_subset}: {e}"
        print(msg)
        if last_subset is not None:
            return [], 0, f"{wiki_repo}/{last_subset}"
        else:
            return [], 0, wiki_repo


def pad_with_fineweb_c(
    language_code: str,
    script_code: str,
    required_padding: float,
    HF_token: str,
):
    data_count = 0
    selected_rows = []
    fineweb_repo = "data-is-better-together/fineweb-c"
    all_subsets = get_dataset_config_names(fineweb_repo)
    matching_subsets = [s for s in all_subsets if s.startswith(f"{language_code}_")]
    last_subset = None
    try:
        for fineweb_subset in matching_subsets:
            last_subset = fineweb_subset
            print(
                f"Loading fineweb-c data for language: {language_code}, subset: {fineweb_subset}"
            )
            fineweb_dataset = load_dataset(
                fineweb_repo,
                name=fineweb_subset,
                split="train",
                streaming=True,
                token=HF_token,
            )
            script_code_fw = fineweb_subset.split("_")[-1]
            pbar = tqdm(total=required_padding, bar_format="{l_bar}{bar}")
            for row in fineweb_dataset:
                if not isinstance(row, dict):
                    continue
                if row.get("problematic_content_label_present", True):
                    continue
                labels = row.get("educational_value_labels", [])
                if len(set(labels)) == 1 and (
                    list(set(labels))[0] is None or list(set(labels))[0] == "None"
                ):
                    continue
                row["script"] = script_code_fw or script_code
                row["category"] = "padding-fineweb-c"
                row["data-source"] = f"{fineweb_repo}/{fineweb_subset}"
                row["age-estimate"] = "n/a"
                row["license"] = "ODC-By"
                num = bytes_in_text(row.get("text", ""))
                data_count += num
                selected_rows.append(deepcopy(row))
                pbar.update(num)
                if check_if_required_padding_met(
                    selected_rows, data_count, required_padding
                ):
                    break
            pbar.close()
            selected_rows, data_count = deduplicate_rows(selected_rows)
            if data_count >= required_padding:
                break
        if last_subset is not None:
            return selected_rows, data_count, f"{fineweb_repo}/{last_subset}"
        else:
            return [], 0, fineweb_repo
    except Exception as e:
        msg = f"fineweb-c dataset not found or error for {fineweb_repo}/{last_subset}: {e}"
        print(msg)
        if last_subset is not None:
            return [], 0, f"{fineweb_repo}/{last_subset}"
        else:
            return [], 0, fineweb_repo


def pad_by_byte_factor(
    dataset_df: pd.DataFrame,
    language_code: str,
    script_code: str,
    factor: float,
    dataset_size: float,
    dataset_tier: str,
    required_padding_in_mb: float,
    HF_token: str,
):
    used_resources = []
    selected_rows = []
    data_count = 0

    # try with language specific first
    if language_code in language_specific_pads:
        lang_rows, lang_count, lang_repo = pad_language_specific(
            required_padding_in_mb - data_count, language_code, script_code, HF_token
        )
        selected_rows.extend(lang_rows)
        data_count += lang_count
        if lang_rows:
            used_resources.append(f"language-specific:{lang_repo}")

    # 1. Try OpenSubtitles
    if data_count < required_padding_in_mb:
        os_rows, os_count, os_repo = pad_with_opensubtitles(
            language_code, required_padding_in_mb - data_count, HF_token
        )
        selected_rows.extend(os_rows)
        data_count += os_count
        if os_rows:
            used_resources.append(f"open-subtitles:{os_repo}")

    # 2. If still not enough, try fineweb-c
    if data_count < required_padding_in_mb:
        fw_rows, fw_count, fw_repo = pad_with_fineweb_c(
            language_code,
            script_code,
            required_padding_in_mb - data_count,
            HF_token,
        )
        selected_rows.extend(fw_rows)
        data_count += fw_count
        if fw_rows:
            used_resources.append(f"fineweb-c:{fw_repo}")

    # 3. If still not enough, try Wikipedia
    if data_count < required_padding_in_mb:
        wiki_rows, wiki_count, wiki_repo = pad_with_wikipedia(
            language_code,
            script_code,
            required_padding_in_mb - data_count,
            HF_token,
        )
        selected_rows.extend(wiki_rows)
        data_count += wiki_count
        if wiki_rows:
            used_resources.append(f"wikipedia:{wiki_repo}")

    if selected_rows:
        padding_df = pd.DataFrame(selected_rows)
        if "script" in padding_df.columns:
            padding_df["script"] = padding_df["script"].apply(normalize_script)

        # pass through builder to validate documents
        docs_padding = dataframe_to_docs(padding_df)

        dataset_padding_config = DatasetConfig(language_code=language_code)
        builder_padding = BabyLMDatasetBuilder(
            dataset_padding_config, merge_existing=False
        )

        builder_padding.add_documents_from_iterable(docs_padding, {})
        dataset_padding_df = preprocess_dataset(builder_padding.create_dataset_table())

        # concatenate the original dataset with the padding dataset
        dataset_df = pd.concat([dataset_df, dataset_padding_df], ignore_index=True)
        dataset_df.reset_index(drop=True, inplace=True)
    else:
        print("No padding data could be loaded from any resource.")
        return {
            "dataset": dataset_df,
            "byte_premium_factor": factor,
            "dataset_size": get_dataset_size(dataset_df),
        }

    final_dataset_size = get_dataset_size(dataset_df)
    tier_words = dataset_tier.split("_")[-1]

    if final_dataset_size < eng_sizes_per_tier[dataset_tier] * factor:
        print(
            f"Warning: Final dataset size {final_dataset_size:.3f} MB is less than required {eng_sizes_per_tier[dataset_tier] * factor:.3f} MB for tier {dataset_tier}."
        )
        print(
            f"Missing {eng_sizes_per_tier[dataset_tier] * factor - final_dataset_size:.3f} MB"
        )

    print(f"\n{'=' * 60}")
    print("PADDING RESULTS")
    print(f"{'=' * 60}")
    print(
        f"Padding language: {language_code} with data from OpenSubtitles, FineWeb-C, and Wikipedia to tier {tier_words} words"
    )
    print(
        f"Downloaded data from repos: {', '.join(used_resources) if used_resources else 'None'}"
    )
    print(f"Initial dataset size: {dataset_size:.3f} MB")
    print(f"Byte Premium factor for {language_code}: {factor}")
    print(
        f"Required dataset size to match {tier_words} words of English ({eng_sizes_per_tier[dataset_tier]} MB) is {eng_sizes_per_tier[dataset_tier] * factor:.3f} MB"
    )
    print(f"Padding data size: {data_count:.3f} MB")
    print(f"Final dataset size after padding: {final_dataset_size:.3f} MB")
    print(f"{'=' * 60}\n")
    return {
        "dataset": dataset_df,
        "byte_premium_factor": factor,
        "dataset_size": final_dataset_size,
    }


def pad_dataset_to_next_tier(
    dataset_df: pd.DataFrame,
    language_code: str,
    script_code: str,
    byte_premium_factor: Optional[float] = None,
) -> dict[str, Any]:
    if byte_premium_factor is None:
        factor = get_byte_premium_factor(language_code, script_code)

    load_dotenv()
    HF_token = os.getenv("HF_TOKEN") or ""

    if factor is not None:
        # MB-based padding (byte premium factor exists)
        dataset_size = get_dataset_size(dataset_df)
        dataset_tier = get_dataset_tier_to_pad(dataset_size, factor)
        if dataset_tier is None:
            return {
                "dataset": dataset_df,
                "byte_premium_factor": factor,
                "dataset_size": dataset_size,
            }
        required_padding_in_mb = (
            eng_sizes_per_tier[dataset_tier] * factor - dataset_size
        )

        return pad_by_byte_factor(
            dataset_df,
            language_code,
            script_code,
            factor,
            dataset_size,
            dataset_tier,
            required_padding_in_mb,
            HF_token,
        )
    else:
        print(f"Byte premium factor not found for {language_code}")
        return {
            "dataset": dataset_df,
            "byte_premium_factor": None,
            "dataset_size": get_dataset_size(dataset_df),
        }
