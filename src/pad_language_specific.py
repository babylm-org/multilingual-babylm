from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from pad_utils import (
    bytes_in_text,
    check_if_required_padding_met,
    deduplicate_rows,
)


def pad_nso(
    required_padding_in_mb, language_code="nso", script_code="Latn", HF_token=None
):
    pairs = [  # machine translation pairs
        "eng-nso",  # already covers our data needs
        # 'fra-nso',
        # 'fuv-nso',
        # 'hau-nso',
        # 'ibo-nso',
        # 'kam-nso',
        # 'kin-nso',
        # 'lug-nso',
        # 'luo-nso',
        # 'nso-nya',
        # 'nso-orm',
        # 'nso-sna',
        # 'nso-som',
        # 'nso-ssw',
        # 'nso-swh',
        # 'nso-tsn',
        # 'nso-tso',
        # 'nso-umb',
        # 'nso-xho',
        # 'nso-yor',
        # 'nso-zul'
    ]
    dataset_all = []
    repo_id = "allenai/wmt22_african"
    print(f"Padding dataset for language {language_code} using {repo_id}")
    for p in pairs:
        dataset_pad = load_dataset(repo_id, p, split="train", trust_remote_code=True)
        dataset_pad = dataset_pad.to_pandas()["translation"].apply(lambda x: x["nso"])
        dataset_all.append(dataset_pad)

    selected_rows = []
    data_count = 0
    dataset_all = pd.concat(dataset_all, ignore_index=True)
    dataset_all = dataset_all.unique()
    pbar = tqdm(total=required_padding_in_mb, bar_format="{l_bar}{bar}")
    for text in dataset_all:
        row = {
            "text": text,
            "category": "padding-mt",
            "data-source": "WMT22 African",
            "script": "Latn",
            "age-estimate": "NA",
            "license": "ODC-BY",
            "misc": "{}",
        }
        num = bytes_in_text(text)
        data_count += num
        selected_rows.append(deepcopy(row))
        pbar.update(num)
        if check_if_required_padding_met(
            selected_rows, data_count, required_padding_in_mb
        ):
            break
    pbar.close()
    selected_rows, data_count = deduplicate_rows(selected_rows)

    return selected_rows, data_count, repo_id


language_specific_pads = {
    "nso": pad_nso,
    # Add other language-specific padding functions here as needed
}


def pad_language_specific(
    required_padding_in_mb, language_code, script_code, HF_token=None
):
    print("Padding using language-specific data for language:", language_code)
    padding_function = language_specific_pads.get(language_code)
    if padding_function:
        return padding_function(
            required_padding_in_mb, language_code, script_code, HF_token
        )
    else:
        return [], 0, None
