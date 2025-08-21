from random import randint
import sys
from pipeline import process_dataset
from hf_uploader import HFDatasetUploader
from datasets import load_dataset
from dotenv import load_dotenv
import json
import os
from pathlib import Path
import pandas as pd
import ast

from loguru import logger

load_dotenv()
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> {message}")

HF_TOKEN = os.getenv("HF_TOKEN")
# import ipdb

# br = ipdb.set_trace



def smart_load(val):
    """
    Try to peel multiple layers of json.dumps() / json.loads()
    until we reach a real Python object.
    Falls back to ast.literal_eval if the final inner value
    is a Python literal (e.g. "{'foo': 'bar'}").

    if we reach a dictionary, return a valid JSON string of it
    else, return pd.NA
    """
    # keep peeling while it's a string
    while isinstance(val, str):
        try:
            val = json.loads(val)
            continue
        except json.JSONDecodeError:
            try:
                val = ast.literal_eval(val)
                continue
            except Exception:
                break
    if isinstance(val, str):
        val = pd.NA
    else:
        val = json.dumps(val, ensure_ascii=False)

    return val


def process_dataframe(dataset_df: pd.DataFrame, lang) -> pd.DataFrame:
    # fix empty misc

    rand_row_id = randint(0, dataset_df.shape[0])
    logger.info(dataset_df.iloc[rand_row_id])

    dataset_df["misc"] = dataset_df["misc"].apply(
        lambda x: x.strip() if x is not None else "{}"
    )
    dataset_df.loc[dataset_df["misc"] == "", "misc"] = "{}"

    misc_fixed = dataset_df["misc"].map(smart_load)
    is_fixed = misc_fixed.notna()
    not_fixed = ~is_fixed
    dataset_df.loc[is_fixed, "misc"] = misc_fixed[is_fixed].astype("string")

    if not_fixed.any():
        logger.warning("Found wrong misc values.")
        logger.warning(
            f"Wrong Misc values - Rows {dataset_df[not_fixed].index.tolist()}: {dataset_df[not_fixed]['misc'].tolist()}"
        )

    # remove empty text
    text_is_na = dataset_df["text"].isna()
    text_empty = dataset_df["text"].str.strip().eq("")
    mask = text_is_na | text_empty
    logger.info(
        f"number of texts with None or empty-string `text` value: {int(mask.sum())}"
    )
    logger.info(f"size before removal: {dataset_df.shape[0]}")
    dataset_df = dataset_df.loc[~mask].reset_index(drop=True)
    logger.info(f"size after removal: {dataset_df.shape[0]}")

    # check for empty license:
    dataset_df["license"] = dataset_df["license"].apply(
        lambda x: x.strip() if x is not None else r""
    )
    empty_license = dataset_df["license"].eq("")
    logger.info(f"number of empty licenses: {int(empty_license.sum())}")
    dataset_df.loc[empty_license, "license"] = "CC-BY-NC-4.0"

    # if lang == "ell":
    #     replacements = [
    #         (r"Creative Commons BY-NC-ND", "CC-BY-NC-ND"),
    #         (r"Creative Commons BY-SA", "CC-BY-SA"),
    #         (r"Creative Commons BY-NC", "CC-BY-NC"),
    #         (
    #             r"Άδεια διανομής:\xa0Creative Commons\xa0BY-SA\xa0 \(Αναφορά δημιουργού – Παρόμοια διανομή\)",
    #             "CC-BY-SA",
    #         ),
    #         (
    #             r"Άδεια διανομής: Ελεύθερη αναπαραγωγή \(με αναφορά της πηγής\)",
    #             "Free distribution (with source attribution)",
    #         ),
    #         (
    #             r"Άδεια διανομής: Ελεύθερη διάθεση για μην εμπορική χρήση",
    #             "Free distribution for non-commercial usage",
    #         ),
    #         (r"Public Domain", "Public Domain"),
    #         (
    #             r"Άδεια διανομής: Ελεύθερη αναπαραγωγή \(με αναφορά της πηγής\)",
    #             "Free distribution (with source attribution)",
    #         ),
    #     ]

    #     license_col = dataset_df["license"].astype("string")

    #     for pattern, replacement in replacements:
    #         mask = license_col.str.contains(pattern, na=False, regex=True)
    #         dataset_df.loc[mask, "license"] = replacement

    #     wrong_script = dataset_df["script"] == "Greek"
    #     dataset_df.loc[wrong_script, "script"] = "Grek"

    logger.info(f"Unique licenses: {dataset_df['license'].unique().tolist()}")
    logger.info(f"number of unique licenses {len(dataset_df['license'].unique())}")

    logger.info(f"Unique scripts: {dataset_df['script'].unique().tolist()}")

    # normalize script
    dataset_df.loc[dataset_df["script"] == "Latin", "script"] = "Latn"

    logger.info(dataset_df)
    logger.info(dataset_df.iloc[rand_row_id])

    return dataset_df


repos_file = "./cache/active_repos.json"
if Path(repos_file).exists():
    active_repos = json.load(open(repos_file))
else:
    data_uploader = HFDatasetUploader(token=HF_TOKEN)
    active_repos = data_uploader._discover_babylm_repos(check_empty=False)
    with open(repos_file, "w") as f:
        json.dump(active_repos, f)

skip = ['zho']
babylm_repos = [repo for repo in active_repos if "subtitles" not in repo]
babylm_repos = [repo for repo in babylm_repos if not any(s in repo for s in skip)]

with open("cache/done.log", "r") as f:
    done_repos = f.read().splitlines()

babylm_repos = [repo for repo in babylm_repos if repo not in done_repos]

for repo in babylm_repos:
    log_id = logger.add(f"cache/{repo.split('/')[-1]}_info.log", mode="w")
    logger.info(f"Processing {repo}")

    language = repo.split("-")[-1]

    if language == 'jpn':
        tokenizer_name = 'tohoku-nlp/bert-base-japanese'
    else:
        tokenizer_name = None

    # load dataset
    dataset = load_dataset(repo, split="train", token=HF_TOKEN, cache_dir="./cache")
    dataset_df = dataset.to_pandas()

    # process dataset
    dataset_df = process_dataframe(dataset_df, lang=language)

    # save dataset
    os.makedirs("tmp_datasets", exist_ok=True)
    data_path = "tmp_datasets/{}.json".format(repo.split("/")[-1])
    dataset_df.to_json(
        data_path, orient="records", lines=False, indent=2, force_ascii=False
    )

    # upload dataset
    upload = True

    script = dataset_df["script"].mode()[0]
    logger.info(f"Chose script {script} out of {dataset_df['script'].unique().tolist()}")
    data_path = data_path
    data_type = "json"
    preprocess = False
    remove_previous_padding = False
    pad = False
    add_multilingual_data = False
    remove_previous_multiling_data = False

    # br()

    process_dataset(
        language_code=language,
        script_code=script,
        data_path=data_path,
        document_config_params={},
        metadata_file=None,
        upload=upload,
        repo_id=repo,
        preprocess_text=preprocess,
        data_type=data_type,
        enable_language_filtering=False,
        language_filter_threshold=0.0,
        pad_opensubtitles=pad,
        tokenizer_name=tokenizer_name,
        overwrite=True,
        add_ririro_data=add_multilingual_data,
        add_glotstorybook_data=add_multilingual_data,
        add_childwiki_data=add_multilingual_data,
        add_childes_data=add_multilingual_data,
        remove_previous_ririro_data=remove_previous_multiling_data,
        remove_previous_glotstorybook_data=remove_previous_multiling_data,
        remove_previous_childwiki_data=remove_previous_multiling_data,
        remove_previous_childes_data=remove_previous_multiling_data,
        remove_previous_padding=remove_previous_padding,
    )

    logger.info("Done")

    updated_dataset = load_dataset(
        repo, split="train", token=HF_TOKEN, download_mode="force_redownload"
    )
    updated_dataset_df = updated_dataset.to_pandas()
    try:
        updated_dataset_df["misc"].apply(json.loads)
    except Exception as e:
        logger.error(f"Error processing 'misc' column: {e}")

    logger.remove(log_id)
    # br()
    with open("cache/done.log", "a") as f:
        f.write(f"{repo}\n")
