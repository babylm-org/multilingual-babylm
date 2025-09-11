from pipeline import process_dataset
from hf_uploader import HFDatasetUploader
from datasets import load_dataset
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def process_dataframe(dataset_df: pd.DataFrame, lang) -> pd.DataFrame:
    # ... do stuff like removing, adding new data
    return dataset_df


def main():
    data_uploader = HFDatasetUploader(token=HF_TOKEN)
    repos = data_uploader._discover_babylm_repos(check_empty=False)
    # ignore subtitles repos
    repos = [repo for repo in repos if "subtitles" not in repo]

    # or define a single repo
    # repos = ["BabyLM-Community/babylm-eng"]

    for repo in repos:
        lang = repo.split("-")[-1]
        print(f"Processing {repo}...")

        # load updated dataset
        dataset = load_dataset(
            repo, split="train", token=HF_TOKEN, download_mode="force_redownload"
        )
        dataset_df = dataset.to_pandas()

        # process dataset
        dataset_df = process_dataframe(dataset_df, lang=lang)

        # save dataset
        os.makedirs("tmp_datasets", exist_ok=True)
        data_path = "tmp_datasets/{}.json".format(repo.split("/")[-1])
        dataset_df.to_json(
            data_path, orient="records", lines=False, indent=2, force_ascii=False
        )

        # upload dataset
        upload = True

        # add known tokenizers used in BabyLMs
        if lang == "jpn":
            tokenizer_name = "tohoku-nlp/bert-base-japanese"
        elif lang == "zho":
            tokenizer_name = "Qwen/Qwen3-0.6B"
        elif lang == "yue":
            tokenizer_name = "Qwen/Qwen1.5-7B-Chat"
        else:
            tokenizer_name = None

        script = dataset_df["script"].mode()[0]  # most frequent value
        print(f"Using script {script}")
        # or define script manually
        # script = "Latn"

        data_path = data_path
        data_type = "json"

        # if adding data turn to True
        preprocess = False

        # re-pad dataset, if desired (e.g., if you add or remove data)
        remove_previous_padding = False
        pad = False

        # add/remove multilingual resources if desired
        remove_previous_multiling_data = False
        add_multilingual_data = False

        # pull request functionality
        create_pr = False
        pr_title = "My PR"
        pr_description = "PR Description"

        print("Running pipeline")

        process_dataset(
            language_code=lang,
            script_code=script,
            data_path=data_path,
            document_config_params={},  # used to fill missing values in documents
            metadata_file=None,  # extra document-level metadata
            upload=upload,
            repo_id=repo,
            preprocess_text=preprocess,
            data_type=data_type,
            enable_language_filtering=False,  # filter documents by language
            language_filter_threshold=0.0,
            pad_opensubtitles=pad,
            tokenizer_name=tokenizer_name,
            overwrite=True,  # overwrite existing dataset in babylm_datasets/ , otherwise append
            add_ririro_data=add_multilingual_data,
            add_glotstorybook_data=add_multilingual_data,
            add_childwiki_data=add_multilingual_data,
            add_childes_data=add_multilingual_data,
            remove_previous_ririro_data=remove_previous_multiling_data,
            remove_previous_glotstorybook_data=remove_previous_multiling_data,
            remove_previous_childwiki_data=remove_previous_multiling_data,
            remove_previous_childes_data=remove_previous_multiling_data,
            remove_previous_padding=remove_previous_padding,
            enable_script_update=False,
            script_update_all=False,
            create_pr=create_pr,
            pr_title=pr_title,
            pr_description=pr_description,
        )
        print("Done")


if __name__ == "__main__":
    main()
