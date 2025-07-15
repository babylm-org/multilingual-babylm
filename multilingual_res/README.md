# multilingual_res

A package for fetching and integrating public multilingual resources (e.g., Ririro, GlotStoryBook, ChildWiki) into the BabyLM pipeline.

## Usage

- Add `--add-ririro-data` to the main pipeline to fetch and add Ririro data for the specified language.
- Add `--add-glotstorybook-data` to fetch and add GlotStoryBook data for the specified language.
- Add `--add-childwiki-data` to fetch and add ChildWiki data for the specified language.
- The package is designed to be extended: add new resource fetchers in their own modules and register them in `manager.py`.

## Available Resource Fetchers

- **Ririro**: Fetches children's books from ririro.com. Use `--add-ririro-data`.
- **GlotStoryBook**: Fetches storybooks for 180 languages from HuggingFace. Use `--add-glotstorybook-data`.
- **ChildWiki**: Fetches child-friendly wiki content from HuggingFace. Use `--add-childwiki-data`.
- **CHILDES**: Fetches child-directed speech transcripts from the CHILDES database via HuggingFace. Use `--add-childes-data`.

## Adding a New Resource

1. Create a new module (e.g., `myresource.py`) with a fetcher class that **derives from `BaseResourceFetcher`** and implements the `fetch()` method (see `ririro.py` for format).
2. Register the fetcher in `manager.py`.
3. Add a CLI argument and integrate it in the pipeline if needed.
