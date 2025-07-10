# multilingual_res

A package for fetching and integrating public multilingual resources (e.g., Ririro) into the BabyLM pipeline.

## Usage

- Add `--add-ririro-data` to the main pipeline to fetch and add Ririro data for the specified language.
- The package is designed to be extended: add new resource fetchers in their own modules and register them in `manager.py`.

## Adding a New Resource

1. Create a new module (e.g., `myresource.py`) with a function that returns a list of dicts (see `ririro.py` for format).
2. Register the fetcher in `manager.py`.
