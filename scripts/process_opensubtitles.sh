#!/bin/bash

set -e

# Prompt user for language list
read -p "Enter space-separated language codes (e.g., 'af ku'): " LANG_LIST
read -p "Do you want to upload your data to HF after processing? (true/false) " UPLOAD


DATA_DIR="prep_subtitles"
# UPLOAD=true  # Set to true or false as needed

if [ ! -d "$DATA_DIR" ]; then
    FILE_BASENAME="title.basics.tsv"
    GZ_FILE="$DATA_DIR/${FILE_BASENAME}.gz"
    TSV_FILE="$DATA_DIR/$FILE_BASENAME"
    CREATE_DB_SCRIPT="create_db.py"

    mkdir -p "$DATA_DIR"

    if [ -f "$CREATE_DB_SCRIPT" ]; then
        cp "$CREATE_DB_SCRIPT" "$DATA_DIR/"
    fi

    if [ ! -f "$TSV_FILE" ]; then
        if [ ! -f "$GZ_FILE" ]; then
            echo "Downloading $GZ_FILE..."
            wget -O "$GZ_FILE" "https://datasets.imdbws.com/title.basics.tsv.gz"
        fi
        echo "Extracting $GZ_FILE..."
        gunzip -c "$GZ_FILE" > "$TSV_FILE"
    else
        echo "$TSV_FILE already exists. Skipping download and extraction."
    fi

    echo "Running create_db.py..."
    cd "$DATA_DIR"
    python "create_db.py"
    cd ..
fi

for lang in $LANG_LIST
do
    # Set upload arguments if UPLOAD is true
    UPLOAD_ARGS=""
    if [ "$UPLOAD" = true ]; then
        UPLOAD_ARGS="--upload --repo-id account-anonymized/babylm-${lang}-subtitles"
    fi
    python process_opensubtitles.py \
        --language $lang \
        --batch-size 50 \
        --imdb-db-path ./prep_subtitles/imdb_mastersheet.db \
        --forbidden-genres Horror News Crime War \
        --script Latn \
        --age-estimate n/a \
        --license cc-by \
        --source-url https://opus.nlpl.eu/OpenSubtitles.php \
        --misc {} \
        $UPLOAD_ARGS

    rm -rf ./output
done

echo "Done."