# Basic command to upload a babyLM dataset from a JSON file
# Recommended options included for: 
# - language filtering
# - text preprocessing
# Fore more data loading options, see the example_usage.sh file
python pipeline.py \
    --language eng\
    --script Latn \
    --data-path "data/dataset.json" \
    --data-type "json" \
    --enable-language-filter \
    --preprocess-text \
    --upload \
    --repo-id "username/babylm-eng"
