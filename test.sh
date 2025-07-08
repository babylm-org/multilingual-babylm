#!/bin/bash

# check document filtering
python pipeline.py \
    --language eng\
    --data-path "examples/" \
    --data-type "text" \
    --metadata-file "examples/text_metadata.json" \
    --script Latn \
    --enable-language-filter \
    --preprocess-text \
    --license "cc-by-sa"
    # should filter out document filter1.txt


# check json
python pipeline.py \
    --language eng\
    --data-path "examples/dataset.json" \
    --data-type "json" \
    --script Latn \
    --enable-language-filter \
    --preprocess-text

# check csv
python pipeline.py \
    --language eng\
    --data-path "examples/dataset.csv" \
    --data-type "csv" \
    --script Latn \
    --enable-language-filter \
    --preprocess-text



# check license validation
python pipeline.py \
    --language eng\
    --data-path "examples/" \
    --data-type "text" \
    --metadata-file "examples/text_metadata.json" \
    --script Latn \
    --enable-language-filter \
    --preprocess-text 
    #--license "cc-by-sa"
    # missing license, should fail

# check script validation
python pipeline.py \
    --language eng\
    --data-path "examples/" \
    --data-type "text" \
    --metadata-file "examples/text_metadata.json" \
    --enable-language-filter \
    --preprocess-text \
    --license "cc-by-sa"\
    --script Latin 
    # non-ISO 15924 script value, should fail

