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