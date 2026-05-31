#!/bin/bash
set -e

pip install gdown

gdown "https://drive.google.com/uc?id=12aehzw_6iYgaWY6N6zP7Nj231cdhr93L" -O data.zip

unzip data.zip

UNZIPPED_DIR=$(unzip -Z1 data.zip | head -1 | cut -d/ -f1)

if [ -n "$UNZIPPED_DIR" ] && [ "$UNZIPPED_DIR" != "data" ]; then
    mv "$UNZIPPED_DIR" data
fi

rm data.zip

mv data ../



pip install -r req.txt

pip install -e .
