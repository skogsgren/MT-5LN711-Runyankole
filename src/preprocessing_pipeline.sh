#!/bin/bash
# AUTHOR=Gustaf Gren
# DOC=Given the provided data.tar preprocesses the data, exporting to ./data

set -e # we want to exit the script if any part fails

if [ -d "./data" ]; then
    echo FATAL: export folder ./data already exists. remove or move to run script.
    exit 1
fi

TMP_DIR=$(mktemp -d)
mkdir -p "$TMP_DIR"/data
echo extracting ./data.tar to "$TMP_DIR"/data
tar xvf ./data.tar --directory "$TMP_DIR"/data

MTHS=("test" "dev" "training")
echo parsing sunbird dataset
for MTH in "${MTHS[@]}"
do
    python3 ./parse_sunbird.py \
        --input "$TMP_DIR"/data/sunbird_"$MTH"_dataset.txt \
        --output "$TMP_DIR"/"$MTH".txt \
        --lowercase \
        --split_punctuation
done

echo parsing bibles
python3 parse_bible.py \
    --source "$TMP_DIR"/data/nyn-x-bible.txt \
    --target "$TMP_DIR"/data/eng-x-bible-kingjames.txt \
    --output_dir "$TMP_DIR" \
    --lowercase

mkdir -p ./data

echo applying bpe to data
./bpe.sh nyn data/ "$TMP_DIR"/nyn*
./bpe.sh eng data/ "$TMP_DIR"/eng*

echo copying original parsed files to data export directory
cp "$TMP_DIR"/*.txt ./data/

echo finished processing. exported to ./data
