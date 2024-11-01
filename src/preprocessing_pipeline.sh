#!/bin/bash
# AUTHOR=Gustaf Gren
# DOC=Given the provided data.tar.xz preprocesses the data, exporting to ./data

set -e # we want to exit the script if any part fails

if [ -d "./data" ]; then
    echo "FATAL: export folder ./data already exists. remove or move to run script."
    exit 1
fi

TMP_DIR=$(mktemp -d)
mkdir -p "$TMP_DIR"/data
echo "extracting ./data.tar.xz to $TMP_DIR/data"
tar xvfz ./data.tar.xz --directory "$TMP_DIR"/data

MTHS=("test" "dev" "training")
echo "parsing sunbird dataset"
for MTH in "${MTHS[@]}"
do
    python3 ./parse_sunbird.py \
        --input "$TMP_DIR"/data/sunbird_"$MTH"_dataset.txt \
        --output "$TMP_DIR"/"$MTH".txt \
        --split_punctuation
done

echo "parsing bibles"
python3 parse_bible.py \
    --source "$TMP_DIR"/data/nyn-x-bible.txt \
    --target "$TMP_DIR"/data/eng-x-bible-kingjames.txt \
    --output_dir "$TMP_DIR" \

mkdir -p ./data

# I know it's ugly that I've hardcoded it, but I didn't want to make jq a
# dependency of the project, as I'd used json otherwise
echo "learning bpe from (original)"
echo "exporting weights to ./data/\$LANG.bpe"
./bpe.sh \
    -l nyn \
    -o data/original/ \
    -d "$TMP_DIR"/nyn_dev.txt \
    -t "$TMP_DIR"/nyn_test.txt \
    "$TMP_DIR"/nyn_training.txt
./bpe.sh \
    -l eng \
    -o data/original/ \
    -d "$TMP_DIR"/eng_dev.txt \
    -t "$TMP_DIR"/eng_test.txt \
    "$TMP_DIR"/eng_training.txt
echo "copying detokenized originals to ./data/\$LANG.detok"
cp "$TMP_DIR"/nyn_training.txt data/original/nyn_train.detok
cp "$TMP_DIR"/nyn_dev.txt data/original/nyn_dev.detok
cp "$TMP_DIR"/nyn_test.txt data/original/nyn_test.detok

cp "$TMP_DIR"/eng_training.txt data/original/eng_train.detok
cp "$TMP_DIR"/eng_dev.txt data/original/eng_dev.detok
cp "$TMP_DIR"/eng_test.txt data/original/eng_test.detok

echo "learning bpe from (original, bible)"
echo "exporting weights to ./data/bible/\$LANG.bpe"
./bpe.sh \
    -l nyn \
    -o data/bible/ \
    -d "$TMP_DIR"/nyn_dev.txt \
    -t "$TMP_DIR"/nyn_test.txt \
    "$TMP_DIR"/nyn_training.txt \
    "$TMP_DIR"/nyn-x-bible_parsed.txt
./bpe.sh \
    -l eng \
    -o data/bible/ \
    -d "$TMP_DIR"/eng_dev.txt \
    -t "$TMP_DIR"/eng_test.txt \
    "$TMP_DIR"/eng_training.txt \
    "$TMP_DIR"/eng-x-bible-kingjames_parsed.txt

echo "copying detokenized originals to ./data/\$LANG.detok"
cat "$TMP_DIR"/nyn_training.txt \
    "$TMP_DIR"/nyn-x-bible_parsed.txt \
    > data/bible/nyn_train.detok
cp "$TMP_DIR"/nyn_dev.txt data/bible/nyn_dev.detok
cp "$TMP_DIR"/nyn_test.txt data/bible/nyn_test.detok

cat "$TMP_DIR"/eng_training.txt \
    "$TMP_DIR"/eng-x-bible-kingjames_parsed.txt \
    > data/bible/eng_train.detok
cp "$TMP_DIR"/eng_dev.txt data/bible/eng_dev.detok
cp "$TMP_DIR"/eng_test.txt data/bible/eng_test.detok

echo "learning bpe from (original, backtranslated)"
echo "exporting weights to ./data/backtranslated/\$LANG.bpe"
./bpe.sh \
    -l nyn \
    -o data/backtranslated/ \
    -d "$TMP_DIR"/nyn_dev.txt \
    -t "$TMP_DIR"/nyn_test.txt \
    "$TMP_DIR"/nyn_training.txt \
    "$TMP_DIR"/data/nyn_backtranslated_training.txt
./bpe.sh \
    -l eng \
    -o data/backtranslated/ \
    -d "$TMP_DIR"/eng_dev.txt \
    -t "$TMP_DIR"/eng_test.txt \
    "$TMP_DIR"/eng_training.txt \
    "$TMP_DIR"/data/eng_backtranslated_training.txt

echo "copying detokenized originals to ./data/\$LANG.detok"
cat "$TMP_DIR"/nyn_training.txt \
    "$TMP_DIR"/data/nyn_backtranslated_training.txt \
    > data/backtranslated/nyn_train.detok
cp "$TMP_DIR"/nyn_dev.txt data/backtranslated/nyn_dev.detok
cp "$TMP_DIR"/nyn_test.txt data/backtranslated/nyn_test.detok

cat "$TMP_DIR"/eng_training.txt \
    "$TMP_DIR"/data/eng_backtranslated_training.txt \
    > data/backtranslated/eng_train.detok
cp "$TMP_DIR"/eng_dev.txt data/backtranslated/eng_dev.detok
cp "$TMP_DIR"/eng_test.txt data/backtranslated/eng_test.detok

echo "learning bpe from (original, bible, backtranslated)"
echo "exporting weights to ./data/all/\$LANG.bpe"
./bpe.sh \
    -l nyn \
    -o data/all/ \
    -d "$TMP_DIR"/nyn_dev.txt \
    -t "$TMP_DIR"/nyn_test.txt \
    "$TMP_DIR"/nyn_training.txt \
    "$TMP_DIR"/nyn-x-bible_parsed.txt \
    "$TMP_DIR"/data/nyn_backtranslated_training.txt
./bpe.sh \
    -l eng \
    -o data/all/ \
    -d "$TMP_DIR"/eng_dev.txt \
    -t "$TMP_DIR"/eng_test.txt \
    "$TMP_DIR"/eng_training.txt \
    "$TMP_DIR"/eng-x-bible-kingjames_parsed.txt \
    "$TMP_DIR"/data/eng_backtranslated_training.txt

echo "copying detokenized originals to ./data/\$LANG.detok"
cat "$TMP_DIR"/nyn_training.txt \
    "$TMP_DIR"/nyn-x-bible_parsed.txt \
    "$TMP_DIR"/data/nyn_backtranslated_training.txt \
    > data/all/nyn_train.detok
cp "$TMP_DIR"/nyn_dev.txt data/all/nyn_dev.detok
cp "$TMP_DIR"/nyn_test.txt data/all/nyn_test.detok

cat "$TMP_DIR"/eng_training.txt \
    "$TMP_DIR"/eng-x-bible-kingjames_parsed.txt \
    "$TMP_DIR"/data/eng_backtranslated_training.txt \
    > data/all/eng_train.detok
cp "$TMP_DIR"/eng_dev.txt data/all/eng_dev.detok
cp "$TMP_DIR"/eng_test.txt data/all/eng_test.detok

echo "finished processing. exported to ./data"
