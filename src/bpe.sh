#!/bin/bash
# AUTHOR=Gustaf Gren
# DOC=shell script for training and encoding BPE for provided data files

set -e # we want script to exit on any failure

display_help() {
    echo "Usage: $0 -l [LANG] -o [DIR] -d [DEV_FILE] -t [TEST_FILE] [FILES*]"
    echo "Description:"
    echo " trains and encodes BPE for provided FILES*"
    echo ""
    echo "Arguments:"
    echo " LANG language prefix for filenames, e.g. nyn_train.txt=nyn"
    echo " FILES* is a list of files to be concatenated for use in training, separated by space"
    echo " DIR output directory (with trailing /) for data files"
    echo "    will export:"
    echo "      DIR/LANG.bpe for BPE weights"
    echo "      DIR/LANG_train.bpe for concatenated training set with applied BPE"
    echo "      DIR/LANG_test.bpe for provided test set with applied BPE"
    echo "      DIR/LANG_dev.bpe for provided dev set with applied BPE"
    echo "    it will also copy the originals to DIR/original/"
    exit 1
}

while getopts "hl:o:d:t:" x; do
    case "${x}" in
        h)
            display_help
            ;;
        l)
            LANG_PREFIX=${OPTARG}
            ;;
        o)
            OUTPUT=${OPTARG}
            ;;
        d)
            DEV=${OPTARG}
            ;;
        t)
            TEST=${OPTARG}
            ;;
        *)
            display_help
            ;;
    esac
done

# we need a tmp file to store concatenated training data for BPE creation
TRAIN="$(mktemp)"
shift $((OPTIND -1)) # removes opts from argument list
for i in "$@"
do
    cat "$i" >>"$TRAIN"
done

if [[ -z "$LANG_PREFIX" && -z "$OUTPUT" ]]; then
    echo "ERR: -l and -o must be provided"
    display_help
fi

mkdir -p "$OUTPUT"

echo concatenated input to BPE is "$(wc -l "$TRAIN")" sequences long

BPE="${OUTPUT}""${LANG_PREFIX}".bpe
echo learning BPE pairs from "$TRAIN", saving to "$BPE"
python3 external_scripts/learn_bpe.py \
    --input "$TRAIN" \
    --output "$BPE"

echo "encoding training set ($TRAIN) using learnt BPE pairs to ${OUTPUT}${LANG_PREFIX}_train.bpe"
python3 external_scripts/apply_bpe.py \
    --input "$TRAIN" \
    --codes "$BPE" \
    --output "${OUTPUT}""${LANG_PREFIX}"_train.bpe

echo "encoding dev set ($DEV) using learnt BPE pairs to ${OUTPUT}${LANG_PREFIX}_dev.bpe"
python3 external_scripts/apply_bpe.py \
    --input "$DEV" \
    --codes "$BPE" \
    --output "${OUTPUT}""${LANG_PREFIX}"_dev.bpe

echo "encoding test set ($TEST) using learnt BPE pairs to ${OUTPUT}${LANG_PREFIX}_test.bpe"
python3 external_scripts/apply_bpe.py \
    --input "$TEST" \
    --codes "$BPE" \
    --output "${OUTPUT}""${LANG_PREFIX}"_test.bpe
