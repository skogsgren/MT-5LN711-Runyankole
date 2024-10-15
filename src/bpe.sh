#!/bin/bash
# AUTHOR=Gustaf Gren
# DOC=shell script for training and encoding BPE for provided data files

display_help() {
    echo "Usage: $0 [LANG] [DIR] [FILES*]"
    echo "Description:"
    echo " trains and encodes BPE for provided FILES*"
    echo ""
    echo "Arguments:"
    echo " LANG language prefix for filenames, e.g. nyn_train.txt=nyn"
    echo " DIR input and output directory (with trailing /) for data files"
    echo "     will export:"
    echo "      LANG.bpe for BPE weights"
    echo "      FILE.bpe for each respective file (NOTE: . in filenames messes with filename parsing)"
    echo " FILES* is a list of files to be concatenated, separated by space"
    exit 1
}

if [[ "$1" == "-h" ]]; then
    display_help
fi

LANG="$1"
OUTPUT="$2"
mkdir -p "$OUTPUT"

# we need a tmp file to store the concatenated data for BPE creation
TMP_IN="$(mktemp)"
for i in "$@"
do
    if [[ "$i" == "$1" ]] || [[ "$i" == "$2" ]];then
        continue
    fi
    cat "$i" >>"$TMP_IN"
done
echo concatenated file is "$(wc -l "$TMP_IN")" sequences long

BPE="$OUTPUT""$LANG".bpe
echo learning BPE pairs from "$TMP_IN", saving to "$BPE"
external_scripts/learn_bpe.py \
    --input "$TMP_IN" \
    --output "$BPE"

for i in "$@"
do
    if [[ "$i" == "$1" ]] || [[ "$i" == "$2" ]];then
        continue
    fi

    OUT="$(echo "$i" | awk -F "/" '{print $NF}' | awk -F "." '{print $1}')"
    OUT="$OUT"_bpe.txt

    echo encoding "$i" using learnt BPE pairs to "$OUTPUT""$OUT"
    external_scripts/apply_bpe.py \
        --input "$i" \
        --codes "$BPE" \
        --output "$OUTPUT""$OUT"
done
