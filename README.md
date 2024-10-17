# Runyankole Low-Resource Machine Translation

Project for MT-5LN711 at Uppsala University.

## Prerequisites

```{bash}
pip3 install -r requirements.txt
```

## Preprocessing

The script in `src/preprocessing_pipeline.sh` does the following:

1. Parses the sunbird dataset and extracts Runyankole-> Eng according to MOSES
   dataset convention (i.e. one sequence per line).
2. Parses the bible verses in Runyankole and matches them with the English King
   James version according to MOSES convention.
3. Trains and applies BPE to those extracted datasets.
4. Copies the extracted datasets to the `./src/data` folder

It should be run from with the current working directory as `src`, and the
`data.tar` file should be there as well.

`data.tar` can be found on `futurum` at `/home/gugre01/data.tar`.

Running the preprocessing stage thus is run like follows (assuming you are in
the `src` directory):

```{bash}
./src/preprocessing_pipeline.sh
```

### Backtranslation

Backtranslation uses Google Translate to go from English to Kiga, which is very
similar to Runyankole. The backtranslation takes a lot of time since
`deep-translator` uses scraping to interact with Google Translate (I'm too
cheap to pay for the API). Even with maximum batching to avoid unnecessary HTTP
calls (and thus also risking getting banned!) this took about 6 hours for all
the sequences in the training data.

The backtranslated data is located on `futurum` at `/home/gugre01/data-bt.tar`

If you want to run the backtranslation yourself, then run it like so:

```{bash}
python3 src/backtranslate_dataset.py \
    --input PATH_TO_INPUT \
    --source_lang LANG_CODE \
    --target_lang LANG_CODE \
    --output PATH_TO_DESIRED_OUTPUT
```

Then to run `BPE` run the following command:

```{bash}
python3 src/external_scripts/apply_bpe.py \
    --input PATH_TO_BACKTRANSLATED_TEXT \
    --codes PATH_TO_BPE_WEIGHTS \
    --output PATH_TO_DESIRED_OUTPUT
```

If you ran the `./src/preprocessing_pipeline.sh` then the BPE weights will be
in the `./src/data` directory as `LANG.bpe`.

### Dataset composition

Using `coreutils` you can compose the datasets according to desired
combination.  For example, if you want to include only the backtranslated data
in the training data:

```{bash}
cat src/data/eng_training_bpe.txt src/data/eng_backtranslated_training_bpe.txt \
    >./eng_concat_training_bpe.txt
cat src/data/nyn_training_bpe.txt src/data/nyn_backtranslated_training_bpe.txt \
    >./nyn_concat_training_bpe.txt
```
