# Runyankole Low-Resource Machine Translation

Project for MT-5LN711 at Uppsala University.

## Prerequisites

In order to run model training, we first have to get our dependencies and our
data.

To get our dependencies simply use `pip install` (preferably in a `venv` or
`conda` environment):

```{bash}
pip3 install -r requirements.txt
```

The data is available unprocessed on `futurum` at `/home/gugre01/data.tar.xz`.
Copy that file to the `./src` directory.

To get it processed a preprocessing pipeline is available, written as a shell
script. The preprocessing pipeline makes sure to process the data in the
correct order (see details below). To run the preprocessing pipeline run:

```{bash}
cd src
./preprocessing_pipeline.sh
```

You will end up with a folder inside `src` called `data` with the following
structure:

```
.
├── all
│   ├── eng.bpe
│   ├── eng_dev.bpe
│   ├── eng_dev.detok
│   ├── eng_test.bpe
│   ├── eng_test.detok
│   ├── eng_train.bpe
│   ├── eng_train.detok
│   ├── nyn.bpe
│   ├── nyn_dev.bpe
│   ├── nyn_dev.detok
│   ├── nyn_test.bpe
│   ├── nyn_test.detok
│   ├── nyn_train.bpe
│   └── nyn_train.detok
├── backtranslated
│   ├── eng.bpe
│   ├── eng_dev.bpe
│   ├── eng_dev.detok
│   ├── eng_test.bpe
│   ├── eng_test.detok
│   ├── eng_train.bpe
│   ├── eng_train.detok
│   ├── nyn.bpe
│   ├── nyn_dev.bpe
│   ├── nyn_dev.detok
│   ├── nyn_test.bpe
│   ├── nyn_test.detok
│   ├── nyn_train.bpe
│   └── nyn_train.detok
├── bible
│   ├── eng.bpe
│   ├── eng_dev.bpe
│   ├── eng_dev.detok
│   ├── eng_test.bpe
│   ├── eng_test.detok
│   ├── eng_train.bpe
│   ├── eng_train.detok
│   ├── nyn.bpe
│   ├── nyn_dev.bpe
│   ├── nyn_dev.detok
│   ├── nyn_test.bpe
│   ├── nyn_test.detok
│   ├── nyn_train.bpe
│   └── nyn_train.detok
└── original
    ├── eng.bpe
    ├── eng_dev.bpe
    ├── eng_dev.detok
    ├── eng_test.bpe
    ├── eng_test.detok
    ├── eng_train.bpe
    ├── eng_train.detok
    ├── nyn.bpe
    ├── nyn_dev.bpe
    ├── nyn_dev.detok
    ├── nyn_test.bpe
    ├── nyn_test.detok
    ├── nyn_train.bpe
    └── nyn_train.detok

5 directories, 56 files
```

These subdirectories of `./src/data` are then finished datasets which can be used
in `OpenNMT`. The `$LANG.bpe` files are the learned BPE weights.

## Backtranslation

Backtranslation uses Google Translate to go from English to Kiga, which
is very similar to Runyankole. The backtranslation takes a lot of time
since `deep-translator` uses scraping to interact with Google Translate
(I'm too cheap to pay for the API). Even with maximum batching to avoid
unnecessary HTTP calls (and thus also risking getting banned!) this took
about 6 hours for all the sequences in the training data.

Before doing that on Kiga, however, you need to patch the `deep-translator`
`constants.py` file to include Kiga in it's predefined list of languages. Use
the diff provided in the repo. For example, if you're using `venv` located at
`$VENV` then the following command will patch it:

```{bash}
patch $VENV/lib/python3.11/site-packages/deep_translator/constants.py \
    < ./deep-translator_constants.diff
```

If you want to run the backtranslation yourself, then run it like so:

```{bash}
python3 ./src/backtranslate_dataset.py \
    --input PATH_TO_INPUT \
    --source_lang LANG_CODE \
    --target_lang LANG_CODE \
    --output PATH_TO_DESIRED_OUTPUT
```

## Preprocessing

The `data.tar.xz` file is unprocessed and contains:
1. The provided project data from the Sunbird-dataset
2. A Nkore-Kiga Bible translation, and the New King James English Bible,
3. The backtranslated output from the provided project data using the
   `./backtranslate_dataset.py` script.

In order to process it, there is a bash-script
`./src/preprocessing_pipeline.sh` which does the following:

- Parses the sunbird dataset and extracts Runyankole-> Eng according to
  MOSES dataset convention (i.e. one sequence per line) using a custom Python
  script (`./src/parse_sunbird.py`).
- Parses the bible verses in Runyankole and matches them with the
  English King James version according to MOSES convention using a custom
  Python script (`./src/parse_bible.py`)
- Trains and applies BPE to those extracted datasets for each respective
  `$METHOD` using an external script from `OpenNMT` called using a bash script
  (`./src/bpe.sh`):
    - `original`: data from sunbird.
    - `bible`: data from sunbird + bible translation.
    - `backtranslated`: data from sunbird + backtranslated data.
    - `all`: data from sunbird + bible translation + backtranslated data.
- Copies the extracted datasets to the `./src/data/$METHOD` folder.

It should be run from with the current working directory as `src`, and
the `data.tar.xz` file should be there as well.

The total time of the entire pipeline for every method should only take a
minute or two.
