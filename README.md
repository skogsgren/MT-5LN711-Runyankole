# Runyankole Low-Resource Machine Translation

Project for MT-5LN711 at Uppsala University.

## Prerequisites

```{bash}
pip3 install -r requirements.txt
```

And then copy the data file located on `futurum` at
`/home/gugre01/data.tar.xz` to the `./src/` directory.

## Backtranslation

Backtranslation uses Google Translate to go from English to Kiga, which
is very similar to Runyankole. The backtranslation takes a lot of time
since `deep-translator` uses scraping to interact with Google Translate
(I'm too cheap to pay for the API). Even with maximum batching to avoid
unnecessary HTTP calls (and thus also risking getting banned!) this took
about 6 hours for all the sequences in the training data.

If you want to run the backtranslation yourself, then run it like so:

```{bash}
python3 src/backtranslate_dataset.py \
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
  MOSES dataset convention (i.e. one sequence per line).
- Parses the bible verses in Runyankole and matches them with the
  English King James version according to MOSES convention.
- Trains and applies BPE to those extracted datasets for each respective
  `$METHOD`:
    - `original`: data from sunbird.
    - `bible`: data from sunbird + bible translation.
    - `all`: data from sunbird + bible translation + backtranslated data.
- Copies the extracted datasets to the `./src/data/$METHOD` folder.

It should be run from with the current working directory as `src`, and
the `data.tar.xz` file should be there as well. The total time of the
entire pipeline for every method should only take a minute or two.

To run the preprocessing pipeline simply run:

```{bash}
cd src
./preprocessing_pipeline.sh
```
