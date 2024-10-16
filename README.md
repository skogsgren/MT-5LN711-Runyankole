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
4. Copies the extracted datasets to the ./data folder

It should be run from with the current working directory as `src`, and the
`data.tar` file should be there as well.

`data.tar` can be found on `futurum` at `/home/gugre01/data.tar`.

Running the preprocessing stage thus is run like follows (assuming you are in
the `src` directory):

```{bash}
./preprocessing_pipeline.sh
```
