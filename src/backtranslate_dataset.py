__doc__ = """
contains code for getting backtranslations for a given file in the desired
source language. it translates in batches due to the 5000 word limit on Google
Translate.
"""
__author__ = "Gustaf Gren"

from tqdm import tqdm
from pathlib import Path
import argparse
from deep_translator import GoogleTranslator
from deep_translator.exceptions import TranslationNotFound
from dataclasses import dataclass
import json


@dataclass
class Batch:
    """dataclass wrapper around a batch for use in calling Google Translate
    since it has a character limit of 5k"""

    char_count: int
    texts: list[str]


def translate(
    lines: list[str], src_lang: str, tgt_lang: str, progress_path: Path
) -> list[str]:
    """given a list of strings, translates them in batches and returns them as
    a list"""
    translator: GoogleTranslator = GoogleTranslator(source=src_lang, target=tgt_lang)

    with open(progress_path) as f:
        progress: dict = json.load(f)

    # generate batches even if progress is provided since it's quick anyways
    lines.reverse()
    batches: dict[int, Batch] = {}
    current_batch: Batch = Batch(0, [])
    n: int = 0
    while lines:
        line: str = lines.pop()
        if (current_batch.char_count + len(line)) > 5000:
            batches[n] = current_batch
            n += 1
            current_batch = Batch(0, [])
        current_batch.char_count += len(line)
        current_batch.texts.append(line)
    # have to add the last batch since it is smaller than 5000 char length
    batches[n] = current_batch

    # filter out already processed batches for accurate tqdm estimations
    filtered_batches: dict[str, Batch] = {}
    for i, batch in batches.items():
        if progress.get(str(i)):
            continue
        filtered_batches[str(i)] = batch

    for i, batch in tqdm(filtered_batches.items()):
        try:
            translated_batch: list[str] = translator.translate_batch(batch.texts)
        # rudimentary recursion (w/o checks!) to keep going if translation fails
        except TranslationNotFound:
            return translate(
                lines=lines,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                progress_path=progress_path,
            )
        # save progress for each batch to increase stability
        with open(progress_path, "r") as f:
            progress = json.load(f)
        progress[i] = translated_batch
        with open(progress_path, "w") as f:
            json.dump(progress, f)

    translated_lines: list[str] = []
    for _, batch in progress.items():
        translated_lines += batch
    return translated_lines


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="given a input file, gets backtranslations using Google Translate"
    )
    parser.add_argument(
        "--input",
        help="path to input file (one sequence per line)",
    )
    parser.add_argument(
        "--source_lang",
        default="en",
        help="language code for source language",
    )
    parser.add_argument(
        "--target_lang",
        default="kiga",
        help="language code for target langugage",
    )
    parser.add_argument(
        "--output",
        help="path to where output should be stored",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite if output already exists",
    )

    args: argparse.Namespace = parser.parse_args()

    inp: Path = Path(args.input)
    out: Path = Path(args.output)
    if out.exists() and not args.overwrite:
        print(f"ERR: {out} already exists and overwrite is not specified.")
        exit(1)
    progress_path: Path = out.parent / (out.stem + "_progress.json")
    if not progress_path.exists():
        with open(progress_path, "w") as f:
            json.dump({}, f)

    with open(inp) as f:
        lines: list[str] = f.readlines()

    translated_lines: list[str] = translate(
        lines=lines,
        src_lang=args.source_lang,
        tgt_lang=args.target_lang,
        progress_path=progress_path,
    )

    with open(out, "w") as f:
        f.writelines([x + "\n" for x in translated_lines])
