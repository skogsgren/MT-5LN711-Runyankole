__doc__ = """ given two bibles formatted with the columns  `verse_number` and `verse_text`, exports that to MOSES formatted dataset (i.e. one row per sequence). One bible is called target and one is called source, where source is assumed to be a subset of target (i.e. from a source language which is lower resource to a target language which is higher resource)."""
__author__ = "Gustaf Gren"

import argparse
from pathlib import Path


def parse_bible(
    source_lines: list[tuple], target_lines: list[tuple], lowercase: bool
) -> list[tuple[str, str]]:
    """given a src and tgt list of tuples[verse_number, verse_text] returns a
    list of tuples of parallel_text between all the verses in both src and
    tgt"""
    target_dict: dict[str, str] = {
        verse_number: verse_text for verse_number, verse_text in target_lines
    }
    parallel_text: list[tuple[str, str]] = []
    for verse_number, verse_text in source_lines:
        if not target_dict.get(verse_number):
            print(f"WARN: {verse_number} not in target")
        if lowercase:
            verse_text = verse_text.lower()
            target_dict[verse_number] = target_dict[verse_number].lower()
        parallel_text.append((verse_text, target_dict[verse_number]))
    return parallel_text


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="given two bibles formatted with the columns `verse_number` and `verse_text` exports that to MOSES formatted dataset."
    )
    parser.add_argument(
        "--source",
        help="path to source bible",
    )
    parser.add_argument(
        "--target",
        help="path to target bible",
    )
    parser.add_argument(
        "--output_dir",
        help="path to output directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite duplicate files in output directory if it exists",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="lowercase each line",
    )

    args: argparse.Namespace = parser.parse_args()

    source: Path = Path(args.source)
    target: Path = Path(args.target)

    with open(source, "r") as f:
        source_lines: list[tuple] = [
            tuple(x.split("\t")) for x in f.readlines() if x[0] != "#"
        ]
    with open(target, "r") as f:
        target_lines: list[tuple] = [
            tuple(x.split("\t")) for x in f.readlines() if x[0] != "#"
        ]

    parsed_source: list[str]
    parsed_target: list[str]
    parallel_text: list[tuple[str, str]] = parse_bible(
        source_lines, target_lines, args.lowercase
    )

    out: Path = Path(args.output_dir)
    source_out: Path = out / (source.stem + "_parsed.txt")
    if source_out.exists() and not args.overwrite:
        print(f"ERR:  {source_out} already exists. use --overwrite if desired")
        exit(1)
    target_out: Path = out / (target.stem + "_parsed.txt")
    if target_out.exists() and not args.overwrite:
        print(f"ERR:  {target_out} already exists. use --overwrite if desired")
        exit(1)

    with open(source_out, "w") as srcf, open(target_out, "w") as tgtf:
        for source_line, target_line in parallel_text:
            srcf.write(source_line)
            tgtf.write(target_line)
