__doc__ = """ this file handles the parsing of sunbird (SALT) data file and exports it to MOSES format (i.e. one sequence per row)"""
__author__ = "Gustaf Gren"

import argparse
from pathlib import Path
from ast import literal_eval


def parse_sunbird(lines: list[str], opts: dict) -> tuple[list[str], list[str]]:
    """given list of lines, return tuple of lines of desired languages (nyn, eng)"""
    eng_text: list[str] = []
    nyn_text: list[str] = []

    line: dict
    for line in [literal_eval(x) for x in lines]:
        nyn_line = line["nyn_text"] + "\n"
        eng_line = line["eng_text"] + "\n"

        if opts["split_punctuation"]:
            nyn_line = (
                nyn_line.translate(
                    str.maketrans({p: f" {p} " for p in ["?", "!", ".", ","]})
                ).rstrip()
                + "\n"
            )
            eng_line = (
                eng_line.translate(
                    str.maketrans({p: f" {p} " for p in ["?", "!", ".", ","]})
                ).rstrip()
                + "\n"
            )
        if opts["lowercase"]:
            nyn_line = nyn_line.lower()
            eng_line = eng_line.lower()

        nyn_text.append(nyn_line)
        eng_text.append(eng_line)
    return (nyn_text, eng_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse and tokenize sunbird data")
    parser.add_argument(
        "--input",
        help="path to input file",
    )
    parser.add_argument(
        "--output",
        help="path to desired output file (format: if args.output is e.g."
        "train.txt then the output will be (nyn_train.txt and eng_train.txt)"
        "respectively.)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite output if it already exists",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="lowercase parsed text",
    )
    parser.add_argument(
        "--split_punctuation",
        action="store_true",
        help="split up punctuation as its own token [?!.,]",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        lines: list[str] = f.readlines()
    parsed_lines = parse_sunbird(
        lines,
        {
            "lowercase": args.lowercase,
            "split_punctuation": args.split_punctuation,
        },
    )

    out: Path = Path(args.output)
    out_stem: str = out.stem

    nyn_name = out.with_name("nyn_" + out.stem + ".txt")
    if not args.overwrite:
        assert not nyn_name.exists()
    with open(nyn_name, "w") as f:
        f.writelines(parsed_lines[0])

    eng_name: Path = out.with_name("eng_" + out.stem + ".txt")
    if not args.overwrite:
        assert not eng_name.exists()
    with open(eng_name, "w") as f:
        f.writelines(parsed_lines[1])
