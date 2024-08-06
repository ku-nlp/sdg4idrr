import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from first_party_modules.utils import PDTB3Utils


def main():
    parser = ArgumentParser(description="script to rebuild synthetic data")
    parser.add_argument("TRAIN", type=Path, help="path to train.jsonl")
    parser.add_argument("ANNOT", type=Path, help="path to annotations")
    parser.add_argument("OUT_DIR", type=Path, help="path to output directory")
    args = parser.parse_args()

    pdtb3_utils = PDTB3Utils(level="l2")
    examples = pdtb3_utils.load_examples(args.TRAIN)
    args.OUT_DIR.mkdir(parents=True, exist_ok=True)
    for annot_file in args.ANNOT.glob("*.jsonl.gz"):
        example_id2annots = defaultdict(list)
        for annot in pdtb3_utils.load_annotations(annot_file):
            example_id = annot.pop("example_id")
            example_id2annots[example_id].append(annot)
        with (args.OUT_DIR / f"{annot_file.name}").open(mode="w") as f:
            for example in examples:
                for annot in example_id2annots[example.example_id]:
                    synthetic_example = {
                        "example_id": example.example_id,
                        "arg1": example.arg1,
                        **annot,
                    }
                    f.write(json.dumps(synthetic_example) + "\n")


if __name__ == "__main__":
    main()
