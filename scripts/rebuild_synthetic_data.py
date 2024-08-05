import gzip
import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Literal

from first_party_modules.constants import (
    PDTB3_L1_SENSE2DEFINITION,
    PDTB3_L1_SENSES,
    SELECTED_PDTB3_L2_SENSE2DEFINITION,
    SELECTED_PDTB3_L2_SENSES,
)
from first_party_modules.utils import ObjectHook


class PDTB3Utils:
    def __init__(self, level: Literal["l2", "l1"] = "l2") -> None:
        self.level = level
        if level == "l2":
            self.senses = SELECTED_PDTB3_L2_SENSES
            self.sense2definition = SELECTED_PDTB3_L2_SENSE2DEFINITION
        elif level == "l1":
            self.senses = PDTB3_L1_SENSES
            self.sense2definition = PDTB3_L1_SENSE2DEFINITION
        else:
            raise ValueError("invalid level")

    @staticmethod
    def load_examples(in_file: Path) -> list[ObjectHook]:
        return [json.loads(line, object_hook=ObjectHook) for line in in_file.read_text().splitlines()]

    @staticmethod
    def load_annotations(in_file: Path) -> list[ObjectHook]:
        with gzip.open(in_file, mode="rb") as f:
            return [json.loads(line, object_hook=ObjectHook) for line in f.readlines()]

    def get_senses(self, example: ObjectHook) -> list[str]:
        senses = []
        for indices in ["1a", "1b", "2a", "2b"]:
            sense = example[f"sclass{indices}_{self.level}"]
            if sense in self.senses and sense not in senses:
                senses.append(sense)
        return senses

    def get_sense2examples(self, examples: list[ObjectHook]) -> dict[str, list[ObjectHook]]:
        sense2examples = defaultdict(list)
        for example in examples:
            for sense in self.get_senses(example):
                sense2examples[sense].append(example)
        return sense2examples


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
