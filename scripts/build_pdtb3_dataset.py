import json
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from collections import defaultdict
from enum import IntEnum
from pathlib import Path
from textwrap import dedent

import numpy as np

from first_party_modules.constants import PDTB3_L1_SENSES, SELECTED_PDTB3_L2_SENSES
from first_party_modules.progress_bar import tqdm


# cf. https://catalog.ldc.upenn.edu/docs/LDC2019T05/PDTB3-Annotation-Manual.pdf (pp. 49-50)
class Index(IntEnum):
    relation_type = 0
    conn1 = 7
    sclass1a = 8
    sclass1b = 9
    conn2 = 10
    sclass2a = 11
    sclass2b = 12
    arg1_span_list = 14
    arg2_span_list = 20


def get_argument(raw_text: str, arg_span_list: str) -> str:
    buffer = []
    for span in arg_span_list.split(";"):
        if span == "":
            continue
        begin, end = map(int, span.split(".."))
        buffer.append(raw_text[begin : end + 1])
    return " ".join(buffer)


# cf. https://github.com/najoungkim/pdtb3/blob/master/preprocess/preprocess_pdtb3.py
def get_section2examples(args: Namespace) -> dict[str, list[dict[str, str]]]:
    article_id2raw_file = {raw_file.stem: raw_file for raw_file in (args.IN_ROOT / "raw").glob("**/wsj_*")}

    example_id = 0
    section2examples = defaultdict(list)
    for gold_file in tqdm((args.IN_ROOT / "gold").glob("**/wsj_*")):
        article_id = gold_file.stem
        for i, gold_line in enumerate(gold_file.read_text(encoding="latin-1").splitlines()):
            values = gold_line.split("|")
            if values[Index.relation_type] != "Implicit":
                continue

            raw_text = article_id2raw_file[article_id].read_text(encoding="latin-1")

            example = {
                "example_id": f"{article_id}-{i}",
                "article_id": article_id,
                "arg1": get_argument(raw_text, values[Index.arg1_span_list]),
                "arg2": get_argument(raw_text, values[Index.arg2_span_list]),
                "conn1": values[Index.conn1],
                "conn2": values[Index.conn2],
            }
            # sclass1b and sclass2b are always empty
            for sense_index in [
                Index.sclass1a,
                Index.sclass1b,
                Index.sclass2a,
                Index.sclass2b,
            ]:
                classes = values[sense_index].split(".")
                example.update(
                    {
                        f"{sense_index.name}_l1": ".".join(classes[:1]),
                        f"{sense_index.name}_l2": ".".join(classes[:2]),
                        f"{sense_index.name}_l3": ".".join(classes[:3]) if len(classes) == 3 else "",
                    }
                )
            section2examples[gold_file.parent.stem].append(example)
            example_id += 1
    return section2examples


def gather_examples(
    section_indices: np.array,
    section2examples: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    return [e for i in section_indices for e in section2examples[f"{i:02}"]]


def save_examples(out_file: Path, examples: list[dict[str, str]], aid_file: Path) -> None:
    article_id2examples = defaultdict(list)
    for example in examples:
        article_id2examples[example["article_id"]].append(example)

    article_ids = [line for line in aid_file.read_text().splitlines()]
    with out_file.open(mode="w") as f:
        for article_id in article_ids:
            for example in article_id2examples[article_id]:
                f.write(json.dumps(example) + "\n")


def get_stats(examples: list[dict[str, str]]) -> dict[str, int]:
    ctr = defaultdict(int)
    for example in examples:
        for sense in {
            example[f"sclass{indices}_{level}"] for indices in ["1a", "1b", "2a", "2b"] for level in ["l1", "l2"]
        }:
            ctr[sense] += 1
    return {
        "num_examples": len(examples),
        "num_L1_labels": sum(ctr[s] for s in PDTB3_L1_SENSES),
        "num_L2_labels": sum(ctr[s] for s in SELECTED_PDTB3_L2_SENSES),
        **{f"num_{s}": ctr[s] for s in PDTB3_L1_SENSES + SELECTED_PDTB3_L2_SENSES},
    }


# cf. https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00142/43281/One-Vector-is-Not-Enough-Entity-Augmented
def ji_split(out_root: Path, section2examples: dict[str, list[dict[str, str]]], aid_dir: Path) -> None:
    expected_ji_stats = {
        "train": {
            "num_examples": 17052,
            "num_L1_labels": 17854,
            "num_L2_labels": 17788,
            "num_Temporal": 1442,
            "num_Contingency": 5892,
            "num_Comparison": 1935,
            "num_Expansion": 8585,
            "num_Temporal.Synchronous": 435,
            "num_Temporal.Asynchronous": 1007,
            "num_Contingency.Cause": 4475,
            "num_Contingency.Cause+Belief": 159,
            "num_Contingency.Purpose": 1092,
            "num_Contingency.Condition": 150,
            "num_Comparison.Concession": 1164,
            "num_Comparison.Contrast": 741,
            "num_Expansion.Conjunction": 3586,
            "num_Expansion.Equivalence": 254,
            "num_Expansion.Instantiation": 1166,
            "num_Expansion.Level-of-detail": 2601,
            "num_Expansion.Manner": 615,
            "num_Expansion.Substitution": 343,
        },
        "dev": {
            "num_examples": 1647,
            "num_L1_labels": 1697,
            "num_L2_labels": 1686,
            "num_Temporal": 138,
            "num_Contingency": 577,
            "num_Comparison": 201,
            "num_Expansion": 781,
            "num_Temporal.Synchronous": 33,
            "num_Temporal.Asynchronous": 105,
            "num_Contingency.Cause": 449,
            "num_Contingency.Cause+Belief": 13,
            "num_Contingency.Purpose": 96,
            "num_Contingency.Condition": 18,
            "num_Comparison.Concession": 105,
            "num_Comparison.Contrast": 91,
            "num_Expansion.Conjunction": 299,
            "num_Expansion.Equivalence": 25,
            "num_Expansion.Instantiation": 118,
            "num_Expansion.Level-of-detail": 274,
            "num_Expansion.Manner": 28,
            "num_Expansion.Substitution": 32,
        },
        "test": {
            "num_examples": 1471,
            "num_L1_labels": 1538,
            "num_L2_labels": 1530,
            "num_Temporal": 151,
            "num_Contingency": 529,
            "num_Comparison": 162,
            "num_Expansion": 696,
            "num_Temporal.Synchronous": 43,
            "num_Temporal.Asynchronous": 108,
            "num_Contingency.Cause": 406,
            "num_Contingency.Cause+Belief": 15,
            "num_Contingency.Purpose": 89,
            "num_Contingency.Condition": 15,
            "num_Comparison.Concession": 97,
            "num_Comparison.Contrast": 63,
            "num_Expansion.Conjunction": 237,
            "num_Expansion.Equivalence": 30,
            "num_Expansion.Instantiation": 128,
            "num_Expansion.Level-of-detail": 214,
            "num_Expansion.Manner": 53,
            "num_Expansion.Substitution": 32,
        },
    }

    section_indices = np.arange(23)
    # Ji split: train/dev/test = 2-20/0-1/21-22
    dataset = {
        "train": gather_examples(section_indices[2:21], section2examples),
        "dev": gather_examples(section_indices[0:2], section2examples),
        "test": gather_examples(section_indices[21:23], section2examples),
    }
    out_dir = out_root / "ji"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, examples in dataset.items():
        save_examples(out_dir / f"{split}.jsonl", examples, aid_dir / f"{split}.txt")
        for key, value in get_stats(examples).items():
            assert value == expected_ji_stats[split][key], f"{split}-{key} isn't reproduced"


def main():
    parser = ArgumentParser(
        description="script for building PDTB3 dataset",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "IN_ROOT",
        type=Path,
        help=dedent(
            """\
            path to input root
            expected directory structure
                input root
                ┣━ raw
                ┃  ┣━ 00
                ┃  ┃  ┣━ wsj_0001
                ┃  ┃  ...
                ┃  ┃  ┗━ wsj_0099
                ┃  ...
                ┃  ┗━ 24
                ┃     ┣━ wsj_2400
                ┃     ...
                ┃     ┗━ wsj_2499
                ┗━ gold
                   ┣━ 00
                   ┃  ┣━ wsj_0001
                   ┃  ...
                   ┃  ┗━ wsj_0098
                   ...
                   ┗━ 24
                      ┣━ wsj_2400
                      ...
                      ┗━ wsj_2454
            """
        ),
    )
    parser.add_argument("OUT_ROOT", type=Path, help="path to output root")
    parser.add_argument("--aid-dir", type=Path, help="path to article id directory for reproduction")
    args = parser.parse_args()

    section2examples = get_section2examples(args)
    ji_split(args.OUT_ROOT, section2examples, args.aid_dir)


if __name__ == "__main__":
    main()
