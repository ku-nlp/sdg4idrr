import json
from argparse import ArgumentParser
from pathlib import Path

from first_party_modules.utils import PDTB3Utils, set_seed


def main():
    parser = ArgumentParser(description="script to organize synthetic data")
    parser.add_argument("FILTERED", type=Path, help="path to filtered synthetic data directory")
    parser.add_argument("DEV_PRED", type=Path, help="path to dev_pred.jsonl")
    parser.add_argument("OUT_FILE", type=Path, help="path to output file")
    parser.add_argument("--top-k", default=3, type=int, help="how many confusing sense pairs to extract")
    args = parser.parse_args()

    set_seed(seed=0)

    pdtb3_utils = PDTB3Utils(level="l2")

    norm_conf_mtx = pdtb3_utils.compute_normalized_confusion_matrix(pdtb3_utils.load_examples(args.DEV_PRED))
    confusing_sense_pairs = pdtb3_utils.get_top_k_confusing_sense_pairs(norm_conf_mtx, top_k=args.top_k)

    filtered_synthetic_examples = []
    for true, pred in confusing_sense_pairs:
        filtered_synthetic_examples += pdtb3_utils.load_examples(args.FILTERED / f"{true}_vs_{pred}.jsonl")

    with args.OUT_FILE.open(mode="w") as f:
        for filtered_synthetic_example in filtered_synthetic_examples:
            f.write(json.dumps(filtered_synthetic_example) + "\n")


if __name__ == "__main__":
    main()
