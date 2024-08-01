import json
from argparse import ArgumentParser
from pathlib import Path
from time import sleep
# from random import sample

from first_party_modules.openai_utils import BaseOpenAIUtils, set_seed
from first_party_modules.progress_bar import tqdm
from first_party_modules.utils import ObjectHook


class OpenAIUtils(BaseOpenAIUtils):
    def get_messages(
        self,
        arg_pair: str,
        nearest_neighbors: list[ObjectHook],
        pred: str,
    ) -> list[dict[str, str]]:
        demonstrations = "\n".join(f"{e.arg1} - {e.arg2}" for e in nearest_neighbors)
        user_prompt = (
            f'Given two arguments, the relation "{pred}" is defined as "{self.sense2definition[pred]}".\n'
            f'Here are examples that have the relation "{pred}":\n'
            f"{demonstrations}\n"
            f'Please answer whether the two arguments "{arg_pair}" have the relation "{pred}" or not.'
            'An answer must end with "No." or "Yes.".'
        )

        if self.monitor["num_examples"] < 3:
            print(
                "---------- confirm user prompt ----------\n"
                f"{user_prompt}\n"
                "--------------------"
            )
        self.monitor["num_examples"] += 1
        self.monitor["max_prompt_tokens"] = max(
            len(self.encoding.encode(user_prompt)), self.monitor["max_prompt_tokens"]
        )

        return [
            {
                "role": "user",
                "content": user_prompt,
            }
        ]


def save_filtered_synthetic_examples(
    filtered_synthetic_examples: list[tuple[ObjectHook, int, str]], out_file: Path
) -> None:
    with out_file.open(mode="w") as f:
        true, pred = out_file.stem.split("_vs_")
        for synthetic_example, j, completion in filtered_synthetic_examples:
            arg2 = synthetic_example.completion.split("- ")[j]
            json_line = json.dumps(
                {
                    "arg1": synthetic_example.arg1,
                    "arg2": arg2,
                    "completion": completion,
                    "true": true,
                    "pred": pred,
                }
            )
            f.write(json_line + "\n")


def main():
    parser = ArgumentParser(description="script to filter synthetic argument pairs")
    parser.add_argument("TRAIN", type=Path, help="path to train.jsonl")
    parser.add_argument("DEV_PRED", type=Path, help="path to dev_pred.jsonl")
    parser.add_argument(
        "SYNTH_DATA_DIR", type=Path, help="path to synthetic data directory"
    )
    parser.add_argument("OUT_DIR", type=Path, help="path to output directory")
    parser.add_argument(
        "--top-k", default=3, type=int, help="how many confusing sense pairs to extract"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="whether to perform a dry run"
    )
    args = parser.parse_args()

    set_seed(seed=0)

    num_few_shot_examples = 8
    openai_utils = OpenAIUtils(
        model_name="gpt-4-0613",
        max_tokens=4096,
        max_completion_tokens=128,
        level="l2",
        num_few_shot_examples=num_few_shot_examples,
    )

    sense2train_examples = openai_utils.get_sense2examples(
        openai_utils.load_examples(args.TRAIN)
    )
    norm_conf_mtx = openai_utils.compute_normalized_confusion_matrix(
        openai_utils.load_examples(args.DEV_PRED)
    )
    confusing_sense_pairs = openai_utils.get_top_k_confusing_sense_pairs(
        norm_conf_mtx, top_k=args.top_k
    )

    args.OUT_DIR.mkdir(parents=True, exist_ok=True)

    num_filtered_synthetic_examples = 0
    for true, pred in confusing_sense_pairs:
        stem = f"{true}_vs_{pred}"
        if (args.OUT_DIR / f"{stem}.jsonl").exists():
            continue
        synthetic_examples = openai_utils.load_examples(
            args.SYNTH_DATA_DIR / f"{true}.jsonl"
        )

        arg_pairs = []
        indices = [0]
        for synthetic_example in synthetic_examples:
            arg_pairs += [
                f"{synthetic_example.arg1} - {arg2}"
                for arg2 in synthetic_example.completion.split("- ")
            ]
            indices.append(len(arg_pairs))

        nearest_neighbors_list = openai_utils.get_nearest_neighbors_list(
            arg_pairs, sense2train_examples[pred]
        )

        filtered_synthetic_examples = []
        bar = tqdm(synthetic_examples)
        for i, synthetic_example in enumerate(bar):
            span = slice(indices[i], indices[i + 1])
            for j, (arg_pair, nearest_neighbors) in enumerate(
                zip(arg_pairs[span], nearest_neighbors_list[span])
            ):
                # random_samples = sample(sense2train_examples[pred], num_few_shot_examples)
                messages = openai_utils.get_messages(arg_pair, nearest_neighbors, pred)
                if args.dry_run is True:
                    continue
                response = openai_utils.get_response(messages)
                if response is None:
                    continue
                filtered_synthetic_examples.append(
                    (synthetic_example, j, response["choices"][0]["message"]["content"])
                )
                if (
                    response["choices"][0]["message"]["content"]
                    .rstrip()
                    .endswith("No.")
                ):
                    num_filtered_synthetic_examples += 1
                sleep(1)
                bar.set_postfix(
                    {
                        "num_filtered_synthetic_examples": f'{num_filtered_synthetic_examples} / {openai_utils.monitor["num_examples"]}',
                        "cost": f"${openai_utils.compute_cost()}",
                    }
                )
        save_filtered_synthetic_examples(
            filtered_synthetic_examples, args.OUT_DIR / stem
        )
    print(
        f'max prompt tokens: {openai_utils.monitor["max_prompt_tokens"]}\n'
        f"total cost: ${openai_utils.compute_cost()}"
    )


if __name__ == "__main__":
    main()
