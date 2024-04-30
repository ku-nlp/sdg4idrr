import json
from argparse import ArgumentParser
from pathlib import Path
from time import sleep

from first_party_modules.openai_utils import BaseOpenAIUtils, set_seed
from first_party_modules.progress_bar import tqdm
from first_party_modules.utils import ObjectHook


class OpenAIUtils(BaseOpenAIUtils):
    def get_messages(
        self,
        train_example: ObjectHook,
        nearest_neighbors: list[ObjectHook],
        true: str,
    ) -> list[dict[str, str]]:
        demonstrations = "\n".join(f"{e.arg1} - {e.arg2}" for e in nearest_neighbors)
        user_prompt = (
            f'Given two arguments, the relation "{true}" is defined as "{self.sense2definition[true]}".\n'
            f'Here are examples that have the relation "{true}":\n'
            f"{demonstrations}\n"
            f'Please write down arguments that have the relation "{true}" to the argument "{train_example.arg1}".'
        )
        assistant_prompt = (
            "Here list several answers:\n" f"- {train_example.arg2}\n" "- "
        )

        # conn = self.get_sense2conns(train_example)[true][0]
        # assistant_prompt = (
        #     "Here list several answers:\n"
        #     f"- {conn} {train_example.arg2}\n"
        #     f"- {conn} "
        # )

        if self.monitor["num_examples"] < 1:
            print(
                "---------- confirm user prompt ----------\n"
                f"{user_prompt}\n"
                "---------- confirm assistant prompt ----------\n"
                f"{assistant_prompt}\n"
                "--------------------"
            )
        self.monitor["num_examples"] += 1
        self.monitor["max_prompt_tokens"] = max(
            len(self.encoding.encode(user_prompt))
            + len(self.encoding.encode(assistant_prompt)),
            self.monitor["max_prompt_tokens"],
        )

        return [
            {
                "role": "user",
                "content": user_prompt,
            },
            {"role": "assistant", "content": assistant_prompt},
        ]


def save_synthetic_examples(
    synthetic_examples: list[tuple[ObjectHook, str]], out_file: Path
) -> None:
    with out_file.open(mode="w") as f:
        for synthetic_examples in synthetic_examples:
            for train_example, completion in synthetic_examples:
                json_line = json.dumps(
                    {
                        **train_example,
                        "completion": completion,
                        "true": out_file.stem,
                    }
                )
                f.write(json_line + "\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("TRAIN", type=Path, help="path to train.jsonl")
    parser.add_argument("DEV_PRED", type=Path, help="path to dev_pred.jsonl")
    parser.add_argument("OUT_DIR", type=Path, help="path to output directory")
    parser.add_argument(
        "--top-k", default=3, type=int, help="how many confusing sense pairs to extract"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="whether to perform a dry run"
    )
    args = parser.parse_args()

    set_seed(seed=0)

    openai_utils = OpenAIUtils(
        model_name="gpt-4-0613",
        max_tokens=8000,
        max_completion_tokens=2048,
        level="l2",
        num_few_shot_examples=8,
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

    for true, _ in confusing_sense_pairs:
        if (args.OUT_DIR / f"{true}.jsonl").exists():  # avoid overwriting
            continue
        synthetic_examples = []
        train_examples = sense2train_examples[true]
        nearest_neighbors_list = openai_utils.get_nearest_neighbors_list(
            [f"{e.arg1} {e.arg2}" for e in train_examples],
            train_examples,
        )
        bar = tqdm(list(zip(train_examples, nearest_neighbors_list)))
        for train_example, nearest_neighbors in bar:
            messages = openai_utils.get_messages(train_example, nearest_neighbors, true)
            if args.dry_run is True:
                continue
            response = openai_utils.get_response(messages)
            if response is None:
                continue
            synthetic_examples.append(
                (train_example, response["choices"][0]["message"]["content"])
            )
            bar.set_postfix({"cost": f"${openai_utils.compute_cost()}"})
            sleep(1)
        if args.dry_run is True:
            continue
        save_synthetic_examples(synthetic_examples, args.OUT_DIR / f"{true}.jsonl")
    print(
        f'max prompt tokens: {openai_utils.monitor["max_prompt_tokens"]}\n'
        f"total cost: ${openai_utils.compute_cost()}"
    )


if __name__ == "__main__":
    main()
