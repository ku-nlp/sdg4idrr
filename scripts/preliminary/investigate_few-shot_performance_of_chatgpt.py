import json
from argparse import ArgumentParser
from pathlib import Path
from time import sleep

from first_party_modules.progress_bar import tqdm
from first_party_modules.utils import BaseOpenAIUtils, ObjectHook, set_seed


class OpenAIUtils(BaseOpenAIUtils):
    def get_messages(
        self,
        test_example: ObjectHook,
        n_way_k_shot_examples: list[tuple[ObjectHook, str]],
    ) -> list[dict[str, str]]:
        definitions = "\n".join(f"- {s}: {d}" for s, d in self.sense2definition.items())
        instruction = (
            "Given two arguments, please answer the most appropriate relation between them from "
            "the following 14 possible relations:\n"
            f"{definitions}\n"
            "Here are examples:"
        )
        test_input = (
            "Please answer the relation between the following arguments.\n"
            f"Arg1: {test_example.arg1}\n"
            f"Arg2: {test_example.arg2}\n"
            "Answer: "
        )
        user_prompt = self.get_user_prompt(instruction, n_way_k_shot_examples, test_input)

        if self.monitor["num_examples"] < 1:
            print("---------- confirm user prompt ----------\n" f"{user_prompt}\n" "--------------------")
        self.monitor["num_examples"] += 1

        return [
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    def get_user_prompt(
        self,
        instruction: str,
        n_way_k_shot_examples: list[tuple[ObjectHook, str]],
        test_input: str,
    ) -> str:
        for i in range(len(n_way_k_shot_examples), 0, -1):
            demonstrations = "\n".join(
                f"Arg1: {e.arg1}\nArg2: {e.arg2}\nAnswer: {s}\n" for e, s in n_way_k_shot_examples[:i]
            )
            user_prompt = f"{instruction}\n" f"{demonstrations}\n" f"{test_input}"
            num_prompt_tokens = len(self.encoding.encode(user_prompt))

            self.monitor["max_prompt_tokens"] = max(num_prompt_tokens, self.monitor["max_prompt_tokens"])

            if num_prompt_tokens <= self.max_prompt_tokens:
                return user_prompt
        else:
            return f"{instruction}\n" f"{test_input}"


def save_results(results: list[tuple[ObjectHook, str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open(mode="w") as f:
        for test_example, completion in results:
            json_line = json.dumps(
                {
                    **test_example,
                    "completion": completion,
                }
            )
            f.write(json_line + "\n")


def main():
    parser = ArgumentParser(description="script to investigate the few-shot performance of ChatGPT on PDTB3 dataset")
    parser.add_argument("TRAIN", type=Path, help="path to train.jsonl")
    parser.add_argument("TEST", type=Path, help="path to test.jsonl")
    parser.add_argument("OUT_FILE", type=Path, help="path to output file")
    parser.add_argument("--dry-run", action="store_true", help="whether to perform a dry run")
    args = parser.parse_args()

    set_seed(seed=0)

    openai_utils = OpenAIUtils(
        model_name="gpt-4-0613",
        # model_name="gpt-3.5-turbo-16k-0613",
        max_tokens=8000,  # gpt-4-0613
        # max_tokens=16000,  # gpt-3.5-turbo-16k-0613
        max_completion_tokens=128,
        level="l2",
        num_few_shot_examples=8,
    )

    sense2train_examples = openai_utils.get_sense2examples(openai_utils.load_examples(args.TRAIN))
    test_examples = openai_utils.load_examples(args.TEST)
    n_way_k_shot_examples_list = openai_utils.get_n_way_k_shot_examples_list(test_examples, sense2train_examples)

    results = []
    bar = tqdm(list(zip(test_examples, n_way_k_shot_examples_list)))
    for test_example, n_way_k_shot_examples in bar:
        messages = openai_utils.get_messages(test_example, n_way_k_shot_examples)
        if args.dry_run is True:
            continue
        response = openai_utils.get_response(messages)
        if response is None:
            continue
        completion = response["choices"][0]["message"]["content"]
        test_example["pred_sense"] = completion.rstrip(".")
        results.append(test_example)
        bar.set_postfix({"cost": f"${openai_utils.compute_cost()}"})
        sleep(3)
    print(
        f'max prompt tokens: {openai_utils.monitor["max_prompt_tokens"]}\n'
        f"total cost: ${openai_utils.compute_cost()}"
    )

    if args.dry_run is True:
        return None

    openai_utils.compute_metrics(results, handling_of_multi_labeled_examples="loose")
    save_results(results, args.OUT_FILE)


if __name__ == "__main__":
    main()
