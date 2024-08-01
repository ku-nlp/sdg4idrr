import json
import random
from collections import defaultdict
from logging import getLogger, ERROR
from os import environ
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import openai
from dotenv import load_dotenv
from simcse import SimCSE
from tenacity import retry, stop_after_attempt, wait_fixed
from tiktoken import encoding_for_model
from torch import as_tensor

from first_party_modules.utils import ObjectHook
from first_party_modules.constants import (
    PDTB3_L1_SENSES,
    PDTB3_L1_SENSE2DEFINITION,
    SELECTED_PDTB3_L2_SENSES,
    SELECTED_PDTB3_L2_SENSE2DEFINITION,
)


load_dotenv()
openai.api_key = environ["OPENAI-API-KEY"]
openai.organization = environ["OPENAI-ORGANIZATION"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class BaseOpenAIUtils:
    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        max_completion_tokens: int,
        level: Literal["l2", "l1"] = "l2",
        num_few_shot_examples: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.encoding = encoding_for_model(model_name)

        self.max_tokens = max_tokens
        self.max_prompt_tokens = max_tokens - max_completion_tokens
        self.max_completion_tokens = max_completion_tokens

        self.level = level
        if level == "l2":
            self.senses = SELECTED_PDTB3_L2_SENSES
            self.sense2definition = SELECTED_PDTB3_L2_SENSE2DEFINITION
        elif level == "l1":
            self.senses = PDTB3_L1_SENSES
            self.sense2definition = PDTB3_L1_SENSE2DEFINITION
        else:
            raise ValueError("invalid level")

        self.num_few_shot_examples = num_few_shot_examples
        if num_few_shot_examples:
            getLogger("simcse").setLevel(ERROR)
            self.retriever = SimCSE("princeton-nlp/sup-simcse-roberta-large")
        else:
            self.retriever = ...

        self.monitor: dict[str, int] = {
            "num_examples": 0,
            "sum_prompt_tokens": 0,
            "sum_completion_tokens": 0,
            "max_prompt_tokens": 0,
        }

    @staticmethod
    def load_examples(in_file: Path) -> list[ObjectHook]:
        return [
            json.loads(line, object_hook=ObjectHook)
            for line in in_file.read_text().splitlines()
        ]

    def get_senses(self, example: ObjectHook) -> list[str]:
        senses = []
        for indices in ["1a", "1b", "2a", "2b"]:
            sense = example[f"sclass{indices}_{self.level}"]
            if sense in self.senses and sense not in senses:
                senses.append(sense)
        return senses

    def get_sense2examples(
        self, examples: list[ObjectHook]
    ) -> dict[str, list[ObjectHook]]:
        sense2examples = defaultdict(list)
        for example in examples:
            for sense in self.get_senses(example):
                sense2examples[sense].append(example)
        return sense2examples

    def get_sense2conns(self, example: ObjectHook) -> dict[str, list[str]]:
        sense2conns = defaultdict(list)
        for conn_index, sclass_index in ["1a", "1b", "2a", "2b"]:
            sense = example[f"sclass{conn_index}{sclass_index}_{self.level}"]
            conn = example[f"conn{conn_index}"]
            if (
                sense in self.senses and conn not in sense2conns[sense]
            ):  # conn != "" if sense in self.senses
                sense2conns[sense].append(conn)
        return sense2conns

    def get_sense_conn2examples(
        self, examples: list[ObjectHook]
    ) -> dict[str, dict[str, list[ObjectHook]]]:
        sense_conn2examples = defaultdict(lambda: defaultdict(list))
        for example in examples:
            for sense, conns in self.get_sense2conns(example).items():
                for conn in conns:
                    sense_conn2examples[sense][conn].append(example)
        return sense_conn2examples

    def get_nearest_neighbors_list(
        self,
        arg_pairs: list[str],
        train_examples: list[ObjectHook],
    ) -> list[list[ObjectHook]]:
        # (num_arg_pairs, num_train_examples) -> (num_arg_pairs, top_k)
        similarities = self.retriever.similarity(
            arg_pairs, [f"{e.arg1} {e.arg2}" for e in train_examples]
        )
        k = min(len(train_examples), self.num_few_shot_examples)
        _, top_k_indices_list = as_tensor(similarities).topk(k, dim=1)
        if k < self.num_few_shot_examples:
            print(f"k < num_few-shot_examples ({k} < {self.num_few_shot_examples})")
        return [
            [train_examples[i] for i in top_k_indices]
            for top_k_indices in top_k_indices_list
        ]

    def get_n_way_k_shot_examples_list(
        self,
        examples: list[ObjectHook],
        sense2train_examples: dict[str, list[ObjectHook]],
    ) -> list[list[tuple[ObjectHook, str]]]:
        num_senses = len(self.senses)
        # placeholder
        n_way_k_shot_examples_list = [
            [...] * num_senses * self.num_few_shot_examples
            for _ in range(len(examples))
        ]
        arg_pairs = [f"{e.arg1} {e.arg2}" for e in examples]
        for j, sense in enumerate(self.senses):
            nearest_neighbors_list = self.get_nearest_neighbors_list(
                arg_pairs, sense2train_examples[sense]
            )
            for i, nearest_neighbors in enumerate(nearest_neighbors_list):
                for k, nearest_neighbor in enumerate(nearest_neighbors):
                    n_way_k_shot_examples_list[i][j + num_senses * k] = (
                        nearest_neighbor,
                        sense,
                    )
        return n_way_k_shot_examples_list

    def get_messages(self, *args) -> list[dict[str, str]]:
        raise NotImplementedError

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_fixed(15),
        retry_error_callback=lambda x: None,  # return None if no response
    )
    def get_response(self, messages: list[dict[str, str]]) -> Optional[dict[str, ...]]:
        # attempt_number = get_response.retry.statistics["attempt_number"]
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_completion_tokens,
            # sampling settings
            temperature=0,
            top_p=1,
            presence_penalty=0,
            frequency_penalty=0,
            n=1,  # number of responses
            request_timeout=15,
        )
        self.monitor["sum_prompt_tokens"] += response["usage"]["prompt_tokens"]
        self.monitor["sum_completion_tokens"] += response["usage"]["completion_tokens"]
        if self.monitor["num_examples"] < 3:
            print(
                "---------- confirm completion ----------\n"
                f'{response["choices"][0]["message"]["content"].rstrip()}\n'
                "--------------------"
            )
        return response

    def compute_cost(self) -> float:
        if self.model_name == "gpt-3.5-turbo-0613":
            return round(
                self.monitor["sum_prompt_tokens"] / 1000 * 0.0015
                + self.monitor["sum_completion_tokens"] / 1000 * 0.0015,
                3,
            )
        elif self.model_name == "gpt-3.5-turbo-16k-0613":
            return round(
                self.monitor["sum_prompt_tokens"] / 1000 * 0.003
                + self.monitor["sum_completion_tokens"] / 1000 * 0.004,
                3,
            )
        elif self.model_name == "gpt-4-0613":
            return round(
                self.monitor["sum_prompt_tokens"] / 1000 * 0.03
                + self.monitor["sum_completion_tokens"] / 1000 * 0.06,
                3,
            )
        else:
            raise ValueError("invalid model")

    def compute_normalized_confusion_matrix(
        self, dev_examples: list[ObjectHook]
    ) -> np.array:
        confusion_matrix = np.zeros((len(self.senses), len(self.senses)), dtype=int)

        for i, dev_example in enumerate(dev_examples):
            labels = [self.senses.index(s) for s in self.get_senses(dev_example)]
            pred = dev_example["prediction"]
            if pred in labels:
                for label in labels:
                    confusion_matrix[label, label] += 1
            else:
                for label in labels:
                    confusion_matrix[label, pred] += 1

        for row in confusion_matrix:
            print(f'[{",".join(str(column).rjust(4, " ") for column in row)}],')

        num_labels = confusion_matrix.sum(axis=1)
        return confusion_matrix / num_labels[:, None]

    def get_top_k_confusing_sense_pairs(
        self,
        normalized_confusion_matrix: np.array,
        top_k: int = 5,
    ) -> list[tuple[str, str]]:
        _, indices = (
            as_tensor(normalized_confusion_matrix)
            .flatten()
            .topk(normalized_confusion_matrix.size)
        )
        confusing_sense_pairs = []
        for index in indices:
            i, j = np.unravel_index(index, normalized_confusion_matrix.shape)
            if i != j:
                confusing_sense_pairs.append((self.senses[i], self.senses[j]))
        return confusing_sense_pairs[:top_k]
