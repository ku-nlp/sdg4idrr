# Synthetic Data Generation for Implicit Discourse Relation Recognition

This repository contains scripts of Synthetic Data Generation for Implicit Discourse Relation Recognition (SDG4IDRR). [WIP]

### Requirements

- Python 3.9
  - see pyproject.toml
- poetry
  ```shell
  pip install poetry
  ```
- Java (required to install hydra)
  ```shell
  wget https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz
  tar -xf openjdk-17.0.2_linux-x64_bin.tar.gz
  mv jdk-17.0.2/ $HOME/.local/
  export PATH="$HOME/.local/jdk-17.0.2/bin:$PATH"
  ```

### Set up Python Virtual Environment

```shell
git clone git@github.com:facebookresearch/hydra.git -b v1.3.2 src/hydra
poetry install [--no-dev]

# make a .env file
echo "OPENAI_API-KEY=<OPENAI_API-KEY>" >> .env
echo "OPENAI-ORGANIZATION=<OPENAI-ORGANIZATION>" >> .env
```

### (optional) Set up pre-commit

```shell
# pip install pre-commit
pre-commit install
```

### Command Examples

##### Build Dataset

```shell
# obtain Penn Discourse Treebank Version 3.0 (cf. https://catalog.ldc.upenn.edu/LDC2019T05)

# confirm help message of IN_ROOT argument
poetry run python scripts/build_pdtb3_dataset.py -h
# build PDTB3 dataset
poetry run python scripts/build_pdtb3_dataset.py \
  IN_ROOT/ \
  OUT_ROOT/
```

##### Investigate Few-Shot Performance of ChatGPT

```shell
# investigate few-shot performance of ChatGPT on PDTB3 dataset
poetry run python scripts/preliminary/investigate_few-shot_performance_of_chatgpt.py \
  path/to/train.jsonl \
  path/to/test.jsonl \
  gpt4_few-shot.jsonl \
  [--dry-run]
```

##### Generate Candidates of Arg2

```shell
# generate candidates of Arg2 based on a confusion matrix
poetry run python scripts/generate_candidates_of_arg2.py \
  path/to/train.jsonl \
  path/to/dev_pred.jsonl \
  OUT_DIR/ \
  [--top-k int] \
  [--dry-run]
```

##### Filter Synthetic Argument Pairs

```shell
# filter synthetic argument pairs using GPT-4
poetry run python scripts/filter_synthetic_argument_pairs.py \
  path/to/train.jsonl \
  path/to/dev_pred.jsonl \
  SYNTH_DATA_DIR/ \
  OUT_DIR/ \
  [--top-k int] \
  [--dry-run]
```

### Reference/Citation

```
@inproceedings{omura-etal-2024-empirical,
  title = "{A}n {E}mpirical {S}tudy of {S}ynthetic {D}ata {G}eneration for {I}mplicit {D}iscourse {R}elation {R}ecognition",
  author = "Omura, Kazumasa and
    Cheng, Fei and
    Kurohashi, Sadao",
  booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING)",
  month = may,
  year = "2024",
  address = "Turin, Italy",
  note = "(to appear)"
}
```
