# Synthetic Data Generation for Implicit Discourse Relation Recognition

This repository contains scripts of Synthetic Data Generation for Implicit Discourse Relation Recognition (SDG4IDRR).

### Requirements

- Python 3.9
  - poetry
    ```shell
    pip install poetry
    ```
  - Dependencies: see pyproject.toml
- Java (required to install hydra)
  ```shell
  # command example
  wget https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz
  tar -xf openjdk-17.0.2_linux-x64_bin.tar.gz
  mv jdk-17.0.2/ $HOME/.local/
  export PATH="$HOME/.local/jdk-17.0.2/bin:$PATH"
  ```

### Set up Python Virtual Environment

```shell
git clone git@github.com:facebookresearch/hydra.git -b v1.3.2 src/hydra
git clone git@github.com:princeton-nlp/SimCSE.git -b 0.4 src/SimCSE
rsync -av patch/src/ src/
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
  path/to/IN_ROOT/ \
  dataset/ \
  --aid-dir data/article_ids/
```

##### Rebuild Synthetic Data

```shell
# rebuild synthetic data from PDTB3 dataset and annotations
poetry run python scripts/rebuild_synthetic_data.py \
  dataset/ji/train.jsonl \
  data/annot/ \
  data/synth/filtered/
```

Since synthetic data was generated using GPT-4, please refer to the OpenAI's terms of use.
For instance, you may not use it to develop models that compete with OpenAI.

##### Compile

```shell
# compile synthetic data based on a confusion matrix
poetry run python scripts/compile.py \
  data/synth/filtered/ \
  results/run_id/dev_pred.jsonl \
  data/synth/compiled/run_id/examples.jsonl \
  [--top-k int]

# zsh
arr=("roberta-base" "roberta-large")
for model in $arr;
  do
    if [[ ${model} == "roberta-base" ]]; then
      lr="2e-05"
      top_k=3
    elif [[ ${model} == "roberta-large" ]]; then
      lr="1e-05"
      top_k=5
    else
      exit 1
    fi
    for seed in {0..2};
      do
        poetry run python scripts/compile.py \
          data/synth/filtered/ \
          results/${model}_${seed}_${lr}/dev_pred.jsonl \
          data/synth/compiled/${model}_${seed}_${lr}/examples.jsonl \
          --top-k ${top_k}
      done
  done
```

---

##### Investigate Few-Shot Performance of ChatGPT

```shell
# investigate few-shot performance of ChatGPT on PDTB3 dataset
poetry run python scripts/preliminary/investigate_few-shot_performance_of_chatgpt.py \
  dataset/ji/train.jsonl \
  dataset/ji/test.jsonl \
  results/few-shot/gpt-4-0613.jsonl \
  [--dry-run]
```

##### Generate Candidates of Arg2

```shell
# generate candidates of Arg2 using GPT-4 based on a confusion matrix
poetry run python scripts/generate_candidates_of_arg2.py \
  dataset/ji/train.jsonl \
  results/run_id/dev_pred.jsonl \
  data/synth/unfiltered/ \
  [--top-k int] \
  [--dry-run]
```

##### Filter Synthetic Argument Pairs

```shell
# filter synthetic argument pairs using GPT-4 based on a confusion matrix
poetry run python scripts/filter_synthetic_argument_pairs.py \
  data/synth/unfiltered/ \
  dataset/ji/train.jsonl \
  results/run_id/dev_pred.jsonl \
  data/synth/filtered/ \
  [--top-k int] \
  [--dry-run]
```

---

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
}
```
