#!/usr/bin/env zsh

usage() {
  cat << _EOT_
Usage: ./scripts/compile.sh [-h | --help]

Options:
    -h | --help    display help message
_EOT_
}

while getopts h-: opt; do
  if [[ $opt = "-" ]]; then
    opt=$(echo "${OPTARG}" | awk -F "=" '{print $1}')
    OPTARG=$(echo "${OPTARG}" | awk -F "=" '{print $2}')
  fi

  case "$opt" in
  h | help)
    usage
    exit 0
    ;;
  *)
    echo "invalid option --$opt"
    exit 1
    ;;
  esac
done

models=("roberta-base" "roberta-large")
for model in $models;
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
        run_id="${model}_${seed}_${lr}"
        poetry run python scripts/compile.py \
          data/synth/filtered/ \
          results/${run_id}/dev_pred.jsonl \
          data/synth/compiled/${run_id}/examples.jsonl \
          --top-k ${top_k}
      done
  done
