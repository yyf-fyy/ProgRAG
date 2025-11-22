#!/usr/bin/env bash

set -x
set -e

TASK="webqsp"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$DATA_ROOT" ]; then
  DATA_ROOT="${DIR}/data"
fi

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DATA_ROOT}/${TASK}"
fi

GRAPH_DIR="${DATA_ROOT}/graphs"
CKPT_DIR="${DIR}/ckpt/${TASK}"

mkdir -p "${DATA_DIR}" "${GRAPH_DIR}" "${CKPT_DIR}"


python3 -u train.py \
--pretrained-model "microsoft/mpnet-base" \
--pooling mean \
--train_path "${DATA_DIR}/train_goldenpath.jsonl" \
--train_graph_path "${GRAPH_DIR}/total_graph_${TASK}.jsonl" \
--triple2id_1 "${GRAPH_DIR}/${TASK}_triple2id.pkl" \
--batch_size 1 \
--print-freq 100 \
--max_num_neg 50 \
--max_num_pos 20 \
--epochs 6 \
--workers 2 \
--max-to-keep 3 "$@" \
--output-dir "${CKPT_DIR}"
