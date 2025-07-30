#!/usr/bin/env bash

set -x
set -e

TASK="webqsp"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="/data/${TASK}"
fi


python3 -u train.py \
--pretrained-model "microsoft/mpnet-base" \
--pooling mean \
--train_path "/data/webqsp/train_goldenpath.jsonl" \
--train_graph_path "/data/webqsp/total_graph_webqsp.jsonl" \
--triple2id_1 "/data/webqsp_triple2id.pkl" \
--batch_size 1 \
--print-freq 100 \
--max_num_neg 50 \
--max_num_pos 20 \
--epochs 6 \
--workers 2 \
--max-to-keep 3 "$@" \
--output-dir "/ckpt/webqsp"
