#!/bin/bash

CKPT="llava-v1.5-7b-pretrain-allava-share4v-100K-unlock-vit_12-llm_all-sft-llava-665K"

python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/llava-v1.5-7b-pretrain-unlock-settings-data-scale/$CKPT \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /mnt/hwfile/mllm/chenlin/llava/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}_result.json
