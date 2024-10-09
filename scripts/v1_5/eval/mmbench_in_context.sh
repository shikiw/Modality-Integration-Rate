#!/bin/bash

SPLIT="mmbench_dev_20230712"

CKPT="llava-v1.5-7b-pretrain-allava-share4v-100K"



## whole model
python -m llava.eval.model_vqa_mmbench_in_context \
    --model-path ./checkpoints/llava-v1.5-7b-pretrain-data-scale/$CKPT \
    --model-base /mnt/hwfile/mllm/chenlin/llava/pretrained/vicuna/vicuna-7b-v1.5 \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --num_beams 1 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT

python playground/data/eval/mmbench/mmbench_excel_test.py ./playground/data/eval/mmbench/answers_upload/$SPLIT/$CKPT.xlsx