#!/bin/bash

CKPT="llava-v1.5-7b-pretrain-allava-share4v-100K"

## whole model
python -m llava.eval.model_vqa_loader_in_context \
    --model-path ./checkpoints/llava-v1.5-7b-pretrain-data-scale/$CKPT \
    --model-base /mnt/hwfile/mllm/chenlin/llava/pretrained/vicuna/vicuna-7b-v1.5 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /mnt/hwfile/mllm/xinglong/llava/llava_1.5/playground/data/eval/mme/MME_Benchmark_release \
    --answers-file ./playground/data/eval/MME/answers/$CKPT.jsonl \
    --temperature 0 \
    --num_beams 1 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py \
    --experiment $CKPT \
    --image-folder /mnt/hwfile/mllm/xinglong/llava/llava_1.5/playground/data/eval/mme/MME_Benchmark_release

cd eval_tool

python calculation.py --results_dir answers/$CKPT
