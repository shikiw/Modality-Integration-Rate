#!/bin/bash

CKPT="llava-v1.5-7b-pretrain-allava-share4v-100K-unlock-vit_12-llm_all-sft-llava-665K"

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-pretrain-unlock-settings-data-scale/$CKPT \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /mnt/hwfile/mllm/wangjiaqi/mllm-data-alg/COCO_2014/ori/val2014/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$CKPT.jsonl


