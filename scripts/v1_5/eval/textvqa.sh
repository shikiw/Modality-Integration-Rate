#!/bin/bash

CKPT="llava-v1.5-7b-pretrain-allava-share4v-100K-unlock-vit_12-llm_all-sft-llava-665K"

# ## lora
# python -m llava.eval.model_vqa_loader \
#     --model-path /mnt/hwfile/mllm/huangqidong/lvlm/llava_ckpts/finetuned/$CKPT \
#     --model-base /mnt/hwfile/mllm/chenlin/llava/pretrained/vicuna/vicuna-7b-v1.5 \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder /mnt/hwfile/mllm/chenlin/llava/data/eval/textvqa/train_images/ \
#     --answers-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl \
#     --temperature 0 \
#     --num_beams 1 \
#     --conv-mode vicuna_v1

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl



## whole model
python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-pretrain-unlock-settings-data-scale/$CKPT \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /mnt/hwfile/mllm/chenlin/llava/data/eval/textvqa/train_images/ \
    --answers-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --num_beams 1 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl



# ## pretrained model
# python -m llava.eval.model_vqa_loader \
#     --model-path /mnt/hwfile/mllm/huangqidong/lvlm/llava_ckpts/pretrained/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5 \
#     --model-base /mnt/hwfile/mllm/chenlin/llava/pretrained/vicuna/vicuna-7b-v1.5 \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder /mnt/hwfile/mllm/chenlin/llava/data/eval/textvqa/train_images/ \
#     --answers-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl \
#     --temperature 0 \
#     --num_beams 1 \
#     --conv-mode vicuna_v1

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl