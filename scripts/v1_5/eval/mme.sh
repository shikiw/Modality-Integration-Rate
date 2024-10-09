#!/bin/bash

CKPT="llava-v1.5-7b-pretrain-allava-share4v-100K-unlock-vit_12-llm_all-sft-llava-665K"

# ## lora
# python -m llava.eval.model_vqa_loader \
#     --model-path checkpoints/llava-v1.5-7b-pretrain-unlock-settings/$CKPT/llava_lora_ckpt \
#     --model-base checkpoints/llava-v1.5-7b-pretrain-unlock-settings/$CKPT \
#     --question-file ./playground/data/eval/MME/llava_mme.jsonl \
#     --image-folder /mnt/hwfile/mllm/xinglong/llava/llava_1.5/playground/data/eval/mme/MME_Benchmark_release \
#     --answers-file ./playground/data/eval/MME/answers/$CKPT.jsonl \
#     --temperature 0 \
#     --num_beams 1 \
#     --conv-mode vicuna_v1

# cd ./playground/data/eval/MME

# python convert_answer_to_mme.py \
#     --experiment $CKPT \
#     --image-folder /mnt/hwfile/mllm/xinglong/llava/llava_1.5/playground/data/eval/mme/MME_Benchmark_release

# cd eval_tool

# python calculation.py --results_dir answers/$CKPT



## whole model
python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-pretrain-unlock-settings-data-scale/$CKPT \
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



# ## pretrained model
# python -m llava.eval.model_vqa_loader \
#     --model-path /mnt/hwfile/mllm/huangqidong/lvlm/llava_ckpts/pretrained/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5 \
#     --model-base /mnt/hwfile/mllm/chenlin/llava/pretrained/vicuna/vicuna-7b-v1.5 \
#     --question-file ./playground/data/eval/MME/llava_mme.jsonl \
#     --image-folder /mnt/hwfile/mllm/xinglong/llava/llava_1.5/playground/data/eval/mme/MME_Benchmark_release \
#     --answers-file ./playground/data/eval/MME/answers/$CKPT.jsonl \
#     --temperature 0 \
#     --num_beams 1 \
#     --conv-mode vicuna_v1

# cd ./playground/data/eval/MME

# python convert_answer_to_mme.py \
#     --experiment $CKPT \
#     --image-folder /mnt/hwfile/mllm/xinglong/llava/llava_1.5/playground/data/eval/mme/MME_Benchmark_release

# cd eval_tool

# python calculation.py --results_dir answers/$CKPT