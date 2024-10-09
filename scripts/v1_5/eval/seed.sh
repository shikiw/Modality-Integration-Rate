#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b-pretrain-allava-share4v-100K-unlock-vit_12-llm_all-sft-llava-665K"

# ## lora
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path /mnt/hwfile/mllm/huangqidong/lvlm/llava_ckpts/finetuned/$CKPT \
#         --model-base /mnt/hwfile/mllm/chenlin/llava/pretrained/vicuna/vicuna-7b-v1.5 \
#         --question-file ./playground/data/eval/seed_bench/llava-seed-bench.jsonl \
#         --image-folder /mnt/hwfile/mllm/chenlin/llava/data/eval/SEED-Bench \
#         --answers-file ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=./playground/data/eval/seed_bench/answers/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# # Evaluate
# python scripts/convert_seed_for_submission.py \
#     --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
#     --result-file $output_file \
#     --result-upload-file ./playground/data/eval/seed_bench/answers_upload/$CKPT.jsonl





# whole model
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints/llava-v1.5-7b-pretrain-unlock-settings-data-scale/$CKPT \
        --question-file ./playground/data/eval/seed_bench/llava-seed-bench.jsonl \
        --image-folder /mnt/hwfile/mllm/chenlin/llava/data/eval/SEED-Bench \
        --answers-file ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --num_beams 1 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/seed_bench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/seed_bench/answers_upload/$CKPT.jsonl



