#!/bin/bash

WANDB_PROJECT=pretrain-vocab-0.01-MSE-KL-margin-800 deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/hwfile/mllm/chenlin/llava/pretrained/vicuna/vicuna-7b-v1.5 \
    --version plain \
    --data_path /mnt/hwfile/mllm/chenlin/llava/data/llava/llava_pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /mnt/hwfile/mllm/chenlin/llava/data/llava/llava_pretrain/images \
    --vision_tower /mnt/hwfile/mllm/chenlin/llava/pretrained/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-pretrain-vocab-0.01-MSE-KL-margin-800 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --vocab_weight 0.01 \
    --vocab_loss_type mse-kl \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb


WANDB_PROJECT=sft-vocab-0.01-MSE-KL-margin-800 deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /mnt/hwfile/mllm/chenlin/llava/pretrained/vicuna/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /mnt/hwfile/mllm/chenlin/llava/data/llava/llava_v1_5_mix665k.json \
    --image_folder /mnt/hwfile/mllm/chenlin/llava/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain-vocab-0.01-MSE-KL-margin-800/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-sft-vocab-0.01-MSE-KL-margin-800 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --vocab_weight 0.01 \
    --vocab_loss_type mse-kl \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb


# CUDA_LAUNCH_BLOCKING=1 WANDB_PROJECT=sft-vocab-0.01-MSE-KL deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path /mnt/hwfile/mllm/chenlin/llava/pretrained/vicuna/vicuna-7b-v1.5 \
#     --version v1 \
#     --data_path /mnt/hwfile/mllm/chenlin/llava/data/llava/llava_v1_5_mix665k.json \
#     --image_folder /mnt/hwfile/mllm/chenlin/llava/data \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --pretrain_mm_mlp_adapter /mnt/hwfile/mllm/huangqidong/lvlm/llava_ckpts/pretrained/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-v1.5-7b-sft-vocab-0.01-MSE-KL-sft-only-zero2 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --vocab_weight 0.01 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb