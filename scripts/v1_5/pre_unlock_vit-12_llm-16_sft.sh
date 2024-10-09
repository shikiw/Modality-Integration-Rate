#!/bin/bash

PRETRAIN_NAME=llava-v1.5-7b-pretrain-unlock-settings/llava-v1.5-7b-pretrain-allava-share4v-1955K-unlock-vit_12-llm_16


WANDB_PROJECT=pretrain MASTER_PORT=29520 deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /mnt/hwfile/mllm/chenlin/llava/pretrained/vicuna/vicuna-7b-v1.5 \
    --version plain \
    --data_path /mnt/hwfile/mllm/huangqidong/data/allava_share-captioner_pretrain_1955k.json \
    --image_folder /mnt/petrelfs/huangqidong/lvlm/ShareGPT4V/data \
    --data_scale -1 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /mnt/hwfile/mllm/huangqidong/lvlm/llava_ckpts/pretrained/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --tune_vision_tower True \
    --tune_vit_from_layer 12 \
    --tune_language_model True \
    --tune_llm_utill_layer 16 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/$PRETRAIN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --save_only_model True \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

PRETRAIN_NAME=llava-v1.5-7b-pretrain-unlock-settings/llava-v1.5-7b-pretrain-allava-share4v-1955K-unlock-vit_12-llm_16
FINETUNE_NAME=llava-v1.5-7b-pretrain-unlock-settings/llava-v1.5-7b-pretrain-allava-share4v-1955K-unlock-vit_12-llm_16-sft-llava-665K


WANDB_PROJECT=sft MASTER_PORT=29520 deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/$PRETRAIN_NAME \
    --version v1 \
    --data_path /mnt/hwfile/mllm/chenlin/llava/data/llava/llava_v1_5_mix665k.json \
    --image_folder /mnt/hwfile/mllm/chenlin/llava/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/$FINETUNE_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --save_only_model True \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb


