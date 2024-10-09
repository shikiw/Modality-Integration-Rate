#!/bin/bash

for i in {6000..7000..1000}
do
    PRETRAIN_NAME=share4v-7b-pretrain-overfitting/share4v-7b-pretrain-2epoch/checkpoint-${i}
    FINETUNE_NAME=share4v-7b-pretrain-overfitting-sft/llava-v1.5-7b-pretrain-step-${i}-sft-share4v-665K


    WANDB_PROJECT=overfitting-sft MASTER_PORT=29532 deepspeed llava/train/train_mem.py \
        --deepspeed ./scripts/zero3.json \
        --model_name_or_path ./checkpoints/$PRETRAIN_NAME \
        --version v1 \
        --data_path /mnt/petrelfs/huangqidong/lvlm/ShareGPT4V/data/sharegpt4v/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json \
        --image_folder /mnt/petrelfs/huangqidong/lvlm/ShareGPT4V/data \
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
        --per_device_train_batch_size 8 \
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
done