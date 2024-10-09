# #!/bin/bash

# PRETRAIN_NAME=llava-v1.5-7b-pretrain-data-scale/llava-v1.5-7b-pretrain-allava-share4v-1.2M


# WANDB_PROJECT=pretrain MASTER_PORT=29515 deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path /mnt/hwfile/mllm/chenlin/llava/pretrained/vicuna/vicuna-7b-v1.5 \
#     --version plain \
#     --data_path /mnt/hwfile/mllm/huangqidong/data/allava_share-captioner_pretrain_1955k.json \
#     --image_folder /mnt/petrelfs/huangqidong/lvlm/ShareGPT4V/data \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir /mnt/hwfile/mllm/huangqidong/lvlm/$PRETRAIN_NAME \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 100000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb


PRETRAIN_NAME=llava-v1.5-7b-pretrain-data-scale/llava-v1.5-7b-pretrain-blip2-allava-400K


WANDB_PROJECT=pretrain MASTER_PORT=29515 deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/hwfile/mllm/chenlin/llava/pretrained/vicuna/vicuna-7b-v1.5 \
    --version plain \
    --data_path /mnt/hwfile/mllm/huangqidong/data/blip2_allava_pretrain_1266k.json \
    --image_folder /mnt/petrelfs/huangqidong/lvlm/MGM/data/MGM-Pretrain \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /mnt/hwfile/mllm/huangqidong/lvlm/$PRETRAIN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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