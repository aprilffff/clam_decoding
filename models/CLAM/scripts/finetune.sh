#!/bin/bash

deepspeed  /root/autodl-tmp/PycharmProjects/llava-decoding-exp/models/CLAM/train_clam.py \
    --deepspeed /root/autodl-tmp/PycharmProjects/llava-decoding-exp/models/CLAM/LLaVA/scripts/zero2.json \
    --model_name_or_path /root/autodl-tmp/PycharmProjects/llm-param-checkpoints/llava-v1.5-7b \
    --version v1 \
    --data_path /root/autodl-tmp/PycharmProjects/benchmark/POVID_preference_data_for_VLLMs_version_1.json \
    --image_folder /root/autodl-tmp/PycharmProjects/benchmark/train2014 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /root/autodl-tmp/PycharmProjects/llava-decoding-exp/checkpoints/llava-cla-scale0.01-reduction256-startlayer16-addnorm \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 500 \
    --tf32 True \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --logging_dir /root/tf-logs \
    --freeze_mm_mlp_adapter True \
    --freeze_backbone True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --use_cross_layer_attention True \
    --cross_layer_attention_start_layer 16 \
    --cross_layer_attention_reduction 256 \
    --cross_layer_attention_scale 0.01 \
    --tune_lm_head False \



