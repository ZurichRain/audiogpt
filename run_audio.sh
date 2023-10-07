# aws --endpoint-url=http://oss.i.basemind.com s3 sync s3://lucaszhao/llm/llama-vicuna-7b /mnt/host0/llama-vicuna-7b
# aws --endpoint-url=http://oss.i.basemind.com s3 sync s3://lucaszhao/llm/clip/vit-large-patch14/ /mnt/host0/vit-large-patch14/
# aws --endpoint-url=http://oss.i.basemind.com s3 sync s3://vision-language-data/VisionPretrainDatasets/annotations/ /data/workspace/data/dataset/LLaVA-Pretrain_cn/
# --model_name_or_path /data/workspace/data/llm/chatglm2 \
# --model_name_or_path /mnt/host0/llama-vicuna-7b \

STAGE_1_MODEL_NAME="Audiogpt-7b-stage1-paimeng_v0"
STAGE_2_MODEL_NAME="mmgpt-7b-stage2-llava_6k_OCR_2k_Math_200_v2"

torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
    train/train.py \
    --output_dir ./checkpoints/$STAGE_1_MODEL_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --bf16 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --ddp_find_unused_parameters True\
    --report_to none

# torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
#     mmgpt/train/train_flash_attn.py \
#     --model_name_or_path /data/public/sharpwang/llm/llama-vicuna-7b \
#     --conversation_version v1 \
#     --conversation_datasets blip_laion_cc_sbu_558k_gen_yana \
#     --vision_tower /data/public/sharpwang/llm/clip-vit-large-patch14 \
#     --freeze_vision_tower True \
#     --freeze_lm_model True \
#     --vision_select_layer -2 \
#     --use_im_start_end \
#     --bf16 True \
#     --output_dir ./checkpoints/$STAGE_1_MODEL_NAME \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 5000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --report_to none

# # exit()

# python3 scripts/extract_mm_projector.py \
#   --model_name_or_path ./checkpoints/$STAGE_1_MODEL_NAME \
#   --output ./checkpoints/$STAGE_1_MODEL_NAME/mm_projector.bin



# aws --endpoint-url=http://oss.i.basemind.com s3 sync s3://lucaszhao/llm/clip/vit-large-patch14/ /mnt/host0/vit-large-patch14/
# --pretrained_stage1_model ./checkpoints/mmglm-7b-pretrain-v0-cc_sbu_558k+cc_sbu_558k_cn-freeze/mm_projector.bin \
# torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
#     mmgpt/train/train_flash_attn.py \
#     --model_name_or_path /data/public/sharpwang/llm/llama-vicuna-7b \
#     --conversation_datasets llava_80k+ocrvqa+Math_200 \
#     --conversation_version v1 \
#     --vision_tower /data/public/sharpwang/llm/clip-vit-large-patch14 \
#     --pretrained_stage1_model ./checkpoints/$STAGE_1_MODEL_NAME/mm_projector.bin \
#     --freeze_vision_tower True \
#     --vision_select_layer -2 \
#     --use_im_start_end True \
#     --bf16 True \
#     --output_dir ./checkpoints/$STAGE_2_MODEL_NAME \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 5000 \
#     --save_total_limit 3 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 0 \
#     --report_to none
