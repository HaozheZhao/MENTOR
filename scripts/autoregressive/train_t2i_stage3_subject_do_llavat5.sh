#!/bin/bash  
node_rank=$1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="~/MLLMG":$PYTHONPATH
NCCL_P2P_DISABLE=1 
NCCL_IB_DISABLE=1 
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=0
# export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_GID_INDEX=3

experiment_name="checkpoints/llamagen_t2i_stage3_subject_instructblip_train_only_qformer_T5_subject_t2i_ti2i_120_w_flux_segment"
experiment_name="checkpoints/llavat5_subject_train_all_T5_subject_t2i_ti2i_120_w_flux_segment_noimageonly"
experiment_name="checkpoints/llavat5_subject_train_all_T5_subject_t2i_ti2i_120_w_flux_segment_noimageonly_newfix_train_only"
experiment_name="checkpoints/llavat5_subject_train_all_T5_subject_t2i_ti2i_120_w_flux_segment_hasimageonly_newfix_unfreeze_output"
# model_name_or_path="/nobackup/zefan/projects/VLGen/model/blip2-flan-t5-xl"
model_name_or_path="/tmp/haozhezhao/model/blip2-flan-t5-xl"

# data_path="/tmp/haozhezhao/MLLMG/jsonl_dir/subject_ti2i_t2i_stage1.jsonl"
data_path="/tmp/haozhezhao/MLLMG/jsonl_dir/subject_ti2i_t2i_stage1_w_flux_segment_mid_700k.jsonl"
val_data_path="/tmp/haozhezhao/MLLMG/jsonl_dir/new_1117_validation_set.jsonl_mid_1k.jsonl"
# val_data_path="/tmp/haozhezhao/MLLMG/jsonl_dir/validation_set.jsonl_mid_1k.jsonl"
load_from_checkpoint="/tmp/haozhezhao/model/llamagen_t2i/t2i_XL_stage2_512.pt"
load_from_checkpoint="/tmp/haozhezhao/MLLMG/checkpoint/FIXed_3M_1epoch_step32000.pt"
subject_embedding="/tmp/haozhezhao/MLLMG/subject_embedding.bin"
subject_embedding="/tmp/haozhezhao/MLLMG/subject_embedding_instructblip.pth"
language_projection="/tmp/haozhezhao/MLLMG/llava-v1.5-flant5_fixed-pretrain/mm_projector.bin"
lr=5e-4
num_workers=4

nnodes=2
nproc_per_node=8
eval_steps=2000
ckpt_every=2000
multimodal_encoder="llava"
MASTER_ADDR=10.0.16.13

MASTER_PORT=25000
# Training script with updated parameters and options  
# python -m debugpy --listen 12345 --wait-for-client \
# pip install nltk matplotlib opencv-python
#  bash scripts/autoregressive/train_t2i_stage3_subject_do_llavat5_node2.sh  

torchrun \
--nnodes=1 \
--nproc_per_node=$nproc_per_node \
autoregressive/train/train_t2i.py \
--vq-ckpt /tmp/haozhezhao/model/llamagen_t2i/vq_ds16_t2i.pt \
--data-path ${data_path} \
--dataset ti2i \
--image-size 512 \
--results-dir ${experiment_name} \
--cloud-save-path ~/MLLMG/checkpoints \
--lr ${lr} \
--val_data_path ${val_data_path} \
--use_vision_tower \
--model_name_or_path ${model_name_or_path} \
--image_place_holder "<image>" \
--do_eval \
--eval_steps ${eval_steps} \
--max_eval_samples 512 \
--cfg-scale 7.5 \
--top-k 16384 \
--load_from_checkpoint ${load_from_checkpoint} \
--global-batch-size 96 \
--num-workers ${num_workers} \
--warmup 0.05 \
--gradient-accumulation-steps 1 \
--train_text_encoder \
--ckpt-every ${ckpt_every} \
--epochs 5 \
--subject_driven \
--reference_data_path /tmp/haozhezhao/MLLMG/cc12m_reference.jsonl \
--multimodal_encoder ${multimodal_encoder} \
--find_unused_parameters \
--cls-token-num 512 \
--load_language_projection ${language_projection} \
--mm_vision_tower "openai/clip-vit-large-patch14" \
--save_total_limit 1 \
--load_fixed_llamagen \
--unfreeze_output \
--with_image_only \
--image_only_rate 0.1 \
--fix 'gpt-empty-fix'
# --train_all \
# --train_all \

# --gpt-ckpt /tmp/haozhezhao/Temp_CKPT_MLLMG/checkpoint/New_multiNode_contineT2I_training_21M/006-GPT-XL/checkpoints/0120000.pt \

# --gpt-ckpt /tmp/haozhezhao/MLLMG/checkpoint/llamagen_t2i_stage3_subject_instructblip_train_only_qformer_T5_subject_t2i_ti2i/030-GPT-XL/checkpoints/0036000.pt \
# --resume \

# --no-compile \

# --gpt-ckpt /tmp/haozhezhao/MLLMG/checkpoint/llamagen_t2i_stage3_subject_instructblip_train_only_qformer_T5_subject_t2i_ti2i/028-GPT-XL

# --load_subject_embedding ${subject_embedding} \




