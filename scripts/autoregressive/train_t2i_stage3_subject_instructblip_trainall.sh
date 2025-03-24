#!/bin/bash  
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="~/MLLMG":$PYTHONPATH

# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=0
# export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1  # 启用 NCCL 异常处理
export NCCL_TIMEOUT=1200  # 调整 NCCL 超时时间（默认600秒）

experiment_name="checkpoints/Fix_posEmbedding_Trainall_llamagen_t2i_stage3_subject_instructblip-flan-t5-xl_train_subject_t2i_ti2i_120_w_flux_segment"
# model_name_or_path="/nobackup/zefan/projects/VLGen/model/blip2-flan-t5-xl"
model_name_or_path="/home/jovyan/model/instructblip-flan-t5-xl"

# data_path="/home/jovyan/MLLMG/jsonl_dir/subject_ti2i_t2i_stage1.jsonl"
# data_path="/home/jovyan/MLLMG/jsonl_dir/subject_ti2i_t2i_stage1_w_flux_segment.jsonl"
data_path="/home/jovyan/MLLMG/jsonl_dir/subject_ti2i_t2i_stage1_w_flux_segment_mid_700k.jsonl"
# val_data_path="/home/jovyan/MLLMG/jsonl_dir/new_1117_validation_set.jsonl"
val_data_path="/home/jovyan/MLLMG/jsonl_dir/new_1117_validation_set.jsonl_mid_1k.jsonl" 
load_from_checkpoint="/home/jovyan/model/llamagen_t2i/t2i_XL_stage2_512.pt"
# subject_embedding="/home/jovyan/MLLMG/subject_embedding.bin"
subject_embedding="/home/jovyan/MLLMG/subject_embedding_instructblip.pth"

lr=5e-4
num_workers=4

nnodes=1
nproc_per_node=8
node_rank=0
eval_steps=2000
ckpt_every=4000
multimodal_encoder="instructblip"
# multimodal_encoder="blip"
master_port=25000
# Training script with updated parameters and options  
# python -m debugpy --listen 12345 --wait-for-client \

torchrun \
--nnodes=1 \
--nproc_per_node=$nproc_per_node \
--node_rank=$node_rank \
autoregressive/train/train_t2i.py \
--vq-ckpt /home/jovyan/model/llamagen_t2i/vq_ds16_t2i.pt \
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
--max_eval_samples 1024 \
--cfg-scale 7.5 \
--top-k 16384 \
--load_from_checkpoint ${load_from_checkpoint} \
--global-batch-size 120 \
--num-workers ${num_workers} \
--warmup 0.05 \
--gradient-accumulation-steps 1 \
--train_text_encoder \
--ckpt-every ${ckpt_every} \
--epochs 3 \
--subject_driven \
--reference_data_path /home/jovyan/MLLMG/cc12m_reference.jsonl \
--multimodal_encoder ${multimodal_encoder} \
--find_unused_parameters \
--cls-token-num 120 \
--load_subject_embedding ${subject_embedding} \
--save_total_limit 2 \
--train_all \

# --gpt-ckpt /home/jovyan/MLLMG/checkpoint/New_multiNode_contineT2I_training_21M/006-GPT-XL/checkpoints/0120000.pt \
# --gpt-ckpt /home/jovyan/MLLMG/checkpoint/llamagen_t2i_stage3_subject_instructblip_train_only_qformer_T5_subject_t2i_ti2i/030-GPT-XL/checkpoints/0036000.pt \
# --resume \

# --no-compile \

# --gpt-ckpt /home/jovyan/MLLMG/checkpoint/llamagen_t2i_stage3_subject_instructblip_train_only_qformer_T5_subject_t2i_ti2i/028-GPT-XL



"$@"