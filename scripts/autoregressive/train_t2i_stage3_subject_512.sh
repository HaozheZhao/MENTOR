#!/bin/bash  
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="~/MLLMG":$PYTHONPATH

# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=0
# export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_GID_INDEX=3

experiment_name="checkpoint/llamagen_t2i_stage3_subject_instructblip_train_only_qformer_T5_subject_t2i_ti2i_512_ckpt_120000"
# model_name_or_path="/nobackup/zefan/projects/VLGen/model/blip2-flan-t5-xl"
model_name_or_path="/tmp/haozhezhao/model/instructblip-flan-t5-xl"

data_path="/tmp/haozhezhao/MLLMG/jsonl_dir/subject_ti2i_t2i_stage1.jsonl"
val_data_path="/tmp/haozhezhao/MLLMG/jsonl_dir/new_1117_validation_set.jsonl"
load_from_checkpoint="/tmp/haozhezhao/model/llamagen_t2i/t2i_XL_stage2_512.pt"
subject_embedding="/tmp/haozhezhao/MLLMG/subject_embedding.bin"
subject_embedding="/tmp/haozhezhao/MLLMG/subject_embedding_instructblip.pth"

lr=1e-3
num_workers=2

nnodes=2
nproc_per_node=8
node_rank=0
eval_steps=1000
ckpt_every=2000
multimodal_encoder="instructblip"
master_port=25000
# Training script with updated parameters and options  
# python -m debugpy --listen 12345 --wait-for-client \

torchrun \
--nnodes=1 \
--nproc_per_node=$nproc_per_node \
--node_rank=$node_rank \
autoregressive/train/train_t2i.py \
--vq-ckpt /tmp/haozhezhao/model/llamagen_t2i/vq_ds16_t2i.pt \
--data-path ${data_path} \
--dataset ti2i \
--image-size 512 \
--results-dir ${experiment_name} \
--cloud-save-path ~/MLLMG/checkpoint \
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
--global-batch-size 96 \
--num-workers ${num_workers} \
--warmup 0.2 \
--gradient-accumulation-steps 1 \
--train_text_encoder \
--ckpt-every ${ckpt_every} \
--epochs 10 \
--subject_driven \
--reference_data_path /tmp/haozhezhao/MLLMG/reference.jsonl \
--multimodal_encoder ${multimodal_encoder} \
--find_unused_parameters \
--cls-token-num 512 \
--gpt-ckpt /tmp/haozhezhao/MLLMG/checkpoint/New_multiNode_contineT2I_training_21M/006-GPT-XL/checkpoints/0120000.pt \
# --gpt-ckpt /tmp/haozhezhao/MLLMG/checkpoint/llamagen_t2i_stage3_subject_instructblip_train_only_qformer_T5_subject_t2i_ti2i/030-GPT-XL/checkpoints/0036000.pt \
# --resume \

# --no-compile \

# --gpt-ckpt /tmp/haozhezhao/MLLMG/checkpoint/llamagen_t2i_stage3_subject_instructblip_train_only_qformer_T5_subject_t2i_ti2i/028-GPT-XL

# --load_subject_embedding ${subject_embedding} \


"$@"