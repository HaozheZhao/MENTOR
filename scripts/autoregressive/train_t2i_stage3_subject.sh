#!/bin/bash  
set -x
export PYTHONPATH="/nobackup/zefan/projects/VLGen/LlamaGen":$PYTHONPATH

# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=0
# export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_GID_INDEX=3

experiment_name="llamagen_t2i_stage3_subject_instructblip"
# model_name_or_path="/nobackup/zefan/projects/VLGen/model/blip2-flan-t5-xl"
model_name_or_path="/nobackup/zefan/projects/VLGen/model/instructblip-flan-t5-xl"

data_path="/nobackup/zefan/projects/VLGen/LlamaGen/training_set.jsonl"
val_data_path="/nobackup/zefan/projects/VLGen/LlamaGen/validation_set.jsonl"
load_from_checkpoint="/nobackup/zefan/projects/VLGen/model/llamagen/t2i_XL_stage2_512.pt"
subject_embedding="/nobackup/zefan/projects/VLGen/model/subject_embedding.bin"
subject_embedding="/nobackup/zefan/projects/VLGen/model/subject_embedding_instructblip.pth"

lr=1e-4
num_workers=32

nnodes=2
nproc_per_node=8
node_rank=0
eval_steps=500
ckpt_every=2000
master_addr="dgx-hyperplane15"
multimodal_encoder="instructblip"
master_port=25000
# Training script with updated parameters and options  
# torchrun \
# --nnodes=$nnodes \
# --nproc_per_node=$nproc_per_node \
# --node_rank=$node_rank \
# --master_addr=$master_addr \
# --master_port=$master_port \

# python3 -m debugpy --listen 5678 --wait-for-client \
# torchrun \
# --nnodes=$nnodes \
# --nproc_per_node=$nproc_per_node \
# --node_rank=$node_rank \
# --master_addr=$master_addr \
# --master_port=$master_port \
torchrun \
--nnodes=1 \
--nproc_per_node=$nproc_per_node \
--node_rank=$node_rank \
autoregressive/train/train_t2i.py \
--vq-ckpt /nobackup/zefan/projects/VLGen/model/llamagen/vq_ds16_t2i.pt \
--data-path ${data_path} \
--dataset ti2i \
--image-size 512 \
--results-dir ${experiment_name} \
--cloud-save-path /nobackup/zefan/projects/VLGen/LlamaGen \
--lr ${lr} \
--val_data_path ${val_data_path} \
--use_vision_tower \
--model_name_or_path ${model_name_or_path} \
--image_place_holder "<image>" \
--do_eval \
--eval_steps ${eval_steps} \
--max_eval_samples 256 \
--cfg-scale 2.0 \
--top-k 5000 \
--load_from_checkpoint ${load_from_checkpoint} \
--global-batch-size 64 \
--gradient-accumulation-steps 2 \
--num-workers ${num_workers} \
--warmup 0.05 \
--gradient-accumulation-steps 1 \
--train_text_encoder \
--ckpt-every ${ckpt_every} \
--epochs 2 \
--subject_driven \
--reference_data_path /nobackup/zefan/projects/VLGen/LlamaGen/reference.jsonl \
--multimodal_encoder ${multimodal_encoder} \
# --load_subject_embedding ${subject_embedding} \

# --gpt-ckpt /nobackup/zefan/projects/VLGen/LlamaGen/llamagen_t2i_stage3_subject/043-GPT-XL/checkpoints/0034000.pt \

"$@"