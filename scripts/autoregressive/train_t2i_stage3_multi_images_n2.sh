#!/bin/bash  
set -x
export PYTHONPATH="/home/zhaohaozhe/projects/LlamaGen":$PYTHONPATH

export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_GID_INDEX=3

experiment_name="llamagen_t2i_stage3_multi"
model_name_or_path="/scratch2/ml/model/blip2-flan-t5-xl"

data_path="/home/zhaohaozhe/data2/kosmos_openimage/"
val_data_path="/home/zhaohaozhe/data2/kosmos_openimage/"
load_from_checkpoint="/home/zhaohaozhe/projects/LlamaGen/llamagen_t2i_stage3/026-GPT-XL/checkpoints/0006000.pt"

lr=1e-5
num_workers=64

nnodes=2
nproc_per_node=8
node_rank=1
eval_steps=500
ckpt_every=1000
master_addr="dgx-hyperplane19"
master_port=25045
# Training script with updated parameters and options  
# torchrun \
# --nnodes=$nnodes \
# --nproc_per_node=$nproc_per_node \
# --node_rank=$node_rank \
# --master_addr=$master_addr \
# --master_port=$master_port \

# python3 -m debugpy --listen 5678 --wait-for-client \


#torchrun \
#--nnodes=1 \
#--nproc_per_node=$nproc_per_node \
#--node_rank=$node_rank \
torchrun \
--nnodes=$nnodes \
--nproc_per_node=$nproc_per_node \
--node_rank=$node_rank \
--master_addr=$master_addr \
--master_port=$master_port \
autoregressive/train/train_t2i.py \
--vq-ckpt /home/zhaohaozhe/model2//llamagen/vq_ds16_t2i.pt \
--data-path ${data_path} \
--dataset ti2i \
--image-size 512 \
--results-dir ${experiment_name} \
--cloud-save-path /home/zhaohaozhe/projects/LlamaGen \
--lr ${lr} \
--val_data_path ${val_data_path} \
--use_vision_tower \
--model_name_or_path ${model_name_or_path} \
--image_place_holder "<image>" \
--do_eval \
--eval_steps ${eval_steps} \
--max_eval_samples 192 \
--cfg-scale 7.5 \
--top-k 5000 \
--load_from_checkpoint ${load_from_checkpoint} \
--global-batch-size 192 \
--gradient-accumulation-steps 2 \
--num-workers ${num_workers} \
--warmup 0.01 \
--train_text_encoder \
--ckpt-every ${ckpt_every} \
--stage2 \
--epochs 12 \
--gpt-ckpt /home/zhaohaozhe/projects/LlamaGen/llamagen_t2i_stage3_multi/000-GPT-XL/checkpoints/0005000.pt \

"$@"