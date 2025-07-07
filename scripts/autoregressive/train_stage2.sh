#!/bin/bash  
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="~/MENTOR":$PYTHONPATH

export NCCL_DEBUG=ERROR

experiment_name="checkpoint/train_result"

model_name_or_path="model/blip2-flan-t5-xl"

data_path="MENTOR_stage1/train_stage2.jsonl"

val_data_path="jsonl_data/valid.jsonl"
load_from_checkpoint="MENTOR/generator_ckpt.pt"

language_projection="llava-v1.5-flant5_fixed-pretrain/mm_projector.bin"

lr=1e-4
num_workers=8

nproc_per_node=8
node_rank=0
eval_steps=2000
ckpt_every=2000
multimodal_encoder="llava"

torchrun \
--nnodes=1 \
--nproc_per_node=$nproc_per_node \
--node_rank=$node_rank \
autoregressive/train/train_t2i.py \
--vq-ckpt model/llamagen_t2i/vq_ds16_t2i.pt \
--data-path ${data_path} \
--dataset ti2i \
--image-size 512 \
--results-dir ${experiment_name} \
--cloud-save-path checkpoint \
--lr ${lr} \
--val_data_path ${val_data_path} \
--use_vision_tower \
--model_name_or_path ${model_name_or_path} \
--image_place_holder "<image>" \
--do_eval \
--eval_steps ${eval_steps} \
--max_eval_samples 500 \
--cfg-scale 7.5 \
--top-k 16384 \
--load_from_checkpoint ${load_from_checkpoint} \
--global-batch-size 64 \
--num-workers ${num_workers} \
--warmup 0.05 \
--gradient-accumulation-steps 4 \
--train_text_encoder \
--ckpt-every ${ckpt_every} \
--epochs 2 \
--subject_driven \
--reference_data_path MENTOR/cc12m_reference_tunnel.jsonl \
--multimodal_encoder ${multimodal_encoder} \
--do_recovery \
--find_unused_parameters \
--cls-token-num 512 \
--dreambench_eval \
--save_total_limit 1 \
--load_language_projection ${language_projection} \
--gpt-ckpt generator_ckpt.pt \
--mm_vision_tower "openai/clip-vit-large-patch14" \
--train_all \
--load_fixed_llamagen \
--fix 'gpt-empty-fix' \

"$@"

