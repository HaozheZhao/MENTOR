#!/bin/bash  
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PYTHONNOUSERSITE=True
export PYTHONPATH="~/MLLMG":$PYTHONPATH

# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=0
# export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_GID_INDEX=3

# experiment_name="llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_mixdata_trainall"
# experiment_name="test"
# experiment_name="llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_Subject200k_mix_300k_t2i_trainall_high_lr_5e4"
# experiment_name="llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_trainall_ckpt_recap_Subject200k_filtered_mix_300k_t2i_trainall_1e4"
# experiment_name="llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_trainall_ckpt_recap_Subject200k_filtered_only_t2i_trainall_1e4"
# experiment_name="llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_trainall_ckpt_recap_Subject200k_filtered_t2i_300k_extract_200k_recovery_200k_trainall_1e4"
# experiment_name="llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_trainall_ckpt_recap_Subject200k_filtered_t2i_300k_extract_200k_trainall_1e4"
experiment_name="checkpoint/llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_trainall_ckpt_recap_Subject200k_filtered_t2i_300k_extract_100k_trainall_1e4"

# model_name_or_path="/nobackup/zefan/projects/VLGen/model/blip2-flan-t5-xl"
model_name_or_path="/tmp/haozhezhao/model/instructblip-flan-t5-xl"

# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/new_1119_train_set_mixed.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/train_set_subject_200k.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/train_set_subject_200k_mixed.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/train_set_subject_200k_mixed_filtered.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/merged_train_set_set_subject_200k_recap_t2i_300k_midjourney.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/train_set_subject_200k_recap.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/merged_train_set_set_subject_200k_recap_t2i_300k_midjourney_200k_grounding_200k_recovery.jsonl"
data_path="/tmp/haozhezhao/MLLMG/merged_train_set_set_subject_200k_recap_t2i_600k_midjourney_200k_recovery.jsonl"
val_data_path="/tmp/haozhezhao/MLLMG/jsonl_data/dreambench_plus_valid.jsonl"
# val_data_path="/nobackup/zefan/projects/VLGen/LlamaGen/eval_test.jsonl"
load_from_checkpoint="/tmp/haozhezhao/model/llamagen_t2i/t2i_XL_stage2_512.pt"
subject_embedding="/tmp/haozhezhao/MLLMG/subject_embedding.bin"
subject_embedding="/tmp/haozhezhao/MLLMG/subject_embedding_instructblip.pth"

lr=1e-4
num_workers=64

nnodes=2
nproc_per_node=8
node_rank=0
eval_steps=1000
ckpt_every=1000
multimodal_encoder="instructblip"
master_port=25000
# Training script with updated parameters and options  
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
--max_eval_samples 512 \
--cfg-scale 7.5 \
--top-k 16384 \
--load_from_checkpoint ${load_from_checkpoint} \
--global-batch-size 96 \
--num-workers ${num_workers} \
--warmup 0.05 \
--gradient-accumulation-steps 4 \
--train_text_encoder \
--ckpt-every ${ckpt_every} \
--epochs 3 \
--subject_driven \
--reference_data_path /tmp/haozhezhao/data/dalle3-llama3.2-11b/reference.jsonl \
--multimodal_encoder ${multimodal_encoder} \
--do_recovery \
--find_unused_parameters \
--no_replace \
--dreambench_eval \
--gpt-ckpt /tmp/haozhezhao/MLLMG/llamagen_t2i_stage3_subject_instructblip_train_except_t5_vit_llamagen_alllayers_no_norm_output/003-GPT-XL/checkpoints/0044000.pt \
--no-compile \

# --gpt-ckpt /nobackup/zefan/projects/VLGen/LlamaGen/llamagen_t2i_stage3_subject_instructblip_train_except_t5_vit_llamagen_alllayers_no_norm_output/003-GPT-XL/checkpoints/0044000.pt \
# --load_visual_encoder \
# /nobackup/zefan/projects/VLGen/LlamaGen/llamagen_t2i_stage3_subject_instructblip_freezeqformer_train_llamagen_projection/008-GPT-XL/checkpoints/0028000.pt
# /nobackup/zefan/projects/VLGen/LlamaGen/llamagen_t2i_stage3_subject_instructblip_train_except_t5_vit_llamagen_alllayers_no_norm_output/003-GPT-XL/checkpoints/0046000.pt
# --gpt-ckpt /nobackup/zefan/projects/VLGen/LlamaGen/llamagen_t2i_stage3_subject_instructblip_train_only_qformer_no_norm_output/000-GPT-XL/checkpoints/0012000.pt \

# --resume \
# --gpt-ckpt /nobackup/zefan/projects/VLGen/LlamaGen/llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_Subject200k_mix_300k_t2i_trainall/002-GPT-XL/checkpoints/0004000.pt \

# --load_subject_embedding ${subject_embedding} \


"$@"