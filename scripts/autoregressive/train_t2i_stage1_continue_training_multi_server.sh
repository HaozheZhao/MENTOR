#!/bin/bash  
node_rank=$1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PYTHONNOUSERSITE=True
export PYTHONPATH="~/MLLMG":$PYTHONPATH
NCCL_P2P_DISABLE=1 
NCCL_IB_DISABLE=1 

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
experiment_name="checkpoint/llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_trainall_ckpt_recap_Subject200k_filtered_t2i_600k_recovery_200k_extract_100k_trainall_1e4"
experiment_name="checkpoint/llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_trainall_ckpt_recap_Subject200k_filtered_t2i_600k_recovery_200k_extract_200k_trainall_1e4"
experiment_name="checkpoint/New_start_train_only_qformer_T5_subject_t2i_ti2i_ckpt48000_llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_trainall_ckpt_recap_Subject200k_filtered_t2i_200k_flux400k_recovery_200k_extract_100k_trainall_1e4"
experiment_name="checkpoint/New_start_train_only_qformer_T5_subject_t2i_ti2i_ckpt48000_llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_trainall_ckpt_recap_Subject200k_filtered_t2i_flux600k_recovery_200k_extract_100k_trainall_1e4"
experiment_name="checkpoint/New_multiNode_contineT2I_training_21M"

# model_name_or_path="/nobackup/zefan/projects/VLGen/model/blip2-flan-t5-xl"
model_name_or_path="/tmp/haozhezhao/model/instructblip-flan-t5-xl"

# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/new_1119_train_set_mixed.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/train_set_subject_200k.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/train_set_subject_200k_mixed.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/train_set_subject_200k_mixed_filtered.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/merged_train_set_set_subject_200k_recap_t2i_300k_midjourney.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/train_set_subject_200k_recap.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/merged_train_set_set_subject_200k_recap_t2i_300k_midjourney_200k_grounding_200k_recovery.jsonl"
data_path="/tmp/haozhezhao/MLLMG/jsonl_data/t2i_extend_training.jsonl"
# data_path="/tmp/haozhezhao/MLLMG/jsonl_dir/midjourney-niji-1m-llavanext.jsonl"

val_data_path="/tmp/haozhezhao/MLLMG/jsonl_data/t2i_extend_valid.jsonl"
# val_data_path="/nobackup/zefan/projects/VLGen/LlamaGen/eval_test.jsonl"
load_from_checkpoint="/tmp/haozhezhao/model/llamagen_t2i/t2i_XL_stage2_512.pt"
subject_embedding="/tmp/haozhezhao/MLLMG/subject_embedding.bin"
subject_embedding="/tmp/haozhezhao/MLLMG/subject_embedding_instructblip.pth"

lr=3e-4
num_workers=6

nnodes=2
nproc_per_node=8
eval_steps=2000
ckpt_every=4000
multimodal_encoder="instructblip"

MASTER_ADDR=10.0.16.5

MASTER_PORT=25001
# Training script with updated parameters and options  
# sleep 2h before lanch

torchrun \
--nnodes ${nnodes} \
--nproc_per_node=$nproc_per_node \
--node_rank ${node_rank} \
--rdzv_backend c10d \
--rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
--master_addr ${MASTER_ADDR} \
--master_port ${MASTER_PORT} \
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
--global-batch-size 224 \
--num-workers ${num_workers} \
--warmup 0.02 \
--gradient-accumulation-steps 1 \
--ckpt-every ${ckpt_every} \
--epochs 3 \
--subject_driven \
--reference_data_path /tmp/haozhezhao/data/dalle3-llama3.2-11b/reference.jsonl \
--multimodal_encoder ${multimodal_encoder} \
--find_unused_parameters \
--continue_stage1 \
--cls-token-num 512 \
--gpt-ckpt /tmp/haozhezhao/MLLMG/checkpoint/New_multiNode_contineT2I_training_21M/006-GPT-XL/checkpoints/0120000.pt \
--resume \

# --do_recovery \

# --no_replace \

# --train_text_encoder \
 
# --dreambench_eval \
# --gpt-ckpt /tmp/haozhezhao/MLLMG/checkpoint/New_start_train_only_qformer_T5_subject_t2i_ti2i_ckptv2_36000_llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_trainall_ckpt_recap_Subject400k_filtered_t2i_flux400k_recovery_100k_extract_100k_trainall_1e4/000-GPT-XL/checkpoints/0030000.pt \
# --resume \
# --gpt-ckpt /tmp/haozhezhao/MLLMG/checkpoint/llamagen_t2i_stage3_subject_instructblip_train_only_qformer_T5_subject_t2i_ti2i/004-GPT-XL/checkpoints/0036000.pt \

# --gpt-ckpt /tmp/haozhezhao/MLLMG/checkpoint/llamagen_t2i_stage3_subject_instructblip_train_only_qformer_T5_subject_t2i_ti2i/031-GPT-XL/checkpoints/0048000.pt \
# --no-compile \

# --gpt-ckpt /nobackup/zefan/projects/VLGen/LlamaGen/llamagen_t2i_stage3_subject_instructblip_train_except_t5_vit_llamagen_alllayers_no_norm_output/003-GPT-XL/checkpoints/0044000.pt \
# --load_visual_encoder \
# /nobackup/zefan/projects/VLGen/LlamaGen/llamagen_t2i_stage3_subject_instructblip_freezeqformer_train_llamagen_projection/008-GPT-XL/checkpoints/0028000.pt
# /nobackup/zefan/projects/VLGen/LlamaGen/llamagen_t2i_stage3_subject_instructblip_train_except_t5_vit_llamagen_alllayers_no_norm_output/003-GPT-XL/checkpoints/0046000.pt
# --gpt-ckpt /nobackup/zefan/projects/VLGen/LlamaGen/llamagen_t2i_stage3_subject_instructblip_train_only_qformer_no_norm_output/000-GPT-XL/checkpoints/0012000.pt \

# --resume \
# --gpt-ckpt /nobackup/zefan/projects/VLGen/LlamaGen/llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_Subject200k_mix_300k_t2i_trainall/002-GPT-XL/checkpoints/0004000.pt \

# --load_subject_embedding ${subject_embedding} \





# import json
# import random 
# from tqdm import tqdm

# # File paths
# file1 = '/tmp/haozhezhao/MLLMG/merged_train_set_set_subject_200k_recap_t2i_600k_midjourney_200k_recovery.jsonl'
# file2 = '/tmp/haozhezhao/MLLMG/jsonl_data/merged_train_set_set_subject_200k_recap_t2i_300k_midjourney_200k_grounding.jsonl'
# output_file = '/tmp/haozhezhao/MLLMG/merged_train_set_set_subject_200k_recap_t2i_600k_midjourney_200k_recovery_200k_grounding.jsonl'

# def replace_paths(data_entry):
#     for key in ["source_image", "image_path"]:
#         if key in data_entry and data_entry[key]:
#             data_entry[key] = data_entry[key].replace("/nobackup/zefan/projects/VLGen/","/tmp/haozhezhao/data/")
#     return data_entry

# # Load data from JSONL file
# def load_jsonl(file_path):
#     with open(file_path, 'r') as file:
#         return [json.loads(line) for line in file]

# # Save data to JSONL file
# def save_jsonl(data, file_path):
#     with open(file_path, 'w') as file:
#         for entry in data:
#             file.write(json.dumps(entry) + '\n')

# # Apply default values to entries
# def apply_defaults(entry):
#     entry['objects'] = entry.get("objects", None)
#     entry['do_mask'] = entry.get("do_mask", False)
#     entry['do_replace'] = entry.get("do_replace", False)
#     entry['generation_only'] = entry.get("generation_only", False)
#     return entry
# # Main processing
# try:
#     # Load the first file
#     left_a = load_jsonl(file1)

#     # Filter for left_a
#     # left_a = [
#     #     apply_defaults(entry) for entry in data1
#     #     if not entry.get("generation_only", False) and not entry.get("do_mask", False) and not entry.get("do_replace", False)
#     # ]

#     data2 = load_jsonl(file2)
#     # Filter for left_b
#     left_b = [apply_defaults(entry) for entry in data2 if "Subjects200K_images" not in entry['source_image'] and entry.get("do_mask", False)]
#     # left_b = random.sample(do_mask_data, len(do_mask_data)//2)

#     # Load the second file

#     # Filter for left_c
#     # left_c = [apply_defaults(entry) for entry in data2 if entry.get("generation_only", False)]

#     # Combine all datasets
#     print(len(left_a),len(left_b))
#     combined_data = left_a + left_b 

#     # Modify paths and apply replacements
#     for entry in tqdm(combined_data, desc="Processing entries"):
#         replace_paths(entry)

#     # Shuffle the combined data
#     random.shuffle(combined_data)

#     # Save the final data
#     save_jsonl(combined_data, output_file)
#     print(f"Data processing complete. Output saved to: {output_file}")

# except FileNotFoundError as e:
#     print(f"File not found: {e}")
