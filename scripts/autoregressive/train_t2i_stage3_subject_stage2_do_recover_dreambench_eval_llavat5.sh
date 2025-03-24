#!/bin/bash  
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PYTHONNOUSERSITE=True
export PYTHONPATH="~/MLLMG":$PYTHONPATH

export NCCL_DEBUG=ERROR
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
experiment_name="checkpoint/New_start_train_only_qformer_T5_subject_t2i_ti2i_ckptv2_36000_llamagen_t2i_stage3_subject_instructblip_train_all_stage2_recover_dreambench_eval_trainall_ckpt_recap_Subject400k_filtered_t2i_flux400k_recovery_100k_extract_100k_trainall_1e4"
experiment_name="checkpoint/New_start_With_FLUX_train_only_qformer_T5_subject_t2i_ti2i_ckptv2_0054000_llamagen_t2i_stage3_dreambench_eval_trainall_ckpt_recap_Subject200k_filtered_t2i_flux400k_200kmidjourney_recovery_150k_extract_150k_100_fluxseg_50samseg_trainall_1e4"
experiment_name="checkpoint/New_FLUX_train_only_qformer_T5_subject_t2i_ti2i_ckptv2_54000_Change_Subject_stage3_dreambench_recap_Subject200k_filtered_t2i_flux400k_200kmid_recovery_150k_extract_150k_100_fluxseg_50samseg_trainall_1e4"

# experiment_name="checkpoint/BLIP2Trainall_stage3_dreambench_recap_Subject200k_filtered_t2i_flux400k_200kmid_recovery_150k_extract_150k_100_fluxseg_50samseg_trainall_1e4"

experiment_name="checkpoint/LLAVAT5Trainall_just_segment_pretrain_stage3_dreambench_recap_Subject400k_filtered_t2i_flux400k_200kmid_recovery_150k_extract_150k_100_fluxseg_50samseg_trainall_1e4_doreplace"

experiment_name="checkpoint/LLAVAT5Trainall_just_segment_pretrain_stage3_dreambench_recap_Subject400k_No_t2i_recovery_150k_extract_150k_100_fluxseg_50samseg_trainall_1e4_doreplace"

experiment_name="checkpoint/LLAVAT5Trainall_just_segment_pretrain_stage3_dreambench__recap_200k_t2i_400k_flux_200k_midjourney_150k_recovery_150k_grounding_100fluxseg_50samseg"

experiment_name="checkpoint/LLAVAT5Fixed_just_segment_pretrain_stage3_dreambench__recap_200k_t2i_400k_flux_200k_midjourney_150k_recovery_150k_grounding_100fluxseg_50samseg"


# model_name_or_path="/nobackup/zefan/projects/VLGen/model/blip2-flan-t5-xl"
model_name_or_path="/home/jovyan/model/blip2-flan-t5-xl"

# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/new_1119_train_set_mixed.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/train_set_subject_200k.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/train_set_subject_200k_mixed.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/train_set_subject_200k_mixed_filtered.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/merged_train_set_set_subject_200k_recap_t2i_300k_midjourney.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/train_set_subject_200k_recap.jsonl"
# data_path="/nobackup/zefan/projects/VLGen/LlamaGen/merged_train_set_set_subject_200k_recap_t2i_300k_midjourney_200k_grounding_200k_recovery.jsonl"
data_path="/tmp/haozhezhao/MLLMG/merged_train_set_set_subject_200k_recap_t2i_600k_midjourney_200k_recovery.jsonl"
data_path="/tmp/haozhezhao/MLLMG/merged_train_set_set_subject_200k_recap_t2i_600k_midjourney_200k_recovery_100k_grounding.jsonl"    
data_path="/tmp/haozhezhao/MLLMG/merged_train_set_set_subject_200k_recap_t2i_600k_midjourney_200k_recovery_200k_grounding.jsonl"
data_path="/tmp/haozhezhao/MLLMG/merged_train_set_set_subject_200k_recap_t2i_600k_400k_flux_200k_midjourney_200k_recovery_100k_grounding.jsonl"
data_path="/tmp/haozhezhao/MLLMG/merged_train_set_set_subject_200k_recap_t2i_600k_flux_200k_recovery_100k_grounding.jsonl"
data_path="/tmp/haozhezhao/MLLMG/merged_train_set_set_subject_200k_recap_t2i_300k_flux_100k_recovery_100k_grounding.jsonl"
data_path="/home/jovyan/MLLMG/merged_train_set_set_subject_400k_recap_t2i_400k_flux_100k_recovery_100k_grounding_server2.jsonl"
data_path="/home/jovyan/MLLMG/jsonl_data/merged_train_set_set_subject_200k_recap_t2i_400k_flux_200k_midjourney_150k_recovery_150k_grounding_100fluxseg_50samseg.jsonl"
# data_path="/home/jovyan/MLLMG/jsonl_data/merged_train_set_set_subject_400k_recap_t2i_400k_flux_200k_midjourney_150k_recovery_150k_grounding_100fluxseg_50samseg.jsonl"
# data_path="/home/jovyan/MLLMG/jsonl_data/merged_train_set_set_subject_400k_recap_no_t2i_recovery_150k_grounding_100fluxseg_50samseg.jsonl"
# data_path="/home/jovyan/MLLMG/jsonl_data/merged_train_set_set_subject_400k_recap.jsonl"
val_data_path="/home/jovyan/MLLMG/jsonl_dir/dreambench_plus_valid.jsonl"
# val_data_path="/nobackup/zefan/projects/VLGen/LlamaGen/eval_test.jsonl"
load_from_checkpoint="/home/jovyan/model/llamagen_t2i/t2i_XL_stage2_512.pt"
load_from_checkpoint="/home/jovyan/MLLMG/checkpoint/FIXed_3M_1epoch_step32000.pt"

subject_embedding="/home/jovyan/MLLMG/subject_embedding.bin"
# subject_embedding="/home/jovyan/MLLMG/subject_embedding_instructblip.pth"
language_projection="/home/jovyan/LLaVA/checkpoints/llava-v1.5-flant5_fixed-pretrain/mm_projector.bin"

lr=1e-4
num_workers=8

nnodes=2
nproc_per_node=8
node_rank=0
eval_steps=2000
ckpt_every=2000
multimodal_encoder="llava"
master_port=25000
# Training script with updated parameters and options  
# sleep 2h before lanch

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
--cloud-save-path ~/MLLMG/checkpoint \
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
--global-batch-size 72 \
--num-workers ${num_workers} \
--warmup 0.05 \
--gradient-accumulation-steps 4 \
--train_text_encoder \
--ckpt-every ${ckpt_every} \
--epochs 4 \
--subject_driven \
--reference_data_path /home/jovyan/MLLMG/cc12m_reference.jsonl \
--multimodal_encoder ${multimodal_encoder} \
--do_recovery \
--find_unused_parameters \
--cls-token-num 512 \
--dreambench_eval \
--save_total_limit 2 \
--load_language_projection ${language_projection} \
--gpt-ckpt /home/jovyan/MLLMG/checkpoints/llavat5_subject_train_all_T5_subject_t2i_ti2i_120_w_flux_segment_noimageonly_newfix_train_only/003-GPT-XL/checkpoints/0056000.pt \
--mm_vision_tower "openai/clip-vit-large-patch14" \
--train_all \
--load_fixed_llamagen \

# --load_subject_embedding ${subject_embedding} \
# --replace_subject \

# --gpt-ckpt /home/jovyan/MLLMG/llamagen_t2i_stage3_subject_instructblip_train_except_t5_vit_llamagen_alllayers_no_norm_output/003-GPT-XL/checkpoints/0044000.pt \
# --replace_subject \
# --gpt-ckpt /home/jovyan/MLLMG/checkpoint/BLIP2Trainall_stage3_dreambench_recap_Subject200k_filtered_t2i_flux400k_200kmid_recovery_150k_extract_150k_100_fluxseg_50samseg_trainall_1e4/004-GPT-XL/checkpoints/0008000.pt \
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



"$@"



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
