#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="~/MLLMG":$PYTHONPATH
export PATH=/tmp/haozhezhao/anaconda3/envs/nlp/bin:$PATH
export NCCL_DEBUG=ERROR

model_name_or_path="/tmp/haozhezhao/model/blip2-flan-t5-xl"
val_data_path="/tmp/haozhezhao/MLLMG/jsonl_data/dreambench_plus_valid.jsonl"
load_from_checkpoint="/tmp/haozhezhao/MLLMG/checkpoint/FIXed_3M_1epoch_step32000.pt"
subject_embedding="/tmp/haozhezhao/MLLMG/subject_embedding.bin"
language_projection="/tmp/haozhezhao/MLLMG/llava-v1.5-flant5_fixed-pretrain/mm_projector.bin"
lr=1e-4
num_workers=8
nnodes=2
nproc_per_node=8
node_rank=0
eval_steps=2000
ckpt_every=2000
multimodal_encoder="llava"
master_port=25000

# Arrays of experiments and data paths
experiment_names=(
  # "checkpoint/Abblation_LLAVAT5Fixed_just_segment_pretrain_stage3_dreambench_recap_400k_t2i_400k_flux_200k_midjourney_150k_recovery_150k_No_segment"
  # "checkpoint/Abblation_LLAVAT5Fixed_just_segment_pretrain_stage3_dreambench_recap_400k_t2i_400k_flux_200k_midjourney_150k_grounding_100fluxseg_50samseg_No_recovery"
  "checkpoint/Abblation_LLAVAT5Fixed_just_segment_pretrain_stage3_dreambench_150k_recovery_150k__grounding_100fluxseg_50samseg_No_t2i"
)

data_paths=(
  # "/tmp/haozhezhao/MLLMG/jsonl_data/Ablation_merged_train_set_set_subject_400k_recap_t2i_400k_flux_200k_midjourney_150k_recovery_150k_Noseg.jsonl"
  # "/tmp/haozhezhao/MLLMG/jsonl_data/Ablation_merged_train_set_set_subject_400k_recap_t2i_400k_flux_200k_midjourney_150k_grounding_100fluxseg_50samseg_No_recover.jsonl"
  "/tmp/haozhezhao/MLLMG/jsonl_data/merged_train_set_set_subject_400k_recap_recovery_150k_grounding_100fluxseg_50samseg.jsonl"
)

for i in ${!experiment_names[@]}; do
  experiment_name=${experiment_names[$i]}
  data_path=${data_paths[$i]}

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
  --cloud-save-path /tmp/haozhezhao/checkpoint \
  --lr ${lr} \
  --val_data_path ${val_data_path} \
  --use_vision_tower \
  --model_name_or_path ${model_name_or_path} \
  --image_place_holder "<image>" \
  --do_eval \
  --eval_steps ${eval_steps} \
  --max_eval_samples 200 \
  --cfg-scale 7.5 \
  --top-k 16384 \
  --load_from_checkpoint ${load_from_checkpoint} \
  --global-batch-size 56 \
  --num-workers ${num_workers} \
  --warmup 0.05 \
  --gradient-accumulation-steps 4 \
  --train_text_encoder \
  --ckpt-every ${ckpt_every} \
  --epochs 2 \
  --subject_driven \
  --reference_data_path /tmp/haozhezhao/MLLMG/cc12m_reference_tunnel.jsonl \
  --multimodal_encoder ${multimodal_encoder} \
  --do_recovery \
  --find_unused_parameters \
  --cls-token-num 512 \
  --dreambench_eval \
  --save_total_limit 1 \
  --load_language_projection ${language_projection} \
  --gpt-ckpt /tmp/haozhezhao/MLLMG/0100000.pt \
  --mm_vision_tower "openai/clip-vit-large-patch14" \
  --train_all \
  --load_fixed_llamagen \
  --fix 'gpt-empty-fix'

  # Upload checkpoint
  # cd ${experiment_name}
  # basename=$(basename ${experiment_name})
  # huggingface-cli upload BleachNick/MLLMG_ckpts . ${basename}/
  # cd /tmp/haozhezhao/MLLMG
done

# Run extra script
# bash scripts/autoregressive/train_t2i_stage3_subject_stage2_do_recover_dreambench_eval_llavat5_skip_stage1.sh

# # Upload final checkpoint
# final_ckpt_dir="checkpoint/Ablation_Skip_stage1_LLAVAT5_empty_Fixed_just_segment_pretrain_stage3_dreambench_recap_400k_t2i_400k_flux_200k_midjourney_150k_recovery_150k_1e4_epoch2"
# cd $final_ckpt_dir
# huggingface-cli upload BleachNick/MLLMG_ckpts . $(basename $final_ckpt_dir)/
