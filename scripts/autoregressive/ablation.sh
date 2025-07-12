#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="~/MENTOR":$PYTHONPATH
export NCCL_DEBUG=ERROR

model_name_or_path="model/blip2-flan-t5-xl"
val_data_path="jsonl_data/valid.jsonl"
load_from_checkpoint="MENTOR/generator_ckpt.pt"
language_projection="MENTOR/llava-v1.5-flant5_fixed-pretrain/mm_projector.bin"
lr=1e-4
num_workers=8
nnodes=2
nproc_per_node=8
node_rank=0
eval_steps=2000
ckpt_every=2000
multimodal_encoder="llava"

# Arrays of experiments and data paths
experiment_names=(
  "checkpoint/Abblation_No_segment"
  "checkpoint/Abblation_No_recovery"
  "checkpoint/Abblation_No_t2i"
)

data_paths=(
  "jsonl_data/Ablation_Noseg.jsonl"
  "jsonl_data/Ablation_No_recover.jsonl"
  "jsonl_data/Ablation_No_t2i.jsonl"
)

for i in ${!experiment_names[@]}; do
  experiment_name=${experiment_names[$i]}
  data_path=${data_paths[$i]}

  torchrun \
  --nnodes=1 \
  --nproc_per_node=$nproc_per_node \
  --node_rank=$node_rank \
  autoregressive/train/train_t2i.py \
  --vq-ckpt MENTOR/vq_ds16_t2i.pt \
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
  --max_eval_samples 200 \
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
  --reference_data_path MENTOR_stage2/cc12m_reference_images.jsonl \
  --multimodal_encoder ${multimodal_encoder} \
  --do_recovery \
  --find_unused_parameters \
  --cls-token-num 512 \
  --dreambench_eval \
  --save_total_limit 1 \
  --load_language_projection ${language_projection} \
  --gpt-ckpt MENTOR/generator_ckpt.pt \
  --mm_vision_tower "openai/clip-vit-large-patch14" \
  --train_all \
  --load_fixed_llamagen \
  --fix 'gpt-empty-fix'

done
