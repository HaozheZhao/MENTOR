import json
from tqdm import tqdm
import random

def filter_jsonl(input_path, output_path):
    # 读取 jsonl 文件
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in tqdm(f)]

    # 去除 objects 数量 > 5 的条目
    filtered = [item for item in data if len(item['objects']) <= 5]

    # 按照 mask 数量进行分类
    one_or_two_mask = [item for item in filtered if 1 <= len(item['image_path']) <= 2]
    other = [item for item in filtered if len(item['image_path']) > 2]

    # 从 1 或 2 个 mask 的列表中随机选择一半
    sampled = random.sample(one_or_two_mask, len(one_or_two_mask) // 3)

    # 合并最终保留的条目
    final_data = sampled + other

    # 写入输出文件
    with open(output_path, 'w') as f:
        for item in tqdm(final_data, desc=f"Writing to {output_path}"):
            f.write(json.dumps(item) + '\n')


# 路径设置
input_paths = [
    "/tmp/haozhezhao/MLLMG/jsonl_data/multiobjects_molom_imagenet_flux_qwen_midsource_cc12m_gen_trained_all_raw_newclean.jsonl",
    # "/tmp/haozhezhao/MLLMG/jsonl_data/multiobjects_molom_imagenet_flux_qwen_midsource_cc12m_gen_val_raw_cleaned.jsonl"
]

output_paths = [
    "/tmp/haozhezhao/MLLMG/jsonl_data/multiobjects_molom_imagenet_flux_qwen_midsource_cc12m_gen_trained_all_raw_newclean_four_mask.jsonl",
    # "/tmp/haozhezhao/MLLMG/jsonl_data/multiobjects_molom_imagenet_flux_qwen_midsource_cc12m_gen_val_raw_four_mask.jsonl"
]

# 执行过滤处理
for in_path, out_path in zip(input_paths, output_paths):
    filter_jsonl(in_path, out_path)
