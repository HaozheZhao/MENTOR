import json
import random
from tqdm import tqdm

# 文件路径
merged_jsonl_path = "/tmp/haozhezhao/MLLMG/jsonl_data/multiobjects_filtered.jsonl"
val_jsonl_path = "/tmp/haozhezhao/MLLMG/jsonl_data/multiobjects_filtered_val.jsonl"
train_jsonl_path = "/tmp/haozhezhao/MLLMG/jsonl_data/multiobjects_filtered_trained.jsonl"

# 读取 JSONL 数据
records = []
with open(merged_jsonl_path, "r") as f:
    for line in tqdm(f, desc="Loading JSONL data"):
        records.append(json.loads(line.strip()))

# 打乱数据，提高随机性
random.shuffle(records)

# 存储验证集和训练集
val_records = []
train_records = []

# 遍历数据并分类
for rec in records:
    source_img = rec.get("source_img", "")
    mask_len = len(rec.get("image_path", []))

    if "VL_Prompt" in source_img and mask_len > 1 and len(val_records) < 300:
        val_records.append(rec)  # 选入验证集
    elif "VL_Prompt" not in source_img and mask_len == 3 and len(val_records) < 500:
        val_records.append(rec)  # 选入验证集
    else:
        train_records.append(rec)  # 其余数据存入训练集

# 再次打乱数据
random.shuffle(val_records)
random.shuffle(train_records)

# 保存验证集
with open(val_jsonl_path, "w") as f:
    for record in tqdm(val_records, desc="Saving Validation JSONL"):
        f.write(json.dumps(record) + "\n")

# 保存训练集
with open(train_jsonl_path, "w") as f:
    for record in tqdm(train_records, desc="Saving Training JSONL"):
        f.write(json.dumps(record) + "\n")

print(f"Validation set saved to: {val_jsonl_path} (Total: {len(val_records)})")
print(f"Training set saved to: {train_jsonl_path} (Total: {len(train_records)})")
