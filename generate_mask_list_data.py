import os
import json
import concurrent.futures
import multiprocessing
from tqdm import tqdm

# 定义要搜索的文件夹路径
folders = [
    "/tmp/haozhezhao/data/segment_results_imgnet",
    "/tmp/haozhezhao/data/segment_results_sam_molom",
    "/tmp/haozhezhao/VL_Prompt/segment_results_FLUX_molom_start_500000_end_600000",
    "/tmp/haozhezhao/VL_Prompt/segment_results_FLUX_molom_start_600000_end_700000",
    "/tmp/haozhezhao/VL_Prompt/segment_results_FLUX_molom_start_700000_end_800000",
    "/tmp/haozhezhao/VL_Prompt/segment_results_FLUX_molom_start_900000_end_950000",
]

# 目标 JSONL 文件路径
output_jsonl_path = "/tmp/haozhezhao/MLLMG/jsonl_data/merged_segment_results.jsonl"

# 额外需要合并的 JSONL 文件路径
extra_jsonl_path = "/tmp/haozhezhao/data/flux_segment/flux_segment_images.jsonl"

def find_metadata_files(folder):
    """
    使用 os.walk 遍历给定文件夹，查找所有名为 metadata.jsonl 的文件
    """
    metadata_files = []
    for root, dirs, files in tqdm(os.walk(folder)):
        for file in files:
            if file == 'metadata.jsonl':
                metadata_files.append(os.path.join(root, file))
    return metadata_files

# 利用多进程对每个文件夹进行并行查找
with multiprocessing.Pool(processes=len(folders)) as pool:
    results = pool.map(find_metadata_files, folders)
# 将所有查找到的文件列表合并为一个列表
jsonl_files = [file for sublist in results for file in sublist]

def process_jsonl_file(file_path):
    """
    读取单个 JSONL 文件，并提取所需字段
    """
    results = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                # 根据数据格式提取所需字段
                if "input_image" in data:
                    extracted = {
                        "prompt": data.get("caption", ""),
                        "source_img": data.get("source_img", ""),
                        "sam_objects": data.get("objects", []),
                        "mask_path": data.get("mask_path", [])
                    }
                else:
                    extracted = {
                        "prompt": data.get("prompt", ""),
                        "source_img": data.get("source_img", ""),
                        "sam_objects": data.get("sam_objects", []),
                        "mask_path": data.get("mask_path", [])
                    }
                results.append(extracted)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return results

# 使用多线程并发处理所有 JSONL 文件，利用 tqdm 显示进度（不设置 total 参数）
merged_data = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    for file_result in tqdm(executor.map(process_jsonl_file, jsonl_files), desc="Processing files"):
        merged_data.extend(file_result)

# 处理额外的 JSONL 文件
merged_data.extend(process_jsonl_file(extra_jsonl_path))

# 保存合并后的数据到输出文件
with open(output_jsonl_path, "w") as out_file:
    for entry in merged_data:
        out_file.write(json.dumps(entry) + "\n")

print(output_jsonl_path)
