import os
import json
import numpy as np
from PIL import Image
import concurrent.futures
from tqdm import tqdm

# 定义全局进度条变量
progress_bar = None

def process_instance(data):
    global progress_bar
    try:
        boxes = []
        # 对每个 mask 图像处理
        for mask_path in data.get("image_path", []):
            if not os.path.exists(mask_path):
                print(f"Mask file not found: {mask_path}")
                continue
            try:
                # 加载为灰度图并 resize 到 512×512
                img = Image.open(mask_path).convert('L')
                img = img.resize((512, 512))
                mask_array = np.array(img)
                # 这里认为非0的像素为 mask 区域
                mask = mask_array > 0
                if not mask.any():
                    continue  # 如果该 mask 没有目标区域，则跳过
                # 得到非零像素的位置，注意 np.where 返回 (y, x)
                ys, xs = np.where(mask)
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                boxes.append((x_min, y_min, x_max, y_max))
            except Exception as e:
                print(f"Error processing mask {mask_path}: {e}")
                continue

        # 如果只有一个 mask，则不存在重叠问题，直接保留
        if len(boxes) < 2:
            return data

        def compute_intersection(box1, box2):
            # 计算两个 bounding box 的交集面积
            x_min = max(box1[0], box2[0])
            y_min = max(box1[1], box2[1])
            x_max = min(box1[2], box2[2])
            y_max = min(box1[3], box2[3])
            if x_max <= x_min or y_max <= y_min:
                return 0
            return (x_max - x_min) * (y_max - y_min)

        def area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        # 两两比较 bounding box 是否有重叠或包含
        n = len(boxes)
        for i in range(n):
            for j in range(i + 1, n):
                inter = compute_intersection(boxes[i], boxes[j])
                if inter > 0:
                    ratio_i = inter / area(boxes[i])
                    ratio_j = inter / area(boxes[j])
                    # 若任一 bounding box 的重叠比例大于 10%
                    if ratio_i > 0.1 or ratio_j > 0.1:
                        return None  # 舍弃该实例
                # 判断包含关系：若一个 box 完全包含另一个
                if (boxes[i][0] >= boxes[j][0] and boxes[i][1] >= boxes[j][1] and
                    boxes[i][2] <= boxes[j][2] and boxes[i][3] <= boxes[j][3]) or \
                   (boxes[j][0] >= boxes[i][0] and boxes[j][1] >= boxes[i][1] and
                    boxes[j][2] <= boxes[i][2] and boxes[j][3] <= boxes[i][3]):
                    return None  # 舍弃该实例

        return data
    finally:
        # 无论如何都更新进度条（不显示总数，仅显示已完成数量）
        if progress_bar is not None:
            progress_bar.update(1)

def main():
    global progress_bar
    input_file = "/tmp/haozhezhao/MLLMG/jsonl_data/multiobjects_molom_imagenet_flux_sam_1_2m.jsonl"
    output_file = "/tmp/haozhezhao/MLLMG/jsonl_data/multiobjects_filtered.jsonl"

    lines = []
    with open(input_file, "r") as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            lines.append(data)

    # 创建全局进度条，不设置 total（或 total=None），只显示累计完成数
    progress_bar = tqdm(desc="Processing", unit="task", total=None)
    processed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=96) as executor:
        # 使用 map 分发任务，内部会自动调用 process_instance，每结束一次就更新进度条
        for result in executor.map(process_instance, lines):
            processed.append(result)
    progress_bar.close()

    # 保存未被舍弃的实例
    with open(output_file, 'w') as f:
        for instance in processed:
            if instance is not None:
                json.dump(instance, f, ensure_ascii=False)
                f.write('\n')

if __name__ == "__main__":
    main()
