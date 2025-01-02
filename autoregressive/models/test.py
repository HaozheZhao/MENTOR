import sys
import os
from PIL import Image
sys.path.append('/tmp/haozhezhao/MLLMG/autoregressive/models')
# Output current sys.path
sys_path_info = sys.path
print(sys_path_info)

from instructblip import InstructBlipForConditionalGeneration, InstructBlipConfig, InstructBlipProcessor
processor = InstructBlipProcessor.from_pretrained("/tmp/haozhezhao/model/instructblip-flan-t5-xl")
image_size = 512
source_path = "/tmp/haozhezhao/data/segment_results_sam/cnt_10000_1000000000_0_15_0/0000000048/ori.jpg"
def load_image( img_path):
    try:
        # 加载图像并转换为 RGB 模式
        img = Image.open(img_path).convert("RGB")
        # 调整图像大小到指定的 self.image_size
        img = img.resize((image_size,image_size), Image.BICUBIC)
        return img
    except Exception as e:
        # 返回默认的白色图像，并打印错误信息
            print(f"Error loading image {img_path}: {e}")
            return Image.new('RGB', (image_size, image_size), (255, 255, 255))
source_image = load_image(source_path) 

process_data = processor(
    images=source_image,
    max_length=120,
    text="Hello",
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="pt",
)

print(process_data)