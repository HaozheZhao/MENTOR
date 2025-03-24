from multiprocessing import process
import os
import json
from platform import processor
from matplotlib.pyplot import isinteractive
import numpy as np
from nltk.corpus import stopwords
import io
import base64
from transformers import BertTokenizer
import cv2 
import torch
from torch.utils.data import Dataset
from PIL import Image
import re
from os.path import isfile, join
from glob import glob

class Text2ImgDatasetImg(Dataset):
    def __init__(self, lst_dir, face_lst_dir, transform):
        img_path_list = []
        valid_file_path = []
        # collect valid jsonl
        for lst_name in sorted(os.listdir(lst_dir)):
            if not lst_name.endswith('.jsonl'):
                continue
            file_path = os.path.join(lst_dir, lst_name)
            valid_file_path.append(file_path)

        # collect valid jsonl for face
        if face_lst_dir is not None:
            for lst_name in sorted(os.listdir(face_lst_dir)):
                if not lst_name.endswith('_face.jsonl'):
                    continue
                file_path = os.path.join(face_lst_dir, lst_name)
                valid_file_path.append(file_path)

        for file_path in valid_file_path:
            with open(file_path, 'r') as file:
                for line_idx, line in enumerate(file):
                    data = json.loads(line)
                    img_path = data['image_path']
                    code_dir = file_path.split('/')[-1].split('.')[0]
                    img_path_list.append((img_path, code_dir, line_idx))
        self.img_path_list = img_path_list
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path, code_dir, code_name = self.img_path_list[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, code_name


class Text2ImgDataset(Dataset):
    def __init__(self, args, transform):
        img_path_list = []
        valid_file_path = []
        # collect valid jsonl file path
        for lst_name in sorted(os.listdir(args.data_path)):
            if not lst_name.endswith('.jsonl'):
                continue
            file_path = os.path.join(args.data_path, lst_name)
            valid_file_path.append(file_path)

        for file_path in valid_file_path:
            with open(file_path, 'r') as file:
                for line_idx, line in enumerate(file):
                    data = json.loads(line)
                    img_path = data['image_path']
                    code_dir = file_path.split('/')[-1].split('.')[0]
                    img_path_list.append((img_path, code_dir, line_idx))
        self.img_path_list = img_path_list
        self.transform = transform

        self.t5_feat_path = args.t5_feat_path
        self.short_t5_feat_path = args.short_t5_feat_path
        self.t5_feat_path_base = self.t5_feat_path.split('/')[-1]
        if self.short_t5_feat_path is not None:
            self.short_t5_feat_path_base = self.short_t5_feat_path.split('/')[-1]
        else:
            self.short_t5_feat_path_base = self.t5_feat_path_base
        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = args.cls_token_num
        self.t5_feature_dim = 2048
        self.max_seq_length = self.t5_feature_max_len + self.code_len

    def __len__(self):
        return len(self.img_path_list)

    def dummy_data(self):
        img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).unsqueeze(0)
        valid = 0
        return img, t5_feat_padding, attn_mask, valid

    def __getitem__(self, index):
        img_path, code_dir, code_name = self.img_path_list[index]
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img, t5_feat_padding, attn_mask, valid = self.dummy_data()
            return img, t5_feat_padding, attn_mask, torch.tensor(valid)

        if min(img.size) < self.image_size:
            img, t5_feat_padding, attn_mask, valid = self.dummy_data()
            return img, t5_feat_padding, attn_mask, torch.tensor(valid)

        if self.transform is not None:
            img = self.transform(img)

        t5_file = os.path.join(self.t5_feat_path, code_dir, f"{code_name}.npy")
        if torch.rand(1) < 0.3:
            t5_file = t5_file.replace(self.t5_feat_path_base, self.short_t5_feat_path_base)

        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        if os.path.isfile(t5_file):
            try:
                t5_feat = torch.from_numpy(np.load(t5_file))
                t5_feat_len = t5_feat.shape[1]
                feat_len = min(self.t5_feature_max_len, t5_feat_len)
                t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
                emb_mask = torch.zeros((self.t5_feature_max_len,))
                emb_mask[-feat_len:] = 1
                attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))
                T = self.t5_feature_max_len
                attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
                eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
                attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
                attn_mask = attn_mask.unsqueeze(0).to(torch.bool)
                valid = 1
            except:
                img, t5_feat_padding, attn_mask, valid = self.dummy_data()
        else:
            img, t5_feat_padding, attn_mask, valid = self.dummy_data()

        return img, t5_feat_padding, attn_mask, torch.tensor(valid)


class Text2ImgDatasetCode(Dataset):
    def __init__(self, args):
        pass


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


import random


# class TextImg2ImgDataset(Dataset):
#     def __init__(self, args, transform, data_path, processor, max_samples=-1, is_val=False, dreambench_eval=False,with_image_only=False,
#                  image_only_rate=0.6):
#         img_path_list = load_jsonl(data_path)

#         # img_path_list.append((img_path, code_dir, line_idx))
#         self.img_path_list = img_path_list

#         if max_samples > 0 and max_samples < len(self.img_path_list):
#             self.img_path_list = random.sample(self.img_path_list, max_samples)
#         self.transform = transform
#         self.processor = processor
#         self.processor.tokenizer.padding_side = "right"  # 设置填充到右侧
#         self.processor.tokenizer.truncation_side = "right"  # 设置截断到右侧
#         self.qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
#         self.qformer_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
#         self.args = args
#         if args.multimodal_encoder == 'llava':
#             image_place_holder_num = 256
#         else:
#             image_place_holder_num = 32
#         self.image_place_holder = "".join([args.image_place_holder] * image_place_holder_num)
#         self.image_size = args.image_size
#         latent_size = args.image_size // args.downsample_size
#         self.code_len = latent_size ** 2
#         self.t5_feature_max_len = args.cls_token_num
#         self.t5_feature_dim = 2048
#         self.max_seq_length = self.t5_feature_max_len + self.code_len

#         self.is_val = is_val
#         self.dreambench_eval = dreambench_eval

#         self.image_only_rate = image_only_rate
#         self.with_image_only = with_image_only

#         if args.reference_data_path is not None:
#             def load_reference_images(reference_data_path):  
#                 with open(reference_data_path, 'r') as file:  
#                     return [json.loads(line) for line in file]  
#             self.reference_images = load_reference_images(args.reference_data_path)

#     def get_random_background(self, reference_images, ori_image_size):  
#         random_reference = np.random.choice(reference_images)  
#         background_image_path = random_reference['input_image']
#         # background_image_path = background_image_path.replace("/tmp/haozhezhao/","/home/jovyan/")  
#         background_image = Image.open(background_image_path)  
#         return self.central_crop_and_resize(background_image, ori_image_size)  

#     def central_crop_and_resize(self,image, target_size):  
#         width, height = image.size  
#         min_dim = min(width, height)  
#         left = (width - min_dim) / 2  
#         top = (height - min_dim) / 2  
#         right = (width + min_dim) / 2  
#         bottom = (height + min_dim) / 2  
        
#         image = image.crop((left, top, right, bottom))  
#         image = image.resize(target_size, Image.Resampling.LANCZOS)  
#         return image  

#     def random_transform(self, image, mask, background_size):
#         bg_width, bg_height = background_size

#         # 获取对象的初始尺寸
#         obj_width, obj_height = image.size

#         # 计算允许的最大缩放比例，确保对象在缩放和旋转后仍能放入背景中
#         max_scale = min(bg_width / obj_width, bg_height / obj_height)

#         # 设置缩放范围，可以根据需要调整
#         min_scale = max(0.5, max_scale * 0.5)  # 最小缩放为允许最大缩放的一半，且不小于0.5
#         scale = random.uniform(min_scale, max_scale)

#         # 缩放对象和遮罩
#         new_width = int(obj_width * scale)
#         new_height = int(obj_height * scale)
#         image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
#         mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)

#         # 旋转角度，确保对象仍在背景内，需要限制旋转角度
#         angle = random.uniform(-10, 10)  # 旋转角度在-10到10度之间

#         # 旋转对象和遮罩
#         image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
#         mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)

#         # 获取旋转后对象的尺寸
#         rotated_width, rotated_height = image.size

#         # 如果旋转后对象尺寸超过背景尺寸，重新调整缩放比例
#         if rotated_width > bg_width or rotated_height > bg_height:
#             scale_factor = min(bg_width / rotated_width, bg_height / rotated_height) * 0.9  # 缩小一点
#             new_width = int(rotated_width * scale_factor)
#             new_height = int(rotated_height * scale_factor)
#             image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
#             mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)

#         return image, mask

#     def __len__(self):
#         return len(self.img_path_list)

#     def dummy_data(self):
#         # img, pixel_values, cond_idx, attention_mask, valid

#         img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
#         valid = 0
#         return img, valid

#     def load_image(self, img_path):
#         try:
#             # 加载图像并转换为 RGB 模式
#             img = Image.open(img_path).convert("RGB")
#             # 调整图像大小到指定的 self.image_size
#             img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
#             return img
#         except Exception as e:
#             # 返回默认的白色图像，并打印错误信息
#             print(f"Error loading image {img_path}: {e}")
#             return Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))

#     def __getitem__(self, index):
#         each = self.img_path_list[index]
#         valid = 1

#         img_path = each['image_path']
#         source_path = each['source_image']
#         # img_path = img_path.replace("/tmp/haozhezhao/","/home/jovyan/")
#         # source_path = source_path.replace("/tmp/haozhezhao/","/home/jovyan/")

#         input_text = each['input_text']
#         generation_only = each['generation_only'] if 'generation_only' in each else False
#         do_replace = each['do_replace'] if 'do_replace' in each else False
#         do_mask = each['do_mask'] if 'do_mask' in each else False
#         input_text = input_text.replace(self.args.image_place_holder, "").strip()
#         if 'objects' in each:
#             objects = each['objects'] 
#             if objects is None:
#                 text_input_ids, text_attention_mask = None, None
#             else:
#                 qformer_inputs = self.qformer_tokenizer(objects, return_tensors="pt", padding=True) 
#                 text_input_ids = qformer_inputs['input_ids'].squeeze(0)
#                 text_attention_mask = qformer_inputs['attention_mask'].squeeze(0)
#                 # input_text = f"The {objects} is "+input_text
#                 if self.args.do_recovery:
#                     if generation_only:
#                         input_text =input_text.replace(self.args.image_place_holder, "")
#                     else:
#                         if do_mask:
#                             input_text = f"The {objects} in {self.image_place_holder}."
#                         else:
#                             if objects in input_text:
#                                 random_change = random.choice([True, False])
#                                 if random_change and self.args.replace_subject:
#                                     # Only replace the first occurrence of 'objects' with placeholder
#                                     input_text = input_text.replace(objects, self.image_place_holder, 1)
#                                 else:
#                                     input_text = f"The {objects} in {self.image_place_holder}.\n{input_text}"
#                             else:
#                                 input_text = f"The {objects} in {self.image_place_holder}.\n{input_text}"

#                             # input_text = f"{self.image_place_holder}\n {input_text}."
#                             # input_text = f"{input_text}.\n The {objects} is in {self.image_place_holder}."

#                 else:
#                     if do_mask:
#                         input_text = f"The {objects} in {self.image_place_holder}."
#                     else:
#                         input_text = f"{self.image_place_holder}\n {input_text}"
#         else:
#             text_input_ids, text_attention_mask = None, None



#         if self.with_image_only:
#             input_text_replace = "{}\n".format(self.image_place_holder)
#             # replace by the image replace rate
#             if random.random() <= self.image_only_rate:
#                 input_text = input_text_replace
#                 qformer_inputs = self.qformer_tokenizer("whole image", return_tensors="pt", padding=True) 
#                 text_input_ids = qformer_inputs['input_ids'].squeeze(0)
#                 text_attention_mask = qformer_inputs['attention_mask'].squeeze(0)
#                 generation_only = False
#         if 'objects' in each and each['objects'] is not None:
            
#             # mask = Image.open(img_path).convert('L')  
#             mask = self.load_image(img_path).convert('L')  
#             mask_array = np.array(mask) 
#             mask_area = np.sum(mask_array > 0)  

#             if self.args.do_recovery:
#                 img = self.load_image(source_path)  # ground truth image is source_image; the input text should be the caption of the source_image
#                 ori_image_area = img.size[0] * img.size[1]
#             else:
#                 source_image = self.load_image(source_path)  # input ori image

#                 ori_image_area = source_image.size[0] * source_image.size[1]
#             try:
#                 # segmented_image = np.array(img) * (mask_array[:, :, None] > 0)  
#                 if self.args.do_recovery:
#                     if self.dreambench_eval and self.is_val:
#                         source_image = Image.open(img_path)
#                     else:
#                         if do_mask:
#                             source_image = self.load_image(source_path)  # input ori image
#                             ori_image_area = source_image.size[0] * source_image.size[1]
#                             segmented_image = np.array(source_image) * (mask_array[:, :, None] > 0)  
#                             segmented_image = Image.fromarray(segmented_image)  
#                             img = segmented_image # segmented mask image
#                         elif do_replace:
#                             mask_image = Image.fromarray(np.uint8(mask_array), mode='L')

#                             # mask_array = (mask_array * 255).astype(np.uint8)  # 确保遮罩在0-255范围内

#                             # 创建分割对象图像
#                             segmented_array = np.array(img) * (mask_array[:, :, None] > 0)
#                             segmented_image = Image.fromarray(segmented_array.astype(np.uint8))

#                             # 获取随机背景
#                             random_background = self.get_random_background(self.reference_images, img.size)
#                             bg_width, bg_height = random_background.size

#                             segmented_image_transformed, mask_image_transformed = self.random_transform(segmented_image, mask_image, (bg_width, bg_height))

#                             # 获取变换后对象的尺寸
#                             obj_width, obj_height = segmented_image_transformed.size

#                             # 计算允许的最大偏移，确保对象完全位于背景内
#                             max_x = bg_width - obj_width
#                             max_y = bg_height - obj_height

#                             # 确保偏移不为负
#                             max_x = max(0, max_x)
#                             max_y = max(0, max_y)

#                             # 随机位置放置对象
#                             x = random.randint(0, max_x)
#                             y = random.randint(0, max_y)

#                             # 创建组合图像
#                             source_image = random_background.copy()
#                             source_image.paste(segmented_image_transformed, (x, y), mask_image_transformed)
#                             # segmented_image = Image.fromarray(np.uint8(segmented_image))  
#                             # source_image = Image.composite(segmented_image, random_background, mask_image)   # input for BLIP model
#                         else:
#                             source_image = self.load_image(img_path) if isinstance(img_path, str) else [self.load_image(each) for
#                                                                                                 each in img_path]
                       
#                 else:
#                     segmented_image = np.array(source_image) * (mask_array[:, :, None] > 0)  
#                     segmented_image = Image.fromarray(segmented_image)  
#                     img = segmented_image # segmented mask image

#                 # save in disk named after time
#                 # from datetime import datetime  
#                 # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")  
#                 # filename = f"{timestamp}.jpg"  
#                 # img.save(f"/nobackup/zefan/projects/VLGen/LlamaGen/llamagen_t2i_stage3_subject/{filename}")
#                 # source_image.save(f"/nobackup/zefan/projects/VLGen/LlamaGen/llamagen_t2i_stage3_subject/{timestamp}_source.jpg")
#             except Exception as e:
#                 print(e)
#                 import traceback
#                 traceback.print_exc()
#                 img, valid = self.dummy_data()
#         else:
#             try:
#                 img = Image.open(img_path).convert("RGB")
#                 if min(img.size) < self.image_size:
#                     img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
#             except Exception as e:
#                 print(f"Error loading image {img_path}: {e}")
#                 img, valid = self.dummy_data()
#             if not generation_only:
#                 source_image = self.load_image(source_path) if isinstance(source_path, str) else [self.load_image(each) for
#                                                                                                 each in source_path]
#             else:
#                 source_image = None
#         if not valid:
#             source_image = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
        

#         process_data = self.processor(
#             images=source_image,
#             max_length=self.t5_feature_max_len,
#             text=input_text,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             add_special_tokens=True,
#             return_tensors="pt"

#         )

#         if not generation_only:

#             pixel_source = self.processor.image_processor(images = source_image,return_tensors="pt").pixel_values

#             pixel_values = process_data['pixel_values']  # num_of_image, 3, 224, 224
#             pixel_values = pixel_values.unsqueeze(0)  # 1, num_of_image, 3, 224, 224
#             image_mask = torch.ones((pixel_values.shape[1]), dtype=torch.bool)
#         else:
#             pixel_values = None
#             image_mask = None
#             pixel_source = self.processor.image_processor(images = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255)),return_tensors="pt" ).pixel_values
#         cond_idx = process_data['input_ids'].squeeze(0)  # 120
#         attention_mask = process_data['attention_mask'].squeeze(0)  # 120



#         if self.transform is not None and valid == 1:
#             img = img.convert('RGB') 
#             img_tensor = self.transform(img)
#             img_pixel = self.processor.image_processor(images = img,return_tensors="pt", do_rescale=True ).pixel_values
#         else:
#             img_tensor = img
#             img_pixel = self.processor.image_processor(images = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255)),return_tensors="pt", do_rescale=True ).pixel_values 
        
        
#         # img/img_tensor GT; pixel_values INPUT
#         common_items = [img_tensor, pixel_values, cond_idx, attention_mask, image_mask, img_pixel,torch.tensor(valid)]  


#         if 'objects' in each:  
#             if self.is_val:  
#                 common_items.extend([pixel_source, text_input_ids, text_attention_mask])  
#             else:  
#                 common_items.extend([text_input_ids, text_attention_mask])  

#             # if pixel_values is None and text_input_ids is not None:
#             #     print("pixel_values is None and text_input_ids is not None: ",each,"=============================")
#             # elif text_input_ids is None and pixel_values is not None:
#             #     print("text_input_ids is None and pixel_values is not None ",each,"=============================")

#         else:  
#             if self.is_val:  
#                 common_items.append(pixel_source)  

#         # 返回构建的列表  
#         return tuple(common_items)  


class TextImg2ImgDataset(Dataset):
    def __init__(self, args, transform, data_path, processor, max_samples=-1, is_val=False, dreambench_eval=False,
                 with_image_only=False, image_only_rate=0.6):
        img_path_list = load_jsonl(data_path)
        self.img_path_list = img_path_list
        if max_samples > 0 and max_samples < len(self.img_path_list):
            self.img_path_list = random.sample(self.img_path_list, max_samples)

        self.transform = transform
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
        self.processor.tokenizer.truncation_side = "right"

        # 初始化 qformer_tokenizer
        self.qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
        self.qformer_tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        self.args = args
        if args.multimodal_encoder == 'llava':
            image_place_holder_num = 256
        else:
            image_place_holder_num = 32
        self.image_place_holder = "".join([args.image_place_holder] * image_place_holder_num)
        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = args.cls_token_num
        self.t5_feature_dim = 2048
        self.max_seq_length = self.t5_feature_max_len + self.code_len

        self.is_val = is_val
        self.dreambench_eval = dreambench_eval
        self.image_only_rate = image_only_rate
        self.with_image_only = with_image_only

        if args.reference_data_path is not None:
            self.reference_images = self.load_reference_images(args.reference_data_path)

    def load_reference_images(self, reference_data_path):
        with open(reference_data_path, 'r') as file:
            return [json.loads(line) for line in file]

    def get_random_background(self, reference_images, ori_image_size):
        random_reference = np.random.choice(reference_images)
        background_image_path = random_reference['input_image']
        background_image = Image.open(background_image_path)
        return self.central_crop_and_resize(background_image, ori_image_size)

    def crop_with_mask(self, mask, original_image):
        """
        根据 mask 的 bounding box 裁剪原图，并将裁剪后的区域填充成正方形，填充部分置为黑色。
        """
        mask_image = mask if mask.mode == "L" else mask.convert("L")
        mask_array = np.array(mask_image)
        coords = np.column_stack(np.where(mask_array > 0))
        
        if coords.size == 0:
            raise ValueError("Mask image does not contain any segmented area.")
        
        # 获取 mask 的边界框
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # 计算裁剪的宽高
        width = x_max - x_min
        height = y_max - y_min
        
        # 计算正方形的边长
        max_side = max(width, height)
        
        # 计算裁剪区域
        cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
        
        # 创建新的正方形画布 (黑色背景)
        square_image = Image.new("RGB", (max_side, max_side), (0, 0, 0))
        
        # 计算粘贴位置，使裁剪图像居中
        paste_x = (max_side - width) // 2
        paste_y = (max_side - height) // 2
        
        square_image.paste(cropped_image, (paste_x, paste_y))
        
        return square_image

    def central_crop_and_resize(self, image, target_size):
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) / 2
        top = (height - min_dim) / 2
        right = (width + min_dim) / 2
        bottom = (height + min_dim) / 2
        image = image.crop((left, top, right, bottom))
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        return image

    def random_transform(self, image, mask, background_size, rotated = False):
        bg_width, bg_height = background_size
        obj_width, obj_height = image.size
        max_scale = min(bg_width / obj_width, bg_height / obj_height)
        min_scale = max(0.5, max_scale * 0.5)
        scale = random.uniform(min_scale, max_scale)
        new_width = int(obj_width * scale)
        new_height = int(obj_height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
        if rotated:
            angle = random.uniform(-10, 10)
            image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)
            rotated_width, rotated_height = image.size
            if rotated_width > bg_width or rotated_height > bg_height:
                scale_factor = min(bg_width / rotated_width, bg_height / rotated_height) * 0.9
                new_width = int(rotated_width * scale_factor)
                new_height = int(rotated_height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
        return image, mask

    def dummy_data(self):
        img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        valid = 0
        return img, valid

    def load_image(self, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))

    def _process_text_info(self, each):
        """
        根据 objects 是否为 list 来调整文本输入的构造方式，并独立计算 qformer_inputs：
          - 如果 objects 为 list，则通过 batch inference 得到 qformer token（形状为 (N, seq_len)）
          - 否则按照原有逻辑处理
        """
        generation_only = each.get('generation_only', False)
        do_replace = each.get('do_replace', False)
        do_mask = each.get('do_mask', False)
        input_text = each['input_text'].replace(self.args.image_place_holder, "").strip()
        text_input_ids, text_attention_mask = None, None

        if 'objects' in each:
            objects = each['objects']
            if objects is not None:
                if isinstance(objects, list):
                    # 多图输入：构造每个 object 对应的前缀，然后通过 batch inference 计算 qformer_inputs

                    qformer_inputs = self.qformer_tokenizer(objects, return_tensors="pt", padding=True)
                    text_input_ids = qformer_inputs['input_ids']   # 形状为 (N, seq_len)
                    text_attention_mask = qformer_inputs['attention_mask']  # 形状为 (N, seq_len)
                    if self.args.do_recovery:
                        if generation_only:
                            input_text = input_text.replace(self.args.image_place_holder, "")
                        else:
                            if do_mask:
                                input_text = "\n".join([f"The {obj} in {self.image_place_holder}." for obj in objects]) 
                            else:
                                if random.choice([True, False]) and self.args.replace_subject:
                                    for obj in objects:
                                        if obj in input_text:
                                                input_text = input_text.replace(obj, self.image_place_holder, 1)
                                else:
                                    input_prefix = "\n".join([f"The {obj} in {self.image_place_holder}." for obj in objects])
                                    input_text = input_prefix + "\n" + input_text
                    else:
                        if do_mask:
                            input_text = "\n".join([f"The {obj} in {self.image_place_holder}." for obj in objects])
                        else:
                            input_text = "\n".join([f"{self.image_place_holder}." for obj in objects]) + "\n" + input_text
                else:
                    qformer_inputs = self.qformer_tokenizer(objects, return_tensors="pt", padding=True)
                    text_input_ids = qformer_inputs['input_ids']
                    text_attention_mask = qformer_inputs['attention_mask']
                    if self.args.do_recovery:
                        if generation_only:
                            input_text = input_text.replace(self.args.image_place_holder, "")
                        else:
                            if do_mask:
                                input_text = f"The {objects} in {self.image_place_holder}."
                            else:
                                if objects in input_text:
                                    if random.choice([True, False]) and self.args.replace_subject:
                                        input_text = input_text.replace(objects, self.image_place_holder, 1)
                                    else:
                                        input_text = f"The {objects} in {self.image_place_holder}.\n{input_text}"
                                else:
                                    input_text = f"The {objects} in {self.image_place_holder}.\n{input_text}"
                    else:
                        if do_mask:
                            input_text = f"The {objects} in {self.image_place_holder}."
                        else:
                            input_text = f"{self.image_place_holder}\n {input_text}"
        if self.with_image_only and random.random() <= self.image_only_rate:
            input_text = "{}\n".format(self.image_place_holder)
            qformer_inputs = self.qformer_tokenizer("whole image", return_tensors="pt", padding=True)
            text_input_ids = qformer_inputs['input_ids']
            text_attention_mask = qformer_inputs['attention_mask']
            generation_only = False
        return input_text, text_input_ids, text_attention_mask, generation_only

    def _process_objects(self, each, do_mask, do_replace, generation_only):
        """
        根据 objects 是否为 list 来处理图像：
          - 单图模式：沿用原有逻辑
          - 多图模式：认为 image_path 为 mask 列表，对单一 source_image 分别利用每个 mask 裁剪，得到多个 crop 结果
        """
        img_path = each['image_path']
        source_path = each['source_image']
        valid = 1

        if isinstance(each.get('objects', None), list):
            # 多图模式：mask 为列表
            if isinstance(img_path, list) and len(img_path) > 0:
                mask_images = []
                for mask_item in img_path:
                    if isinstance(mask_item, str):
                        mask_img = self.load_image(mask_item).convert("L")
                    else:
                        mask_img = mask_item.convert("L") if hasattr(mask_item, "convert") else mask_item
                    mask_images.append(mask_img)
                source_image_orig = self.load_image(source_path) if isinstance(source_path, str) else self.load_image(source_path[0])
                cropped_images = []
                for mask in mask_images:
                    try:
                        cropped = self.crop_with_mask(mask, source_image_orig)
                    except Exception as e:
                        print(e)
                        cropped = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
                    cropped_images.append(cropped)
                img = source_image_orig
                source_image = cropped_images
            # else:
            #     img = self.load_image(img_path).convert("L")
            #     source_image = self.load_image(source_path)
        else:
            # 单图模式
            mask = self.load_image(img_path).convert("L")
            mask_array = np.array(mask)
            if self.args.do_recovery:
                img = self.load_image(source_path)
            else:
                source_image = self.load_image(source_path)
            try:
                if self.args.do_recovery:
                    if self.dreambench_eval and self.is_val:
                        source_image = Image.open(img_path)
                    else:
                        if do_mask:
                            source_image = self.load_image(source_path)
                            segmented_image = np.array(source_image) * (mask_array[:, :, None] > 0)
                            segmented_image = Image.fromarray(segmented_image)
                            img = segmented_image
                        elif do_replace:
                            mask_image = Image.fromarray(np.uint8(mask_array), mode='L')
                            segmented_array = np.array(img) * (mask_array[:, :, None] > 0)
                            segmented_image = Image.fromarray(segmented_array.astype(np.uint8))
                            random_background = self.get_random_background(self.reference_images, img.size)
                            bg_width, bg_height = random_background.size
                            segmented_image_transformed, mask_image_transformed = self.random_transform(segmented_image, mask_image, (bg_width, bg_height))
                            obj_width, obj_height = segmented_image_transformed.size
                            max_x = max(0, bg_width - obj_width)
                            max_y = max(0, bg_height - obj_height)
                            x = random.randint(0, max_x)
                            y = random.randint(0, max_y)
                            source_image = random_background.copy()
                            source_image.paste(segmented_image_transformed, (x, y), mask_image_transformed)
                        else:
                            source_image = self.load_image(img_path) if isinstance(img_path, str) else [self.load_image(item) for item in img_path]
                else:
                    segmented_image = np.array(source_image) * (mask_array[:, :, None] > 0)
                    segmented_image = Image.fromarray(segmented_image)
                    img = segmented_image
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                img, valid = self.dummy_data()
                source_image = None
        if not valid:
            source_image = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
        return img, source_image, valid

    def _process_default_image(self, each, generation_only):
        img_path = each['image_path']
        source_path = each['source_image']
        valid = 1
        try:
            img = Image.open(img_path).convert("RGB")
            if min(img.size) < self.image_size:
                img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img, valid = self.dummy_data()
        if not generation_only:
            source_image = self.load_image(source_path) if isinstance(source_path, str) else [self.load_image(item) for item in source_path]
        else:
            source_image = None
        return img, source_image, valid

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        each = self.img_path_list[index]
        input_text, text_input_ids, text_attention_mask, generation_only = self._process_text_info(each)
        do_mask = each.get('do_mask', False)
        do_replace = each.get('do_replace', False)

        if 'objects' in each and each['objects'] is not None:
            img, source_image, valid = self._process_objects(each, do_mask, do_replace, generation_only)
        else:
            img, source_image, valid = self._process_default_image(each, generation_only)
        try:
            process_data = self.processor(
                images=source_image,
                max_length=self.t5_feature_max_len,
                text=input_text,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )
        except Exception as e:
            print(f"Error processing text info: {e}")
            source_image = [ Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255)) for _ in range(len(source_image))]
            process_data = self.processor(
                images=source_image,
                max_length=self.t5_feature_max_len,
                text=input_text,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )

        if not generation_only:
            if isinstance(source_image, list):
                pixel_source = self.processor.image_processor(images=source_image[0], return_tensors="pt").pixel_values
            else:
                pixel_source = self.processor.image_processor(images=source_image, return_tensors="pt").pixel_values
            pixel_values = process_data['pixel_values'].unsqueeze(0)
            image_mask = torch.ones((pixel_values.shape[1]), dtype=torch.bool)
        else:
            pixel_values = None
            image_mask = None
            pixel_source = self.processor.image_processor(
                images=Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255)),
                return_tensors="pt"
            ).pixel_values

        cond_idx = process_data['input_ids'].squeeze(0)
        attention_mask = process_data['attention_mask'].squeeze(0)

        if self.transform is not None and valid == 1:
            if isinstance(img, list):
                img_tensor = [self.transform(im.convert('RGB')) for im in img]
                img_pixel = [self.processor.image_processor(images=im, return_tensors="pt", do_rescale=True).pixel_values for im in img]
            else:
                img = img.convert('RGB')
                img_tensor = self.transform(img)
                img_pixel = self.processor.image_processor(images=img, return_tensors="pt", do_rescale=True).pixel_values
        else:
            img_tensor = img
            img_pixel = self.processor.image_processor(
                images=Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255)),
                return_tensors="pt", do_rescale=True
            ).pixel_values

        common_items = [img_tensor, pixel_values, cond_idx, attention_mask, image_mask, img_pixel, torch.tensor(valid)]
        if 'objects' in each:
            if self.is_val:
                common_items.extend([pixel_source, text_input_ids, text_attention_mask])
            else:
                common_items.extend([text_input_ids, text_attention_mask])
        else:
            if self.is_val:
                common_items.append(pixel_source)

        return tuple(common_items)




from datasets import concatenate_datasets, load_dataset
from datasets import Dataset as Huggingface_Dataset

# sss
class TextImg2ImgStage2Dataset(Dataset):
    def __init__(self, args, transform, data_path, processor, max_samples=-1, is_val=False, with_image_only=False,
                 image_only_rate=0.6):
        # img_path_list = load_jsonl(data_path)
        #
        self.is_val = is_val

        # # img_path_list.append((img_path, code_dir, line_idx))
        # self.img_path_list = img_path_list
        dataset_list = []

        train_dataset, eval_dataset, predict_dataset = self.load_dataset_from_arrow_split_val_test(
            data_path)
        if self.is_val:
            dataset_list.append(eval_dataset)
            # val_ffhq = load_dataset('json', data_files='/home/zhaohaozhe/data/ffhq_wild_files/new_test.jsonl',
            #                         split="train")
            # dataset_list.append(val_ffhq)
            del train_dataset, predict_dataset
        else:
            dataset_list.append(train_dataset)
            train_ffhq = load_dataset('json', data_files='/home/zhaohaozhe/data/ffhq_wild_files/new_train.jsonl',
                                      split="train")
            dataset_list.append(train_ffhq)

            del eval_dataset, predict_dataset

        self.img_path_list = concatenate_datasets(dataset_list)

        # self.eval_dataset = self.load_dataset_from_arrow(self.data_args.validation_file)
        # self.predict_dataset = self.load_dataset_from_arrow(self.data_args.test_file)
        # extra dataset

        if max_samples > 0 and max_samples < len(self.img_path_list):
            self.img_path_list = self.img_path_list.shuffle(seed=args.global_seed).select(range(max_samples))
        self.stop_words = set(stopwords.words('english'))

        self.transform = transform
        self.processor = processor
        self.args = args
        if args.multimodal_encoder == 'llava':
            image_place_holder_num = 256
        else:
            image_place_holder_num = 32
        self.image_place_holder = "".join([args.image_place_holder] * image_place_holder_num)
        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = args.cls_token_num
        self.t5_feature_dim = 2048
        self.max_seq_length = self.t5_feature_max_len + self.code_len

        self.image_only_rate = image_only_rate
        self.with_image_only = with_image_only

    def __len__(self):
        return len(self.img_path_list)

    def load_dataset_from_arrow_split_val_test(self, data_files):
        files = glob(join(data_files, "cnt*.arrow"))
        files = [f for f in files if 'cache' not in f]
        dataset_list = [Huggingface_Dataset.from_file(score) for score in files]
        random.shuffle(dataset_list)
        # test = dataset_list.pop(0)
        val = dataset_list.pop(0)
        train = concatenate_datasets(dataset_list)
        # return train,val,test
        return train, val, val

    def dummy_data(self):
        # img, pixel_values, cond_idx, attention_mask, valid

        img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        valid = 0
        return img, valid

    def load_image(self, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            return img
        except:
            return Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))

    def read_image(self, img_path, postfix=None, add_source=True):
        if postfix == 'png':
            image = Image.open(join("/home/zhaohaozhe/data", img_path))
        else:
            if '.jpg' in img_path or '.png' in img_path:
                if add_source and 'MIC_tool' not in img_path:
                    img_path = join("/home/zhaohaozhe/data", img_path)
                image = Image.open(img_path)
            else:
                try:
                    image = Image.open(io.BytesIO(base64.b64decode(img_path)))
                except Exception as e:
                    print('Error:', e)
                    return Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
        return image.convert('RGB')

    def tag_filter(self, tag, tags, caption):
        if tag in ['yellow', 'blue', 'red', 'green', 'white', 'black', 'brown', 'grey', 'gray', 'purple',
                   'pink', 'orange', 'cyan', 'silver', 'gold', 'golden']:
            return False

        if re.search(r'\b{}\b'.format(re.escape(tag)), caption, re.IGNORECASE) is None:
            return False

        if sum([re.search(r'\b{}\b'.format(tag), t) is not None for t in tags]) > 1:
            return False

        return True

    def preprocess_phrase(self, phrase):
        tokens = phrase.split()
        # NLTK Stop words
        cleaned_tokens = [token for token in tokens if token not in self.stop_words]
        cleaned_phrase = ' '.join(cleaned_tokens)
        return cleaned_phrase


    def build_prompt(self, prompt, idx, tgt_subject):
        # prompt = "\n".join([ f"{self.preprocess_phrase(tgt)} is {self.replace_token}." for tgt in tgt_subject])+ ", ".join([f"{prompt}\n"]* int(prompt_strength * prompt_reps))
        # a trick to amplify the prompt

        # insert_text = [ f", {self.replace_token}," for _ in  range(len(tgt_subject)) ]
        # insert_text = [ f"{self.replace_token}" for _ in  range(len(tgt_subject)) ]
        # prompt = self.replace_tgt_subject_prompt_by_index(prompt,idx, insert_text,tgt_subject)
        # if prompt[-1] ==',':
        #     prompt[-1] = '.'

        # if idx is None:
        #     prompt = f"{self.replace_token}\n"+ prompt
        # else:
        #     prompt = ". ".join([ f"{self.preprocess_phrase(tgt)} is {self.replace_token}" for tgt in tgt_subject]) +prompt
        prompt = ". ".join(
            [f"{self.preprocess_phrase(tgt)} in {self.image_place_holder}" for tgt in tgt_subject]) + "\n" + prompt


        return prompt


    def __getitem__(self, index):
        each = self.img_path_list[index]
        valid = 1

        img_path = each['output_images'][0]
        if ('index' in each and each['index'] is None or 'index' not in each) and each['subjects'] is not None:  # kosmos
            image_list = each['images']
            tags = each['subjects']
            caption = each['input_text']
            tags = {tag: tag_id for tag_id, tag in enumerate(tags) if self.tag_filter(tag, tags, caption)}
            tags = sorted(tags.items(),
                          key=lambda x: re.search(r'\b{}\b'.format(re.escape(x[0])), caption, re.IGNORECASE).end())
            index = []
            source_image = []
            subjects = []
            for tag, tag_id in tags:
                tag_loc = tag_id * 2 + (1 if random.random() <= 0.5 else 0)
                index.append(re.search(r'\b{}\b'.format(re.escape(tag)), caption, re.IGNORECASE).end())
                source_image.append(self.read_image(image_list[tag_loc]))
                subjects.append(tag)

        else:
            source_image = [self.read_image(each) for each in each['images']]
            index = each['index']
            subjects = each['subjects']

        input_text = each['input_text']

        # input_text = input_text.replace(self.args.image_place_holder, self.image_place_holder)

        input_text = self.build_prompt(input_text, index, subjects)

        # if self.with_image_only:
        #     input_text_replace = "The image 0 is {}\n".format(self.image_place_holder)
        #     # replace by the image replace rate
        #     if random.random() <= self.image_only_rate:
        #         input_text = input_text_replace

        try:
            img = self.read_image(img_path)
            if  min(img.size) < self.image_size:
                img, valid = self.dummy_data()
        except:
            img, valid = self.dummy_data()

        max_length = args.cls_token_num if self.is_val else 512
        process_data = self.processor(
            images=source_image,
            max_length=max_length,
            text=input_text,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"

        )

        pixel_source = self.processor.image_processor(images = img,return_tensors="pt", do_rescale=True ).pixel_values


        pixel_values = process_data['pixel_values']  # num_of_image, 3, 224, 224
        pixel_values = pixel_values.unsqueeze(0)  # 1, num_of_image, 3, 224, 224
        cond_idx = process_data['input_ids'].squeeze(0)  # 120
        attention_mask = process_data['attention_mask'].squeeze(0)  # 120
        image_mask = torch.ones((pixel_values.shape[1]), dtype=torch.bool)
        if self.transform is not None and valid == 1:
            img = self.transform(img)

        if self.is_val:
            return img, pixel_values, cond_idx, attention_mask, image_mask, torch.tensor(valid), pixel_source
        else:
            return img, pixel_values, cond_idx, attention_mask, image_mask, torch.tensor(valid)


def build_t2i_image(args, transform):
    return Text2ImgDatasetImg(args.data_path, args.data_face_path, transform)


def build_t2i(args, transform):
    return Text2ImgDataset(args, transform)


def build_t2i_code(args):
    return Text2ImgDatasetCode(args)


def build_ti2i(args, transform, data_path, processor, max_samples=-1, is_val=False, dreambench_eval=False, with_image_only=False,
               image_only_rate=0.6, stage2 =False):

    if not stage2:
        return TextImg2ImgDataset(args, transform, data_path, processor, max_samples=max_samples, is_val=is_val,dreambench_eval=dreambench_eval,
                              with_image_only=with_image_only, image_only_rate=image_only_rate)
    else:
        return TextImg2ImgStage2Dataset(args, transform, data_path, processor, max_samples=max_samples, is_val=is_val,
                              with_image_only=with_image_only, image_only_rate=image_only_rate)