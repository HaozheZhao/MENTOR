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
import random

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data



class TextImg2ImgDataset(Dataset):
    def __init__(self, args, transform, data_path, processor, max_samples=-1, is_val=False, dreambench_eval=False,
                 with_image_only=False, image_only_rate=0.6):
        img_path_list = load_jsonl(data_path)
        self.img_path_list = img_path_list
        if max_samples > 0 and max_samples < len(self.img_path_list):
            self.img_path_list = random.sample(self.img_path_list, max_samples)
        temp_text = load_jsonl("/tmp/haozhezhao/MLLMG/temp_text.jsonl")
        self.img_path_list.extend(temp_text)
        random.shuffle(self.img_path_list)
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
        self.do_central_crop = args.do_central_crop if hasattr(args, "do_central_crop") else False

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


    def crop_with_mask(self, mask, original_image, method="bbox"):
        """
        根据 mask 裁剪原图，并将裁剪后的区域填充成正方形，填充部分置为黑色。

        参数:
            mask: PIL.Image, 分割掩膜图像
            original_image: PIL.Image, 原图
            method: str, 裁剪方式:
                - "bbox": 基于 mask 的 bounding box 裁剪
                - "segment": 根据 mask 直接 segment 出 object
        返回:
            PIL.Image: 裁剪并填充后的正方形图像
        """
        mask_image = mask if mask.mode == "L" else mask.convert("L")
        mask_array = np.array(mask_image)
        coords = np.column_stack(np.where(mask_array > 0))

        if coords.size == 0:
            raise ValueError("Mask image does not contain any segmented area.")

        # 获取 mask 的边界框
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        width = x_max - x_min
        height = y_max - y_min
        max_side = max(width, height)

        if method == "bbox":
            # 仅使用 bounding box 进行裁剪
            cropped_image = original_image.crop((x_min, y_min, x_max, y_max))

        elif method == "segment":
            # Segment 掩膜区域，背景设为黑色
            original_array = np.array(original_image)
            segmented_array = np.zeros_like(original_array)
            mask_binary = mask_array > 0
            segmented_array[mask_binary] = original_array[mask_binary]
            segmented_image = Image.fromarray(segmented_array)
            cropped_image = segmented_image.crop((x_min, y_min, x_max, y_max))

        else:
            raise ValueError(f"Unknown cropping method: {method}")

        # 创建正方形黑底画布
        square_image = Image.new("RGB", (max_side, max_side), (0, 0, 0))

        # 计算粘贴位置，使图像居中
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



    def random_transform(
        self,
        image, 
        mask, 
        background_size, 
        rotated=False, 
        stretch=False, 
        perspective=False
    ):
        bg_width, bg_height = background_size
        obj_width, obj_height = image.size

        # 缩放比例
        max_scale = min(bg_width / obj_width, bg_height / obj_height)
        min_scale = max(0.2, max_scale * 0.1) # 
        scale = random.uniform(min_scale, max_scale)
        new_width = int(obj_width * scale)
        new_height = int(obj_height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)

        # 拉伸（仿射变换）
        if stretch:
            scale_x = random.uniform(0.7, 1.3)
            scale_y = random.uniform(0.7, 1.3)
            shear = random.uniform(-0.3, 0.3)
            w, h = image.size
            matrix = [scale_x, shear, 0, 0, scale_y, 0]
            image = image.transform((int(w * scale_x), int(h * scale_y)), Image.AFFINE, matrix, resample=Image.Resampling.BICUBIC)
            mask = mask.transform((int(w * scale_x), int(h * scale_y)), Image.AFFINE, matrix, resample=Image.Resampling.NEAREST)

        # 旋转
        if rotated:
            angle = random.uniform(-20, 20)
            image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)

            rotated_width, rotated_height = image.size
            if rotated_width > bg_width or rotated_height > bg_height:
                scale_factor = min(bg_width / rotated_width, bg_height / rotated_height) * 0.9
                new_width = int(rotated_width * scale_factor)
                new_height = int(rotated_height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)

        # 透视扭曲（模拟 3D 旋转感）
        if perspective:
            image_np = np.array(image)
            mask_np = np.array(mask)

            h, w = image_np.shape[:2]
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [random.uniform(0, w * 0.2), random.uniform(0, h * 0.2)],
                [random.uniform(w * 0.8, w), random.uniform(0, h * 0.2)],
                [random.uniform(w * 0.8, w), random.uniform(h * 0.8, h)],
                [random.uniform(0, w * 0.2), random.uniform(h * 0.8, h)]
            ])
            M = cv2.getPerspectiveTransform(src, dst)
            image_np = cv2.warpPerspective(image_np, M, (w, h), flags=cv2.INTER_CUBIC)
            mask_np = cv2.warpPerspective(mask_np, M, (w, h), flags=cv2.INTER_NEAREST)

            image = Image.fromarray(image_np)
            mask = Image.fromarray(mask_np)

        return image, mask

    def dummy_data(self):
        img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        valid = 0
        return img, valid

    def load_image(self, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
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
        objects = each.get('objects', None)
        if generation_only:
            text_input_ids, text_attention_mask = None, None
        elif not generation_only and not do_replace and not do_mask and objects is None:
            input_text = f"The image is {self.image_place_holder}\n" + input_text
            qformer_inputs = self.qformer_tokenizer("whole image", return_tensors="pt", padding=True)
            text_input_ids = qformer_inputs['input_ids']
            text_attention_mask = qformer_inputs['attention_mask']

        if 'objects' in each:
            objects = each['objects']
            if objects is not None:
                if isinstance(objects, list):

                    qformer_inputs = self.qformer_tokenizer(objects, return_tensors="pt", padding=True)
                    text_input_ids = qformer_inputs['input_ids']   # 形状为 (N, seq_len)
                    text_attention_mask = qformer_inputs['attention_mask']  # 形状为 (N, seq_len)
                    if not each.get('input_img_list', False):
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
                        input_text = each['input_text'].replace(self.args.image_place_holder, self.image_place_holder).strip()
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

    def load_image_nocrop(self, path_or_img):
        """
        加载一张图片，不做中心裁剪：
          1. 读入 PIL.Image
          2. pad 到正方形（短边两侧各加黑边）
          3. resize 到 (self.image_size, self.image_size)
        """
        try:
            if isinstance(path_or_img, str):
                img = Image.open(path_or_img).convert("RGB")
            else:
                # 已经是 PIL.Image 或类似对象
                img = path_or_img.convert("RGB") if hasattr(path_or_img, "convert") else path_or_img

            w, h = img.size
            m = max(w, h)
            # pad 成正方形，背景色设置为黑色
            padded = Image.new("RGB", (m, m), (0, 0, 0))
            paste_x = (m - w) // 2
            paste_y = (m - h) // 2
            padded.paste(img, (paste_x, paste_y))
        except Exception as e:
            print(f"Error loading image {path_or_img}: {e}")
            padded = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
        # resize
        return padded.resize((self.image_size, self.image_size), Image.LANCZOS)
        
    def _process_objects(self, each, do_mask, do_replace, generation_only):

        img_path = each['image_path']
        source_path = each['source_image']
        valid = 1

        if isinstance(each.get('objects', None), list):
            if isinstance(img_path, list) and len(img_path) > 0:
                if each.get('input_img_list', False):
                    loaded_inputs = []
                    for item in img_path:
                        if self.do_central_crop:
                            loaded_inputs.append(self.load_image(item))
                        else:
                            loaded_inputs.append(self.load_image_nocrop(item))
                    img = self.load_image(source_path) if isinstance(source_path, str) else self.load_image(source_path[0])
                    source_image = loaded_inputs
                else:
                    mask_images = []
                    for mask_item in img_path:
                        if isinstance(mask_item, str):
                            mask_img = self.load_image(mask_item).convert("L")
                        else:
                            mask_img = mask_item.convert("L") if hasattr(mask_item, "convert") else mask_item
                        mask_images.append(mask_img)
                    source_image_orig = self.load_image(source_path) if isinstance(source_path, str) else self.load_image(source_path[0])
                    cropped_images = []
                    segment_type = random.choice(["bbox", "segment"])
                    for mask in mask_images:
                        try:
                            cropped = self.crop_with_mask(mask, source_image_orig,method=segment_type)
                        except Exception as e:
                            print(e)
                            cropped = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
                        cropped_images.append(cropped)
                    img = source_image_orig
                    source_image = cropped_images

        else:
            mask = self.load_image(img_path).convert("L")
            mask_array = np.array(mask)
            if self.args.do_recovery:
                img = self.load_image(source_path)  # source_path is ground truth image
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
                            if random.random() < 0.5:
                                rotated, stretch, perspective =True, True,True
                            else:
                            # rotated, stretch, perspective =True, True,True
                                rotated, stretch, perspective =False, False,False
                            segmented_image_transformed, mask_image_transformed = self.random_transform(segmented_image, mask_image, (bg_width, bg_height),rotated=rotated, stretch=stretch,perspective=perspective)
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


def build_ti2i(args, transform, data_path, processor, max_samples=-1, is_val=False, dreambench_eval=False, with_image_only=False,
               image_only_rate=0.6):
    return TextImg2ImgDataset(args, transform, data_path, processor, max_samples=max_samples, is_val=is_val,dreambench_eval=dreambench_eval,
                            with_image_only=with_image_only, image_only_rate=image_only_rate)