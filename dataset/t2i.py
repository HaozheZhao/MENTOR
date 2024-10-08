from multiprocessing import process
import os
import json
from platform import processor
from matplotlib.pyplot import isinteractive
import numpy as np
from nltk.corpus import stopwords
import io
import base64
from openai import images
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
        self.t5_feature_max_len = 120
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


class TextImg2ImgDataset(Dataset):
    def __init__(self, args, transform, data_path, processor, max_samples=-1, is_val=False, with_image_only=False,
                 image_only_rate=0.6):
        img_path_list = load_jsonl(data_path)

        # img_path_list.append((img_path, code_dir, line_idx))
        self.img_path_list = img_path_list

        if max_samples > 0 and max_samples < len(self.img_path_list):
            self.img_path_list = random.sample(self.img_path_list, max_samples)
        self.transform = transform
        self.processor = processor
        self.qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
        self.qformer_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.args = args
        self.image_place_holder = "".join([args.image_place_holder] * 32)
        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = 120
        self.t5_feature_dim = 2048
        self.max_seq_length = self.t5_feature_max_len + self.code_len

        self.is_val = is_val

        self.image_only_rate = image_only_rate
        self.with_image_only = with_image_only

    def __len__(self):
        return len(self.img_path_list)

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

    def __getitem__(self, index):
        each = self.img_path_list[index]
        valid = 1

        img_path = each['image_path']
        source_image = each['source_image']
        input_text = each['input_text']
        if 'objects' in each:
            objects = each['objects']
            qformer_inputs = self.qformer_tokenizer(objects, return_tensors="pt", padding=True) 
            text_input_ids = qformer_inputs['input_ids'].squeeze(0)
            text_attention_mask = qformer_inputs['attention_mask'].squeeze(0)

        input_text = input_text.replace(self.args.image_place_holder, self.image_place_holder)

        if self.with_image_only:
            input_text_replace = "The image 0 is {}\n".format(self.image_place_holder)
            # replace by the image replace rate
            if random.random() <= self.image_only_rate:
                input_text = input_text_replace

        try:
            img = Image.open(img_path).convert("RGB")
            if min(img.size) < self.image_size:
                img, valid = self.dummy_data()
        except:
            img, valid = self.dummy_data()

        source_image = self.load_image(source_image) if isinstance(source_image, str) else [self.load_image(each) for
                                                                                            each in source_image]
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

        pixel_source = self.processor.image_processor(images = source_image,return_tensors="pt").pixel_values

        pixel_values = process_data['pixel_values']  # num_of_image, 3, 224, 224
        pixel_values = pixel_values.unsqueeze(0)  # 1, num_of_image, 3, 224, 224
        cond_idx = process_data['input_ids'].squeeze(0)  # 120
        attention_mask = process_data['attention_mask'].squeeze(0)  # 120



        image_mask = torch.ones((pixel_values.shape[1]), dtype=torch.bool)
        if self.transform is not None and valid == 1:
            img = self.transform(img)

        common_items = [img, pixel_values, cond_idx, attention_mask, image_mask, torch.tensor(valid)]  


        if 'objects' in each:  
            if self.is_val:  
                common_items.extend([pixel_source, text_input_ids, text_attention_mask])  
            else:  
                common_items.extend([text_input_ids, text_attention_mask])  
        else:  
            if self.is_val:  
                common_items.append(pixel_source)  

        # 返回构建的列表  
        return tuple(common_items)  


from datasets import load_metric, concatenate_datasets, load_dataset
from datasets import Dataset as Huggingface_Dataset


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
        self.image_place_holder = "".join([args.image_place_holder] * 32)
        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = 120
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
            [f"{self.preprocess_phrase(tgt)} is {self.image_place_holder}" for tgt in tgt_subject]) + "\n" + prompt


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

        max_length = 120 if self.is_val else 512
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


def build_ti2i(args, transform, data_path, processor, max_samples=-1, is_val=False, with_image_only=False,
               image_only_rate=0.6, stage2 =False):

    if not stage2:
        return TextImg2ImgDataset(args, transform, data_path, processor, max_samples=max_samples, is_val=is_val,
                              with_image_only=with_image_only, image_only_rate=image_only_rate)
    else:
        return TextImg2ImgStage2Dataset(args, transform, data_path, processor, max_samples=max_samples, is_val=is_val,
                              with_image_only=with_image_only, image_only_rate=image_only_rate)