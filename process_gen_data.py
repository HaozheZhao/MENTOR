import json  
import os  
from PIL import Image  
import numpy as np  
from tqdm import tqdm  
from multiprocessing import Pool, cpu_count  

def load_reference_images(reference_data_path):  
    with open(reference_data_path, 'r') as file:  
        return [json.loads(line) for line in file]  

def central_crop_and_resize(image, target_size):  
    width, height = image.size  
    min_dim = min(width, height)  
    left = (width - min_dim) / 2  
    top = (height - min_dim) / 2  
    right = (width + min_dim) / 2  
    bottom = (height + min_dim) / 2  
    
    image = image.crop((left, top, right, bottom))  
    image = image.resize(target_size, Image.Resampling.LANCZOS)  
    return image  

def get_random_background(reference_images, ori_image_size):  
    random_reference = np.random.choice(reference_images)  
    background_image_path = random_reference['input_image']  
    background_image = Image.open(background_image_path)  
    return central_crop_and_resize(background_image, ori_image_size)  

def process_example(args):  
    example, reference_images, save_dir = args 
    try:  
        source_img_path = example['source_img']  
        global_idx = example['global_idx']  
        caption = example['caption']  
        sam_objects = example['sam_objects']  

        ori_image = Image.open(source_img_path)  
        ori_image_size = ori_image.size  
        ori_image_area = ori_image_size[0] * ori_image_size[1]  

        masks = {}  
        for sam_obj in sam_objects:  
            segment_mask_path = source_img_path.replace("ori.jpg", f"mask_{sam_obj}.jpg")  
            mask = Image.open(segment_mask_path).convert('L')  
            mask_array = np.array(mask)  
            # if np.any(mask_array):  
            #     masks[sam_obj] = mask_array  
            # Calculate mask area  
            mask_area = np.sum(mask_array > 0)  

            # Only keep masks that cover more than 10% of the original image area  
            if mask_area >= 0.05 * ori_image_area and mask_area <= 0.95 * ori_image_area:  
                masks[sam_obj] = mask_array  

        output_data = []  
        segment_idx = 0
        for sam_obj, mask_array in masks.items():  
            if not mask_array.any():  
                continue  

            segmented_image = np.array(ori_image) * (mask_array[:, :, None] > 0)  
            segmented_image = Image.fromarray(segmented_image)  
            
            random_background = get_random_background(reference_images, ori_image_size)  
            segmented_image_with_background = Image.composite(segmented_image, random_background, Image.fromarray(mask_array))  
            
            image_save_path = os.path.join(save_dir, f"image_net_seg_{sam_obj}_{global_idx}.jpg")  
            segmented_image_with_background.save(image_save_path)  
            
            output_data.append({  
                "image_path": image_save_path,  
                "objects": sam_obj,  
                "source_image": source_img_path,  
                "global_idx": global_idx, 
                "segment_idx": segment_idx,
                "input_text": f"The image 0 is <image>\n{caption}"  
            })  
            segment_idx+=1
        
        return output_data  

    except Exception as e:  
        print(f"Error processing example {global_idx}: {e}")  

def main():  
    source_data_path = '/nobackup/zefan/projects/VLGen/segment_results_imgnet/total.jsonl'  
    reference_data_path = '/nobackup/zefan/projects/VLGen/sam-1b/image_paths_50.jsonl'  
    save_dir = '/nobackup/zefan/projects/VLGen/training_data/imagenet_seg/'  

    os.makedirs(save_dir, exist_ok=True)  

    with open(source_data_path, 'r') as file:  
        examples = [json.loads(line) for line in file]  

    reference_images = load_reference_images(reference_data_path)  
    num_workers = cpu_count()  

    with Pool(num_workers) as pool:  
        results = list(tqdm(pool.imap(process_example, [(ex, reference_images, save_dir) for ex in examples]), total=len(examples)))  

    results = [item for sublist in results for item in sublist if item]  

    with open(os.path.join(save_dir, 'output.jsonl'), 'w') as file:  
        for result in results:  
            file.write(json.dumps(result) + '\n')  

if __name__ == "__main__":  
    main()