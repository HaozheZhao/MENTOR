# from https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py
import math
import random
import numpy as np
from PIL import Image


# def center_crop_arr(pil_image, image_size):
#     """
#     Center cropping implementation from ADM.
#     https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
#     """
#     while min(*pil_image.size) >= 2 * image_size:
#         pil_image = pil_image.resize(
#             tuple(x // 2 for x in pil_image.size), resample=Image.BOX
#         )

#     scale = image_size / min(*pil_image.size)
#     pil_image = pil_image.resize(
#         tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
#     )

#     arr = np.array(pil_image)
#     crop_y = (arr.shape[0] - image_size) // 2
#     crop_x = (arr.shape[1] - image_size) // 2
#     return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

# def center_crop_arr(pil_image, image_size):
#     # if isinstance(pil_image, Image.Image):
#     try:
#         original_width, original_height = pil_image.size
#     except Exception as e:
#         print(e,pil_image, type(pil_image),  isinstance(pil_image, Image.Image) )
#         import traceback
#         traceback.print_exc()
#         assert False
#
#     # else:
#     #     print("pil_image is not a valid Image object.")
#
#     if min(original_width, original_height) < image_size:
#         aspect_ratio = original_width / original_height
#
#         if aspect_ratio > 1:
#             new_height = image_size
#             new_width = round(image_size * aspect_ratio)
#         else:
#             new_width = image_size
#             new_height = round(image_size / aspect_ratio)
#
#         pil_image = pil_image.resize((new_width, new_height), resample=Image.BICUBIC)
#
#     else:
#         while min(*pil_image.size) >= 2 * image_size:
#             pil_image = pil_image.resize(
#                 tuple(x // 2 for x in pil_image.size), resample=Image.BOX
#             )
#
#         scale = image_size / min(*pil_image.size)
#         pil_image = pil_image.resize(
#             tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
#         )
#
#     arr = np.array(pil_image)
#     crop_y = (arr.shape[0] - image_size) // 2
#     crop_x = (arr.shape[1] - image_size) // 2
#
#     return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def center_crop_arr(pil_image, image_size):
    try:
        original_width, original_height = pil_image.size
    except Exception as e:
        print(e, pil_image, type(pil_image), isinstance(pil_image, Image.Image))
        import traceback
        traceback.print_exc()
        raise  # 改成raise，这样在异常时不会继续执行而是在调用点中捕获异常

    # 如果图像最小维小于目标大小，按比例扩大图像
    if min(original_width, original_height) < image_size:
        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:  # 宽大于高
            new_height = image_size
            new_width = round(image_size * aspect_ratio)
        else:  # 高大于或等于宽
            new_width = image_size
            new_height = round(image_size / aspect_ratio)

        pil_image = pil_image.resize((max(1, new_width), max(1, new_height)), resample=Image.BICUBIC)

    else:
        # 当图像大于两倍于目标大小时，持续缩小至小于两倍
        while min(pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(max(1, x // 2) for x in pil_image.size), resample=Image.BOX
            )

        scale = image_size / min(pil_image.size)
        final_size = tuple(max(1, round(x * scale)) for x in pil_image.size)

        pil_image = pil_image.resize(final_size, resample=Image.BICUBIC)

    # 转换为矩阵切割图像中心部分
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2

    # 确保不会切出边界
    crop_y = max(0, crop_y)
    crop_x = max(0, crop_x)

    cropped_arr = arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

    # 如果终切掉区域小于期望的尺寸，填充
    if cropped_arr.shape[0] < image_size or cropped_arr.shape[1] < image_size:
        padded_arr = np.zeros((image_size, image_size, cropped_arr.shape[2]), dtype=cropped_arr.dtype)
        y_offset = (image_size - cropped_arr.shape[0]) // 2
        x_offset = (image_size - cropped_arr.shape[1]) // 2
        padded_arr[y_offset:y_offset + cropped_arr.shape[0], x_offset:x_offset + cropped_arr.shape[1]] = cropped_arr
        cropped_arr = padded_arr

    return Image.fromarray(cropped_arr)  

def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])

