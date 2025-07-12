from dataset.imagenet import build_imagenet, build_imagenet_code
from dataset.coco import build_coco
from dataset.openimage import build_openimage
from dataset.pexels import build_pexels
from dataset.t2i import build_t2i, build_t2i_code, build_t2i_image,build_ti2i


def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'ti2i':
        return build_ti2i(args, **kwargs)   
    raise ValueError(f'dataset {args.dataset} is not supported')