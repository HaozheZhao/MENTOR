# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT
#   nanoGPT: https://github.com/karpathy/nanoGPT

# import debugpy
#
# # Listen for a debugger client to attach
# debugpy.listen(("0.0.0.0", 5678))
#
# # Wait for the client to attach to the debugger before proceeding
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from textwrap import fill
import torch
import socket
import wandb
# torch._dynamo.config.capture_scalar_outputs = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from glob import glob
from transformers import BertTokenizer
import time
import argparse
import os
from torch.nn.utils.rnn import pad_sequence  
from utils.distributed import init_distributed_mode
from utils.logger import create_logger
from dataset.build import build_dataset
from dataset.augmentation import center_crop_arr
from autoregressive.train.train_c2i import creat_optimizer
from autoregressive.models.gpt import GPT_models
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.generate import generate
from torchvision.utils import save_image
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint
from scheduler import AnnealingLR
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
def text_to_image(text, width, height, font_size=30, padding=10):
    """Converts a single text string to an image with automatic line wrapping based on width."""
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Split text into lines based on the image width
    wrapped_text = fill(text, width=width // (font_size // 2))
    lines = wrapped_text.split('\n')
    total_text_height = sum(draw.textbbox((0, 0), line, font=font)[3] for line in lines) + padding * (len(lines) - 1)
    y_text = (height - total_text_height) / 2

    for line in lines:
        bounding_box = draw.textbbox((0, 0), line, font=font)
        line_width = bounding_box[2]
        line_height = bounding_box[3]
        x_text = (width - line_width) / 2
        draw.text((x_text, y_text), line, font=font, fill=(0, 0, 0))
        y_text += line_height + padding

    return img

def postprocess(tensors, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]):
    # Denormalize tensors
    def clip_processor_denormalize(tensors, mean, std):
        for tensor, m, s in zip(tensors, mean, std):
            tensor.mul_(s).add_(m)
        return tensors

    # Convert tensors to numpy arrays and then to PIL Images
    image_pil_list = []
    for image_tensor in tensors:
        denormalized_tensor = clip_processor_denormalize(image_tensor, mean, std)
        # Convert back to the range [0, 255], change data type and format
        image_np = denormalized_tensor.mul(255).byte().permute(1, 2, 0).detach().cpu().numpy()
        # Create PIL image and append to list
        image_pil_list.append(Image.fromarray(image_np))

    return image_pil_list

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    print(f"Using seed: {args.global_seed}")
    print(f"Using visual encoder type: {args.multimodal_encoder}")
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-") 
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        # cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        # os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        # logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


    # Setup model
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
        # blip2 input embedding
        use_vision_tower = args.use_vision_tower,
        latent_size = latent_size,
        model_name_or_path = args.model_name_or_path,
        mixed_precision = args.mixed_precision,
        image_place_holder = args.image_place_holder,
        processor_path = args.processor_path,
        max_seq_length = args.cls_token_num,
        multimodal_encoder = args.multimodal_encoder
        
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")



    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)



    # Setup data:
    if args.dataset == 't2i' or args.dataset == 'ti2i':     # create and load model
        vq_model = VQ_models[args.vq_model](
            codebook_size=args.codebook_size,
            codebook_embed_dim=args.codebook_embed_dim)
        vq_model.to(device)
        vq_model.eval()
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        del checkpoint        
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    def custom_collate_fn(batch):  
        # Extract individual fields from batch data:
        expected_length = len(batch[0])

        if expected_length == 7:
            # Unpack according to the expected size of 6
            images, pixel_values_list, cond_idxs, attention_masks, image_masks,img_pixel, valids = zip(*batch)
            img_pixel = torch.stack(img_pixel)
        elif expected_length == 8:
            # Unpack according to the expected size of 7
            images, pixel_values_list, cond_idxs, attention_masks, image_masks,img_pixel, valids, pixel_source = zip(*batch)
            pixel_source = torch.stack(pixel_source)
            img_pixel = torch.stack(img_pixel)
        elif expected_length == 9:
            # Unpack according to the expected size of 8
            images, pixel_values_list, cond_idxs, attention_masks, image_masks, img_pixel,valids, text_input_ids, text_attention_mask = zip(*batch)
            img_pixel = torch.stack(img_pixel)
        elif expected_length == 10:
            # Unpack according to the expected size of 9
            images, pixel_values_list, cond_idxs, attention_masks, image_masks,img_pixel, valids, pixel_source, text_input_ids, text_attention_mask = zip(*batch)
            pixel_source = torch.stack(pixel_source)
            img_pixel = torch.stack(img_pixel)
        else:
            # Handle unexpected cases
            raise ValueError(f"Unexpected batch element size: {expected_length}. Expecting 6 or 7.")

            # Stack images (assuming they are already of equal size)
        images = torch.stack(images)  

        # Identify the maximum number of images in the batch for padding purposes  
        max_num_of_images = max(pv.shape[1] for pv in pixel_values_list)

        # Pad and stack pixel_values  
        padded_pixel_values = []  
        for pv in pixel_values_list:  
            num_of_images = pv.shape[1]
            # Calculate padding required  
            padding_needed = max_num_of_images - num_of_images  
            # Pad the tensor to the desired size  
            # Padding size format: (lastdim_pad_before, lastdim_pad_after, ..., firstdim_pad_before, firstdim_pad_after)  
            # padded_pv = torch.nn.functional.pad(pv, (0, 0, 0, 0, 0, 0, 0, padding_needed), "constant", 0)
            padded_pv = torch.nn.functional.pad(pv, (0, 0, 0, 0, 0, 0,padding_needed, 0),
                                                "constant", 0)
            padded_pixel_values.append(padded_pv)  
        
        # Stack all padded pixel_values into a single tensor
        # for each in padded_pixel_values:
        #     logger.info("each shape: {}".format(each.shape))
        pixel_values = torch.concat(padded_pixel_values)  
        
        cond_idxs_padded = pad_sequence(cond_idxs, batch_first=True, padding_value=0)  
        attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)  

        # Pad image masks with `False` as padding value  
        padded_image_masks = pad_sequence(image_masks, batch_first=True, padding_value=False)  

        # Stack valid flags  
        valids = torch.stack(valids)  

        if expected_length == 9 or expected_length == 10:
            text_input_ids_padded = pad_sequence(text_input_ids, batch_first=True, padding_value=0)
            text_attention_mask_padded = pad_sequence(text_attention_mask, batch_first=True, padding_value=0)
        
        # print(f"images shape: {images.shape}")
        # print(f"pixel_values shape: {pixel_values.shape}")
        # print(f"cond_idxs_padded shape: {cond_idxs_padded.shape}")
        # print(f"attention_masks_padded shape: {attention_masks_padded.shape}")
        # print(f"padded_image_masks shape: {padded_image_masks.shape}")
        # print(f"valids shape: {valids.shape}")
        # print(f"text_input_ids_padded shape: {text_input_ids_padded.shape}")
        # print(f"text_attention_mask_padded shape: {text_attention_mask_padded.shape}")
        # print("+==========================")

        # if expected_length == 9:
        #     return images, pixel_values, cond_idxs_padded, attention_masks_padded, padded_image_masks, img_pixel, valids, text_input_ids_padded, text_attention_mask_padded
        # elif expected_length == 10:
        #     return images, pixel_values, cond_idxs_padded, attention_masks_padded, padded_image_masks,img_pixel, valids, pixel_source, text_input_ids_padded, text_attention_mask_padded
        
        valids_bool = valids.bool()

        if valids_bool.sum() > 0:
            common_items = [images[valids_bool], pixel_values[valids_bool], cond_idxs_padded[valids_bool], attention_masks_padded[valids_bool], padded_image_masks[valids_bool],img_pixel[valids_bool], valids[valids_bool]]  

            if expected_length == 8:
                common_items.extend([pixel_source[valids_bool]])
            elif expected_length == 9:
                common_items.extend([text_input_ids_padded[valids_bool], text_attention_mask_padded[valids_bool]])
            elif expected_length == 10:
                common_items.extend([pixel_source[valids_bool], text_input_ids_padded[valids_bool], text_attention_mask_padded[valids_bool]])
        else:
            common_items = [images, pixel_values, cond_idxs_padded, attention_masks_padded, padded_image_masks,img_pixel, valids]  

            if expected_length == 8:
                common_items.extend([pixel_source])
            elif expected_length == 9:
                common_items.extend([text_input_ids_padded, text_attention_mask_padded])
            elif expected_length == 10:
                common_items.extend([pixel_source, text_input_ids_padded, text_attention_mask_padded])

        return tuple(common_items)


    if args.use_vision_tower:
        dataset = build_dataset(args, transform=transform, data_path = args.data_path, processor = model.multimodal_processor,with_image_only=args.with_image_only, image_only_rate = args.image_only_rate, stage2 = args.stage2 )
    else:
        dataset = build_dataset(args, transform=transform)


    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )


    if args.use_vision_tower:
        loader = DataLoader(
            dataset,
            batch_size=int(args.global_batch_size // dist.get_world_size()),
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=int(args.global_batch_size // dist.get_world_size()),
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
    logger.info(f"Dataset contains {len(dataset):,} images")

    train_iters = len(loader) * args.epochs
    # define lr_scheduler
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.warmup * train_iters,
        num_iters=train_iters,
        decay_style=args.lr_decay_style,
        last_iter=-1,
        decay_ratio=args.lr_decay_ratio
    )


    if args.do_eval:
        val_dataset = build_dataset(args, transform=transform, data_path = args.val_data_path, processor = model.multimodal_processor, max_samples = args.max_eval_samples,is_val = True, with_image_only=args.with_image_only, image_only_rate = args.image_only_rate, stage2 = args.stage2 )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.global_batch_size // dist.get_world_size()),
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )
 
        
        
    # Prepare models for training:

    if args.load_from_checkpoint is not None:
        strict = True
        if not args.stage2:
            result = load_sharded_checkpoint(model.multimodal_encoder,args.model_name_or_path,strict=False)
            print(f'load multimodal_encoder from pretrained CKPT: {args.model_name_or_path}  Result: ',result)
            if args.subject_driven:
                if args.multimodal_encoder == 'blip':
                    subject_embedding_ckpt = torch.load(args.load_subject_embedding, map_location="cpu")
                    model.multimodal_encoder.qformer.embeddings.load_state_dict(subject_embedding_ckpt, strict=True) 
                    del subject_embedding_ckpt
            strict = False
        checkpoint = torch.load(args.load_from_checkpoint, map_location="cpu")
        result_llama = model.load_state_dict(checkpoint["model"], strict=strict)
        print(f'load generator from pretrained CKPT: {args.load_from_checkpoint} Result: ', result_llama)
        del checkpoint
        
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
        steps_per_epoch = len(dataset) // args.global_batch_size  
        start_epoch = train_steps // steps_per_epoch  
        skip_epoch_steps =  train_steps % steps_per_epoch
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        skip_epoch_steps =0

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0
    if args.use_vision_tower:
        logger.info(f"freeze the vit")
        for para in model.multimodal_encoder.parameters():
            para.requires_grad = False

        if args.train_text_encoder:
            logger.info(f"set text encoder as trainable")

            for para in model.multimodal_encoder.language_projection.parameters():
                para.requires_grad = True
            if args.subject_driven and not args.stage2:
                for para in model.multimodal_encoder.language_model.parameters():
                    para.requires_grad = False
            else:
                for para in model.multimodal_encoder.language_model.parameters():
                    para.requires_grad = True

    all_param = 0
    trained_param=0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad ==True:
            trained_param+=param.numel()
    total_param = all_param

    logger.info('***** total param is {} *****'.format(total_param))
    logger.info('***** total trained param is {} *****'.format(trained_param))
    model.cuda(torch.cuda.current_device())
    model = DDP(model, device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()
    if torch.distributed.get_rank() == 0:
        print('initializing wandb')
        wandb.init(config=args,
                project=args.project_name,
                notes=socket.gethostname(),
                name=args.results_dir+"_"+str(args.global_seed),
                dir=str(args.results_dir),
                job_type="training",
                reinit=True)
    qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
    qformer_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        # images, pixel_values, cond_idxs_padded, attention_masks_padded, padded_image_masks, valids  
        logger.info(f"Beginning epoch {epoch}...")
        current_step = 0 
        for sample in tqdm(loader):

            if current_step < skip_epoch_steps:  
                current_step += 1  
                continue  # Skip processing of this sample and move to the next  
            elif current_step == skip_epoch_steps and skip_epoch_steps > 0:
                skip_epoch_steps == 0
            current_step += 1  
            # logger.info(f"Training {train_steps} steps...")
            if args.use_vision_tower:
                if args.subject_driven:
                    x, pixel_values, c_indices, cond_attn_mask, image_masks,gt_img, valid, text_input_ids, text_attention_mask = sample
                else:
                    x, pixel_values, c_indices, cond_attn_mask, image_masks,gt_img, valid = sample
                
                x = x.to(device, non_blocking=True)
                pixel_values = pixel_values.to(device, non_blocking=True)
                c_indices = c_indices.to(device, non_blocking=True)
                cond_attn_mask = cond_attn_mask.to(device, non_blocking=True)
                image_masks = image_masks.to(device, non_blocking=True)
                if args.dataset == 't2i' or args.dataset == 'ti2i':
                    img = x
                    with torch.no_grad():
                        _, _, [_, _, indices] = vq_model.encode(img)
                    x = indices.reshape(img.shape[0], -1)
                z_indices = x.reshape(x.shape[0], -1)
                # assert z_indices.shape[0] == c_indices.shape[0]
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    # c_indices t5 input_ids z_indices image tokens
                    if args.subject_driven:
                        _, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1],pixel_values=pixel_values,cond_idx_mask= cond_attn_mask,img_mask=image_masks, targets=z_indices, valid=valid, text_input_ids=text_input_ids, text_attention_mask=text_attention_mask)
                    else:
                        _, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1],pixel_values=pixel_values,cond_idx_mask= cond_attn_mask,img_mask=image_masks, targets=z_indices, valid=valid)
            
            else:
                x, y, attn_mask, valid  = sample
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                if args.dataset == 't2i' or args.dataset == 'ti2i':
                    img = x
                    with torch.no_grad():
                        _, _, [_, _, indices] = vq_model.encode(img)
                    x = indices.reshape(img.shape[0], -1)
                z_indices = x.reshape(x.shape[0], -1)
                c_indices = y.reshape(y.shape[0], y.shape[-2], y.shape[-1])
                assert z_indices.shape[0] == c_indices.shape[0]
                attn_mask = attn_mask.reshape(attn_mask.shape[0], 1, attn_mask.shape[-2], attn_mask.shape[-1]) # (bs, n_head, seq_len, seq_len)
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    # c_indices t5 embedding z_indices image tokens
                    _, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices, mask=attn_mask[:, :, :-1,:-1], valid=valid)
                # backward pass, with gradient scaling if training in fp16         
            
 # backward pass, with gradient scaling if training in fp16         
            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            lr_scheduler.step()

            optimizer.zero_grad(set_to_none=True)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if torch.distributed.get_rank() == 0:
                    wandb_log = {
                        'train lr': optimizer.param_groups[0]['lr'],
                        'train loss': avg_loss,
                        'Train Steps/Sec': steps_per_sec,
                    }
                    wandb.log(wandb_log
                              , step=train_steps)

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            if args.do_eval and train_steps % args.eval_steps == 0 and train_steps >0:
                if torch.distributed.get_rank() == 0:
                    with torch.no_grad():
                        torch.cuda.synchronize()
                        eval_model = GPT_models[args.gpt_model](
                            vocab_size=args.vocab_size,
                            block_size=latent_size ** 2,
                            num_classes=args.num_classes,
                            cls_token_num=args.cls_token_num,
                            model_type=args.gpt_type,
                            resid_dropout_p=args.dropout_p,
                            ffn_dropout_p=args.dropout_p,
                            token_dropout_p=args.token_dropout_p,
                            class_dropout_prob=args.class_dropout_prob,
                            # blip2 input embedding
                            use_vision_tower=args.use_vision_tower,
                            latent_size=latent_size,
                            model_name_or_path=args.model_name_or_path,
                            mixed_precision=args.mixed_precision,
                            image_place_holder=args.image_place_holder,
                            processor_path=args.processor_path,
                            max_seq_length=args.cls_token_num,
                            multimodal_encoder=args.multimodal_encoder

                        )

                        if not args.no_compile:
                            model_weight = model.module._orig_mod.state_dict()
                        else:
                            model_weight = model.module.state_dict()
                        eval_model.load_state_dict(model_weight, strict=True)

                        eval_model = eval_model.to(device, ptdtype, non_blocking=True)
                        eval_model.eval()
                        for eval_idx, patch in enumerate(tqdm(val_loader)):  
                            if args.subject_driven:
                                eval_x, eval_pixel_values, eval_c_indices, eval_cond_attn_mask, eval_image_masks,eval_gt_img, eval_valid, pixual_value_source, text_input_ids, text_attention_mask = patch
                                text_input_ids = text_input_ids.to(device, non_blocking=True)
                                text_attention_mask = text_attention_mask.to(device, non_blocking=True)
                            else:
                                eval_x, eval_pixel_values, eval_c_indices, eval_cond_attn_mask, eval_image_masks,eval_gt_img, eval_valid, pixual_value_source = patch
                                text_input_ids = None
                                text_attention_mask = None
                            eval_x = eval_x.to(device,ptdtype, non_blocking=True)
                            eval_pixel_values = eval_pixel_values.to(device,ptdtype, non_blocking=True)
                            eval_c_indices = eval_c_indices.to(device, non_blocking=True)
                            eval_cond_attn_mask = eval_cond_attn_mask.to(device, non_blocking=True)
                            eval_image_masks = eval_image_masks.to(device, non_blocking=True)

                            caption_embs = eval_model.get_multmodal_embeddings(
                                pixel_values=eval_pixel_values,
                                cond_idx=eval_c_indices,
                                cond_idx_mask=eval_cond_attn_mask,
                                img_mask=eval_image_masks,
                                text_input_ids = text_input_ids,
                                text_attention_mask = text_attention_mask
                            )
                            emb_masks = eval_cond_attn_mask
                            # if args.stage2:
                            #     caption_embs = caption_embs[:, :args.cls_token_num, :]
                            #     emb_masks = emb_masks[:, :args.cls_token_num]
                            if not args.no_left_padding:
                                print(f"processing left-padding...")
                                # a naive way to implement left-padding
                                new_emb_masks = torch.flip(emb_masks, dims=[-1])
                                new_caption_embs = []
                                for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                                    valid_num = int(emb_mask.sum().item())
                                    new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                                    new_caption_embs.append(new_caption_emb)
                                new_caption_embs = torch.stack(new_caption_embs)
                            else:
                                new_caption_embs, new_emb_masks = caption_embs, emb_masks
                            new_c_indices = new_caption_embs * new_emb_masks[:,:, None]
                            c_emb_masks = new_emb_masks
                            qzshape = [len(new_c_indices), args.codebook_embed_dim, latent_size, latent_size]
                            index_sample = generate(
                                eval_model, new_c_indices, latent_size ** 2,
                                c_emb_masks,
                                cfg_scale=args.cfg_scale,
                                temperature=args.temperature, top_k=args.top_k,
                                top_p=args.top_p, sample_logits=True,
                                )
                            samples = vq_model.decode_code(index_sample, qzshape)
                            eval_dir = f"{checkpoint_dir}/eval_step_{train_steps}"
                            os.makedirs(eval_dir, exist_ok=True)
                            sample_save_path = f"{eval_dir}/batch_{eval_idx}_cfg_{args.cfg_scale}_topk_{args.top_k}.png"
                            upload_path = f"{eval_dir}/batch_0_cfg_{args.cfg_scale}_topk_{args.top_k}.png"
                            ori_save_path = f"{eval_dir}/ori.png"
                            sample_text = eval_model.multimodal_processor.tokenizer.batch_decode(eval_c_indices, skip_special_tokens=True)


                            if text_input_ids is not None and args.subject_driven:
                                subject_text = qformer_tokenizer.batch_decode(text_input_ids, skip_special_tokens=True)

                            try:
                                if not args.stage2:
                                    p_values = eval_pixel_values[:,0] if len(eval_pixel_values.shape) == 5  else eval_pixel_values
                                else:
                                    p_values = pixual_value_source[:,0] if len(pixual_value_source.shape) == 5  else pixual_value_source
                                gt_values = eval_gt_img[:,0] if len(eval_gt_img.shape) == 5  else eval_gt_img



                                pixel_values_img_list = postprocess(p_values, mean=eval_model.multimodal_processor.image_processor.image_mean,
                                                               std=eval_model.multimodal_processor.image_processor.image_std)
                                eval_gt_img_list = postprocess(gt_values, mean=eval_model.multimodal_processor.image_processor.image_mean,
                                                                std=eval_model.multimodal_processor.image_processor.image_std)
                                transformed_images = [transform(img.resize(
                                    (args.image_size, args.image_size), Image.LANCZOS)) for img in pixel_values_img_list] # input images
                                transformed_images = torch.stack(transformed_images)
                                transformed_gt_imgs = [transform(img.resize(
                                    (args.image_size, args.image_size), Image.LANCZOS)) for img in eval_gt_img_list] # input images
                                transformed_gt_imgs = torch.stack(transformed_gt_imgs)
                                samples = samples.to(transformed_images.device)

                                if text_input_ids is not None and args.subject_driven:
                                    text_images_tensors = [
                                        transform(text_to_image(text+f" ## subject: {sbj}", args.image_size, args.image_size ))
                                        for sbj, text in zip(subject_text,sample_text)
                                    ]
                                else:

                                    text_images_tensors = [
                                        transform(text_to_image(text, args.image_size, args.image_size ))
                                        for  text in sample_text
                                    ]

                                text_images_tensors = torch.stack(text_images_tensors)


                                images_to_save = torch.cat((text_images_tensors,transformed_images, samples, transformed_gt_imgs), dim=0) # input text, input images, generated images, ground truth images
                                save_image(images_to_save, sample_save_path, nrow=eval_x.size(0), normalize=True,
                                           value_range=(-1, 1))
                            except Exception as e:
                            # If `samples` is a tensor: collect ori_images tensors
                                print(e)
                                import traceback
                                traceback.print_exc()
                                img_denormalized = eval_x * 0.5 + 0.5
                                img_denormalized = torch.clamp(img_denormalized, 0, 1)
                                save_image(img_denormalized, ori_save_path, nrow=img_denormalized.shape[0])

                                save_image(samples, sample_save_path, nrow=samples.shape[0], normalize=True, value_range=(-1, 1))
                        wandb.log(
                            {"eval_samples": wandb.Image(upload_path) },
                            step=train_steps,
                        )

                        # dist.barrier()
                        # model.train()
                    del eval_model
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.info(f"Finish Eval in {train_steps} steps...")

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    # torch.save(checkpoint, cloud_checkpoint_path)
                    # logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()
            

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    # parser.add_argument("--t5-feat-path", type=str, required=True)
    # parser.add_argument("--short-t5-feat-path", type=str, default=None, help="short caption of t5_feat_path")
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path", type=float, default=0.0, help="drop_path_rate of attention and ffn")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='ti2i')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    
    # blip2 inputs args
    
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--use_vision_tower", action='store_true')
    parser.add_argument("--model_name_or_path", type=str, default=None, help="ckpt path for blip2 model")
    parser.add_argument("--image_place_holder", type=str, default="<image>", help="image_place_holder")
    parser.add_argument("--processor_path", type=str, default=None, help=" path for image processor ")
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--max_eval_samples", type=int, default=250)

    parser.add_argument("--train_text_encoder", action='store_true')

    # generation eval:
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--top-k", type=int, default=5000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=0.9, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--project_name", type=str, default='llamagen_ti2i')
    
    parser.add_argument("--load_from_checkpoint", type=str, default=None)
    
    # lr

    parser.add_argument('--warmup', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                            'training iters). Default 0.01')

    parser.add_argument('--lr-decay-style', type=str, default='cosine',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='learning rate decay function')
    parser.add_argument('--lr-decay-ratio', type=float, default=0.1)
    
    parser.add_argument('--train-iters', type=int, default=500000,
                       help='total number of iterations to train over all training runs')
    
    parser.add_argument("--class-dropout-prob", type=float, default=0.1, help="cfg uncond prob")

    # image only
    parser.add_argument("--with_image_only", action='store_true')

    parser.add_argument("--image_only_rate", type=float, default=0.6, help="image_only_rate")

    parser.add_argument("--stage2", action='store_true')

    # subject driven
    parser.add_argument("--subject_driven", action='store_true')

    parser.add_argument("--load_subject_embedding", type=str, default=None)

    parser.add_argument("--reference_data_path", type=str, default=None)

    # 
    parser.add_argument("--multimodal_encoder", type=str, default="blip")


    args = parser.parse_args()
    main(args)
