#!/usr/bin/env python
# demo.py
"""
Single-image inference.
Example:
    python demo.py \
        --image_path tests/cat.jpg \
        --prompt "A cat in <image>.\n A cat in a 16-bit fantasy pixel-art scene" \
        --gpt_ckpt /path/to/gpt.pt \
        --vq_ckpt /path/to/vq.pt \
        --output out/cat_pixel.jpg \
        --mm_vision_tower "openai/clip-vit-large-patch14" \
        --multimodal_encoder llava \
"""

import argparse, os, torch
from PIL import Image
from tqdm import tqdm

from autoregressive.models.ori_gpt import GPT_models 
from autoregressive.models.empty_fix_gpt import GPT_models as GPT_models_fix_rope
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.generate import generate
from transformers import BertTokenizer


def load_models(args, device):
    torch_dtype = getattr(torch, args.torch_dtype)
    latent_size = args.image_size // args.downsample_size

    # GPT backbone
    gpt_cls = GPT_models_fix_rope if args.fixed_version else GPT_models
    gpt = gpt_cls[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        use_vision_tower=args.use_vision_tower,
        latent_size=latent_size,
        model_name_or_path=args.model_name_or_path,
        mixed_precision=args.mixed_precision,
        image_place_holder=args.image_place_holder,
        processor_path=args.processor_path,
        max_seq_length=args.cls_token_num,
        multimodal_encoder=args.multimodal_encoder,
        mm_vision_tower=args.mm_vision_tower,
    )
    state = torch.load(args.gpt_ckpt, map_location="cpu")
    state_dict = checkpoint.get("model", state)
    model_dict = gpt.state_dict()
    pretrained_dict = {
        k: v
        for k, v in state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(pretrained_dict)
    gpt.load_state_dict(model_dict, strict=False)
    gpt.eval().to(device, dtype=torch_dtype)

    # VQ decoder
    vq = VQ_models[args.vq_model](
        codebook_size=args.codebook_size, codebook_embed_dim=args.codebook_embed_dim
    )
    vq.load_state_dict(torch.load(args.vq_ckpt, map_location="cpu"))
    vq.eval().to(device, dtype=torch_dtype)

    if args.use_qformer:
        tok = BertTokenizer.from_pretrained(args.qformer_tokenizer_path, truncation_side="right"
    )
        tok.add_special_tokens({"bos_token": "[DEC]"})
    else:
        tok = None

    return gpt, vq, tok, latent_size, torch_dtype


def build_prompt(args, text_only: bool):
    if args.just_prompt or text_only:
        return args.prompt
    placeholder = args.image_place_holder * args.placeholder_num
    return args.prompt.replace(args.image_place_holder, placeholder)


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt, vq, qformer_tok, latent_size, dtype = load_models(args, device)

    cond_images = []
    if args.image_path:
        img = Image.open(args.image_path).convert("RGB")
        cond_images.append(img)
        text_only = False
    else:
        text_only = True

    # ── prompt & tokenizer ─────────────────────────────────
    full_prompt = build_prompt(args, text_only)
    multimodal_inputs = gpt.multimodal_processor(
        images=cond_images if not text_only else None,
        text=[full_prompt],
        max_length=args.cls_token_num,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    cond_idx = multimodal_inputs["input_ids"].to(device)
    att_mask = multimodal_inputs["attention_mask"].to(device)
    pixel_values = multimodal_inputs.get("pixel_values")
    if pixel_values is not None:
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
        pixel_values = pixel_values.to(device, dtype=dtype)
        img_mask = torch.ones(pixel_values.shape[:2], dtype=torch.bool).to(device)
    else:
        img_mask = None

    # Q-Former text
    if args.use_qformer:
        qformer_inputs = qformer_tok(
            ["image" if text_only else args.subject_name],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.qformer_max_len,
        )
        q_ids = qformer_inputs["input_ids"].to(device)
        q_mask = qformer_inputs["attention_mask"].to(device)
    else:
        q_ids = q_mask = None

    # ── Embedding & generation ─────────────────────────────
    with torch.no_grad():
        embs = gpt.get_multmodal_embeddings(
            pixel_values=pixel_values,
            cond_idx=cond_idx,
            cond_idx_mask=att_mask,
            img_mask=img_mask,
            text_input_ids=q_ids,
            text_attention_mask=q_mask,
        )
        emb_masks = att_mask
        if not args.no_left_padding:
            new_emb_masks = torch.flip(emb_masks, dims=[-1])
            new_caption_embs = []
            for caption_emb, emb_mask in zip(caption_embs, emb_masks):
                valid_num = int(emb_mask.sum().item())
                new_caption_emb = torch.cat(
                    [caption_emb[valid_num:], caption_emb[:valid_num]]
                )
                new_caption_embs.append(new_caption_emb)
            new_caption_embs = torch.stack(new_caption_embs)
        else:
            new_caption_embs, new_emb_masks = caption_embs, emb_masks

        new_c_indices = new_caption_embs * new_emb_masks[:, :, None]
        c_emb_masks = new_emb_masks
        qzshape = [
            len(new_c_indices),
            args.codebook_embed_dim,
            latent_size,
            latent_size,
        ]

        out_len = latent_size ** 2
        indices = generate(
            gpt,
            new_c_indices,
            out_len,
            c_emb_masks,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_logits=True,
        )
        img_tensor = vq.decode_code(indices, qzshape)[0].cpu()
        img_tensor = (img_tensor + 1) / 2
        img_tensor.clamp_(0, 1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    from torchvision.utils import save_image
    save_image(img_tensor, args.output)
    print(f"[✔] Saved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", type=str, default=None, help="image path")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--output", type=str, required=True)

    p.add_argument("--gpt_ckpt", type=str, required=True)
    p.add_argument("--vq_ckpt", type=str, required=True)
    p.add_argument("--gpt_model", type=str, default="GPT-XL", choices=list(GPT_models.keys()))
    p.add_argument("--vq_model", type=str, default="VQ-16", choices=list(VQ_models.keys()))
    p.add_argument("--fixed_version", action="store_true")
    p.add_argument("--gpt_type", type=str, default="t2i", choices=["c2i", "t2i"])
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--downsample_size", type=int, default=16)
    p.add_argument("--cls_token_num", type=int, default=120)
    p.add_argument("--multimodal_encoder", type=str, default="llava")
    p.add_argument("--model_name_or_path", type=str, default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--processor_path", type=str, default=None)
    p.add_argument("--mm_vision_tower", type=str, default=None)
    p.add_argument("--use_qformer", action="store_true")
    p.add_argument("--qformer_tokenizer_path", type=str, default="bert-base-uncased")
    p.add_argument("--vocab_size", type=int, default=16384)
    p.add_argument("--codebook_size", type=int, default=16384)
    p.add_argument("--codebook_embed_dim", type=int, default=8)
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    p.add_argument("--cfg_scale", type=float, default=2.5)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=5000)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--image_place_holder", type=str, default="<image>")
    p.add_argument("--placeholder_num", type=int, default=32,choices=[32,256] )
    p.add_argument("--qformer_max_len", type=int, default=32)
    p.add_argument("--just_prompt", action="store_true")
    p.add_argument("--subject_name", type=str, default=None)

    args = p.parse_args()

    if not args.use_qformer and args.mm_vision_tower is None:
        p.error("--mm_vision_tower is required")
    run_inference(args)
