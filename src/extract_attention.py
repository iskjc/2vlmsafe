from __future__ import annotations

import argparse
import os

import numpy as np
import torch

try:
    from src.generate import (
        import_transformers,
        load_model,
        resolve_device_and_dtype,
        select_model_class,
    )
    from src.models.input_builder import InputBuilder
    from src.models.learnable_tokens import LearnableTokens, LearnableTokensConfig
except ModuleNotFoundError:
    from generate import import_transformers, load_model, resolve_device_and_dtype, select_model_class
    from models.input_builder import InputBuilder
    from models.learnable_tokens import LearnableTokens, LearnableTokensConfig


def load_model_for_attention(auto_model_cls, model_name: str, dtype: torch.dtype, device: torch.device):
    """
    output_attentions=True requires eager attention for many decoder models.
    Try loading with eager first; fallback to project loader for compatibility.
    """
    eager_error = None
    load_kwargs = {"torch_dtype": dtype}

    for use_safetensors in (True, False):
        kwargs = dict(load_kwargs)
        kwargs["attn_implementation"] = "eager"
        if use_safetensors:
            kwargs["use_safetensors"] = True
        try:
            model = auto_model_cls.from_pretrained(model_name, **kwargs)
            return model.to(device)
        except TypeError as exc:
            eager_error = exc
            break
        except Exception as exc:
            eager_error = exc

    model = load_model(auto_model_cls, model_name, dtype=dtype, device=device)
    if hasattr(model, "config"):
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
        except Exception:
            pass
    if eager_error is not None:
        print(f"[warn] failed to load with attn_implementation='eager': {eager_error}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--n_plugin", type=int, default=16)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--vision_tokens", type=int, default=8)
    ap.add_argument("--out_dir", type=str, default="outputs/attn")
    ap.add_argument("--max_new_tokens", type=int, default=1)
    args = ap.parse_args()

    AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, AutoConfig = import_transformers()
    device, dtype = resolve_device_and_dtype(args.device, args.dtype)

    # FIX(local): keep tokenizer loading consistent for VLM checkpoints.
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    model_cls = select_model_class(
        args.model_name,
        auto_config_cls=AutoConfig,
        causal_cls=AutoModelForCausalLM,
        vl_cls=AutoModelForImageTextToText,
    )
    # FIX(local): select VL-capable model class when needed.
    model = load_model_for_attention(model_cls, args.model_name, dtype=dtype, device=device)
    model.config.output_attentions = True
    model.eval()

    hidden = model.config.hidden_size
    lt = LearnableTokens(LearnableTokensConfig(n_tokens=args.n_plugin, hidden_size=hidden)).to(device)
    lt.load(args.ckpt, map_location=device)

    text = tok(args.prompt, return_tensors="pt").to(device)
    input_ids = text.input_ids
    batch_size, text_len = input_ids.shape

    vision_len = args.vision_tokens
    vision_embeds = torch.zeros((batch_size, vision_len, hidden), device=device, dtype=dtype)

    base_builder = InputBuilder(model, plugin_len=0, use_position_ids=True)
    plug_builder = InputBuilder(model, plugin_len=args.n_plugin, use_position_ids=True)

    base = base_builder.build(
        vision_embeds=vision_embeds,
        learnable_embeds=torch.zeros((batch_size, 0, hidden), device=device, dtype=dtype),
        text_input_ids=input_ids,
        text_attention_mask=text.attention_mask,
    )

    learnable = lt(batch_size, device=device, dtype=dtype)
    plug = plug_builder.build(
        vision_embeds=vision_embeds,
        learnable_embeds=learnable,
        text_input_ids=input_ids,
        text_attention_mask=text.attention_mask,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        try:
            base_out = model(
                inputs_embeds=base.inputs_embeds,
                attention_mask=base.attention_mask,
                position_ids=base.position_ids,
                output_attentions=True,
                use_cache=False,
            )
            plug_out = model(
                inputs_embeds=plug.inputs_embeds,
                attention_mask=plug.attention_mask,
                position_ids=plug.position_ids,
                output_attentions=True,
                use_cache=False,
            )
        except Exception as exc:
            msg = str(exc)
            if "sdpa" in msg.lower() and "output_attentions" in msg.lower():
                raise RuntimeError(
                    "SDPA does not support output_attentions=True for this model. "
                    "Please use a transformers version that supports "
                    "attn_implementation='eager' at load time."
                ) from exc
            raise

    base_attn = base_out.attentions
    plug_attn = plug_out.attentions
    if base_attn is None or plug_attn is None:
        raise RuntimeError(
            "Model did not return attentions. Some model families do not expose attentions in this mode."
        )

    base_np = np.stack([a[0].float().cpu().numpy() for a in base_attn], axis=0)
    plug_np = np.stack([a[0].float().cpu().numpy() for a in plug_attn], axis=0)

    np.save(os.path.join(args.out_dir, "base_attn.npy"), base_np)
    np.save(os.path.join(args.out_dir, "plug_attn.npy"), plug_np)

    meta = {
        "V": vision_len,
        "N": int(args.n_plugin),
        "T": int(text_len),
        "L_base": int(base_np.shape[-1]),
        "L_plug": int(plug_np.shape[-1]),
        "prompt": args.prompt,
    }
    with open(os.path.join(args.out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")

    print("[saved]", os.path.join(args.out_dir, "base_attn.npy"))
    print("[saved]", os.path.join(args.out_dir, "plug_attn.npy"))
    print("[meta ]", os.path.join(args.out_dir, "meta.txt"))


if __name__ == "__main__":
    main()
