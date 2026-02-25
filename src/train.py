from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

try:
    from src.models.learnable_tokens import LearnableTokens, LearnableTokensConfig
    from src.models.input_builder import InputBuilder
except ModuleNotFoundError:
    # Support direct execution: `python src/train.py`
    from models.learnable_tokens import LearnableTokens, LearnableTokensConfig
    from models.input_builder import InputBuilder


def freeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment-specific
        msg = str(exc)
        hint = ""
        if "numpy.dtype size changed" in msg:
            hint = (
                " Detected NumPy/scikit-learn binary mismatch. "
                "Try reinstalling compatible versions, e.g. "
                "`pip install --upgrade --force-reinstall numpy scikit-learn`."
            )
        raise SystemExit(f"Failed to import transformers: {msg}.{hint}")
    return AutoTokenizer, AutoModelForCausalLM


def resolve_device_and_dtype(device_name: str, dtype_name: str) -> tuple[torch.device, torch.dtype]:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        device_name = "cpu"

    try:
        device = torch.device(device_name)
    except Exception as exc:
        raise SystemExit(f"Invalid --device value '{device_name}': {exc}")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[dtype_name]

    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        print(f"{dtype_name} on CPU may be unsupported; using fp32.")
        dtype = torch.float32

    if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("bf16 is not supported on this CUDA device; falling back to fp16.")
        dtype = torch.float16

    return device, dtype


def _is_torch_too_old_error(msg: str) -> bool:
    return "upgrade torch to at least v2.6" in msg.lower()


def _is_missing_safetensors_error(msg: str) -> bool:
    text = msg.lower()
    return "safetensors" in text and (
        "does not appear to have a file named" in text
        or "no file named" in text
        or "cannot be found" in text
    )


def load_model(
    auto_model_cls,
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
):
    load_kwargs = {"dtype": dtype}

    try:
        model = auto_model_cls.from_pretrained(model_name, use_safetensors=True, **load_kwargs)
        return model.to(device)
    except Exception as exc:
        msg = str(exc)

    if _is_missing_safetensors_error(msg):
        try:
            model = auto_model_cls.from_pretrained(model_name, **load_kwargs)
            return model.to(device)
        except Exception as exc:
            inner_msg = str(exc)
            if _is_torch_too_old_error(inner_msg):
                raise SystemExit(
                    "Model loading failed: this checkpoint likely requires .bin loading, "
                    "which now needs torch>=2.6 in recent transformers. "
                    "Upgrade torch or use a safetensors model."
                )
            raise SystemExit(f"Failed to load model '{model_name}': {inner_msg}")

    if _is_torch_too_old_error(msg):
        raise SystemExit(
            "Model loading failed due to torch version. "
            "Upgrade torch to >=2.6 or use a safetensors checkpoint."
        )

    raise SystemExit(f"Failed to load model '{model_name}': {msg}")


def main() -> None:
    import math
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])

    # 插件与训练超参
    ap.add_argument("--n_plugin", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # Loss 权重
    ap.add_argument("--lambda_safe", type=float, default=1.0)
    ap.add_argument("--lambda_util", type=float, default=1.0)
    ap.add_argument("--lambda_gate", type=float, default=0.2)

    # Gate（门控）
    ap.add_argument("--use_gate", action="store_true")
    ap.add_argument("--gate_hidden", type=int, default=256)

    # 视觉 token（暂时用占位；接真 VLM 时替换 vision_embeds 构造）
    ap.add_argument("--vision_tokens", type=int, default=8)

    # 仅用于 toy 数据演示 gate：给 harmful/benign 注入不同强度噪声，使 gate 有可学信号
    ap.add_argument("--toy_vision_signal", action="store_true")

    ap.add_argument("--save_path", type=str, default="outputs/learnable_tokens.pt")
    ap.add_argument("--save_gate_path", type=str, default="outputs/gate.pt")
    args = ap.parse_args()

    if args.n_plugin <= 0:
        raise SystemExit("--n_plugin must be > 0")
    if args.steps <= 0:
        raise SystemExit("--steps must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be > 0")
    if args.vision_tokens < 0:
        raise SystemExit("--vision_tokens must be >= 0")

    AutoTokenizer, AutoModelForCausalLM = import_transformers()
    device, dtype = resolve_device_and_dtype(args.device, args.dtype)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = load_model(AutoModelForCausalLM, args.model_name, dtype=dtype, device=device)
    freeze_model(model)

    hidden_size = model.config.hidden_size

    # Learnable Tokens
    cfg = LearnableTokensConfig(
        n_tokens=args.n_plugin,
        hidden_size=hidden_size,
        init_from_token_id=tok.pad_token_id if tok.pad_token_id is not None else None,
    )
    lt = LearnableTokens(cfg).to(device=device,dtype=dtype)
    lt.initialize(model.get_input_embeddings())

    # Gate（可选）
    gate = None
    if args.use_gate:
        try:
            from src.models.gate import VisionGate, VisionGateConfig
        except ModuleNotFoundError:
            from models.gate import VisionGate, VisionGateConfig

        gate = VisionGate(
            VisionGateConfig(
                input_size=hidden_size,
                hidden_size=args.gate_hidden,
                dropout=0.0,
                min_scale=0.0,
                max_scale=1.0,
            )
        ).to(device=device, dtype=dtype)

    # InputBuilders
    builder = InputBuilder(model, plugin_len=args.n_plugin, use_position_ids=True)
    base_builder = InputBuilder(model, plugin_len=0, use_position_ids=True)

    # Dataset / Loader（目前用 toy， 后面接真实 jailbreak/VQA 只要替换 build_dataset）
    try:
        from src.data.datasets import build_toy_dataset
        from src.data.collate import collate_prompt_target_batch
    except ModuleNotFoundError:
        from data.datasets import build_toy_dataset
        from data.collate import collate_prompt_target_batch

    ds = build_toy_dataset()
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_prompt_target_batch(b, tok, device=device),
        drop_last=False,
    )

    # Optimizer：只更新插件（+ gate）
    params = [lt.emb]
    if gate is not None:
        params += list(gate.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr)

    model.train(False)
    lt.train(True)
    if gate is not None:
        gate.train(True)

    # 训练循环
    it = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        input_ids = batch["input_ids"]          # [B, T]
        text_attn = batch["attention_mask"]     # [B, T]
        labels_text = batch["labels"]           # [B, T] prompt=-100, target=token_id
        is_harmful = batch["is_harmful"]        # [B]
        B, T = input_ids.shape

        # ----- 构造 vision_embeds（占位：接真 VLM 后用真实视觉特征替换） -----
        V = int(args.vision_tokens)
        if V > 0:
            vision_embeds = torch.zeros((B, V, hidden_size), device=device, dtype=dtype)
            if args.toy_vision_signal:
                # 让 harmful 样本“看起来更敏感”，使 gate 在 toy 数据上有可学信号
                strength = (0.2 + 0.8 * is_harmful.float()).view(B, 1, 1).to(device=device,dtype=dtype)  # benign=0.2, harmful=1.0
                vision_embeds = vision_embeds + strength * torch.randn_like(vision_embeds) * 0.1
        else:
            vision_embeds = torch.zeros((B, 0, hidden_size), device=device, dtype=dtype)

        # ----- Learnable embeds（可选 gate 缩放） -----
        learnable_embeds = lt(batch_size=B, device=device, dtype=dtype)    # [B, N, H]
        gate_value = None
        if gate is not None:
            learnable_embeds, gate_value = gate.apply_to_embeddings(learnable_embeds, vision_embeds)  # gate_value: [B]

        # ----- build plugin inputs -----
        built = builder.build(
            vision_embeds=vision_embeds,
            learnable_embeds=learnable_embeds,
            text_input_ids=input_ids,
            text_attention_mask=text_attn,
        )

        N = args.n_plugin
        L = V + N + T

        # full labels: [B, V+N+T]
        full_labels = torch.full((B, L), -100, device=device, dtype=torch.long)
        full_labels[:, V + N : V + N + T] = labels_text

        # forward plugin（拿 logits 手算每类 loss，便于 harmful/benign 分开）
        out_plug = model(
            inputs_embeds=built.inputs_embeds,
            attention_mask=built.attention_mask,
            position_ids=built.position_ids,
        )
        logits_plug = out_plug.logits  # [B, L, vocab]

        # ----- Safety Loss：只在 harmful 样本上算 CE -----
        # token-level CE（reduction='none'）→ per-sample mean
        ce_tok = F.cross_entropy(
            logits_plug.transpose(1, 2),
            full_labels,
            ignore_index=-100,
            reduction="none",
        )  # [B, L]

        mask_tok = (full_labels != -100).float()
        denom = mask_tok.sum(dim=1).clamp(min=1.0)
        ce_per_sample = (ce_tok * mask_tok).sum(dim=1) / denom

        if is_harmful.any():
            loss_safe = ce_per_sample[is_harmful].mean()
        else:
            loss_safe = torch.zeros((), device=device,dtype=dtype)

        # ----- Utility Loss：只在 benign 样本上做 KL(p_base || p_plugin) -----
        # baseline forward（N=0），注意对齐到 text 区间
        base = base_builder.build(
            vision_embeds=vision_embeds,
            learnable_embeds=torch.zeros((B, 0, hidden_size), device=device, dtype=dtype),
            text_input_ids=input_ids,
            text_attention_mask=text_attn,
        )
        out_base = model(
            inputs_embeds=base.inputs_embeds,
            attention_mask=base.attention_mask,
            position_ids=base.position_ids,
        )
        logits_base = out_base.logits  # [B, V+T, vocab]

        # 取 text 区间 logits 对齐（不包含 vision/plugin）
        plug_text_logits = logits_plug[:, V + N : V + N + T, :]
        base_text_logits = logits_base[:, V : V + T, :].detach()

        # 只在 target token（labels!=-100）上做 KL
        text_mask = (labels_text != -100).float()  # [B, T]
        # KL( base || plug )：把 base 当 teacher（teacher，教师模型），plug 当 student（student，学生模型）
        logp = F.log_softmax(plug_text_logits, dim=-1)
        q = F.softmax(base_text_logits, dim=-1)
        kl_tok = F.kl_div(logp, q, reduction="none").sum(dim=-1)  # [B, T]
        kl_per_sample = (kl_tok * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1.0)

        benign = ~is_harmful
        if benign.any():
            loss_util = kl_per_sample[benign].mean()
        else:
            loss_util = torch.zeros((), device=device)

        # ----- Gate Loss：benign->0, harmful->1 -----
        loss_gate = torch.zeros((), device=device, dtype=dtype)
        if gate_value is not None:
            y = is_harmful.to(device=device, dtype=dtype)
            loss_gate = F.binary_cross_entropy(gate_value.clamp(1e-6, 1 - 1e-6), y)

        # total
        loss = (
            args.lambda_safe * loss_safe
            + args.lambda_util * loss_util
            + args.lambda_gate * loss_gate
        )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optim.step()

        if step % 20 == 0:
            msg = (
                f"step={step} loss={loss.item():.4f} "
                f"safe={loss_safe.item():.4f} util={loss_util.item():.4f} gate={loss_gate.item():.4f} "
                f"V={V} N={N} T={T}"
            )
            if gate_value is not None:
                msg += f" gate_mean={gate_value.mean().item():.3f}"
            print(msg)

    # save
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    lt.save(args.save_path)
    print(f"saved learnable tokens -> {args.save_path}")

    if gate is not None:
        torch.save({"state_dict": gate.state_dict()}, args.save_gate_path)
        print(f"saved gate -> {args.save_gate_path}")


if __name__ == "__main__":
    main()