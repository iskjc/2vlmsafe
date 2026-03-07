from __future__ import annotations
import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoProcessor
try:
    from src.models.learnable_tokens import LearnableTokens, LearnableTokensConfig
    from src.models.input_builder import InputBuilder
except ModuleNotFoundError:
    # Support direct execution: `python src/train.py`
    from models.learnable_tokens import LearnableTokens, LearnableTokensConfig
    from models.input_builder import InputBuilder
try:
    from src.data.datasets import build_jsonl_dataset
except ModuleNotFoundError:
    from data.datasets import build_jsonl_dataset

import random
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler[list[int]]):
    """
    每个 batch 强制包含 roughly 50/50 的 harmful/benign。
    drop_last=True 时 batch 大小恒定
    会自动过采样少数类
    """
    def __init__(self, labels, batch_size: int, seed: int = 0, drop_last: bool = True):
        assert batch_size >= 2, "batch_size must be >= 2 for balanced sampling"
        self.labels = [bool(x) for x in labels]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        self.h_idx = [i for i, y in enumerate(self.labels) if y]
        self.b_idx = [i for i, y in enumerate(self.labels) if not y]
        if len(self.h_idx) == 0 or len(self.b_idx) == 0:
            raise ValueError(f"Need both harmful and benign samples. got harmful={len(self.h_idx)}, benign={len(self.b_idx)}")

        self.n_h = batch_size // 2
        self.n_b = batch_size - self.n_h  # 处理奇数 batch_size

        # 一个 epoch 产出多少 batch：按多数类估算
        self.num_batches = (max(len(self.h_idx), len(self.b_idx)) // max(1, min(self.n_h, self.n_b)))
        if self.num_batches <= 0:
            self.num_batches = 1

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        h = self.h_idx[:]
        b = self.b_idx[:]
        self.rng.shuffle(h)
        self.rng.shuffle(b)

        # 指针 + 过采样
        hp = 0
        bp = 0
        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.n_h):
                if hp >= len(h):
                    self.rng.shuffle(h)
                    hp = 0
                batch.append(h[hp]); hp += 1
            for _ in range(self.n_b):
                if bp >= len(b):
                    self.rng.shuffle(b)
                    bp = 0
                batch.append(b[bp]); bp += 1

            self.rng.shuffle(batch)
            yield batch


def freeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

def import_transformers():
    try:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
        )
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
    return AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, AutoConfig


def select_model_class(model_name: str, auto_config_cls, causal_cls, vl_cls):
    # FIX(local): use VL model class for Qwen2.5-VL configs.
    cfg = auto_config_cls.from_pretrained(model_name, trust_remote_code=True)
    cfg_name = type(cfg).__name__.lower()
    if "vl" in cfg_name:
        return vl_cls
    return causal_cls


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
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])

    # plug与训练超参
    ap.add_argument("--n_plugin", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # Loss
    ap.add_argument("--lambda_safe", type=float, default=0.02)
    ap.add_argument("--lambda_util", type=float, default=0.2)
    ap.add_argument("--lambda_gate", type=float, default=0.1)
    # Gate
    ap.add_argument("--use_gate", action="store_true")
    ap.add_argument("--gate_hidden", type=int, default=256)
    # 视觉 token（暂时用占位；接真 VLM 时替换 vision_embeds 构造）
    ap.add_argument("--vision_tokens", type=int, default=8)
    ap.add_argument("--save_path", type=str, default="outputs/learnable_tokens.pt")
    ap.add_argument("--save_gate_path", type=str, default="outputs/gate.pt")
    ap.add_argument("--data_path", type=str, default="", help="Path to training data jsonl")
    ap.add_argument("--kl_cap", type=float, default=10.0, help="Cap per-token KL to stabilize training")
    ap.add_argument(
        "--balanced_sampling",
        action="store_true",
        default=True,
        help="Use class-balanced batch sampling (requires --batch_size >= 2).",
    )
    ap.add_argument(
        "--no_balanced_sampling",
        action="store_false",
        dest="balanced_sampling",
        help="Disable class-balanced batch sampling.",
    )
    args = ap.parse_args()


    min_pixels = 128 * 28 * 28
    max_pixels = 256 * 28 * 28
    # FIX(local): ensure processor works with remote-code VLM checkpoints.
    processor = AutoProcessor.from_pretrained(args.model_name, min_pixels = min_pixels,max_pixels = max_pixels,trust_remote_code=True)

    if args.n_plugin <= 0:
        raise SystemExit("--n_plugin must be > 0")
    if args.steps <= 0:
        raise SystemExit("--steps must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be > 0")
    if args.vision_tokens < 0:
        raise SystemExit("--vision_tokens must be >= 0")

    AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, AutoConfig = import_transformers()
    device, dtype = resolve_device_and_dtype(args.device, args.dtype)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    model_cls = select_model_class(
        args.model_name,
        auto_config_cls=AutoConfig,
        causal_cls=AutoModelForCausalLM,
        vl_cls=AutoModelForImageTextToText,
    )
    model = load_model(model_cls, args.model_name, dtype=dtype, device=device)
    model.gradient_checkpointing_enable()

    print("config type:", type(model.config))
    print("config has hidden_size:", hasattr(model.config, "hidden_size"))
    for k in ["text_config", "language_config", "llm_config"]:
        sub = getattr(model.config, k, None)
        print(k, "exists:", sub is not None, "hidden_size:", getattr(sub, "hidden_size", None))

    freeze_model(model)

    def get_hidden_size(cfg) -> int:
        if hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)

        for sub_name in ["text_config", "language_config", "llm_config"]:
            sub = getattr(cfg, sub_name, None)
            if sub is not None and hasattr(sub, "hidden_size"):
                return int(sub.hidden_size)

        for sub_name in ["text_config", "language_config", "llm_config"]:
            sub = getattr(cfg, sub_name, None)
            if sub is not None and hasattr(sub, "d_model"):
                return int(sub.d_model)
        raise AttributeError(
            f"Cannot find hidden size in config. Top-level keys: {list(getattr(cfg, 'to_dict', lambda: {})().keys())}"
        )
    hidden_size = get_hidden_size(model.config)

    # Learnable Tokens
    cfg = LearnableTokensConfig(
        n_tokens=args.n_plugin,
        hidden_size=hidden_size,
        init_from_token_id=tok.pad_token_id if tok.pad_token_id is not None else None,
    )
    lt = LearnableTokens(cfg).to(device=device,dtype=torch.float32)
    lt.initialize(model.get_input_embeddings())


    def chk(name, x):
        if x is None: 
            return
        if not torch.isfinite(x).all():
            bad = x[~torch.isfinite(x)]
            raise RuntimeError(f"{name} has non-finite: {bad[:5]}")


    # Gate
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
        ).to(device=device, dtype=torch.float32)



    # InputBuilders
    builder = InputBuilder(model, plugin_len=args.n_plugin, use_position_ids=True)


    base_builder = InputBuilder(model, plugin_len=0, use_position_ids=True)



    # Dataset / Loader
    try:
        from src.data.collate import collate_prompt_target_batch
    except ModuleNotFoundError:
        from data.collate import collate_prompt_target_batch

    if not args.data_path:
        raise SystemExit("--data_path is required.")
    ds = build_jsonl_dataset(args.data_path)

    def get_is_harmful(sample):
    # dict
        if isinstance(sample, dict):
            return bool(sample.get("is_harmful", False))
        # object / dataclass
        if hasattr(sample, "is_harmful"):
            return bool(getattr(sample, "is_harmful"))
        raise TypeError(f"Sample has no is_harmful field: type={type(sample)}")
    
    print(f'ds:{len(ds)}')
    
    labels = [get_is_harmful(ds[i]) for i in range(len(ds))]
    print("label stats:", sum(labels), "/", len(labels), "harmful")

    if args.balanced_sampling and args.batch_size < 2:
        print("Warning: --balanced_sampling requires --batch_size >= 2. Falling back to shuffle sampling.")
        args.balanced_sampling = False

    loader_kwargs = dict(
        collate_fn=lambda b: collate_prompt_target_batch(
            b, processor=processor, model_name=args.model_name, device=device, tokenizer=tok
        ),
        num_workers=0,  # 先保持0，后续再调
        pin_memory=False,
    )

    if args.balanced_sampling:
        batch_sampler = BalancedBatchSampler(
            labels=labels,
            batch_size=args.batch_size,
            seed=42,
            drop_last=True,  # True保证每步都有同样数量 harmful/benign
        )
        loader = DataLoader(ds, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)

    # Optimizer：只更新插件（+ gate）
    params = list(lt.parameters())
    if gate is not None:
        params += list(gate.parameters())

    lt_params = list(lt.parameters())
    gate_params = list(gate.parameters()) if gate is not None else []

    optimizer = torch.optim.AdamW(
        [
            {"params": lt_params, "lr": args.lr, "weight_decay": 0.0},          # LT 不建议 weight_decay
            {"params": gate_params, "lr": args.lr * 0.1, "weight_decay": 0.01}, # Gate 更小 lr
        ]
    )

    def count_trainable(m):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    # 解冻 plugin / gate
    for p in lt.parameters():   # LearnableTokens（可学习 token）
        p.requires_grad = True
    if gate is not None:
        for p in gate.parameters():
            p.requires_grad = True

    # 打印确认
    print("Backbone trainable:", sum(p.requires_grad for p in model.parameters()))
    print("LT total/trainable:", count_trainable(lt))

    if gate is not None:
        print("Gate total/trainable:", count_trainable(gate))


    #设置训练状态
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

        # ----- 构造 vision_embeds -----
        has_vision_inputs = ("pixel_values" in batch and "image_grid_thw" in batch)
        if has_vision_inputs and args.vision_tokens <= 0:
            raise RuntimeError("Batch contains vision inputs, but --vision_tokens is <= 0.")
        if has_vision_inputs and not hasattr(model, "get_image_features"):
            raise RuntimeError("Batch contains vision inputs, but model has no get_image_features method.")

        if args.vision_tokens > 0 and has_vision_inputs and hasattr(model, "get_image_features"):
            with torch.no_grad():
                raw_feats = model.get_image_features(
                    batch["pixel_values"].to(device=device),
                    batch["image_grid_thw"].to(device=device),
                )
            # Qwen2.5-VL get_image_features returns BaseModelOutputWithPooling:
            # - last_hidden_state: vision hidden dim (e.g. 1280)
            # - pooler_output: merged features aligned to LLM hidden dim (e.g. 3584)
            if hasattr(raw_feats, "pooler_output") and raw_feats.pooler_output is not None:
                raw_feats = raw_feats.pooler_output
            elif hasattr(raw_feats, "last_hidden_state"):
                raw_feats = raw_feats.last_hidden_state

                if isinstance(raw_feats, torch.Tensor):
                    chk("raw_feats", raw_feats)
                elif hasattr(raw_feats, "pooler_output") and raw_feats.pooler_output is not None:
                    chk("raw_feats.pooler_output", raw_feats.pooler_output)
                elif hasattr(raw_feats, "last_hidden_state"):
                    chk("raw_feats.last_hidden_state", raw_feats.last_hidden_state)

            if isinstance(raw_feats, torch.Tensor):
                if raw_feats.ndim == 2:
                    raw_feats = raw_feats.unsqueeze(0)
                if raw_feats.ndim != 3:
                    raise RuntimeError(f"Unsupported tensor vision feature shape: {tuple(raw_feats.shape)}")
                if raw_feats.shape[0] != B:
                    raise RuntimeError(f"Vision batch mismatch: expected {B}, got {raw_feats.shape[0]}")
                if raw_feats.shape[-1] != hidden_size:
                    raise RuntimeError(
                        f"Vision hidden mismatch: {raw_feats.shape[-1]} vs model hidden={hidden_size}"
                    )
                vision_embeds = raw_feats.to(device=device, dtype=dtype)
                Vmax = int(vision_embeds.shape[1])
                vision_mask = torch.ones((B, Vmax), device=device, dtype=torch.long)
            elif isinstance(raw_feats, (tuple, list)):
                if len(raw_feats) != B:
                    raise RuntimeError(f"Vision feature count mismatch: expected {B}, got {len(raw_feats)}")
                Vmax = max((int(x.shape[0]) for x in raw_feats), default=0)
                vision_embeds = torch.zeros((B, Vmax, hidden_size), device=device, dtype=dtype)
                vision_mask = torch.zeros((B, Vmax), device=device, dtype=torch.long)
                for i, x in enumerate(raw_feats):
                    if x.ndim != 2:
                        raise RuntimeError(f"Each vision feature must be [V,H], got {tuple(x.shape)}")
                    v_i, h_i = x.shape
                    if h_i != hidden_size:
                        raise RuntimeError(
                            f"Vision hidden mismatch at sample {i}: {h_i} vs model hidden={hidden_size}"
                        )
                    if v_i > 0:
                        vision_embeds[i, :v_i] = x.to(device=device, dtype=dtype)
                        vision_mask[i, :v_i] = 1
            else:
                raise RuntimeError(f"Unsupported vision feature type: {type(raw_feats)}")
            chk("vision_embeds", vision_embeds)
        else:
            vision_embeds = torch.zeros((B, 0, hidden_size), device=device, dtype=dtype)
            vision_mask = torch.zeros((B, 0), device=device, dtype=torch.long)

        # ----- Learnable embeds（可选 gate 缩放）-----
        learnable_fp32 = lt(batch_size=B, device=device, dtype=torch.float32)
        chk("learnable_embeds_pre_gate_fp32", learnable_fp32)

        # 先清洗 LT 输出
        learnable_fp32 = torch.nan_to_num(
            learnable_fp32, nan=0.0, posinf=1e4, neginf=-1e4
        ).clamp(-5.0, 5.0)

        gate_value = None

        if gate is not None:
            # gate 前先确保输入干净
            vision_fp32 = torch.nan_to_num(
                vision_embeds.float(), nan=0.0, posinf=1e4, neginf=-1e4
            ).clamp(-5.0, 5.0)

            learnable_fp32 = torch.nan_to_num(
                learnable_fp32, nan=0.0, posinf=1e4, neginf=-1e4
            ).clamp(-5.0, 5.0)

            # 只调用一次
            learnable_fp32, gate_value = gate.apply_to_embeddings(learnable_fp32, vision_fp32)
            gate_value = gate_value.float()

            # 检查 gate 输出
            if torch.isnan(learnable_fp32).any():
                print(f"Step {step}: NaN in learnable_fp32 after gate!")
                learnable_fp32 = torch.nan_to_num(learnable_fp32, nan=0.0, posinf=1e4, neginf=-1e4)

            if torch.isnan(gate_value).any():
                print(f"Step {step}: NaN in gate_value!")
                gate_value = torch.nan_to_num(gate_value, nan=0.5)

            gate_value = torch.clamp(gate_value, 0.0, 1.0)
            chk("gate_value", gate_value)

            # 最后再转给 backbone
            learnable_embeds = learnable_fp32.to(dtype=dtype)


        # ----- build plugin inputs -----
        built = builder.build(
            vision_embeds=vision_embeds,
            learnable_embeds=learnable_embeds,
            text_input_ids=input_ids,
            vision_attention_mask=vision_mask,
            text_attention_mask=text_attn,
        )

        V = int(vision_embeds.shape[1])
        N = args.n_plugin

        chk("built.inputs_embeds", built.inputs_embeds)
        chk("built.attention_mask", built.attention_mask)
        chk("built.position_ids", built.position_ids)

        # forward plugin（拿 logits 手算每类 loss，便于 harmful/benign 分开）
        out_plug = model(
            inputs_embeds=built.inputs_embeds,
            attention_mask=built.attention_mask,
            position_ids=built.position_ids,
            use_cache=False,
        )
        logits_plug = out_plug.logits  # [B, V+N+T, vocab]
        plug_text_logits = logits_plug[:, V + N : V + N + T, :]

        chk("logits_plug", logits_plug)


        # ----- Safety Loss：只在 harmful 样本上算 CE -----
        # 仅在 text 区间计算 CE，避免对 vision/plugin 区间做无效大张量运算
        ce_tok = F.cross_entropy(
            plug_text_logits.transpose(1, 2),
            labels_text,
            ignore_index=-100,
            reduction="none",
        )  # [B, T]

        mask_tok = (labels_text != -100).float()
        denom = mask_tok.sum(dim=1).clamp(min=1.0)
        ce_per_sample = (ce_tok * mask_tok).sum(dim=1) / denom

        if is_harmful.any():
            loss_safe = ce_per_sample[is_harmful].mean()
        else:
            loss_safe = torch.zeros((), device=device,dtype=torch.float32)



        # Utility Loss：只在 benign 样本上做 KL(p_base || p_plugin)
        # baseline forward（N=0），注意对齐到 text 区间
        #base = base_builder.build(
        #    vision_embeds=vision_embeds,
        #    learnable_embeds=torch.zeros((B, 0, hidden_size), device=device, dtype=dtype),
        #    text_input_ids=input_ids,
        #    vision_attention_mask=vision_mask,
        #    text_attention_mask=text_attn,
        #)
        #with torch.inference_mode():
        #    out_base = model(
        #        inputs_embeds=base.inputs_embeds,
        #        attention_mask=base.attention_mask,
        #        position_ids=base.position_ids,
        #    )
        #    logits_base = out_base.logits  # [B, V+T, vocab]
#
        ## 取 text 区间 logits 对齐（不包含 vision/plugin）
        #base_text_logits = logits_base[:, V : V + T, :]
        #plug_text_logits = plug_text_logits                      # [B, T, vocab]
#
        ## 只在 target token（labels!=-100）上做 KL
        #mask = (labels_text != -100)                              # [B, T] bool
        #if mask.any():
        #    # 选出所有需要算 KL 的 token：形状变成 [N, vocab]
        #    base_sel = base_text_logits[mask]                     # [N, vocab]
        #    plug_sel = plug_text_logits[mask]                     # [N, vocab]
#
        #    # teacher: base 的 log-prob（不需要梯度）
        #    with torch.no_grad():
        #        lq = F.log_softmax(base_sel, dim=-1)              # [N, vocab]
#
        #    # student: plugin 的 log-prob（需要梯度）
        #    lp = F.log_softmax(plug_sel, dim=-1)                  # [N, vocab]
        #    # KL(q || p) = sum q * (log q - log p)
        #    # F.kl_div 的约定：input=log p, target=log q 且 log_target=True
        #    kl_tok = F.kl_div(lp, lq, log_target=True, reduction="none").sum(dim=-1)  # [N]
        #    kl_tok=kl_tok.clamp(max=args.kl_cap)
        #    # 把 token 归回每个 sample：每个 token 属于哪个 batch
        #    b_idx = mask.nonzero(as_tuple=False)[:, 0]            # [N]
        #    kl_sum = torch.zeros((B,), device=device, dtype=kl_tok.dtype)
        #    cnt = torch.zeros((B,), device=device, dtype=kl_tok.dtype)
#
        #    kl_sum.scatter_add_(0, b_idx, kl_tok)
        #    cnt.scatter_add_(0, b_idx, torch.ones_like(kl_tok))
#
        #    kl_per_sample = kl_sum / cnt.clamp(min=1.0)           # [B]
        #else:
        #    kl_per_sample = torch.zeros((B,), device=device, dtype=torch.float32)
#
#
        #benign = ~is_harmful
        #loss_util = kl_per_sample[benign].mean() if benign.any() else torch.zeros((), device=device, dtype=torch.float32)
        loss_util = torch.zeros((), device=device, dtype=torch.float32)
        # ----- Gate Loss：benign->0, harmful->1 -----
        loss_gate = torch.zeros((), device=device, dtype=torch.float32)
        if gate_value is not None:
            y = is_harmful.to(device=device, dtype=torch.float32).view(-1)
            if torch.isnan(gate_value).any():
                print(f"Step {step}: Skipping gate loss due to NaN")
                loss_gate = torch.zeros((), device=device, dtype=torch.float32)
            else:
                # 使用 label smoothing 避免过拟合
                y_smooth = y * 0.9 + 0.05  # label smoothing
                gate_clamped=torch.clamp(gate_value, min=1e-7, max=1-1e-7)
                loss_gate = F.binary_cross_entropy(gate_clamped, y_smooth)
                if torch.isnan(loss_gate):
                    print(f"Step {step}: NaN in gate loss!")
                    print(f"gate_value: {gate_value}")
                    print(f"y: {y}")
                    loss_gate = torch.zeros((), device=device, dtype=torch.float32)

        # total
        loss = (
            args.lambda_safe * loss_safe
            + args.lambda_util * loss_util
            + args.lambda_gate * loss_gate
        )
        if step % 100==0:
            print("components:", loss_safe.item(), loss_util.item(), loss_gate.item(), "total:", loss.item())

        optimizer.zero_grad(set_to_none=True)

        if step %100==0:
            print(
             "[dbg]",
             "harmful:", int(batch["is_harmful"].sum().item()),
             "ls:", float(loss_safe.detach().cpu()),
             "lu:", float(loss_util.detach().cpu()),
             "lg:", float(loss_gate.detach().cpu()),
             "lam:", args.lambda_safe, args.lambda_util, args.lambda_gate,
             "total:", float(loss.detach().cpu()),
             "check_total_minus:", float((loss - (args.lambda_safe*loss_safe + args.lambda_util*loss_util + args.lambda_gate*loss_gate)).detach().cpu())
            )


        loss.backward()

        # 检查 gate 梯度
        if gate is not None:
            for name, param in gate.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"Step {step}: NaN gradient in gate.{name}")
                        param.grad = torch.nan_to_num(param.grad, nan=0.0)
                    if torch.isinf(param.grad).any():
                        print(f"Step {step}: Inf gradient in gate.{name}")
                        param.grad = torch.nan_to_num(param.grad, posinf=1.0, neginf=-1.0)

        # 梯度裁剪
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)






        # 检查 backbone 永远没梯度
        backbone_has_grad = any(p.grad is not None for p in model.parameters())
        lt_has_grad = any(p.grad is not None for p in lt.parameters())
        gate_has_grad = (gate is not None) and any(p.grad is not None for p in gate.parameters())

        if step == 1:
            print("backbone_has_grad:", backbone_has_grad)  # 必须 False
            print("lt_has_grad:", lt_has_grad)              # 必须 True
            if gate is not None:
                print("gate_has_grad:", gate_has_grad)      # 必须 True

        # 梯度裁剪
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()

        if step % 100 == 0:
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
