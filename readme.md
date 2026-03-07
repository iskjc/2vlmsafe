训练真实视觉分支：用带 image 字段的 jsonl（比如 VLGuard），并用 VL 模型（如Qwen/Qwen2。5-VL-7B-Instruct）。


推理真实视觉分支：加 --image_path <图片路径>。
如果是纯文本模型/无图数据，会自动走 fallback 占位，不会崩。