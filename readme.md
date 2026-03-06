训练真实视觉分支：用带 image 字段的 jsonl（比如 VLGuard），并用 VL 模型（如Qwen/Qwen2。5-VL-7B-Instruct）。


推理真实视觉分支：加 --image_path <图片路径>。
如果是纯文本模型/无图数据，会自动走 fallback 占位，不会崩。

### mini run
```python
python src/train.py \
--model_name /s/models/Qwen2.5-VL-3B-Instruct \
--device cuda \
--dtype fp16 \
--n_plugin 16 \
--lr 5e-5 \
--steps 200 \
--batch_size 1 \
--use_gate \
--data_path data/vlguard/processed/vlguard_train.jsonl \
--save_path outputs/learnable_tokens.pt \
--save_gate_path outputs/gate.pt
--balanced_sampling

```