# LLaMA-Factory 本地修改记录

基于上游 `hiyouga/LLaMA-Factory` (commit `e67ab9e2`) 的修改。

## 1. `src/llamafactory/data/template.py` (L2061)
- **修改**: 注释掉了 Qwen3.5 VL 的 `mm_plugin` 注册
- **原因**: 纯文本训练不需要多模态插件，避免不必要的依赖
- **状态**: 保留注释

## 2. `src/llamafactory/train/{sft,pt,dpo,kto,rm}/trainer.py`
- **修改**: `create_optimizer(self)` → `create_optimizer(self, model=None)`，并传递 `model` 给 `super()`
- **涉及文件**: sft/trainer.py, pt/trainer.py, dpo/trainer.py, kto/trainer.py, rm/trainer.py
- **原因**: 适配当前 transformers/trl 版本的 API 签名变化
- **状态**: 保留修改

## 3. `pyproject.toml`
- **修改**: 移除了多个依赖的版本上限（transformers, datasets, accelerate, peft, trl, torchdata, gradio, av）
- **原因**: 避免 ROCm 环境下的依赖版本冲突
- **状态**: 保留修改

## 4. `src/llamafactory/model/loader.py` (L206-213) — 已恢复上游
- **曾经修改**: 注释掉了 torch 2.9.x + Conv3D 的检查
- **恢复原因**: 纯文本 LLM 不含 Conv3D 层，检查不会触发；且该 bug 仅影响 CUDA 后端，ROCm 不受影响
- **状态**: ✅ 已恢复为上游版本

## 环境信息
- torch: 2.9.1+git8907517 (ROCm 7.0, HIP 7.0.51831)
- 上游 remote: https://github.com/hiyouga/LLaMA-Factory.git
