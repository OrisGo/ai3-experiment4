# 实验四：GSM8K 指令微调

使用 Qwen/Qwen2.5-0.5B 基座模型，对比 LoRA r=4/8/16、QLoRA r=8 和全参微调。
正式训练每组 1 epoch，统一评测官方 test 集前 128 题，并与未微调基座比较。

本项目可以直接在远程 Jupyter 的 Python Notebook 中运行 `.py` 文件，无需转换训练代码。
需要容器实际分配 NVIDIA GPU，且当前 Python 环境中的 PyTorch 能访问 CUDA。
本地 CPU 可以做数据准备和代码检查，训练与评测脚本会要求 CUDA。

## 1. 上传项目与模型

把项目放在平台的**持久化存储目录**；该目录的实际位置以平台说明为准。
上传代码、依赖文件和 `data/gsm8k/`，不要上传本地 `.venv/` 和缓存。
如果通过 Git 同步，先提交代码与原始数据；只有本地文件而未提交时，远程 clone 不会得到它们。

建议目录结构如下（`work/` 只是示例名称）：

```text
work/
├── Qwen2.5-0.5B/                 # 完整基座模型与分词器
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── experiment4/
    ├── README.md
    ├── requirements-local-uv.txt
    ├── config_alignment/
    │   ├── local_sft.py
    │   ├── runtime.py
    │   └── prompts/r1_zero.prompt
    ├── scripts/
    │   ├── prepare_gsm8k_sft.py
    │   ├── train_gsm8k_sft.py
    │   └── evaluate_gsm8k_sft.py
    └── data/gsm8k/
        ├── train.jsonl
        └── test.jsonl
```

脚本使用 `local_files_only=True`，运行时不会自动下载缺失的模型。
可以上传本地完整模型目录；远程有网络时，也可在安装依赖后按第 4 节下载。

## 2. 配置 Notebook 环境

选择 Python 3.11 内核。先查看当前解释器及目录：

```python
import sys
from pathlib import Path

print(sys.version)
print(sys.executable)
print(Path.cwd())
```

进入上传的项目根目录。把下面的路径替换为远程真实路径：

```python
%cd /你的持久化目录/experiment4
```

如果平台已提供可用的 CUDA PyTorch，先保留它。其他依赖按项目版本安装：

```python
%pip install -r requirements-local-uv.txt
```

安装完成后**重启 Notebook 内核**，再从第 3 节继续。
如果没有 PyTorch，或安装的是 CPU 版，使用平台推荐的 GPU 镜像或安装与宿主机 NVIDIA 驱动兼容的 CUDA PyTorch。
不要照抄课件中的 Windows 路径 `.venv/Scripts/python.exe`；Linux 虚拟环境对应 `.venv/bin/python`。
本地检查环境使用过 `torch==2.10.0+cu126`，这不代表所有远程驱动都适配这一版本，也不代表已经通过 GPU 训练验证。

## 3. 检查 GPU，并定义运行入口

重启内核后运行以下单元格，修改 `PROJECT` 一处路径即可。
后面的示例都依赖本节定义的 `run`、`PROJECT`、`MODEL` 和 `RUNS`。

```python
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import torch

PROJECT = Path('/你的持久化目录/experiment4').resolve()
assert (PROJECT / 'scripts/train_gsm8k_sft.py').is_file(), '请检查 PROJECT 路径'
os.chdir(PROJECT)
MODEL = PROJECT.parent / 'Qwen2.5-0.5B'
RUNS = PROJECT / 'runs'
RUNS.mkdir(exist_ok=True)

assert torch.cuda.is_available(), '当前内核无法使用 CUDA，请检查 GPU 分配和 PyTorch'
gpu = torch.cuda.get_device_properties(0)
native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
environment = {
    'python': sys.version,
    'executable': sys.executable,
    'gpu': gpu.name,
    'gpu_total_gib': gpu.total_memory / 1024**3,
    'torch_cuda': torch.version.cuda,
    'native_bf16': native_bf16,
    'packages': {
        name: importlib.metadata.version(name)
        for name in ['torch', 'transformers', 'trl', 'peft',
                     'datasets', 'accelerate', 'bitsandbytes']
    },
}
print(json.dumps(environment, ensure_ascii=False, indent=2))
(RUNS / 'environment.json').write_text(
    json.dumps(environment, ensure_ascii=False, indent=2), encoding='utf-8'
)

def run(script, *args):
    command = [sys.executable, '-u', str(PROJECT / script), *map(str, args)]
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)

def fresh_output(name):
    path = RUNS / name
    if path.exists():
        raise FileExistsError(f'{path} 已存在，请使用新名称或先备份旧结果')
    return path
```

`run()` 使用 Notebook 当前解释器，实时输出进度；子进程失败时会报错并停止当前单元格。
每组训练在独立进程中执行，进程结束后释放其 GPU 资源。一次只运行一组，不要并行启动不同训练单元格。

训练和评测统一使用原生 BF16（支持时），否则回退到 FP32；QLoRA 的计算精度也一致。
FP32 回退不改变 QLoRA 的 4-bit 基座存储，但会增加部分计算和内存开销。
若显存不足，改用资源足够的 GPU，并保持各正式实验使用同一设备和设置。
结果中的 `compute_dtype` 记录计算精度，不代表所有参数或优化器状态均使用该精度。

## 4. 准备模型与数据

已经上传完整模型目录时，跳过下载。需要在线下载时运行：

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id='Qwen/Qwen2.5-0.5B', local_dir=str(MODEL))
```

模型在其他位置时，修改第 3 节中的 `MODEL`；不要改为 Instruct 版本，否则不再是课件要求的同一基座。
检查目录并准备数据：

```python
assert (MODEL / 'config.json').is_file(), '模型目录不存在或不完整'
assert list(MODEL.glob('*.safetensors')), '缺少模型权重'
run('scripts/prepare_gsm8k_sft.py', '--seed', 42, '--validation-size', 128)
```

原始 train 共 7473 条，固定留出 128 条验证，剩余 7345 条为训练池；test 共 1319 条。
训练会过滤超过 `--max-length` 的完整样本，避免截断末尾答案。
本地用当前模型分词器、seed=42、长度384检查时，保留训练7172条、验证127条；以实际运行的 `summary.json` 为准。
过滤不会改变独立的官方 test 集。SFTTrainer 为 completion 补充 EOS，且只对 completion 计算损失。

## 5. 用 128 条训练数据试跑

先检查 LoRA 全流程，输出单独放在 `smoke_lora_r8`，不作为正式结果：

```python
run('scripts/train_gsm8k_sft.py',
    '--model-path', MODEL, '--method', 'lora', '--rank', 8,
    '--train-size', 128, '--epochs', 1, '--max-length', 384,
    '--output-dir', fresh_output('smoke_lora_r8'))

run('scripts/evaluate_gsm8k_sft.py',
    '--model-path', MODEL, '--adapter-path', RUNS / 'smoke_lora_r8/model',
    '--split', 'test', '--limit', 8, '--max-new-tokens', 160,
    '--output-dir', fresh_output('smoke_lora_r8_test8'))
```

也建议独立检查 QLoRA 的 CUDA/bitsandbytes 兼容性：

```python
run('scripts/train_gsm8k_sft.py',
    '--model-path', MODEL, '--method', 'qlora', '--rank', 8,
    '--train-size', 128, '--epochs', 1, '--max-length', 384,
    '--output-dir', fresh_output('smoke_qlora_r8'))
```

## 6. 正式实验：五组训练、六组评测

按顺序执行以下单元格。所有正式评测都采用相同 test 前128题、贪心解码、最多160个新 token。
若改成全部1319题，将所有评测的 `--limit` 统一改为0，不能混用不同题数比较。

### 6.1 基座评测

```python
run('scripts/evaluate_gsm8k_sft.py',
    '--model-path', MODEL, '--split', 'test', '--limit', 128,
    '--max-input-tokens', 384, '--max-new-tokens', 160,
    '--output-dir', fresh_output('base_test128'))
```

### 6.2 LoRA r=4、8、16：依次训练并评测

```python
for rank in (4, 8, 16):
    name = f'lora_r{rank}'
    run('scripts/train_gsm8k_sft.py',
        '--model-path', MODEL, '--method', 'lora', '--rank', rank,
        '--train-size', 0, '--epochs', 1, '--max-length', 384,
        '--batch-size', 1, '--gradient-accumulation', 8,
        '--output-dir', fresh_output(name))
    run('scripts/evaluate_gsm8k_sft.py',
        '--model-path', MODEL, '--adapter-path', RUNS / name / 'model',
        '--split', 'test', '--limit', 128,
        '--max-input-tokens', 384, '--max-new-tokens', 160,
        '--output-dir', fresh_output(f'{name}_test128'))
```

### 6.3 QLoRA r=8

```python
run('scripts/train_gsm8k_sft.py',
    '--model-path', MODEL, '--method', 'qlora', '--rank', 8,
    '--train-size', 0, '--epochs', 1, '--max-length', 384,
    '--batch-size', 1, '--gradient-accumulation', 8,
    '--output-dir', fresh_output('qlora_r8'))

run('scripts/evaluate_gsm8k_sft.py',
    '--model-path', MODEL, '--quantized',
    '--adapter-path', RUNS / 'qlora_r8/model',
    '--split', 'test', '--limit', 128,
    '--max-input-tokens', 384, '--max-new-tokens', 160,
    '--output-dir', fresh_output('qlora_r8_test128'))
```

QLoRA 评测必须保留 `--quantized`，与训练的 NF4、双重量化设置一致。

### 6.4 全参微调

```python
run('scripts/train_gsm8k_sft.py',
    '--model-path', MODEL, '--method', 'full', '--optimizer', 'adamw_bnb_8bit',
    '--train-size', 0, '--epochs', 1, '--max-length', 384,
    '--batch-size', 1, '--gradient-accumulation', 8,
    '--output-dir', fresh_output('full'))

run('scripts/evaluate_gsm8k_sft.py',
    '--model-path', RUNS / 'full/model',
    '--split', 'test', '--limit', 128,
    '--max-input-tokens', 384, '--max-new-tokens', 160,
    '--output-dir', fresh_output('full_test128'))
```

全参评测直接加载保存后的完整模型，不传 `--adapter-path`。
LoRA/QLoRA 默认学习率为2e-4，全参为2e-5；全参命令还使用8-bit AdamW，报告需要说明这些差异。

## 7. 结果保存与后续提交

五组训练目录：`lora_r4`、`lora_r8`、`lora_r16`、`qlora_r8`、`full`。
每组包含 `summary.json`、`training_log.json` 和 `model/`。
六组正式评测目录为 `base_test128` 以及五组对应的 `*_test128`，各含 `summary.json` 和 `predictions.jsonl`。

- `answer_accuracy`、`strict_accuracy`、`format_rate` 是0到1的比例，绘制百分比时乘100。
- 训练主表显存使用 `peak_cuda_allocated_gib`，同时保留 reserved 数据。
- `elapsed_seconds` 除以60得到分钟；训练计时包含 epoch 验证，不包含加载和最终保存。
- `training_log.json` 用于 loss 曲线；`predictions.jsonl` 用于前后变化的逐题分析。
- 记录 `runs/environment.json`，并补充保存模型目录的磁盘大小。
- 后续还需编写绘图脚本、汇总 CSV、完成报告；本 README 不代表这些提交项已经完成。

`.gitignore` 忽略虚拟环境、缓存、生成的数据、`runs/` 下的模型和检查点，保留原始数据、指标、预测及训练日志。
**忽略只影响 Git，不会删除本地文件，也不会自动备份远程文件。**
停止或释放容器前，下载需要的实验结果；模型权重需要单独备份，因为不会随 Git 同步。
报告及结果按课件要求打包为“学号_姓名_实验四.zip”。

## 8. 常见问题

- **没有 Terminal：**上述代码都在 Python Notebook 单元格执行即可。
- **找不到模块：**确认 `%pip` 安装成功并重启内核，检查第3节打印的解释器和包版本。
- **找不到数据：**先执行准备数据脚本；`data/gsm8k_sft/` 不会通过 Git 自动带过来。
- **找不到模型：**确认 `MODEL` 指向实际文件目录；上传时若用 Git LFS，需要完整权重而非指针文件。
- **显存不足：**检查是否同时运行其他训练，确认精度及设备；保持正式比较条件一致，不要只给某一组缩短序列或减少数据。
- **运行要求登录 W&B：**项目已关闭自动实验平台上报，日志仍写入本地 JSON 文件。
- **试跑成功后怎么正式跑：**使用第6节新目录，`--train-size 0` 表示使用全部通过长度过滤的训练数据。
- **中途断开后怎么恢复：**不要直接从头重跑整个循环；先查看已经完成的目录，单独运行剩余步骤。脚本目前没有提供断点续训参数，失败训练可备份原目录后用新目录重新开始。
- **关闭网页后是否继续：**取决于平台的内核与容器保活策略。此运行方式不保证断线续跑；重启内核或停止容器也不应视为正常后台运行。
- **`WindowsSFTTrainer` 能在 Linux 用吗：**可以；它仅覆盖损失计算方法，名称沿用原始项目，不调用 Windows 专属接口。

参考：[IPython 的 `%pip` / `%run` 文档](https://ipython.readthedocs.io/en/stable/interactive/magics.html)、[Hugging Face 模型下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)。
