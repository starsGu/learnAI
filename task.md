# 第一个目标：Qwen3-0.6B 中文预训练

## 目标与边界

- [x] 使用 Qwen3-0.6B 架构，所有语言模型参数随机初始化。
- [x] 复用官方 Qwen3 分词器，但禁止下载或加载官方模型权重。
- [x] 实现中文上下文续写所需的数据、训练、恢复、生成、评测和导出管线。
- [ ] 完成 100M token 试跑。
- [ ] 完成 1B token 原型训练。
- [ ] 完成 10B token 正式里程碑。

本阶段不训练聊天对齐、指令遵循、生物专家知识或高考答题能力。

本机验证使用现有 Conda `cuda` 环境。运行前执行 `conda activate cuda`；不要在该环境中升级或重装 PyTorch、CUDA、cuDNN。

## 固定技术方案

- 模型：自行实现 Qwen3-0.6B Base，参数名兼容 Hugging Face `Qwen3ForCausalLM`。
- 分词器：`Qwen/Qwen3-0.6B-Base`，下载时只允许分词器和配置文件。
- 本机：RTX 5070 12GB，仅执行微型模型冒烟测试和完整模型单步显存探测。
- 正式训练：保留 `torchrun` DDP 多卡训练结构；当前 Linux 云服务器配置 `num_gpus=1`，使用单张 RTX 5090 32GB。默认单卡 micro batch 为 8，通过 32 次梯度累计保持每步 524,288 tokens；增加 GPU 时同步调整 `num_gpus` 和 `--nproc-per-node`。
- 正式上下文长度：2048；模型配置保留 32768 最大位置长度。
- 随机种子：42。

## 数据计划

正式训练按有效 token 数混合：

| 来源 | 比例 | 用途 |
|---|---:|---|
| Fineweb-Edu-Chinese-V2.1 高质量部分 | 55% | 通用中文与教育文本 |
| Chinese Cosmopedia | 25% | 连贯说明、教材和叙事文本 |
| CCI3-HQ | 20% | 通用中文互联网文本 |

处理规则：Unicode NFKC、空白规范化、中文比例与长度过滤、文档精确去重、段落精确去重、跨来源去重。使用文档哈希确定训练/验证/最终测试划分。输出 `uint32` token shards、SHA-256、来源统计、过滤统计和 manifest。

商业使用前必须重新检查各数据集许可证。本仓库只记录来源与许可证信息，不替用户作授权判断。

## 稳定命令

```powershell
python -m datasets.prepare --config configs/smoke/data.json --data data
python -m datasets.verify --manifest data/processed/smoke/manifest.json
python -m pretrain.train --config configs/smoke/train.json --data data
python -m pretrain.train --config configs/smoke/train_resume.json --data data
python -m pretrain.evaluate --checkpoint checkpoints/pretrain/qwen3-tiny/latest --config configs/smoke/eval.json
python -m pretrain.generate --checkpoint checkpoints/pretrain/qwen3-tiny/latest --prompt "春天到了"
python -m pretrain.export --checkpoint checkpoints/pretrain/qwen3-tiny/latest --output checkpoints/pretrain/qwen3-tiny-hf
python -m pretrain.probe --model-config configs/formal/model.json
python -m pretrain.score --dev-csv <dev-scores.csv> --final-csv <final-scores.csv>
```

正式训练：

python -c "
import json
s = json.load(open('/root/autodl-tmp/datasets/processed/chinese-10b/processing_state.json'))
st = s['stats']; d = s['split_tokens_on_disk']
print('已完成文件:', len(s['completed_files']), '个')
print('已处理文档:', st.get('documents_seen',0), '条')
print('已落盘 tokens: 共', sum(d.values()), '  (train', d.get('train',0), '/ val', d.get('validation',0), '/ test', d.get('test',0), ')')
print('训练进度:', round(d.get('train',0)/1e10*100,1), '% of 10B')
"

```bash
python -m datasets.prepare --config configs/formal/data.json --data /root/autodl-tmp/datasets --modelscope
# 100M 试跑（新 output_dir，从 step 0 开始）
torchrun --nproc-per-node 1 -m pretrain.train --config configs/formal/train_100m_v2.json --data /root/autodl-tmp/datasets
# 10B 正式训练（含 checkpoint 滚动保留，避免磁盘打满）
torchrun --nproc-per-node 2 -m pretrain.train --config configs/formal/train.json --data /root/autodl-tmp/datasets \
--resume-from checkpoints/pretrain/qwen3-0.6B/latest

# 监视
python scripts/monitor_metrics.py --config configs/formal/train_10b.json --data /root/autodl-tmp/datasets
```

### 10B 开跑前必须确认

- **设备自检**：`python scripts/doctor_device.py`。检查驱动、内核模块、设备节点、torch 的 CUDA 构建，
  并真正执行一次 kernel（唯一能暴露驱动版本不匹配的方法），不可用时以非零状态退出。
- 训练配置的 `"device"`：`"cuda"` = 要求 GPU，不可用即**启动失败**；`"cpu"` 用于冒烟测试；`"auto"` 允许降级到 CPU（仅本机调试）。
  正式配置都是 `"cuda"`，避免出现"静默跑 CPU"——那会慢 50–100 倍，看着在跑其实毫无意义。
- 磁盘：`save_every_steps=500` 全量保留会产生 38 个 checkpoint、实测约 **211GB**，因此 `train_10b.json` 设置了
  `keep_recent_checkpoints=2`（只有最新 2 个保留优化器/调度器/RNG）与 `keep_total_checkpoints=12`（目录数上限，最旧的整体删除），
  峰值约 **22.6GB**。优化器状态最大的 0.6B 约 4.44GB/份。
- `log_every_steps` / `eval_every_steps` / `save_every_steps` 只影响观测与落盘，**不改变训练轨迹**
  （`tests/test_interval_independence.py`：只改这三个值，最终权重逐元素相同），可随时按需调整。
- 被裁剪过的 checkpoint 只有模型权重，`load_training_checkpoint` 会明确报错，续训只能用最新的 checkpoint。
- 数据：`chinese-10b` 正式清单已校验（train 10B / val 10M / test 10M tokens，4480 分片，零缺失）。

### 中途加卡（1 卡暂停 → N 卡续训）

全局 batch 是 `global_tokens_per_step / sequence_length` 条序列，与卡数无关：`train.py` 里
`accumulation_steps = global_tokens_per_step // (sequence_length * micro_batch_size * world_size)`，
所以加卡时 `micro_batch_size` 与 `global_tokens_per_step` 保持不变，只是梯度累积步数变小
（1 卡 32 → 2 卡 16），每步仍是 524288 tokens，`total_steps`、学习率调度、optimizer step 数都不变。
每个 rank 的 loss 除以自己的 accumulation、再由 DDP 平均，等价于在全局 batch 上求平均，
因此梯度定义与单卡一致（`tests/test_world_size_equivalence.py` 固化：全局 batch 组成一致 + 梯度一致）。

- 必须同步修改配置里的 `"num_gpus"`，否则与 `WORLD_SIZE` 不一致会直接报错。
- 只能从最新的 checkpoint 续训（更早的已被滚动保留裁剪掉优化器状态）。
- 单卡 checkpoint 只有 `rng-rank-00000.pt`，新 rank 会回退到 rank0 的随机状态，不会报错（有测试覆盖）。
- 浮点层面不是逐 bit 相同：DDP 的 all-reduce 改变了梯度加法顺序，差异在 1e-10 量级的舍入误差。
- 日志里的 `train_loss` 已做全局平均（等价于单卡口径），可与加卡前的曲线直接比较。

```bash
# 例：1 卡暂停后改 2 卡（先把配置里的 num_gpus 改成 2）
torchrun --nproc-per-node 2 -m pretrain.train \
  --config configs/formal/train_10b.json --data /root/autodl-tmp/datasets \
  --resume-from checkpoints/pretrain/qwen3-0.6B-10b/latest
```

- 训练中随时查看健康度（NaN、验证 loss 趋势、吞吐漂移、磁盘）：

```bash
python scripts/monitor_metrics.py --config configs/formal/train_10b.json --data /root/autodl-tmp/datasets
```

### 续写生成参数（评测可比性）

基座模型在早期 checkpoint 上**贪心解码会陷入自我强化重复**（如 `1.1.1.1...`、`学生可以在教学中` 循环），
因此固定续写评测必须用采样并配重复惩罚，且**所有 checkpoint 使用同一套参数**，否则人工评分不可比：

- `do_sample=true`、`temperature=0.8`、`top_k=40`、`top_p=0.95`、`seed=42`、`repetition_penalty=1.2`
  （已写入 `configs/formal/eval_dev.json` 与 `eval_final.json`）。
- `repetition_penalty` 只作用于解码，**不影响训练结果**；`--temperature 0` 是贪心，只适合当"重复程度"诊断指标。

```bash
# 查看某个 checkpoint 的续写
python -m pretrain.infer --config configs/formal/train.json --data /root/autodl-tmp/datasets \
  --checkpoint checkpoints/pretrain/qwen3-0.6B/latest \
  --prompt "冯小刚执导、雷佳音胡歌主演的《抓特务》，上映后口碑不差，被视为他近年最有诚意的作品，但网友并不买账，吐槽影片枯燥老套，韩红站台“走个面儿”更被批道德绑架，争议发酵，票房。" \
  --max-new-tokens 1024 --repetition-penalty 1.0
```

注意：`PackedTokenCorpus` 的全局采样位置 `global_sequence_offset` 只在分片清单不变时才有效。
checkpoint 的 `trainer_state.json` 会记录 `train_corpus_fingerprint`（分片路径 + token 数的 SHA-256），
续训时若指纹不一致会直接报错，必须改新 `output_dir` 从头训练，避免出现"部分数据重复、部分数据从未见过"。

Hugging Face 下载慢或不通时，加 `--modelscope` 改从 modelscope.cn 直连下载（忽略代理环境变量，国内 CDN 直达；分词器同样从 ModelScope 获取）：

```bash
python -m datasets.prepare --config configs/formal/data.json --modelscope
```

`datasets.prepare` 和 `pretrain.train` 均支持 `--data <目录>`。它指定原始下载、分词器和处理后训练数据的根目录，默认是 `D:/datasets/llm`。目录结构为 `raw/`、`tokenizer/qwen3/` 和 `processed/<dataset_name>/`；checkpoint 仍由训练配置的 `output_dir` 指定。

## 里程碑

### 远程环境安装

远程 Linux 服务器使用已有的 Conda `cuda` 环境，一键安装项目依赖：

```bash
bash scripts/install_remote.sh
```

脚本不安装或升级 PyTorch/CUDA，并会核对安装前后的 PyTorch 版本及 CUDA 构建标识。环境名不是 `cuda` 时，可使用 `CONDA_ENV_NAME=<名称> bash scripts/install_remote.sh`。

### M0：微型冒烟测试（已完成）

- 输入：仓库内置中文测试文本。
- 输出：处理后 shards、微型 checkpoint、续写样例和 Hugging Face 导出目录。
- 通过条件：单元测试通过；训练 loss 下降；中断恢复状态一致；导出模型可由官方 Qwen3 类加载。
- 失败处理：先在 CPU 上过拟合一个 batch，再检查遮罩、标签位移、学习率和初始化。

### M1：100M tokens

- 使用正式 0.6B 配置，先跑数据和显存基准，再开始训练。
- 数据现状：`/root/autodl-tmp/datasets` 下 `chinese-10b` 正式清单已就绪（train 10B / val 10M / test 10M tokens，共 4479 个 shard）。
- 第一次 100M 试跑（`qwen3-0.6B-100m`）因数据排列算法更换作废，用 `configs/formal/train_100m_v2.json` 从 step 0 重跑，**不要** `--resume-from` 旧 checkpoint。
- 每 10 步验证、每 10 步保存（试跑便于观察），确认无 NaN、吞吐稳定、验证 loss 下降；正式 10B 按每 250 步验证、每 500 步保存。
- 若 OOM，先降低每卡 micro batch；不改变全局 tokens/step，由梯度累积补足。

### M2：1B tokens

- 比较随机初始化、100M 和 1B checkpoint 的验证 loss 与固定续写集。
- 若验证 loss 停滞，优先审查数据过滤和学习率，不直接扩大数据规模。

### M3：10B tokens

- 训练目标：10B 个训练 tokens，另保留 10M 验证 tokens 和 10M 最终测试 tokens。
- 自动验收：验证 loss 较初始化下降至少 30%；无 NaN；测试 loss 与验证 loss 差距不超过 10%；生成异常率低于 10%。
- 人工验收：语法通顺满分比例至少 80%；上下文相关满分比例至少 70%；五项平均至少 1.4/2；盲测与开发集平均分差不超过 0.2。
- 10B 结果未达标时，根据错误类型决定清洗数据、调整训练配置或扩展到 30B，不默认继续烧算力。

## Checkpoint 内容

每个训练 checkpoint 保存：模型、优化器、学习率调度器、训练步数、累计 tokens、配置、每个 rank 的随机状态、可恢复的数据全局位置，以及数据分片清单指纹 `train_corpus_fingerprint`（清单变化时拒绝续训）。最终 Hugging Face 导出只保留配置、分词器和 `.safetensors` 权重。

滚动保留：配置 `keep_recent_checkpoints` 指定最近几个 checkpoint 保留完整的优化器/调度器/RNG（只有这些能精确续训），更早的只留模型权重与元数据；`keep_total_checkpoints` 限制目录总数，最旧的目录整体删除（`latest.txt` 指向的最新 checkpoint 永不删除）。

## 完成定义

- [x] 模型结构与官方微型 Qwen3 实现 logits/loss 对齐，误差不超过 `1e-5`。
- [x] 数据划分无泄漏，分片哈希和统计可复核。
- [x] 微型模型能过拟合小数据，训练 loss 明显下降。
- [x] 恢复训练的下一批 token、步数、学习率与不中断运行一致。
- [x] 固定提示生成和异常统计可重复。
- [x] 导出目录能由 `AutoModelForCausalLM.from_pretrained()` 加载。
- [x] CPU 单元测试通过。
- [x] RTX 5070 CUDA 冒烟测试通过：在 Conda `cuda` 环境中完成 596,049,920 参数模型、序列长度 16 的单步 AdamW，峰值约 4.50 GiB。

## 已知风险

- 12GB 显存不适合高效完成 0.6B/10B-token 正式训练。
- 公开中文语料存在事实错误、模板化和污染，自动清洗不能代替抽样审查。
- 10B tokens 是工程里程碑，不保证获得成熟模型的稳定语言和逻辑能力。
- Qwen3 大词表使嵌入层和输出层成本较高；权重绑定与分块 loss 必须保持启用。
