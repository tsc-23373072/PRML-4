# Transformer 复现与 Q/K/V 必要性对照实验

本工程用于完成两件事：

1. **复现 2017 年论文《Attention Is All You Need》中的经典 Transformer 架构**：
   - Encoder–Decoder 堆叠结构
   - Scaled Dot-Product Attention
   - Multi-Head Attention
   - Sinusoidal Positional Encoding
   - Position-wise FFN
   - Residual Connection + LayerNorm
   - Adam + Noam 学习率调度
   - Label Smoothing

2. **完成“Q、K、V 的必要性”实验**：
   - `standard`：标准 QKV 注意力  
     \[
     \operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt{d_k})V
     \]
   - `shared_kv`：K 与 V 合并为一个矩阵，即
     \[
     \operatorname{Attention}(Q,KV,KV)=\operatorname{softmax}(QKV^\top/\sqrt{d_k})KV
     \]
     代码中使用同一个线性投影生成 K/V，使“用于匹配的表示”和“用于传递内容的表示”被迫共享。

---

## 1. 为什么使用合成数据集

论文原始实验使用 WMT 2014 英德、英法机器翻译数据集，训练成本很高。为了在课程大作业条件下研究 **K 与 V 分离是否必要**，本工程构造了一个更适合做消融实验的 **键值检索型序列到序列任务**。

### 数据样例

| 字段 | 示例 |
|---|---|
| `src` | `K3 V7 K1 V4 K5 V2 <sep> Q1 Q5 Q3` |
| `tgt` | `V4 V2 V7` |

含义：
- 输入序列前半部分表示若干“键–值”关系：
  - `K3 -> V7`
  - `K1 -> V4`
  - `K5 -> V2`
- `<sep>` 后面的 `Q1 Q5 Q3` 是查询顺序
- 输出应依次返回相应的值：`V4 V2 V7`

这个任务天然区分：
- **Key**：负责被查询与匹配
- **Value**：负责输出被检索到的内容

因此，它非常适合比较：
- 标准 QKV
- K/V 合并的 QK-only 变体

---

## 2. 项目结构

```text
transformer_qkv_reproduction/
├── README.md
├── REPORT_GUIDE.md
├── requirements.txt
├── generate_dataset.py
├── train.py
├── compare_experiment.py
├── visualize_attention.py
├── data/
│   ├── synthetic_kv_retrieval.xlsx
│   ├── synthetic_kv_retrieval.csv
│   └── dataset_description.md
├── src/
│   ├── __init__.py
│   ├── attention.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── positional_encoding.py
│   ├── scheduler.py
│   └── training.py
├── checkpoints/
└── outputs/
```

---

## 3. 安装依赖

```bash
pip install -r requirements.txt
```

建议：
- 有 NVIDIA GPU 时使用 CUDA 训练。
- 没有 GPU 也可以运行，但训练会慢一些。

---


---

## 3.1 论文 base Transformer 结构参数复现方式

本工程默认使用的是便于课程作业快速训练的“小模型”参数。若希望在结构上切换到论文 **Transformer (base)** 的主要超参数，可运行：

```bash
python train.py \
  --data_path data/synthetic_kv_retrieval.xlsx \
  --attention_mode standard \
  --d_model 512 \
  --num_layers 6 \
  --num_heads 8 \
  --d_ff 2048 \
  --dropout 0.1 \
  --warmup_steps 4000 \
  --epochs 20 \
  --device auto
```

说明：
- 这会复现论文 base 模型的核心结构超参数。
- 由于本工程使用的是合成数据集而非 WMT 2014 翻译集，因此训练结果不会等同于论文 BLEU。
- 若仅进行课程实验与 Q/K/V 消融，建议使用默认的小模型或适度缩小模型，训练更快。

---

## 3.2 CPU 快速试跑命令

仅用于确认环境与代码可运行：

```bash
python compare_experiment.py \
  --data_path data/synthetic_kv_retrieval.xlsx \
  --epochs 3 \
  --batch_size 128 \
  --d_model 64 \
  --num_layers 1 \
  --num_heads 4 \
  --d_ff 128 \
  --warmup_steps 100 \
  --device cpu
```


## 4. 直接运行对照实验

### 4.1 推荐命令

```bash
python compare_experiment.py \
  --data_path data/synthetic_kv_retrieval.xlsx \
  --epochs 20 \
  --batch_size 64 \
  --device auto
```

### 4.2 输出结果

运行完成后，会在 `outputs/` 目录生成：

- `comparison_summary.csv`：两种模型最终指标对比
- `standard_metrics.csv`：标准 QKV 每轮训练日志
- `shared_kv_metrics.csv`：K/V 合并模型每轮训练日志
- `val_loss_comparison.png`：验证集损失曲线对比
- `val_token_accuracy_comparison.png`：验证集 token 准确率曲线对比
- `val_exact_match_comparison.png`：验证集序列完全匹配率对比

主要比较指标：
- `val_loss`
- `val_token_accuracy`
- `val_exact_match`

---

## 5. 单独训练某一种模型

### 5.1 标准 QKV

```bash
python train.py \
  --data_path data/synthetic_kv_retrieval.xlsx \
  --attention_mode standard \
  --epochs 20 \
  --device auto
```

### 5.2 K/V 合并

```bash
python train.py \
  --data_path data/synthetic_kv_retrieval.xlsx \
  --attention_mode shared_kv \
  --epochs 20 \
  --device auto
```

---

## 6. 重新生成数据集

压缩包内已经包含可直接使用的数据集。需要重新生成时：

```bash
python generate_dataset.py \
  --output_xlsx data/synthetic_kv_retrieval.xlsx \
  --output_csv data/synthetic_kv_retrieval.csv \
  --train_size 5000 \
  --val_size 800 \
  --test_size 800 \
  --seed 2026
```

---

## 7. 查看注意力图

训练完成后，可视化某条样本的 decoder cross-attention：

```bash
python visualize_attention.py \
  --checkpoint checkpoints/standard_best.pt \
  --data_path data/synthetic_kv_retrieval.xlsx \
  --sample_index 0 \
  --output_path outputs/standard_attention_heatmap.png \
  --device auto
```

说明：
- 热力图横轴：输入 token
- 热力图纵轴：decoder 查询位置
- 颜色越深表示注意力权重越高

---

## 8. 实验结论应该怎么写

建议从以下角度分析：

1. **收敛速度**
   - 标准 QKV 是否更快降低 loss？
   - `shared_kv` 是否需要更多 epoch 才能达到相近性能？

2. **最终精度**
   - 对比 `val_token_accuracy`
   - 对比 `val_exact_match`

3. **结构原因**
   - K 的主要职责是“用于匹配”
   - V 的主要职责是“用于传递被汇聚的信息”
   - 将 K 和 V 强制合并，会降低表征自由度，使同一表示既要便于匹配，又要便于输出内容，可能造成任务冲突。

4. **可能出现的现象**
   - 在简单任务、小数据集上，`shared_kv` 仍可能学得不错。
   - 在“检索地址”和“传递内容”角色差异更大的任务上，标准 QKV 通常更占优势。
   - 实验结果受随机种子、模型规模和训练轮数影响，建议至少固定随机种子并重复 2–3 次验证趋势。

---

## 9. 与原论文复现范围的关系

本工程复现的是 **Transformer 的核心结构与训练思想**，而不是完整复现论文中的 WMT 2014 BLEU 数值。若要严格复现实验表格，需要：
- WMT 2014 英德/英法数据
- BPE 或 word-piece 分词
- 大规模训练资源
- BLEU 评估与 beam search 解码

本工程更适合作为课程实验：
- 结构完整
- 代码可运行
- 对照实验明确
- 能直接支撑“Q/K/V 必要性”分析
