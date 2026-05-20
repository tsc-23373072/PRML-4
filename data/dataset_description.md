# 合成数据集说明

文件：
- `synthetic_kv_retrieval.xlsx`
- `synthetic_kv_retrieval.csv`

字段：
- `split`：`train` / `val` / `test`
- `src`：源序列
- `tgt`：目标序列
- `num_pairs`：该样本中包含的键值对数量
- `num_queries`：该样本中包含的查询数量

## 示例

```text
src = K3 V7 K1 V4 K5 V2 <sep> Q1 Q5 Q3
tgt = V4 V2 V7
```

## 任务含义

- 读取 `<sep>` 之前的键值关系。
- 读取 `<sep>` 之后的查询顺序。
- 输出对应值。

该数据集用于突出：
- K：检索/匹配角色
- V：被读取/汇聚的信息角色
