# DERA: Dense Entity Retrieval for Entity Alignment

DERA 是一个基于 PyTorch 实现的实体对齐工具集，支持从三元组转换为文本描述、通过检索模型获得初步对齐结果，再由 rerank 模型进行最终精排的方法

## 安装依赖

确保你的 Python 版本不低于 3.6，然后使用以下命令安装依赖：

```bash
git clone https://github.com/XChen-Zero/aligncraft.git
cd aligncraft
pip install -e .
```

## 项目准备

- 在第一阶段的三元组到实体描述的转换实验，我们将所有的结果数据存放在google云盘上[google](还在传输....)，可以点击直接下载，存放在根目录下的".cache/aligncraft"中

- 数据集我们同样放在云盘上，可以下载完成后放入benchmark目录中

## Quick Start

AlignCraft 分为三个阶段：

1. **三元组转文本描述**：将知识图谱中的三元组转换为自然语言描述。
2. **检索模型阶段**：使用嵌入模型对实体描述进行向量化并进行粗匹配。
3. **Rerank 阶段**：对候选实体进行 rerank，进一步提升对齐精度。

所有完整流程的脚本在：

```
aligncraft/examples/retrievalea/pipeline/
```

以 `DBP15K fr-en` 数据集为例，可运行以下脚本开始完整流程：

```bash
bash aligncraft/examples/retrievalea/pipeline/attr/pipeline_fr_en.sh
```

### 输出目录结构

- `.cache/aligncraft/`: 存储中间生成的文本描述（MD5格式）
- `logs/attr/`: 各阶段运行日志
- `config/attr/`: 各语言对的配置文件（`.yaml`）
- 云盘数据链接：请访问 [xxx]() 下载预处理文件和模型

### 各阶段说明

- `generate_seq.py`: 将三元组转换为文本序列
- `retrieval_test.py`: 使用未训练的检索模型进行初步匹配
- `generate_retrieval_sft_data.py`: 生成训练数据
- `hn_mine_retrieval_sft_data.py`: 硬负采样
- `retriever_finetune.py`: 训练检索模型
- `reranker_finetune.py`: 训练 reranker 模型
- `rerank_test.py`: 多种组合条件下评估对齐效果



