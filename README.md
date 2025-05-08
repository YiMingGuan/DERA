# DERA: Dense Entity Retrieval for Entity Alignment

**DERA** is a large language model (LLM)-based framework for entity alignment. In this framework, entity information from knowledge graphs is first uniformly transformed into natural language descriptions. These descriptions are then encoded using LLM-based embedding models, enabling nearest-neighbor entity search across different KGs.

------

## Installation

Ensure your Python version is 3.6 or higher. Then run the following commands to install the package and its dependencies:

```bash
git clone https://github.com/XChen-Zero/aligncraft.git
cd aligncraft
pip install -e .
```

------

## Quick Start

**AlignCraft** follows a three-stage pipeline:

1. **Triple-to-Text Conversion**: Transforms KG triples into natural language descriptions.
2. **Retrieval Stage**: Uses a dual-encoder model to embed descriptions and perform initial candidate search.
3. **Reranking Stage**: Refines the candidate pairs with an interaction-based reranker to improve alignment precision.

All pipeline scripts are located in:

```
aligncraft/examples/retrievalea/pipeline/
```

To run a full example using the `DBP15K fr-en` dataset:

```bash
bash aligncraft/examples/retrievalea/pipeline/attr/pipeline_fr_en.sh
```

------

### Output Structure

- `~/.cache/aligncraft/`: Stores intermediate text descriptions (MD5-keyed, HDF5 format)
- `aligncraft/examples/retrievalea/logs/`: Logs for each pipeline stage
- `aligncraft/examples/retrievalea/config/`: YAML configuration files for different language pairs

------

### Script Descriptions

- `generate_seq.py`: Converts KG triples into text descriptions
- `retrieval_test.py`: Evaluates untrained retrieval models
- `generate_retrieval_sft_data.py`: Generates supervised training data for retrieval
- `hn_mine_retrieval_sft_data.py`: Performs hard negative mining for retrieval training
- `retriever_finetune.py`: Fine-tunes the retrieval model
- `reranker_finetune.py`: Fine-tunes the reranker model
- `rerank_test.py`: Evaluates reranking performance under different configurations



