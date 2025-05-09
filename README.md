# DERA: Dense Entity Retrieval for Entity Alignment

DERA is a dense entity retrieval framework for EA, leveraging language models to uniformly encode various features of entities and facilitate nearest entity search across KGs. Candidate alignments are first generated through entity retrieval, which are subsequently reranked to determine the final alignments.

------

## Installation

Ensure your Python version is 3.6 or higher. Run the following commands to install the project in development mode:

```bash
git clone https://github.com/YiMingGuan/DERA.git
cd aligncraft
pip install -e .
```

------

## Project Preparation

Before running the full pipeline, make sure to download the fine-tuned models for each stage and place them in the appropriate directories. Here are the model links:

- **Entity Verbalization (EV) Stage Model**:
   [Download EV model](https://drive.google.com/file/d/1wfWLUMYdjDhcCLPRFIsoVhruVyF8r09X/view?usp=drive_link)
- **Entity Retrieval (ER) Stage Embedding Model**:
   [Download ER model](https://drive.google.com/file/d/1lz-vmYW4ZUt30cMxXfPnCMgNrGgIujQX/view?usp=drive_link)
- **Alignment Reranking (AR) Stage Model**:
   [Download AR model](https://drive.google.com/file/d/1zSnxn1ydpac622fyr6_ZtixQjJv5TYM1/view?usp=drive_link)

Place the downloaded models in your model directory, and make sure the configuration YAML files correctly reference these paths.

------

## Quick Start

DERA is structured into a three-stage pipeline:

1. **Entity Verbalization**: Transforms knowledge graph triples into natural language entity descriptions.
2. **Entity Retrieval**: Encodes textual descriptions into embeddings and retrieves top-k similar entities.
3. **Alignment Reranking**: Refines the alignment results through an interaction-based reranking model.

All full pipeline scripts are located in:

```
aligncraft/examples/retrievalea/pipeline/
```

To run the complete DERA pipeline on the `DBP15K fr-en` dataset:

```bash
bash aligncraft/examples/retrievalea/pipeline/attr/pipeline_fr_en.sh
```

------

### Output Directory Structure

- `~/.cache/aligncraft/`: Stores intermediate text descriptions (HDF5 format, keyed by MD5)
- `aligncraft/examples/retrievalea/logs/`: Logs for each pipeline step
- `aligncraft/examples/retrievalea/config/`: YAML configuration files for each language pair

------



- `generate_seq.py`: Converts KG triples into text descriptions
- `retrieval_test.py`: Evaluates untrained retrieval models
- `generate_retrieval_sft_data.py`: Generates supervised training data for retrieval
- `hn_mine_retrieval_sft_data.py`: Performs hard negative mining for retrieval training
- `retriever_finetune.py`: Fine-tunes the retrieval model
- `reranker_finetune.py`: Fine-tunes the reranker model
- `rerank_test.py`: Evaluates reranking performance under different configurations



