#! /bin/bash
# export RAY_memory_monitor_refresh_ms=0
# export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
set -e
# This script is used to run the pipeline for DBP15K (ZH-EN).
export CUDA_VISIBLE_DEVICES=4,5
MASTER_PORT=29503
CONFIG_PATH=config/url_rel/dbp15k_zh_en_qwen2-chat-translation.yaml
LOG_FILE=logs/url_rel/dbp15k_zh_en_qwen2-chat-translation.log
GPU_NUM=2

if [ ! -d "logs/url_rel" ]; then
  mkdir -p logs/url_rel
fi
echo ">>>>>> Start pipeline for DBP15K (ZH-EN)." > $LOG_FILE
# Step 1: generate entity sequence
echo ">>>>>> Step 1: generate entity sequence." >> $LOG_FILE
python generate_seq.py --config_path $CONFIG_PATH >> $LOG_FILE 2>&1

# Step 2: test via untrained retrieval model
echo ">>>>>> Step 2: test via untrained retrieval model." >> $LOG_FILE
python retrieval_test.py --config_path $CONFIG_PATH >> $LOG_FILE 2>&1

# Step 3: generate retrieval sft data
echo ">>>>>> Step 3: generate retrieval sft data." >> $LOG_FILE
python generate_retrieval_sft_data.py --config_path $CONFIG_PATH >> $LOG_FILE 2>&1

# Step 4: hard negative mining retrieval sft data
echo ">>>>>> Step 4: hard negative mining retrieval sft data." >> $LOG_FILE
python hn_mine_retrieval_sft_data.py --config_path $CONFIG_PATH --step retrieval >> $LOG_FILE 2>&1

# Step 5: train retrieval model
echo ">>>>>> Step 5: train retrieval model." >> $LOG_FILE
torchrun --nproc_per_node $GPU_NUM --master_port $MASTER_PORT retriever_finetune.py --config_path $CONFIG_PATH >> $LOG_FILE 2>&1

# Step 6: test retrieval model
echo ">>>>>> Step 6: test retrieval model." >> $LOG_FILE
python retrieval_test.py --config_path $CONFIG_PATH --trained >> $LOG_FILE 2>&1

# Step 7: hard negative mining reranking sft data
echo ">>>>>> Step 7: hard negative mining reranking sft data." >> $LOG_FILE
python hn_mine_rerank_sft_data.py --config_path $CONFIG_PATH --step rerank >> $LOG_FILE 2>&1

# Step 8: train reranking model
echo ">>>>>> Step 8: train reranking model." >> $LOG_FILE
torchrun --nproc_per_node $GPU_NUM --master_port $MASTER_PORT reranker_finetune.py --config_path $CONFIG_PATH >> $LOG_FILE 2>&1

# Step 9: test reranking model
echo ">>>>>> Step 9: test reranking model." >> $LOG_FILE

echo ">>>>>> Step 9.1 retriever:N, rerank:N" >> $LOG_FILE
python rerank_test.py --config_path $CONFIG_PATH >> $LOG_FILE 2>&1

echo ">>>>>> Step 9.2 retriever:Y, rerank:N" >> $LOG_FILE
python rerank_test.py --config_path $CONFIG_PATH --retriever_trained >> $LOG_FILE 2>&1

echo ">>>>>> Step 9.3 retriever:Y, rerank:Y" >> $LOG_FILE
python rerank_test.py --config_path $CONFIG_PATH --retriever_trained --reranker_trained >> $LOG_FILE 2>&1

echo ">>>>>> Step 9.4 retriever:N, rerank:Y" >> $LOG_FILE
python rerank_test.py --config_path $CONFIG_PATH --reranker_trained >> $LOG_FILE 2>&1
