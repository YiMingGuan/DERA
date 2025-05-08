#! /bin/bash
# export RAY_memory_monitor_refresh_ms=0
# export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
set -e
# This script is used to run the pipeline for DBP15K (fr-EN).
# export CUDA_VISIBLE_DEVICES=0,1
# MASTER_PORT=29503
CONFIG_PATH=config/url_attr_rel/dbp15k_zh_en_none_llm.yaml
LOG_FILE=logs/url_attr_rel/dbp15k_zh_en_none_llm_testrun.log
# GPU_NUM=2

if [ ! -d "logs/url_attr_rel" ]; then
  mkdir -p logs/url_attr_rel
fi

echo ">>>>>> Start pipeline for DBP15K (zh-EN)." > $LOG_FILE
# Step 1: generate entity sequence
echo ">>>>>> Step 1: generate entity sequence." >> $LOG_FILE
python generate_seq.py --config_path $CONFIG_PATH >> $LOG_FILE 2>&1


# echo ">>>>>> Step 9.3 retriever:Y, rerank:Y" >> $LOG_FILE
# python rerank_test.py --config_path $CONFIG_PATH --retriever_trained --reranker_trained >> $LOG_FILE 2>&1