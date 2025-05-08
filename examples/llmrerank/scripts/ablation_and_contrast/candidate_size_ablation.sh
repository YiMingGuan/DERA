#!/bin/bash

LOG_DIR=logs/ablation_and_contrast
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

SAMPLE_NUM=315
DATASET_FILE=("dbp15k_zh_en" "dbp15k_ja_en" "dbp15k_fr_en")
# CONFIG_FILE=("config/url_attr/dbp15k_zh_en.yaml" "config/url_attr/dbp15k_ja_en.yaml" "config/url_attr/dbp15k_fr_en.yaml")

for dataset_file in ${DATASET_FILE[@]}
do
    for topk in {20..50..10}
    do
        echo "Running one shot experiment with parameter Seed: 0 ..." > ${LOG_DIR}/${dataset_file}-shot-qwen32b-chat-seed-0-candidate-size-${topk}-new.log
        python -u test.py \
                    --seed 0 \
                    --config_path config/url_attr/${dataset_file}.yaml \
                    --trained \
                    --topk ${topk} \
                    --vllm \
                    --llm_path /public/home/chenxuan/models/Qwen1.5-32B-Chat-AWQ \
                    --quantization AWQ \
                    --bootstraping \
                    --bootstraping_num 1 \
                    --max_tokens 1024 \
                    --use_name \
                    --attr_sampled \
                    --attr_strategy str_cooccurrence \
                    --attr_max_num 7 \
                    --test_strategy at_topk \
                    --sample_num ${SAMPLE_NUM} \
                    --verbose \
                    --tensor_parallel_size 1 \
                    --swap_space 4 \
                    --experiment without_bootstraping_and_only_name >> ${LOG_DIR}/${dataset_file}-shot-qwen32b-chat-seed-0-candidate-size-${topk}-new.log 2>&1
    done
done
echo "Without Bootstraping 32B Done!"
for dataset_file in ${DATASET_FILE[@]}
do
    for topk in {10..10..10}
    do
        echo "Running one shot experiment with parameter Seed: 0 ..." > ${LOG_DIR}/${dataset_file}-shot-qwen14b-chat-seed-0-candidate-size-${topk}-new.log
        python -u test.py \
                    --seed 0 \
                    --config_path config/url_attr/${dataset_file}.yaml \
                    --trained \
                    --topk ${topk} \
                    --vllm \
                    --llm_path /public/home/chenxuan/models/Qwen1.5-14B-Chat \
                    --bootstraping \
                    --bootstraping_num 1 \
                    --max_tokens 1024 \
                    --use_name \
                    --attr_sampled \
                    --attr_strategy str_cooccurrence \
                    --attr_max_num 7 \
                    --test_strategy at_topk \
                    --sample_num ${SAMPLE_NUM} \
                    --verbose \
                    --tensor_parallel_size 1 \
                    --swap_space 4 \
                    --experiment without_bootstraping_and_only_name >> ${LOG_DIR}/${dataset_file}-shot-qwen14b-chat-seed-0-candidate-size-${topk}-new.log 2>&1
    done
done
echo "Without Bootstraping 14B Done!"