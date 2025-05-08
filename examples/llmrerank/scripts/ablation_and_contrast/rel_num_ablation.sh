#!/bin/bash

LOG_DIR=logs/ablation_and_contrast
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

SAMPLE_NUM=150
DATASET_FILE=("dbp15k_zh_en" "dbp15k_ja_en" "dbp15k_fr_en")
# CONFIG_FILE=("config/url_attr/dbp15k_zh_en.yaml" "config/url_attr/dbp15k_ja_en.yaml" "config/url_attr/dbp15k_fr_en.yaml")
rel_num_start=1
rel_num_end=9
for dataset_file in ${DATASET_FILE[@]}
do
    for rel_num in {1..9..2}
    do
        echo "Running one shot experiment with parameter Seed: 0 ..." > ${LOG_DIR}/${dataset_file}-shot-qwen32b-chat-attr-num-7-seed-0-ablation-rel-num-${rel_num}.log
        python -u test.py \
                    --seed 0 \
                    --config_path config/url_attr/${dataset_file}.yaml \
                    --trained \
                    --topk 10 \
                    --vllm \
                    --llm_path /public/home/chenxuan/models/Qwen1.5-32B-Chat-AWQ \
                    --quantization AWQ \
                    --bootstraping \
                    --bootstraping_num 1 \
                    --max_tokens 1024 \
                    --use_name \
                    --attr_sampled \
                    --attr_strategy str_cooccurrence \
                    --rel_sampled \
                    --rel_max_num ${rel_num} \
                    --test_strategy same_proportion_as_test_set \
                    --sample_num ${SAMPLE_NUM} \
                    --verbose \
                    --tensor_parallel_size 1 \
                    --swap_space 4 \
                    --experiment only_rel >> ${LOG_DIR}/${dataset_file}-shot-qwen32b-chat-attr-num-7-seed-0-ablation-rel-num-${rel_num}.log 2>&1
    done
done
echo "Without Rel Ablation 32B Done!"

for dataset_file in ${DATASET_FILE[@]}
do
    for rel_num in {1..9..2}
    do
        echo "Running one shot experiment with parameter Seed: 0 ..." > ${LOG_DIR}/${dataset_file}-shot-qwen14b-chat-attr-num-7-seed-0-ablation-rel-num-${rel_num}.log
        python -u test.py \
                    --seed 0 \
                    --config_path config/url_attr/${dataset_file}.yaml \
                    --trained \
                    --topk 10 \
                    --vllm \
                    --llm_path /public/home/chenxuan/models/Qwen1.5-14B-Chat \
                    --bootstraping \
                    --bootstraping_num 1 \
                    --max_tokens 1024 \
                    --use_name \
                    --attr_sampled \
                    --attr_strategy str_cooccurrence \
                    --rel_sampled \
                    --rel_max_num ${rel_num} \
                    --test_strategy same_proportion_as_test_set \
                    --sample_num 150 \
                    --verbose \
                    --tensor_parallel_size 1 \
                    --swap_space 4 \
                    --experiment only_rel >> ${LOG_DIR}/${dataset_file}-shot-qwen14b-chat-attr-num-7-seed-0-ablation-rel-num-${rel_num}.log 2>&1
    done
done
echo "Without Rel Ablation 14B Done!"