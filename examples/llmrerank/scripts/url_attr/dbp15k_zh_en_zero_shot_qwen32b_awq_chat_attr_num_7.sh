#!/bin/bash

start=0
end=3

LOG_DIR=logs/url_attr
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

SAMPLE_NUM=105
CONFIG_FILE=config/url_attr/dbp15k_zh_en.yaml
# TEST_STRATEGY=("at_topk_and_not_at_rank_1" "only_at_rank_1" "at_topk" "same_proportion_as_test_set")
TEST_STRATEGY=("at_topk" "same_proportion_as_test_set")
for test_strategy in ${TEST_STRATEGY[@]}
do
    for seed in $(seq $start $end)
    do
        echo "Running one shot experiment with parameter Seed: ${seed} TEST_STRATEGY:${test_strategy} ..." > ${LOG_DIR}/dbp15k-zh-en-zero-shot-qwen32b-awq-chat-attr-num-7-seed-${seed}-test_strategy-${test_strategy}.log
        python -u test.py \
                --seed ${seed} \
                --config_path ${CONFIG_FILE} \
                --trained \
                --topk 10 \
                --vllm \
                --llm_path /public/home/chenxuan/models/Qwen1.5-32B-Chat-AWQ \
                --quantization AWQ \
                --bootstraping \
                --bootstraping_num 7 \
                --max_tokens 1024 \
                --use_name \
                --use_attr \
                --attr_sampled \
                --attr_strategy str_cooccurrence \
                --attr_max_num 7 \
                --test_strategy ${test_strategy} \
                --sample_num ${SAMPLE_NUM} \
                --verbose \
                --tensor_parallel_size 1 \
                --swap_space 4 >> ${LOG_DIR}/dbp15k-zh-en-zero-shot-qwen32b-awq-chat-attr-num-7-seed-${seed}-test_strategy-${test_strategy}.log 2>&1

        echo "One shot experiment with parameter Seed: ${seed} TEST_STRATEGY:${test_strategy} done..."
    done
done

echo "All experiments completed"