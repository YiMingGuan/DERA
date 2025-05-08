#!/bin/bash

# 假设我们要控制的参数是从1开始，到10结束

LOG_DIR=logs/url_attr
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

CONFIG_FILE=config/url_attr/dbp15k_ja_en_one_shot_gpt3.5.yaml
# 循环从start到end


echo "Running one shot experiment with parameter: 3 ..." > ${LOG_DIR}/dbp15k_ja_en_one_shot_attr_num_1_gpt3.5.log
python -u test.py \
    --config_path ${CONFIG_FILE} \
    --attr_format_max_num 3 \
    --output_path "cases/url_attr/dbp15k_ja_en_one_shot_attr_num_1_gpt3.5.json" >> ${LOG_DIR}/dbp15k_ja_en_one_shot_attr_num_1_gpt3.5.log 2>&1

echo "Experiment with parameter 3 completed"


