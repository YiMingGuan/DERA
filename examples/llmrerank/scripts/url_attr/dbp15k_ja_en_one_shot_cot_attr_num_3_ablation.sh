#!/bin/bash

# 假设我们要控制的参数是从1开始，到10结束
start=0
end=10

LOG_DIR=logs/url_attr
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

CONFIG_FILE=config/url_attr/dbp15k_ja_en_i2d2.yaml
# 循环从start到end
for attr_num in $(seq $start $end)
do
   echo "Running one shot experiment with parameter: ${attr_num} ..." > ${LOG_DIR}/dbp15k-ja-en-i2d2-one-shot-cot-attr-num-${attr_num}.log
   python -u test.py \
       --config_path ${CONFIG_FILE} \
       --attr_format_max_num ${attr_num} \
       --output_path "cases/url_attr/dbp15k_ja_en_i2d2_one_shot_cot_attr_num_ablation_param_${attr_num}.json" >> ${LOG_DIR}/dbp15k-ja-en-i2d2-one-shot-cot-attr-num-${attr_num}.log 2>&1

   echo "Experiment with parameter ${attr_num} completed"
done

echo "All experiments completed"