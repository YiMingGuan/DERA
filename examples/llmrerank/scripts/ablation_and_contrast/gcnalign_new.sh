#!/bin/bash

LOG_DIR=logs/ablation_and_contrast
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

SAMPLE_NUM=300

# python -u test.py \
#                 --seed 0 \
#                 --config_path config/gcnalign/dbp15k_zh_en.yaml \
#                 --trained \
#                 --topk 10 \
#                 --vllm \
#                 --llm_path /public/home/chenxuan/models/Qwen1.5-32B-Chat-AWQ \
#                 --quantization AWQ \
#                 --bootstraping \
#                 --bootstraping_num 7 \
#                 --max_tokens 1024 \
#                 --use_name \
#                 --attr_sampled \
#                 --attr_strategy str_cooccurrence \
#                 --attr_max_num 7 \
#                 --test_strategy same_proportion_as_test_set \
#                 --sample_num ${SAMPLE_NUM} \
#                 --verbose \
#                 --tensor_parallel_size 1 \
#                 --ent_embs_path /public/home/chenxuan/Code/EAkit/embs/gcnalign/zh_en_gcnalign_20240409-0935_enh_ins.npy \
#                 --swap_space 4 >> ${LOG_DIR}/zh-en-shot-qwen32b-chat-seed-0-gcnalign-new.log 2>&1


# python -u test.py \
#                 --seed 0 \
#                 --config_path config/gcnalign/dbp15k_ja_en.yaml \
#                 --trained \
#                 --topk 10 \
#                 --vllm \
#                 --llm_path /public/home/chenxuan/models/Qwen1.5-32B-Chat-AWQ \
#                 --quantization AWQ \
#                 --bootstraping \
#                 --bootstraping_num 7 \
#                 --max_tokens 1024 \
#                 --use_name \
#                 --attr_sampled \
#                 --attr_strategy str_cooccurrence \
#                 --attr_max_num 7 \
#                 --test_strategy same_proportion_as_test_set \
#                 --sample_num ${SAMPLE_NUM} \
#                 --verbose \
#                 --tensor_parallel_size 1 \
#                 --ent_embs_path /public/home/chenxuan/Code/EAkit/embs/gcnalign/ja_en_gcnalign_20240409-0940_enh_ins.npy \
#                 --swap_space 4 >> ${LOG_DIR}/ja-en-shot-qwen32b-chat-seed-0-gcnalign-new.log 2>&1

python -u test.py \
                --seed 0 \
                --config_path config/gcnalign/dbp15k_fr_en.yaml \
                --trained \
                --topk 10 \
                --vllm \
                --llm_path /public/home/chenxuan/models/Qwen1.5-32B-Chat-AWQ \
                --quantization AWQ \
                --bootstraping \
                --bootstraping_num 7 \
                --max_tokens 1024 \
                --use_name \
                --attr_sampled \
                --attr_strategy str_cooccurrence \
                --attr_max_num 7 \
                --test_strategy same_proportion_as_test_set \
                --sample_num ${SAMPLE_NUM} \
                --verbose \
                --tensor_parallel_size 1 \
                --ent_embs_path /public/home/chenxuan/Code/EAkit/embs/gcnalign/fr_en_gcnalign_20240409-0946_enh_ins.npy \
                --swap_space 4 > ${LOG_DIR}/fr-en-shot-qwen32b-chat-seed-0-gcnalign-new.log 2>&1


python -u test.py \
                --seed 0 \
                --config_path config/gcnalign/dbp15k_zh_en.yaml \
                --trained \
                --topk 10 \
                --vllm \
                --llm_path /public/home/chenxuan/models/Qwen1.5-14B-Chat \
                --bootstraping \
                --bootstraping_num 7 \
                --max_tokens 1024 \
                --use_name \
                --attr_sampled \
                --attr_strategy str_cooccurrence \
                --attr_max_num 7 \
                --test_strategy same_proportion_as_test_set \
                --sample_num ${SAMPLE_NUM} \
                --verbose \
                --tensor_parallel_size 1 \
                --ent_embs_path /public/home/chenxuan/Code/EAkit/embs/gcnalign/zh_en_gcnalign_20240409-0935_enh_ins.npy \
                --swap_space 4 > ${LOG_DIR}/zh-en-shot-qwen14b-chat-seed-0-gcnalign-new.log 2>&1


python -u test.py \
                --seed 0 \
                --config_path config/gcnalign/dbp15k_ja_en.yaml \
                --trained \
                --topk 10 \
                --vllm \
                --llm_path /public/home/chenxuan/models/Qwen1.5-14B-Chat \
                --bootstraping \
                --bootstraping_num 7 \
                --max_tokens 1024 \
                --use_name \
                --attr_sampled \
                --attr_strategy str_cooccurrence \
                --attr_max_num 7 \
                --test_strategy same_proportion_as_test_set \
                --sample_num ${SAMPLE_NUM} \
                --verbose \
                --tensor_parallel_size 1 \
                --ent_embs_path /public/home/chenxuan/Code/EAkit/embs/gcnalign/ja_en_gcnalign_20240409-0940_enh_ins.npy \
                --swap_space 4 > ${LOG_DIR}/ja-en-shot-qwen14b-chat-seed-0-gcnalign-new.log 2>&1

python -u test.py \
                --seed 0 \
                --config_path config/gcnalign/dbp15k_fr_en.yaml \
                --trained \
                --topk 10 \
                --vllm \
                --llm_path /public/home/chenxuan/models/Qwen1.5-14B-Chat \
                --bootstraping \
                --bootstraping_num 7 \
                --max_tokens 1024 \
                --use_name \
                --attr_sampled \
                --attr_strategy str_cooccurrence \
                --attr_max_num 7 \
                --test_strategy same_proportion_as_test_set \
                --sample_num ${SAMPLE_NUM} \
                --verbose \
                --tensor_parallel_size 1 \
                --ent_embs_path /public/home/chenxuan/Code/EAkit/embs/gcnalign/fr_en_gcnalign_20240409-0946_enh_ins.npy \
                --swap_space 4 > ${LOG_DIR}/fr-en-shot-qwen14b-chat-seed-0-gcnalign-new.log 2>&1


