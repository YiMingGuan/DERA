#!/bin/bash
# 对于attr的任务，ja_en不使用rel效果不好，所以这里加上rel再次训练
./pipeline/attr/pipeline_ja_en_add_rel.sh

# 对于url_attr_trans的任务，原始的ja_en任务url没有处理干净，所以这里重新处理后再次训练
./pipeline/url_attr_trans/pipeline_ja_en_new.sh

# 未知原因中断，从Step 9.1开始(Step 9.3 已经跑完)
./pipeline/url_attr_trans/pipeline_zh_en.sh