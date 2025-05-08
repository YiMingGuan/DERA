#!/bin/bash

# echo "ZH-EN Url Attr Rerank 30"
# ./pipeline/attr/pipeline_zh_en_rerank_top30.sh

# echo "JA-EN Url Attr Rerank 30"
# ./pipeline/attr/pipeline_ja_en_rerank_top30.sh

echo "FR-EN Url Attr Rerank 30"
./pipeline/attr/pipeline_fr_en_rerank_top30.sh

echo "zh en  large retreival"
./pipeline/url_attr/pipeline_zh_en_large_retrieval.sh

echo "ja en  small retreival"
./pipeline/url_attr/pipeline_ja_en_small_retrieval.sh
echo "ja en  large retreival"
./pipeline/url_attr/pipeline_ja_en_large_retrieval.sh