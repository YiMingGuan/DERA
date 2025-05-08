

# echo "Running ablation and contrast experiments for without_attr"

# ./scripts/ablation_and_contrast/without_attr.sh

# echo "Running ablation and contrast experiments for without_rel"

# ./scripts/ablation_and_contrast/without_rel.sh

# echo "Running ablation and contrast experiments for without_attr_rel.sh"

# ./scripts/ablation_and_contrast/without_attr_rel.sh


# echo "Running ablation and contrast experiments for without_bootstraping_and_only_attr.sh"
# ./scripts/ablation_and_contrast/without_bootstraping_and_only_attr.sh

# echo "Running ablation and contrast experiments for without_bootstraping_and_only_name.sh"
# ./scripts/ablation_and_contrast/without_bootstraping_and_only_name.sh

# echo "Running ablation and contrast experiments for without_bootstraping_and_only_rel.sh"
# ./scripts/ablation_and_contrast/without_bootstraping_and_only_rel.sh

# echo "Running ablation and contrast experiments for entity_order_ablation_name.sh"
# ./scripts/ablation_and_contrast/entity_order_ablation_name.sh

echo "Running ablation and contrast experiments for entity_order_ablation_name_attr.sh"
./scripts/ablation_and_contrast/entity_order_ablation_name_attr.sh

echo "Running ablation and contrast experiments for gcnalign..."
./scripts/ablation_and_contrast/gcnalign_new.sh

echo "Running ablation and contrast experiments for attr_num_ablation.sh"
./scripts/ablation_and_contrast/attr_num_ablation.sh


echo "Running ablation and contrast experiments for rel_num_ablation.sh"
./scripts/ablation_and_contrast/rel_num_ablation.sh


echo "Running ablation and contrast experiments for dbp15k_zh_en_zero_shot_qwen7b_chat_attr_num_7_last.sh"
./scripts/ablation_and_contrast/dbp15k_zh_en_zero_shot_qwen7b_chat_attr_num_7_last.sh