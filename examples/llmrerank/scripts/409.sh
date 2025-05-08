echo "Running ablation and contrast experiments for gcnalign..."
./scripts/ablation_and_contrast/gcnalign_new.sh

echo "Running ablation and contrast experiments for entity_order_ablation_name_attr_zh_en_14b"
./scripts/ablation_and_contrast/entity_order_ablation_name_attr_zh_en_14b.sh

echo "Running ablation and contrast experiments for attr_num_ablation.sh"
./scripts/ablation_and_contrast/attr_num_ablation.sh


echo "Running ablation and contrast experiments for rel_num_ablation.sh"
./scripts/ablation_and_contrast/rel_num_ablation.sh

echo "Running ablation and contrast experiments for candidate_size_ablation"
./scripts/ablation_and_contrast/candidate_size_ablation.sh

echo "Running ablation and contrast experiments for dbp15k_zh_en_zero_shot_qwen7b_chat_attr_num_7_last.sh"
./scripts/ablation_and_contrast/dbp15k_zh_en_zero_shot_qwen7b_chat_attr_num_7_last.sh