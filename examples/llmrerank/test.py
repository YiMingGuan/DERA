from aligncraft.models.retrievalea.retrieval import Retrieval
from aligncraft.models.retrievalea.llm_reranker import LLMsReranker
import argparse
import yaml
import logging
import json
from transformers import set_seed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--trained', action='store_true', help='Whether to use the trained model')
    parser.add_argument('--topk', type=int, default=3, help='Top k candidate entities to retrieve')
    parser.add_argument('--sample_num', type=int, default=None, help='Number of samples to test')
    parser.add_argument('--test_strategy', type=str, default=None, help='Test pair sample strategy')
    parser.add_argument('--ent_embs_path', type=str, default=None, help='Path to the entity embeddings')
    parser.add_argument('--openai', action='store_true', help='Whether to use openai')
    parser.add_argument('--vllm', action='store_true', help='Whether to use vllm')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Size of tensor parallel')
    parser.add_argument('--swap_space', type=int, default=4, help='Size of swap space')
    parser.add_argument('--llm_path', type=str, default=None, help='Path to the LLM model')
    parser.add_argument('--quantization', type=str, default=None, help='Quantization of the model')
    parser.add_argument('--openai_model', type=str, default=None, help='OpenAI model name')
    parser.add_argument('--openai_key', type=str, default=None, help='OpenAI key')
    parser.add_argument('--openai_url', type=str, default=None, help='OpenAI url')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature of the generation')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top p of the generation')
    parser.add_argument('--max_tokens', type=int, default=128, help='Maximum length of the generation')
    parser.add_argument('--repetition_penalty', type=float, default=1.1, help='Repetition penalty of the generation')
    parser.add_argument('--idealTest', action='store_true', help='Whether to use ideal test')
    parser.add_argument('--bootstraping', action='store_true', help='Whether to use bootstraping')
    parser.add_argument('--bootstraping_num', type=int, default=10, help='Number of bootstraping')
    parser.add_argument('--verbose', action='store_true', help='Whether to use verbose')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--candidate_filter', action='store_true', help='Whether to use candidate filter')
    parser.add_argument('--maxUncertain', type=int, default=1, help='Maximum number of uncertain rerank')
    parser.add_argument('--bootstraping_strategy', type=str, default='shuffle', help='Bootstraping strategy')
    parser.add_argument('--max_infer_iter', type=int, default=2, help='Maximum number of inference iteration')

    """
    Name
    """
    parser.add_argument('--use_name', action='store_true', help='Whether to use name')

    """
    Attr
    """
    parser.add_argument('--use_attr', action='store_true', help='Whether to use attr')
    parser.add_argument('--attr_strategy', type=str, default='functionality', help='Attr strategy')
    parser.add_argument('--attr_max_num', type=int, default=3, help='Maximum number of attr')
    parser.add_argument('--attr_sampled', action='store_true', help='Whether to use sampled attr')
    parser.add_argument('--attr_sample_strategy', type=str, default='linspace', help='Attr sample strategy')

    """
    Rel
    """
    parser.add_argument('--use_rel', action='store_true', help='Whether to use rel')
    parser.add_argument('--rel_strategy', type=str, default='functionality', help='Rel strategy')
    parser.add_argument('--rel_max_num', type=int, default=5, help='Maximum number of rel')
    parser.add_argument('--rel_sampled', action='store_true', help='Whether to use sampled rel')
    parser.add_argument('--rel_sample_strategy', type=str, default='linspace', help='Rel sample strategy')

    """
    experiment setting
    """
    parser.add_argument('--experiment', type=str, default='all_framework', help='Experiment setting')
    parser.add_argument('--position', type=int, default=0, help='Position of the true candidate')
    return parser.parse_args()

from aligncraft.models.retrievalea.llm_reranker import RelationshipStrategy
def main(config, args):
    set_seed(args.seed)
    retriever = Retrieval(config, args.trained)
    
    candidate_url_set, candidate_id_set, score_dict, ents, entity_info_dict, test_pair_url, entities = retriever.generate_candidate_set_topk(topk=args.topk, additional_embedding=args.ent_embs_path)

    # rels = RelationshipStrategy(
    #     url2EntityInfo_dict=entity_info_dict,
    #     url2EntityObject_dict=entities,
    # )
    # func = rels.calculate_rel_functionality()
    # sorted_func = sorted(func.items(), key=lambda x: x[1], reverse=True)
    # with open('functionality.log', 'w', encoding='utf-8') as f:
        # for k, v in sorted_func:
            # f.write(f'{k}\t{v}\n')
    llmreranker = LLMsReranker(args,
                               candidate_url_set, 
                               candidate_id_set, 
                               score_dict, 
                               ents, 
                               entity_info_dict, 
                               test_pair_url, 
                               entities)
    llmreranker.inference()
    # all_analysis_json_list = llmreranker.pilot_test(sample_num=args.sample_num, strategy=args.strategy)
    # all_analysis_json_list = llmreranker.run()

    # with open(args.output_path, 'w', encoding='utf-8') as f:
        # json.dump(all_analysis_json_list, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    args = get_args()
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    main(config, args)
