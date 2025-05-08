from aligncraft.models.retrievalea.rerank import Rerank
import argparse
import yaml
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--retriever_trained', action='store_true', help='Whether to use the trained retrieval model')
    parser.add_argument('--reranker_trained', action='store_true', help='Whether to use the trained reranker model')
    return parser.parse_args()

def main(config, retriever_trained, reranker_trained):
    rerank = Rerank(config, retriever_trained, reranker_trained)
    rerank.run()
    logging.info(f"Successfully rerank {config['dataset']['type']}-{config['dataset']['kg1']}-{config['dataset']['kg2']}")
if __name__ == "__main__":
    args = get_args()
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    main(config, args.retriever_trained, args.reranker_trained)
