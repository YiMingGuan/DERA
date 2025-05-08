from aligncraft.models.retrievalea.retrieval import Retrieval
import argparse
import yaml
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--trained', action='store_true', help='Whether to use the trained model')
    return parser.parse_args()

def main(config, trained):
    retriever = Retrieval(config, trained)
    retriever.save_embs_ordered_by_id()
    logging.info(f"Successfully encoding {config['dataset']['type']}-{config['dataset']['kg1']}-{config['dataset']['kg2']}")

if __name__ == '__main__':
    args = get_args()
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    main(config, args.trained)
