from aligncraft.models.retrievalea.retrieval import Retrieval
import argparse
import yaml
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    return parser.parse_args()

def main(config):
    retriever = Retrieval(config)
    retriever.generate_retrieval_sft_data()
    logging.info(f"Successfully generated retrieval sft data for {config['dataset']['type']}-{config['dataset']['kg1']}-{config['dataset']['kg2']}")
if __name__ == "__main__":
    args = get_args()
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    main(config)
