from aligncraft.models.retrievalea.rerank_sft import RerankSFT
import argparse
import yaml
import logging

logger = logging.getLogger(__name__)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    return parser.parse_args()

def main(config):
    trainer = RerankSFT(config)
    trainer.run()

if __name__ == "__main__":
    args = get_args()
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    main(config)