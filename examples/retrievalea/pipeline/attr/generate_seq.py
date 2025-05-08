from aligncraft.models.retrievalea.entity2sequence import Entity2Sequence
import argparse
import yaml
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    return parser.parse_args()

def main(config):
    e2s = Entity2Sequence(config)
    e2s.run()
    logging.info(f"Successfully generated sequences for {config['dataset']['type']}-{config['dataset']['kg1']}-{config['dataset']['kg2']}")

if __name__ == "__main__":
    args = get_args()
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    main(config)