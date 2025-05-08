from aligncraft.models.retrievalea.hn_mine import HN_Mine
import argparse
import yaml
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--step', type=str)
    return parser.parse_args()

def main(config, step):
    mine = HN_Mine(config, step)
    mine.run()
    logging.info(f"Successfully mined hard negatives for {config['dataset']['type']}-{config['dataset']['kg1']}-{config['dataset']['kg2']} in {step} step.")
if __name__ == "__main__":
    args = get_args()
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    main(config, args.step)