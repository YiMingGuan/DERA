import numpy as np
from aligncraft.models.retrievalea.post_ot import EAData, GWEA
import argparse
import time
from aligncraft.models.retrievalea.retrieval import Retrieval
import argparse
import yaml
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--kg_abs_path', type=str, help='The absolute path of the KG file')
    parser.add_argument('--trained', action='store_true', help='Whether to use the trained model')
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--gw_ss", type=float, default=0.01)
    parser.add_argument("--gw_iter", type=int, default=2000)
    args = parser.parse_args()
    return args

def main(args):
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    print(config)
    retriever = Retrieval(config, args.trained)
    ent_emb = retriever.get_embs_ordered_by_id()
    logging.info(f"Successfully encoding {config['dataset']['type']}-{config['dataset']['kg1']}-{config['dataset']['kg2']}")
    data = EAData(retriever.eadata)
    gwea = GWEA(data=data, args=args, ent_emb=ent_emb, kg_abs_path=args.kg_abs_path, trained=args.trained)
    time_st = time.time()
    thre = 0.5/gwea.n
    gwea.cal_cost_st(w_homo=0, w_rel=0)  # stage 1: Semantic Alignment
    X = gwea.ot_align()
    for ii in range(args.epochs):
        print('iteration: {}, threshold: {}'.format(ii, thre))
        gwea.update_anchor(X, thre)
        gwea.rel_align(emb_w=1, seed_w=1)
        gwea.cal_cost_st()  # stage 2: Multi-view OT Alignment
        X = gwea.ot_align()
    gwea.update_anchor(X, thre)
    """
    if args.gw_iter > 0:
        X = gwea.gw_align(X, lr=args.gw_ss, iter=args.gw_iter)  # stage 3: Gromov-Wasserstein Refinement
    time_ed = time.time()
    a1, a10, mrr = test_align(X, gwea.test_pair)
    p, r, f1 = gwea.update_anchor(X, 1e-5)

    with open('result.txt', 'a+') as f:
        f.write('Dataset: {}; use_attr: {}; use_name: {}; use_trans: {}; use_stru: {}; use_rel: {}; GW_ss: {}; GW_iter: {} \n'.format(
            args.dataset_id, args.use_attr, args.use_name, args.use_trans, args.use_stru, args.use_rel, args.gw_ss, gwea.iters))
        f.write('{:.3f}\t{:.3f}\t{:.3f}\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(a1,a10,mrr,time_ed-time_st,p,r,f1))
    """
if __name__ == '__main__':
    args = get_args()
    main(args)