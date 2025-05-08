import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import numpy as np
from aligncraft.models.retrievalea.retrieval import Retrieval
import argparse
import yaml
import logging
import random
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path_1', type=str)
    parser.add_argument('--config_path_2', type=str)
    parser.add_argument('--trained_1', action='store_true', help='Whether to use the trained model')
    parser.add_argument('--trained_2', action='store_true', help='Whether to use the trained model')
    parser.add_argument('--key_pairs_num', type=int, default=4, help='Number of key entity pairs to visualize')
    return parser.parse_args()

def umap_process(embeddings_source, embeddings_target, key_entity_pairs, save_path):
    # 假设 embeddings_source 和 embeddings_target 分别是来源于source KG和target KG的实体的高维embedding

    # 选取关键实体对的索引（示例）

    # 选择t-SNE或UMAP进行降维
    # 使用TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(np.vstack((embeddings_source, embeddings_target)))

    # 或者使用UMAP
    # reducer = umap.UMAP()
    # embeddings_2d = reducer.fit_transform(np.vstack((embeddings_source, embeddings_target)))

    # 分开source和target的降维结果
    embeddings_2d_source = embeddings_2d[:10000, :]
    embeddings_2d_target = embeddings_2d[10000:, :]

    # 绘图
    fig, ax = plt.subplots()
    # 画出所有实体点
    ax.scatter(embeddings_2d_source[:, 0], embeddings_2d_source[:, 1], c='blue', label='Source KG', alpha=0.5, s=10)
    ax.scatter(embeddings_2d_target[:, 0], embeddings_2d_target[:, 1], c='red', label='Target KG', alpha=0.5, s=10)

    # 突出表示关键实体对
    for source_idx, target_idx in key_entity_pairs:
        ax.scatter(embeddings_2d_source[source_idx, 0], embeddings_2d_source[source_idx, 1], c='blue', edgecolors='black', s=100)
        ax.scatter(embeddings_2d_target[target_idx, 0], embeddings_2d_target[target_idx, 1], c='red', edgecolors='black', s=100)
        ax.plot([embeddings_2d_source[source_idx, 0], embeddings_2d_target[target_idx, 0]],
                [embeddings_2d_source[source_idx, 1], embeddings_2d_target[target_idx, 1]], 
                c='green') # 用绿色线连接关键实体对

    ax.legend()
    plt.savefig(f"{save_path}.svg", format='svg')
    # plt.show()

def source_and_target(emb, eadata):
    source_emb = []
    target_emb = []
    key_entity_pairs = []
    for i, (source_id, target_id) in enumerate(list(eadata.test_pairs.items())):
        source_emb.append(emb[source_id])
        target_emb.append(emb[target_id])
        key_entity_pairs.append((i, i))
    return np.array(source_emb), np.array(target_emb), key_entity_pairs

if __name__ == '__main__':
    args = get_args()
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    config_1 = yaml.load(open(args.config_path_1, 'r'), Loader=yaml.FullLoader)
    config_2 = yaml.load(open(args.config_path_2, 'r'), Loader=yaml.FullLoader)

    retriever_1 = Retrieval(config_1, args.trained_1)
    retriever_2 = Retrieval(config_2, args.trained_2)
    emb_1 = retriever_1.get_embs_ordered_by_id()
    emb_2 = retriever_2.get_embs_ordered_by_id()

    source_1, target_1, key_entity_pairs_1 = source_and_target(emb_1, retriever_1.eadata)
    source_2, target_2, key_entity_pairs_2 = source_and_target(emb_2, retriever_2.eadata)

    key_pairs = random.choices(key_entity_pairs_1, k=args.key_pairs_num)

    umap_process(source_1, target_1, key_pairs, f'source_target_1_{args.trained_1}')
    umap_process(source_2, target_2, key_pairs, f'source_target_2_{args.trained_2}')