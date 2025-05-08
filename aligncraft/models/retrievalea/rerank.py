from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification
from .utils import get_trained_rerank_model
from .retrieval import Retrieval
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

class RankingModel:
    def __init__(self, model_path: str, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = device
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        self.model.eval()
        self.model.to(self.device)

    def rank(self, pairs: List[List[str]], batch_size=4096, **params):
        try:
            processed_pairs = self.preprocess(pairs)
            scores = []
            with torch.no_grad():
                # for i in range(0, len(pairs), batch_size):
                for i in tqdm(range(0, len(processed_pairs), batch_size), desc="Reranking"):
                    batch = processed_pairs[i : i + batch_size]
                    inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device) 
                    batch_scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
                    scores.extend(batch_scores.tolist())
        except Exception as e:
            logger.error(f"Reranker failed to rank the pairs: {e}")
            return False, str(e)
        return True, scores
    
    def preprocess(self, pairs: List[List[str]]):
        processed_pairs = []
        max_len = 512 - 3
        for one_pair in pairs:
            qry_encoding, doc_encoding = one_pair[0], one_pair[1]
            qry_tokens = self.tokenizer.tokenize(qry_encoding)
            doc_tokens = self.tokenizer.tokenize(doc_encoding)
            
            # 检查总长度是否超过max_len
            total_length = len(qry_tokens) + len(doc_tokens)
            if total_length > max_len:
                # 需要截断
                # 确定每部分应该占据的token数量
                qry_length = int(len(qry_tokens) / total_length * max_len)
                doc_length = max_len - qry_length  # 确保总和为max_len
                
                # 截断
                qry_tokens = qry_tokens[:qry_length]
                doc_tokens = doc_tokens[:doc_length]
            processed_pairs.append([
                self.tokenizer.convert_tokens_to_string(qry_tokens),
                self.tokenizer.convert_tokens_to_string(doc_tokens)
            ])
        return processed_pairs 
    

class Rerank:
    def __init__(self, config, retriever_trained, reranker_trained):
        self.config = config
        self.retriever_trained = retriever_trained
        self.reranker_trained = reranker_trained
        self.retriever = Retrieval(config, retriever_trained)

    def load_reranker(self):
        path = self.config['rerank'].get('model_path', None) if self.reranker_trained is False else get_trained_rerank_model(self.config)
        if path is None:
            return "Reranker not loaded"
        self.reranker = RankingModel(path)        
        return "Reranker loaded"
    
    def plot_similarity_curves(self, scores, hits_at_1):
        """
        绘制相似度曲线。
        
        参数:
        scores (numpy array): 一个形状为 (n, m) 的二维数组，其中每一行表示一个查询与其他文档的相似度。
        hits_at_1 (list or numpy array): 一个长度为 n 的布尔型列表，指示每个查询的hits@1是否命中。
        """
        save_path = '/public/home/chenxuan/aligncraft/examples/retrievalea/similar_curves_new'
        n = scores.shape[0]

        # 绘制命中的曲线
        plt.figure(figsize=(10, 6))
        for i in range(n):
            if hits_at_1[i]:
                plt.plot(scores[i], color='blue', alpha=0.3)  # 增加透明度

        plt.title('Similarity Curves for Hits')
        plt.xlabel('Documents')
        plt.ylabel('Similarity')
        plt.savefig(f"{save_path}_hits.png")

            # 绘制未命中的曲线
        plt.figure(figsize=(10, 6))
        for i in range(n):
            if not hits_at_1[i]:
                plt.plot(scores[i], color='red', linestyle='--')  # 使用虚线

        plt.title('Similarity Curves for Misses')
        plt.xlabel('Documents')
        plt.ylabel('Similarity')
        plt.savefig(f"{save_path}_misses.png")

    def plot_difference_curves(self, scores, hits_at_1):
        n = scores.shape[0]
        save_path = '/public/home/chenxuan/aligncraft/examples/retrievalea/similar_curves'
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        # 绘制命中的差分曲线
        ax1.set_title('Hit Difference in Similarity Scores')
        for i in range(n):
            if hits_at_1[i]:
                diffs = -np.diff(scores[i])
                ax1.plot(diffs, color='blue', alpha=0.5)

        ax1.set_xlabel('Document Transition')
        ax1.set_ylabel('Difference in Similarity')

        # 绘制未命中的差分曲线
        ax2.set_title('Miss Difference in Similarity Scores')
        for i in range(n):
            if not hits_at_1[i]:
                diffs = np.diff(scores[i])
                ax2.plot(diffs, color='red', alpha=0.5)

        ax2.set_xlabel('Document Transition')
        ax2.set_ylabel('Difference in Similarity')

        plt.tight_layout()
        plt.savefig(f"{save_path}_difference.png")

    def find_cutoff_point(self, scores):
        diffs = np.diff(scores, axis=1)
        cutoff = np.argmin(diffs, axis=1) + 1
        return cutoff
    def find_cutoff_point_multi_average(self, scores, window_size=3):
        """
        使用多点平均方法找到截断点。

        :param scores: 一个形状为(n, m)的二维numpy数组，每一行代表一个查询与其他文档的相似度。
        :param window_size: 滑动窗口的大小。
        :return: 一个长度为n的列表，包含每个查询的截断点。
        """
        cutoff_points = []
        for score in scores:
            # 初始化最大平均下降速率和对应的截断点
            max_avg_drop = 0
            cutoff = len(score) - 1  # 如果没有找到合适的截断点，默认为最后一个元素

            # 计算窗口内的平均下降速率
            for i in range(len(score) - window_size):
                window = score[i:i + window_size]
                avg_drop = np.mean(np.diff(window))
                if avg_drop < max_avg_drop:
                    max_avg_drop = avg_drop
                    cutoff = i + window_size - 1  # 截断点设为窗口末尾

            cutoff_points.append(cutoff)
        return cutoff_points
    def test(self, scores, candidates, kg1_sequence, kg2_sequence, rerank_topk=50):
        if hasattr(self, 'reranker') is False:
            self.load_reranker()
        np_candidates = np.array(candidates)
        pairs = []
        hits_at_1_bool = []
        unrecalled_num = 0
        unrecalled_mrr = 0
        cnt = 100
        for i, candidate_row in enumerate(np_candidates):
            if candidate_row[0] == i:
                hits_at_1_bool.append(True)
            else:
                hits_at_1_bool.append(False)
            if i not in candidate_row[: rerank_topk]:
                unrecalled_num += 1
                unrecalled_mrr += (1 / (np.where(candidate_row == i)[0] + 1))
                """
                logger.info(f"Unrecalled: {kg1_sequence[i]}")
                logger.info(f"Unrecalled: {kg2_sequence[candidate_row[0]]}")
                logger.info(f"The correct candidate: {kg2_sequence[i]}")
                logger.info(f"The position of the correct pair: {np.where(candidate_row == i)[0] + 1}")
                logger.info(f"The score of the correct pair: {scores[i][np.where(candidate_row == i)[0]]}")
                logger.info(f"The top {rerank_topk} scores: {scores[i][:rerank_topk]}")
                logger.info("-----------------------------------")
                """
            else:
                
                tmp_pairs = []
                for j in range(rerank_topk):
                    if candidate_row[j] == i:
                        tmp_pairs.insert(0, [kg1_sequence[i], kg2_sequence[candidate_row[j]]])
                    else:
                        tmp_pairs.append([kg1_sequence[i], kg2_sequence[candidate_row[j]]])
                pairs.extend(tmp_pairs)
                cnt += 1
                if cnt <= 100:
                    logger.info(f"Recalled: {kg1_sequence[i]}")
                    logger.info(f"Recalled: {kg2_sequence[candidate_row[0]]}")
                    logger.info(f"The position of the correct pair: {np.where(candidate_row == i)[0] + 1}")
                    logger.info(f"The score of the correct pair: {scores[i][np.where(candidate_row == i)[0]]}")
                    logger.info(f"The top {rerank_topk} scores: {scores[i][:rerank_topk]}")
                    logger.info("-----------------------------------")
        
        if unrecalled_num == 0:
            unrecalled_mrr = 0
        else:
            unrecalled_mrr /= unrecalled_num
        recalled_num = len(np_candidates) - unrecalled_num
        self.plot_similarity_curves(np.array(scores), np.array(hits_at_1_bool))
        self.plot_difference_curves(np.array(scores), np.array(hits_at_1_bool))
        logger.info(f"unrecalled_num: {unrecalled_num}, unrecalled_mrr: {unrecalled_mrr}")
        logger.info(f"The number of pairs to rerank: {len(pairs)}")
        rerank_batch_size = self.config['rerank'].get('rerank_batch_size', 4096)
        success, rerank_scores = self.reranker.rank(pairs, batch_size=rerank_batch_size)
        if success is False:
            logger.error("Reranker failed to rank the pairs")

        np_rerank_scores = np.array(rerank_scores)
        logger.info(f"Rerank scores shape: {np_rerank_scores.shape}")
       
        np_rerank_scores = np_rerank_scores.reshape(-1, rerank_topk)

        logger.info(f"The case of rerank score {np_rerank_scores[0]}")
        sorted_indices = np_rerank_scores.argsort(axis=1)[:, ::-1]

        zero_indices = sorted_indices.argsort(axis=1)[:, 0]

        def hits_at_k(ranks, k):
            return (ranks < k).sum() / (recalled_num + unrecalled_num)

        def mrr(ranks):
            recalled_mrr = (1 / (ranks + 1)).mean()
            return (recalled_mrr * recalled_num + unrecalled_mrr * unrecalled_num) / (recalled_num + unrecalled_num)
    
        hits_at_1 = hits_at_k(zero_indices, 1)
        hits_at_3 = hits_at_k(zero_indices, 3)
        hits_at_5 = hits_at_k(zero_indices, 5)
        hits_at_6 = hits_at_k(zero_indices, 6)
        hits_at_7 = hits_at_k(zero_indices, 7)
        hits_at_8 = hits_at_k(zero_indices, 8)
        hits_at_9 = hits_at_k(zero_indices, 9)
        hits_at_10 = hits_at_k(zero_indices, 10)
        mrr_score = mrr(zero_indices)

        metrics = {
            "hits@1": hits_at_1,
            "hits@3": hits_at_3,
            "hits@5": hits_at_5,
            "hits@6": hits_at_6,
            "hits@7": hits_at_7,
            "hits@8": hits_at_8,
            "hits@9": hits_at_9,
            "hits@10": hits_at_10,
            "mrr": mrr_score
        }
        logger.info(f"Rerank result: {metrics}")
        return metrics

    def run(self):
        retrieval_output = self.retriever.run()
        retrieval_scores = retrieval_output['retrieval_scores']
        retrieval_candidates = retrieval_output['retrieval_candidates']
        kg1_sequence = retrieval_output['kg1_sequence']
        kg2_sequence = retrieval_output['kg2_sequence']
        if hasattr(self.retriever, 'retriever'):
            self.retriever.retriever.model.to('cpu')
            logger.info("To ease the need for caching, move unused retrievers to the cpu.")
        self.test(
            retrieval_scores, 
            retrieval_candidates, 
            kg1_sequence, 
            kg2_sequence, 
            rerank_topk=self.config['rerank'].get('rerank_topk', 50)
        )

    