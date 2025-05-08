import torch
import os
from ...data_loading.load_dataset import load_dataset
import yaml
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import copy
import logging
import json
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import gc
from .utils import plot_calibration_curve
from .utils import get_retrieval_data_path, save_cache_file, load_cache_file, get_trained_retrieval_model
logger = logging.getLogger(__name__)
import ot
from .utils import entity_info, get_llmrerank_data_ori_name, sim
class EmbeddingModel:
    def __init__(
        self,
        model_path: str,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = SentenceTransformer(model_path,device=device)

    def encode_multi_process(self, sentences, **params):
        try:
            pool = self.model.start_multi_process_pool(['cuda:0', 'cuda:1'])
            emb = self.model.encode_multi_process(sentences, pool, **params)
            logger.info(f"Embedding shape: {emb.shape}")
            self.model.stop_multi_process_pool(pool)
        except Exception as e:
            return False, str(e)
        return True, emb


    def encode(self, sentences, **params):
        try:
            embedding = self.model.encode(sentences, **params)
        except Exception as e:
            return False, str(e)
        return True, embedding


class Retrieval:
    def __init__(self, config, trained=False):
        self.config = config
        self.trained = trained
        self.eadata = self.load_eadata()

    def load_eadata(self):
        data_config_path = self.config['dataset']['config_path']
        data_type = self.config['dataset']['type'].upper()
        data_config = yaml.load(open(data_config_path, 'r'), Loader=yaml.FullLoader)
        eadata = load_dataset(data_type, data_config)
        return eadata

    def load_retriever(self):
        path = self.config['retrieval'].get('model_path', None) if self.trained is False else get_trained_retrieval_model(self.config)
        if self.trained is not False and self.config['retrieval'].get('retriever_suffix_path', None) is not None:
            path = os.path.join(path, self.config['retrieval']['retriever_suffix_path'])
        if path is None:
            return "Retriever not loaded"

        device = "cuda" if torch.cuda.is_available() and self.config['retrieval']['multi_gpu_encoding'] is False else "cpu"
        self.retriever = EmbeddingModel(path, device=device)
        return "Retriever loaded"

    def build_index(self, documents_emb):
        """
        Build a Faiss index for the documents using GPU if GPU is available.
        If the index is already built, load it from the cache.
        If the index is not built, build it and save it to the cache.
        If the length of the documents is more than 2048, build the index in chunks.
        """
        emb = copy.deepcopy(documents_emb)
        if self.config['faiss'].get('using_gpu', False):
            res = faiss.StandardGpuResources()
        index_type = self.config['faiss'].get('index_type', 'l2')
        if index_type == 'l2':
            tmp_index = faiss.IndexFlatL2(documents_emb.shape[1])
        elif index_type == 'cosine':
            faiss.normalize_L2(emb)
            tmp_index = faiss.IndexFlatIP(documents_emb.shape[1])
        elif index_type == 'hnsw':
            tmp_index = faiss.IndexHNSWFlat(documents_emb.shape[1], 32)
        else:
            raise ValueError(f"Index type {index_type} not supported")
        if self.config.get('using_faiss_gpu', False):
            index = faiss.index_cpu_to_gpu(res, 0, tmp_index)
        else:
            index = tmp_index
        if len(emb) > 2048:
            for i in range(0, len(emb), 2048):
                index.add(emb[i:i+2048])
        else:
            index.add(emb)
        return index

    def retrieval(self, queries, documents):
        queries_emb = self.get_retrieval_embeddings(queries)
        if self.config['faiss'].get('index_type', 'l2') == 'cosine':
            faiss.normalize_L2(queries_emb)
        documents_emb = self.get_retrieval_embeddings(documents)
        index = self.build_index(documents_emb)
        scores, candidates = self.search(queries_emb, index)
        return scores, candidates
    
    def get_retrieval_embeddings(self, sentences):
        if hasattr(self, 'retriever') is False:
            load_retriever_info = self.load_retriever()
            logger.info(load_retriever_info)
        multi_gpu_encoding = self.config['retrieval'].get('multi_gpu_encoding', False)
        if multi_gpu_encoding:
            state, embeddings = self.retriever.encode_multi_process(
                sentences=sentences,
                batch_size=self.config['retrieval'].get('retrieval_batch_size', 32)
            )
        else:
            state, embeddings = self.retriever.encode(
                sentences=sentences,
                batch_size=self.config['retrieval'].get('retrieval_batch_size', 32)
            )
        if not state:
            raise ValueError("Error in getting retrieval embeddings")
        return embeddings

    def search(self, queries_emb, index):
        retrieval_topk = self.config['retrieval'].get('retrieval_topk')
        if retrieval_topk is None:
            retrieval_topk = index.ntotal
        retrieval_batch_size = self.config['retrieval'].get('retrieval_batch_size', 2048)
        candidates = []
        scores = []
        for i in range(0, len(queries_emb), retrieval_batch_size):
            batch_queries_emb = queries_emb[i:i+retrieval_batch_size]
            D, I = index.search(batch_queries_emb, retrieval_topk)
            candidates.extend(I)
            scores.extend(D)
        return scores, candidates

    def get_query_and_documents_4_test(self, kg1_sequence, kg2_sequence):
        retrieval_direction = self.config['retrieval'].get('retrieval_direction', 'l2r')
        if retrieval_direction == 'l2r':
            queries = kg1_sequence
            documents = kg2_sequence
            url2q = {line['url']: line['sequence'] for line in queries}
            url2d = {line['url']: line['sequence'] for line in documents}
            test_pairs = [(id1, id2) for (id1, id2) in self.eadata.test_pairs.items()]
            q = [url2q[self.eadata.kg1.ent_ids[id1]] for (id1, id2) in test_pairs]
            d = [url2d[self.eadata.kg2.ent_ids[id2]] for (id1, id2) in test_pairs]
        elif retrieval_direction == 'r2l':
            queries = kg2_sequence
            documents = kg1_sequence
            url2q = {line['url']: line['sequence'] for line in queries}
            url2d = {line['url']: line['sequence'] for line in documents}
            test_pairs = [(id2, id1) for (id1, id2) in self.eadata.test_pairs.items()]
            q = [url2q[self.kg2.ent_ids[id2]] for (id1, id2) in test_pairs]
            d = [url2d[self.kg1.ent_ids[id1]] for (id1, id2) in test_pairs]
        else:
            raise ValueError(f"Retrieval direction {retrieval_direction} not supported")
        return q, d

    def metrics_retrieval(self, scores, candidates, threshold=None, step='untrained'):
        """
        Hits@1, Hits@10, Hits@50, MRR
        """
        assert step in ['untrained', 'trained'], "Step should be either untrained or trained."
        def hitsk(rank, k):
            assert len(rank[0]) >= k
            ids = np.array(range(len(rank)))
            ids_expanded = ids[:, np.newaxis]
            matches = rank == ids_expanded
            indices = np.argmax(matches, axis=1)
            return np.sum(indices < k) / len(rank)
        
        def mrr(rank):
            ids = np.array(range(len(rank)))
            ids_expanded = ids[:, np.newaxis]
            matches = rank == ids_expanded
            indices = np.argmax(matches, axis=1)
            return np.mean(1 / (indices + 1))

        def calibration(rank, rank_score, n_bins=20, pic_path=None):
            ids = np.array(range(len(rank)))
            ids_expanded = ids[:, np.newaxis]
            matches = rank == ids_expanded
            y_labels = matches[:, 0] * 1

            y_prob = np.array(rank_score)[:, 0]
            plot_calibration_curve(y_labels, y_prob, n_bins=n_bins, pic_path=pic_path)
        
        def OT(score, candidates):
            """
            sinkhorn
            """
            # Rank the score according to the candidates
            # score: n * m
            # candidates: n * m
            score = np.array(score)
            candidates = np.array(candidates)
            permunate = np.argsort(candidates, axis=1)
            
            C = np.zeros((len(candidates), len(candidates)))
            for i in range(len(candidates)):
                C[i] = score[i][permunate[i]]

            n, m = score.shape
            mu = np.ones(n) / n
            nu = np.ones(m) / m
            P = ot.sinkhorn(mu, nu, C, reg=0.01)

            ot_candidate = np.argsort(P, axis=1)
            return ot_candidate


        candidates = np.array(candidates)
        hits_at_1 = hitsk(candidates, 1)
        hits_at_10 = hitsk(candidates, 10)
        hits_at_50 = hitsk(candidates, 50)
        hits_at_100 = hitsk(candidates, 100)
        hits_at_300 = hitsk(candidates, 300)
        mrr_score = mrr(candidates)

        """
        eval after ot
        """
        ot_candidate = OT(scores, candidates)
        hits_at_1_ot = hitsk(ot_candidate, 1)
        hits_at_10_ot = hitsk(ot_candidate, 10)
        hits_at_50_ot = hitsk(ot_candidate, 50)
        mrr_score_ot = mrr(ot_candidate)


        
        """
        calibration
        y_true: 1 if the query is in the top 1, 0 otherwise
        y_prob: the retrieval score
        """
        # calibration(candidates, scores, n_bins=20, pic_path=f"/public/home/chenxuan/aligncraft/examples/retrievalea/calibration_curve_{self.config['dataset']['type']}-{self.config['dataset']['kg1']}-{self.config['dataset']['kg2']}_{step}")

        return {
            "hits_at_1": hits_at_1,
            "hits_at_10": hits_at_10,
            "hits_at_50": hits_at_50,
            "hits_at_100": hits_at_100,
            "hits_at_300": hits_at_300,
            "mrr": mrr_score,
            "hits_at_1_ot": hits_at_1_ot,
            "hits_at_10_ot": hits_at_10_ot,
            "hits_at_50_ot": hits_at_50_ot,
            "mrr_ot": mrr_score_ot,
        }

    def get_sequence(self):
        kg1_sequence = self.generate_sequence(
            dataset_type=self.config['dataset']['type'],
            kg_name=self.config['dataset']['kg1'],
            entity_info_method=self.config['entity_info_method'],
        )
        kg2_sequence = self.generate_sequence(
            dataset_type=self.config['dataset']['type'],
            kg_name=self.config['dataset']['kg2'],
            entity_info_method=self.config['entity_info_method'],
        )
        return kg1_sequence, kg2_sequence

    def terminate_vllm(self):
        if self.config.get('using_vllm', False) and hasattr(self, 'llms'):
            destroy_model_parallel()
            del self.llms
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()
            logger.info("VLLM terminated and memory released")
        else:
            logger.info("VLLM is not loaded, no need to terminate")

    def get_query_and_documents_4_train(self, kg1_sequence, kg2_sequence):
        retrieval_direction = self.config['retrieval'].get('retrieval_direction', 'l2r')
        if retrieval_direction == 'l2r':
            queries = kg1_sequence
            documents = kg2_sequence
            url2q = {line['url']: line['sequence'] for line in queries}
            url2d = {line['url']: line['sequence'] for line in documents}
            train_pairs = [(id1, id2) for (id1, id2) in self.eadata.train_pairs.items()]
            q = [url2q[self.eadata.kg1.ent_ids[id1]] for (id1, id2) in train_pairs]
            d = [url2d[self.eadata.kg2.ent_ids[id2]] for (id1, id2) in train_pairs]
        elif retrieval_direction == 'r2l':
            queries = kg2_sequence
            documents = kg1_sequence
            url2q = {line['url']: line['sequence'] for line in queries}
            url2d = {line['url']: line['sequence'] for line in documents}
            train_pairs = [(id2, id1) for (id1, id2) in self.eadata.train_pairs.items()]
            q = [url2q[self.kg2.ent_ids[id2]] for (id1, id2) in train_pairs]
            d = [url2d[self.kg1.ent_ids[id1]] for (id1, id2) in train_pairs]
        else:
            raise ValueError(f"Retrieval direction {retrieval_direction} not supported")
        return q, d

    def write_retrieval_sft_data(self, train_pairs, other_corpus):
        retrieval_sft_data = [
            {
                "query": query,
                "pos": [pos],
                "neg": [],
            } for (query, pos) in train_pairs
        ]
        logger.info(f"Retrieval sft data: {retrieval_sft_data[0]}")
        assert len(retrieval_sft_data) > 0, "No retrieval sft data generated..."
        retrieval_sft_data[0]['neg'].extend(other_corpus)
        retrieval_sft_data_path = get_retrieval_data_path(self.config)
        with open(retrieval_sft_data_path, 'w') as f:
            for line in retrieval_sft_data:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        logger.info(f"Retrieval sft data saved to {retrieval_sft_data_path}")
    
    def generate_sequence(self, dataset_type, kg_name, entity_info_method):
        """
        Generate a sequence for each entity in the KG using vllm.
        """

        using_llms = self.config.get('using_llms', False)
        method = entity_info_method if using_llms else "None"
        use_url = self.config['entity_info'].get('use_url', False)
        use_name = self.config['entity_info'].get('use_name', False)
        use_translated_url = self.config['entity_info'].get('use_translated_url', False)
        use_relationship = self.config['entity_info'].get('use_relationship', False)
        use_attributes = self.config['entity_info'].get('use_attributes', False)
        use_inverse = self.config['entity_info'].get('use_inverse', False)
        neighbors_use_attr = self.config['entity_info'].get('neighbors_use_attr', False)
        head_use_attr = self.config['entity_info'].get('head_use_attr', False)
        llms_name = self.config['vllm'].get('llms_name', None)
        if using_llms is False:
            llms_name = None

        prompt_type = self.config.get('prompt_type', None)
        if using_llms is False:
            prompt_type = None
        kg1_name = self.config['dataset']['kg1']
        kg2_name = self.config['dataset']['kg2']

        kg_cache_sequence_file_name = f"{dataset_type}-{kg1_name}-{kg2_name}_{kg_name}_using_llms{using_llms}_llms_name{llms_name}_prompt_type{prompt_type}_sequence_{method}_use_url_{use_url}_use_name_{use_name}_use_translated_url_{use_translated_url}_use_relationship_{use_relationship}_use_attributes_{use_attributes}_use_inverse_{use_inverse}_neighbors_use_attr_{neighbors_use_attr}_head_use_attr_{head_use_attr}"
        kg_cache_sequence = load_cache_file(kg_cache_sequence_file_name)
        assert kg_cache_sequence is not None and len(kg_cache_sequence) > 0
        return kg_cache_sequence

    def run(self):
        logger.info(self.load_retriever())
        kg1_sequence, kg2_sequence = self.get_sequence()
        test_queries, test_documents = self.get_query_and_documents_4_test(
            kg1_sequence=kg1_sequence,
            kg2_sequence=kg2_sequence,
        )
        retrieval_scores, retrieval_candidates = self.retrieval(test_queries, test_documents)
        test_result = self.metrics_retrieval(scores=retrieval_scores, candidates=retrieval_candidates, step='untrained' if self.trained is False or self.trained is None else 'trained')
        logger.info(f"Retrieval test result: {test_result}")
        return {
            "retrieval_scores": retrieval_scores,
            "retrieval_candidates": retrieval_candidates,
            "kg1_sequence": test_queries,
            "kg2_sequence": test_documents,
        }

    def get_kgs_all_url2sequence(self):
        kg1_sequence, kg2_sequence = self.get_sequence()
        kg1_u2s = {line['url']: line['sequence'] for line in kg1_sequence}
        kg2_u2s = {line['url']: line['sequence'] for line in kg2_sequence}
        id2url = {}
        ent_str_list = {}
        for ent_id in self.eadata.kg1.ent_ids:
            id2url[ent_id] = self.eadata.kg1.ent_ids[ent_id]
            url = self.eadata.kg1.ent_ids[ent_id]
            if url in kg1_u2s:
                ent_str_list[ent_id] = kg1_u2s[url]
            else:
                tmp_str = self.eadata.kg1.entities[url].parse_url()
                ent_str_list[ent_id] = tmp_str
        
        for ent_id in self.eadata.kg2.ent_ids:
            id2url[ent_id] = self.eadata.kg2.ent_ids[ent_id]
            url = self.eadata.kg2.ent_ids[ent_id]
            if url in kg2_u2s:
                ent_str_list[ent_id] = kg2_u2s[url]
            else:
                tmp_str = self.eadata.kg2.entities[url].parse_url()
                ent_str_list[ent_id] = tmp_str

        assert len(ent_str_list) == len(self.eadata.kg1.ent_ids) + len(self.eadata.kg2.ent_ids)
        assert len(id2url) == len(self.eadata.kg1.ent_ids) + len(self.eadata.kg2.ent_ids)    
        # ents is a list of sequence
        ents = [ent_str_list[ent_id] for ent_id in range(len(ent_str_list))]
        urlents = {}
        for ent_id in range(len(ent_str_list)):
            urlents[id2url[ent_id]] = ents[ent_id]
        return urlents

    def get_kgs_all_url2entitiesinfo(self):
        entity_info_list_kg1 = {}
        for kg1_entity_info_one_id in self.eadata.kg1.ent_ids:
            entity_info_list_kg1[self.eadata.kg1.ent_ids[kg1_entity_info_one_id]] = entity_info(
               entity=self.eadata.kg1.entities[self.eadata.kg1.ent_ids[kg1_entity_info_one_id]],
               method=self.config['entity_info_method'],
               **self.config['entity_info']
            ) 
        entity_info_list_kg2 = {}
        for kg2_entity_info_one_id in self.eadata.kg2.ent_ids:
            entity_info_list_kg2[self.eadata.kg2.ent_ids[kg2_entity_info_one_id]] = entity_info(
               entity=self.eadata.kg2.entities[self.eadata.kg2.ent_ids[kg2_entity_info_one_id]],
               method=self.config['entity_info_method'],
               **self.config['entity_info']
            )
        entity_info_dict = {**entity_info_list_kg1, **entity_info_list_kg2}
        return entity_info_dict
    
    def get_test_pair_url2url_dict(self):
        retrieval_direction = self.config['retrieval'].get('retrieval_direction', 'l2r')
        if retrieval_direction == 'l2r':
            test_pairs = [(id1, id2) for (id1, id2) in self.eadata.test_pairs.items()]
        elif retrieval_direction == 'r2l':
            test_pairs = [(id2, id1) for (id1, id2) in self.eadata.test_pairs.items()]
        else:
            raise ValueError(f"Retrieval direction {retrieval_direction} not supported")
        test_pair_url2url = {}
        for (id1, id2) in test_pairs:
            if retrieval_direction == 'l2r':
                test_pair_url2url[self.eadata.kg1.ent_ids[id1]] = self.eadata.kg2.ent_ids[id2]
            else:
                test_pair_url2url[self.eadata.kg2.ent_ids[id1]] = self.eadata.kg1.ent_ids[id2]
        return test_pair_url2url


    def generate_candidate_set_topk(self, topk=10, additional_embedding=None):
        """
        generate candidate set
        return:
            1. candidate set url dict: dict, key: entity url, value: list of entity url
            2. candidate set id dict: dict, key: entity id, value: list of entity id
            3. score dict: dict, key: entity url, value: list of score
            4. sequence list: list of sequence
            5. entity info dict: dict, key: entity url, value: entity info
            6. test pair ground truth url: list of tuple
            7. entities: list of entity

        """
        # ----------------- get sequence -----------------
        kg1_sequence, kg2_sequence = self.get_sequence()
        kg1_u2s = {line['url']: line['sequence'] for line in kg1_sequence}
        kg2_u2s = {line['url']: line['sequence'] for line in kg2_sequence}
        ent_str_list = {}
        for ent_id in self.eadata.kg1.ent_ids:
            url = self.eadata.kg1.ent_ids[ent_id]
            if url in kg1_u2s:
                ent_str_list[ent_id] = kg1_u2s[url]
            else:
                tmp_str = self.eadata.kg1.entities[url].parse_url()
                ent_str_list[ent_id] = tmp_str
        
        for ent_id in self.eadata.kg2.ent_ids:
            url = self.eadata.kg2.ent_ids[ent_id]
            if url in kg2_u2s:
                ent_str_list[ent_id] = kg2_u2s[url]
            else:
                tmp_str = self.eadata.kg2.entities[url].parse_url()
                ent_str_list[ent_id] = tmp_str

        assert len(ent_str_list) == len(self.eadata.kg1.ent_ids) + len(self.eadata.kg2.ent_ids)
        
        # ents is a list of sequence
        ents = [ent_str_list[ent_id] for ent_id in range(len(ent_str_list))]

        # ----------------- get test query and document -----------------
        retrieval_direction = self.config['retrieval'].get('retrieval_direction', 'l2r')

        if retrieval_direction == 'l2r':
            test_pairs = [(id1, id2) for (id1, id2) in self.eadata.test_pairs.items()]
        elif retrieval_direction == 'r2l':
            test_pairs = [(id2, id1) for (id1, id2) in self.eadata.test_pairs.items()]
        else:
            raise ValueError(f"Retrieval direction {retrieval_direction} not supported")
        q = [ents[id1] for (id1, id2) in test_pairs]
        d = [ents[id2] for (id1, id2) in test_pairs]

        # ----------------- get retrieval embeddings -----------------
        if additional_embedding is None:
            load_llmrerank_input = self.config.get('llmrerank', {}).get('rerank_input', {}).get('load_cache', False)
            load_cache_success = False
            llmrerank_data_name = get_llmrerank_data_ori_name(self.config)
            if load_llmrerank_input:
                all_save = load_cache_file(llmrerank_data_name)
                if all_save is not None:
                    logger.info(f"Load llmrerank data from {llmrerank_data_name}")
                    retrieval_scores = all_save['retrieval_scores']
                    retrieval_candidates = all_save['retrieval_candidates']
                    load_cache_success = True
            
            if load_cache_success is False:
                logger.info(f"Load llmrerank data from {llmrerank_data_name} failed. Start to generate.")
                if hasattr(self, 'retriever') is False:
                    load_retriever_info = self.load_retriever()
                    logger.info(load_retriever_info)
                retrieval_scores, retrieval_candidates = self.retrieval(q, d)
        else:
            assert type(additional_embedding) == str
            logger.info(f"Load additional embedding from {additional_embedding}")
            additional_embs = np.load(additional_embedding)
            source_kg_embs = [additional_embs[id1] for (id1, id2) in test_pairs]
            target_kg_embs = [additional_embs[id2] for (id1, id2) in test_pairs]
            source_kg_embs = np.array(source_kg_embs)
            target_kg_embs = np.array(target_kg_embs)
            sim_mat = sim(source_kg_embs, target_kg_embs, metric='euclidean', normalize=True)
            sim_mat_sorted = np.argsort(sim_mat, axis=1)[:, ::-1]
            retrieval_candidates = copy.deepcopy(sim_mat_sorted)
            """
            score should be according to the candidates
            """
            retrieval_scores = np.zeros_like(sim_mat)
            for i in range(len(sim_mat)):
                retrieval_scores[i] = sim_mat[i][sim_mat_sorted[i]]
            

        
        # ----------------- get topk candidate set -----------------
        test_ent_kg1 = [id1 for (id1, id2) in test_pairs]
        test_ent_kg2 = [id2 for (id1, id2) in test_pairs]
        test_ent_kg1 = np.array(test_ent_kg1)
        test_ent_kg2 = np.array(test_ent_kg2)

        candidate_id_set = {}
        score_dict = {}
        for i, (test_id_1, test_id_2) in enumerate(test_pairs):
            candidate_id_set_test_id_1_tmp = list(retrieval_candidates[i][:topk])
            candidate_id_set[test_id_1] = list(test_ent_kg2[candidate_id_set_test_id_1_tmp])
            score_dict[test_id_1] = list(retrieval_scores[i][:topk])
        
        candidate_url_set = {}
        
        for test_id_1 in candidate_id_set:
            if retrieval_direction == 'l2r':
                candidate_url_set[self.eadata.kg1.ent_ids[test_id_1]] = [self.eadata.kg2.ent_ids[ent_id] for ent_id in candidate_id_set[test_id_1]]
            else:
                candidate_url_set[self.eadata.kg2.ent_ids[test_id_1]] = [self.eadata.kg1.ent_ids[ent_id] for ent_id in candidate_id_set[test_id_1]]
        
        
        # ------------------ get entity info ------------------
        entity_info_list_kg1 = {}
        for kg1_entity_info_one_id in self.eadata.kg1.ent_ids:
            entity_info_list_kg1[self.eadata.kg1.ent_ids[kg1_entity_info_one_id]] = entity_info(
               entity=self.eadata.kg1.entities[self.eadata.kg1.ent_ids[kg1_entity_info_one_id]],
               method=self.config['entity_info_method'],
               **self.config['entity_info']
            ) 
        entity_info_list_kg2 = {}
        for kg2_entity_info_one_id in self.eadata.kg2.ent_ids:
            entity_info_list_kg2[self.eadata.kg2.ent_ids[kg2_entity_info_one_id]] = entity_info(
               entity=self.eadata.kg2.entities[self.eadata.kg2.ent_ids[kg2_entity_info_one_id]],
               method=self.config['entity_info_method'],
               **self.config['entity_info']
            )
        entity_info_dict = {**entity_info_list_kg1, **entity_info_list_kg2}

        # ------------------ test pair groundtruth url ------------------
        test_pair_ground_truth_url = []
        for (id1, id2) in test_pairs:
            if retrieval_direction == 'l2r':
                test_pair_ground_truth_url.append((self.eadata.kg1.ent_ids[id1], self.eadata.kg2.ent_ids[id2]))
            else:
                test_pair_ground_truth_url.append((self.eadata.kg2.ent_ids[id1], self.eadata.kg1.ent_ids[id2]))
        
        # ------------------ entities ------------------
        entities = {**self.eadata.kg1.entities, **self.eadata.kg2.entities}

        all_save = {
            "retrieval_scores": retrieval_scores,
            "retrieval_candidates": retrieval_candidates,
        }

        llmrerank_cache_save = self.config.get('llmrerank', {}).get('rerank_input', {}).get('save_cache', False)
        llmrerank_cache_overwrite = self.config.get('llmrerank', {}).get('rerank_input', {}).get('overwrite', False)

        if llmrerank_cache_save:
            try:
                llmrerank_data_name = get_llmrerank_data_ori_name(self.config)
                save_cache_file(llmrerank_data_name, all_save, overwrite=llmrerank_cache_overwrite)
                logger.info(f"Save llmrerank data to {llmrerank_data_name}")
            except:
                logger.info(f"Save llmrerank data to {llmrerank_data_name} failed")

        return candidate_url_set, candidate_id_set, score_dict, ents, entity_info_dict, test_pair_ground_truth_url, entities


    def get_embs_ordered_by_id(self):
        if hasattr(self, 'retriever') is False:
            load_retriever_info = self.load_retriever()
            logger.info(load_retriever_info)
        kg1_sequence, kg2_sequence = self.get_sequence()
        kg1_u2s = {line['url']: line['sequence'] for line in kg1_sequence}
        kg2_u2s = {line['url']: line['sequence'] for line in kg2_sequence}
        ent_str_list = {}
        for ent_id in self.eadata.kg1.ent_ids:
            url = self.eadata.kg1.ent_ids[ent_id]
            if url in kg1_u2s:
                ent_str_list[ent_id] = kg1_u2s[url]
            else:
                tmp_str = self.eadata.kg1.entities[url].parse_url()
                ent_str_list[ent_id] = tmp_str
        
        for ent_id in self.eadata.kg2.ent_ids:
            url = self.eadata.kg2.ent_ids[ent_id]
            if url in kg2_u2s:
                ent_str_list[ent_id] = kg2_u2s[url]
            else:
                tmp_str = self.eadata.kg2.entities[url].parse_url()
                ent_str_list[ent_id] = tmp_str

        assert len(ent_str_list) == len(self.eadata.kg1.ent_ids) + len(self.eadata.kg2.ent_ids)
        
        ents = [ent_str_list[ent_id] for ent_id in range(len(ent_str_list))]

        batch_size = self.config['retrieval'].get('retrieval_batch_size', 32)
        multi_gpu_encoding = self.config['retrieval'].get('multi_gpu_encoding', False)
        logger.info(batch_size)
        logger.info(multi_gpu_encoding)
        logger.info(ents[0])
        logger.info(len(ents))
        if multi_gpu_encoding:
            state, embs = self.retriever.encode_multi_process(sentences=ents, batch_size=batch_size)
        else:
            state, embs = self.retriever.encode(sentences=ents, batch_size=batch_size)
        if not state:
            raise ValueError("Error in getting retrieval embeddings")
        embs = np.array(embs)
        return embs

    def save_embs_ordered_by_id(self):
        if hasattr(self, 'retriever') is False:
            load_retriever_info = self.load_retriever()
            logger.info(load_retriever_info)
        kg1_sequence, kg2_sequence = self.get_sequence()
        kg1_u2s = {line['url']: line['sequence'] for line in kg1_sequence}
        kg2_u2s = {line['url']: line['sequence'] for line in kg2_sequence}
        ent_str_list = {}
        for ent_id in self.eadata.kg1.ent_ids:
            url = self.eadata.kg1.ent_ids[ent_id]
            if url in kg1_u2s:
                ent_str_list[ent_id] = kg1_u2s[url]
            else:
                tmp_str = self.eadata.kg1.entities[url].parse_url()
                ent_str_list[ent_id] = tmp_str
        
        for ent_id in self.eadata.kg2.ent_ids:
            url = self.eadata.kg2.ent_ids[ent_id]
            if url in kg2_u2s:
                ent_str_list[ent_id] = kg2_u2s[url]
            else:
                tmp_str = self.eadata.kg2.entities[url].parse_url()
                ent_str_list[ent_id] = tmp_str

        assert len(ent_str_list) == len(self.eadata.kg1.ent_ids) + len(self.eadata.kg2.ent_ids)
        
        ents = [ent_str_list[ent_id] for ent_id in range(len(ent_str_list))]

        batch_size = self.config['retrieval'].get('retrieval_batch_size', 32)
        multi_gpu_encoding = self.config['retrieval'].get('multi_gpu_encoding', False)
        if multi_gpu_encoding:
            state, embs = self.retriever.encode_multi_process(sentences=ents, batch_size=batch_size)
        else:
            state, embs = self.retriever.encode(sentences=ents, batch_size=batch_size)
        if not state:
            raise ValueError("Error in getting retrieval embeddings")
        embs = np.array(embs)

        path_name = f"{self.config['dataset']['type']}-{self.config['dataset']['kg1']}-{self.config['dataset']['kg2']}_"
        ok = False
        if self.config['entity_info']['use_name'] is True:
            path_name += "name_"
            ok = True
        
        if self.config['entity_info']['use_url'] is True:
            path_name += "url_"
            ok = True
        
        if self.config['entity_info']['use_translated_url'] is True:
            path_name += "translatedurl_"
            ok = True
        
        if self.config['entity_info']['use_attributes'] is True:
            path_name += "attributes_"
            ok = True

        if self.trained:
            path_name += "trained_"
        else:
            path_name += "untrained_"

        assert ok, "No entity info used"
        
        home_directory = os.path.expanduser('~')
        file_name = os.path.join(home_directory, '.cache', 'aligncraft')
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        
        embs_path = os.path.join(file_name, f"{path_name}.npy")
        np.save(embs_path, embs)
        logger.info(f"Embeddings saved to {embs_path}")






    def retrieval_before_rerank(self):
        if hasattr(self, 'retriever') is False:
            load_retriever_info = self.load_retriever()
            logger.info(load_retriever_info)
        kg1_sequence, kg2_sequence = self.get_sequence()
        test_queries, test_documents = self.get_query_and_documents_4_test(
            kg1_sequence=kg1_sequence,
            kg2_sequence=kg2_sequence,
        )
        retrieval_scores, retrieval_candidates = self.retrieval(test_queries, test_documents)
        return {
            "retrieval_scores": retrieval_scores,
            "retrieval_candidates": retrieval_candidates,
            "kg1_sequence": test_queries,
            "kg2_sequence": test_documents,
        }

    def generate_retrieval_sft_data(self):
        kg1_sequence, kg2_sequence = self.get_sequence()
        if self.config['retrieval']['training'].get('strategy', 'supervised') == 'supervised':
            train_queries, train_documents = self.get_query_and_documents_4_train(
                kg1_sequence=kg1_sequence,
                kg2_sequence=kg2_sequence,
            )
            _, test_documents = self.get_query_and_documents_4_test(
                kg1_sequence=kg1_sequence,
                kg2_sequence=kg2_sequence,
            )
            train_pairs = [(s1, s2) for s1, s2 in zip(train_queries, train_documents)]
            other_corpus  = copy.deepcopy(test_documents)
            self.write_retrieval_sft_data(train_pairs, other_corpus)
        else:
            raise NotImplementedError("Unsupervised training not implemented")