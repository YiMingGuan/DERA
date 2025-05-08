import json
import random
import numpy as np
import faiss
from tqdm import tqdm

from .retriever_sft import FlagModel
from .utils import get_retrieval_hn_mine_data_name, get_retrieval_data_path, get_rerank_hn_mine_data_name, get_rerank_data_path, get_trained_retrieval_model
import logging
logger = logging.getLogger(__name__)
class HN_Mine:
    def __init__(self, config, step):
        assert step in ['retrieval', 'rerank']
        self.config = config
        self.step = step

    def create_index(self, embeddings, use_gpu):
        index = faiss.IndexFlatIP(len(embeddings[0]))
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if use_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index, co=co)
        index.add(embeddings)
        return index


    def batch_search(self,
                     index,
                    query,
                    topk: int = 200,
                    batch_size: int = 64):
        all_scores, all_inxs = [], []
        for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
            batch_query = query[start_index:start_index + batch_size]
            batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
            all_scores.extend(batch_scores.tolist())
            all_inxs.extend(batch_inxs.tolist())
        return all_scores, all_inxs


    def get_corpus(self, candidate_pool):
        corpus = []
        for line in open(candidate_pool):
            line = json.loads(line.strip())
            corpus.append(line['text'])
        return corpus


    def find_knn_neg(self, model, input_file, candidate_pool, output_file, sample_range, negative_number, use_gpu):
        corpus = []
        queries = []
        train_data = []
        for line in open(input_file):
            line = json.loads(line.strip())
            train_data.append(line)
            corpus.extend(line['pos'])
            if 'neg' in line:
                corpus.extend(line['neg'])
            queries.append(line['query'])

        if candidate_pool is not None:
            if not isinstance(candidate_pool, list):
                candidate_pool = self.get_corpus(candidate_pool)
            corpus = list(set(candidate_pool))
        else:
            corpus = list(set(corpus))
        logger.info(f'corpus size: {len(corpus)}')
        logger.info(f'query size: {len(queries)}')
        logger.info(f'{queries[0]}')
        logger.info(f'inferencing embedding for corpus (number={len(corpus)})--------------')
        p_vecs = model.encode(corpus, batch_size=256)
        logger.info(f'inferencing embedding for queries (number={len(queries)})--------------')
        q_vecs = model.encode_queries(queries, batch_size=256)

        logger.info('create index and search------------------')
        index = self.create_index(p_vecs, use_gpu=use_gpu)
        _, all_inxs = self.batch_search(index, q_vecs, topk=sample_range[-1])
        assert len(all_inxs) == len(train_data)

        for i, data in enumerate(train_data):
            query = data['query']
            inxs = all_inxs[i][sample_range[0]:sample_range[1]]
            filtered_inx = []
            for inx in inxs:
                if inx == -1: break
                if corpus[inx] not in data['pos'] and corpus[inx] != query:
                    filtered_inx.append(inx)

            if len(filtered_inx) > negative_number:
                filtered_inx = random.sample(filtered_inx, negative_number)
            data['neg'] = [corpus[inx] for inx in filtered_inx]

        with open(output_file, 'w') as f:
            for data in train_data:
                if len(data['neg']) < negative_number:
                    data['neg'].extend(random.sample(corpus, negative_number - len(data['neg'])))
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        logger.info(f"hn_mine data saved to {output_file}")
    
    def run(self):
        sample_range = self.config[self.step]['hn_mine'].get('range_for_sampling', '2-200').split('-')
        sample_range = [int(x) for x in sample_range]

        pseudo_model_path = self.config[self.step]['hn_mine']['model_path']
        model_path = get_trained_retrieval_model(self.config) if self.step == 'rerank' and pseudo_model_path == 'sft_retrieval' else pseudo_model_path
        logger.info(f"Loading hn minde model from {model_path}")
        model = FlagModel(model_path, query_instruction_for_retrieval=self.config[self.step]['hn_mine'].get('query_instruction_for_retrieval', ""))

        args = self.config[self.step]['hn_mine']
        input_file = get_retrieval_data_path(self.config) if self.step == 'retrieval' else get_rerank_data_path(self.config)
        output_file = get_retrieval_hn_mine_data_name(self.config) if self.step == 'retrieval' else get_rerank_hn_mine_data_name(self.config)
        logger.info(f"start mining hard negatives for {input_file}--------------")
        self.find_knn_neg(model,
                    input_file=input_file,
                    candidate_pool=args.get('candidate_pool', None),
                    output_file=output_file,
                    sample_range=sample_range,
                    negative_number=args.get('negative_number', 15),
                    use_gpu=args.get('use_gpu_for_searching', True))
