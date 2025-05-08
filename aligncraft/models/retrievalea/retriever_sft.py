from typing import cast, List, Union, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, is_torch_npu_available
from transformers.trainer_utils import EvalPrediction, PredictionOutput
import logging
from dataclasses import dataclass
from typing import Dict, Optional
import faiss
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput

import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

logger = logging.getLogger(__name__)

class FlagModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            use_fp16: bool = True
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            logger.info(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)

    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int = 256,
                       max_length: int = 512,
                       convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            if isinstance(queries, str):
                input_texts = self.query_instruction_for_retrieval + queries
            else:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    def encode_corpus(self,
                      corpus: Union[List[str], str],
                      batch_size: int = 256,
                      max_length: int = 512,
                      convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        return self.encode(corpus, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    @torch.no_grad()
    def encode(self,
               sentences: Union[List[str], str],
               batch_size: int = 256,
               max_length: int = 512,
               convert_to_numpy: bool = True) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.stack(all_embeddings)

        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor = None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d


import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizer


import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )



@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: str= field(
        default=None, metadata={"help": "instruction for query"}
    )
    passage_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for passage"}
    )
    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean"})
    normlized: bool = field(default=True)
    use_inbatch_neg: bool = field(default=True, metadata={"help": "use passages in the same batch as negatives"})
    report_to: Optional[List[str]] = field(default=None, metadata={"help": "The list of integrations to report the results"})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Whether or not to load the best model found during training at the end of training."})
    metric_for_best_model: Optional[str] = field(default='eval_mrr', metadata={"help": "The metric to use to compare two different models."})
    greater_is_better: bool = field(default=True, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."})
    evaluation_strategy: str = field(default='steps', metadata={"help": "The evaluation strategy to adopt during training."})
    save_strategy: str = field(default='steps', metadata={"help": "The checkpoint save strategy to adopt during training."})
    save_total_limit: int = field(default=2, metadata={"help": "Limit the total amount of checkpoints."})
    eval_steps: int = field(default=10, metadata={"help": "Number of update steps between two evaluations if evaluation_strategy='steps'."})
    logging_steps: int = field(default=10, metadata={"help": "Number of update steps between two logs if logging_strategy='steps'."})
    save_steps: int = field(default=10, metadata={"help": "Number of updates steps before two checkpoint saves if save_strategy='steps'."})
    eval_build_faiss_emb_batch_size: int = field(
        default=1024, metadata={"help": "the batch size for building faiss index"}
    )


def load_retrieval_original_dataset(args, data_path):
    if os.path.isdir(data_path):
        train_datasets = []
        for file in os.listdir(data_path):
            temp_dataset = datasets.load_dataset('json', data_files=os.path.join(data_path, file),
                                                 split='train')
            if len(temp_dataset) > args.max_example_num_per_dataset:
                temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)
        dataset = datasets.concatenate_datasets(train_datasets)
    else:
        dataset = datasets.load_dataset('json', data_files=data_path, split='train')
    return dataset
        
class EvalDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer,
            train_dataset: datasets.Dataset,
            eval_dataset: datasets.Dataset
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.corpus = self.build_corpus(train_dataset, eval_dataset)
        self.dataset = self.build_eval_dataset(eval_dataset)
        self.corpus_ids = self.corpus_tokenized()
        logger.info(f"corpus size: {len(self.corpus)}")
    
    def corpus_tokenized(self):
        corpus_ids = self.tokenizer(self.corpus, padding=True, truncation=True, max_length=self.args.passage_max_len,
                                    return_tensors="pt")
        return corpus_ids


    def build_corpus(self, train_dataset, eval_dataset):
        corpus = []
        for item in train_dataset:
            if isinstance(item['pos'], str):
                corpus.append(item['pos'])
            elif isinstance(item['pos'], list):
                corpus.extend(item['pos'])
            
            if isinstance(item['neg'], str):
                corpus.append(item['neg'])
            elif isinstance(item['neg'], list):
                corpus.extend(item['neg'])
        
        for item in eval_dataset:
            if isinstance(item['pos'], str):
                corpus.append(item['pos'])
            elif isinstance(item['pos'], list):
                corpus.extend(item['pos'])
            
            if isinstance(item['neg'], str):
                corpus.append(item['neg'])
            elif isinstance(item['neg'], list):
                corpus.extend(item['neg'])
        return list(set(corpus))
    
    def build_eval_dataset(self, eval_dataset):
        eval_data = []
        doc2index = {
            doc: idx for idx, doc in enumerate(self.corpus)
        }
        for item in eval_dataset:
            query = item['query']
            assert len(item['pos']) == 1
            pos = item['pos'][0]
            eval_data.append(
                {
                    'query': query,
                    'pos_index': doc2index[pos]
                }
            )
        return eval_data

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        return self.dataset[item]

@dataclass
class EvalCollator(DataCollatorWithPadding):
    query_max_len: int = 512

    def __call__(self, features):
        query = [f['query'] for f in features]
        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        q_index = [f['pos_index'] for f in features]
        return {"query": q_collated, "pos_index": q_index}


class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer,
            dataset: datasets.Dataset,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query

        passages = []
        assert self.dataset[item]['pos']
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)

        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        passages.extend(negs)

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval+p for p in passages]
        return query, passages


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 512
    passage_max_len: int = 512

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated}



@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config

        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                scores = scores.view(q_reps.size(0), -1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size
                loss = self.compute_loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G
                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss = self.compute_loss(scores, target)
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)


from sentence_transformers import SentenceTransformer, models
from transformers.trainer import Trainer


def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normlized: bool=True):
    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    if normlized:
        normlize_layer = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normlize_layer], device='cpu')
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    model.save(ckpt_dir)
import copy
def build_faiss_index(embeddings):
    """
    embedding in cuda
    """
    emb = copy.deepcopy(embeddings)
    emb = emb.astype('float32')
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index

class BiTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save the checkpoint for sentence-transformers library
        if self.is_world_process_zero():
            save_ckpt_for_sentence_transformers(output_dir,
                                                pooling_mode=self.args.sentence_pooling_method,
                                                normlized=self.args.normlized)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


    def evaluate(self, eval_dataset: Optional[Dataset]=None, ignore_keys: Optional[List[str]] = None):
        self.model.eval()
        self.eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        corpus_embs = []
        corpus_ids = self.eval_dataset.corpus_ids
        logger.info(f">>> Len of corpus: {len(self.eval_dataset.corpus)}")
        """
        build faiss index for corpus
        """
        with torch.no_grad():
            for i in range(0, len(self.eval_dataset.corpus), self.args.eval_build_faiss_emb_batch_size):
                batch = corpus_ids[i: i+self.args.eval_build_faiss_emb_batch_size]
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                batch_embs = self.model.encode(batch)
                corpus_embs.append(batch_embs.cpu())
        corpus_embs = torch.cat(corpus_embs, dim=0)
        """
        main process
        """
        index = build_faiss_index(corpus_embs.numpy())
        eval_dataloader = DataLoader(
            self.eval_dataset,
            collate_fn=EvalCollator(self.tokenizer),
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
        )
        """
        evaluate the model
        """
        ranks = []
        pos_indexs = []
        logger.info("ntotal: %d", index.ntotal)
        with torch.no_grad():
            for step, inputs in enumerate(eval_dataloader):
                batch = {k: v.to(self.args.device) for k, v in inputs['query'].items()}
                query_embs = self.model.encode(batch).cpu().numpy()
                query_embs = query_embs.astype('float32')
                faiss.normalize_L2(query_embs)
                D, I = index.search(query_embs, k=index.ntotal)
                pos_indexs.extend(inputs['pos_index'])
                ranks.extend(I)
        
        eval_prediction = EvalPrediction(predictions=np.array(ranks), label_ids=np.array(pos_indexs))
        metrics = self.compute_metrics(eval_prediction)
        self.log(metrics)
        return metrics

from .utils import get_trained_retrieval_model, get_retrieval_hn_mine_data_name


def compute_metrics(eval_output: EvalPrediction):
    prediction = eval_output.predictions
    label_id = eval_output.label_ids
   
    def mrr(ranks, pos_indexs):
        pos_indexs_expanded = pos_indexs[:, np.newaxis]
        matches = ranks == pos_indexs_expanded
        """
        if matches[i] isn't including True, set matches[i][-1] = True
        """
        for i in range(len(matches)):
            if not True in matches[i]:
                matches[i][-1] = True
        indices = np.argmax(matches, axis=1)

        return np.mean(1 / (indices + 1))

    def hits_at_k(ranks, pos_indexs, k):
        assert len(ranks[0]) >= k
        pos_indexs_expanded = pos_indexs[:, np.newaxis]
        matches = ranks == pos_indexs_expanded
        for i in range(len(matches)):
            if not True in matches[i]:
                matches[i][-1] = True
        indices = np.argmax(matches, axis=1)
        return np.sum(indices < k) / len(ranks)

    MRR = mrr(prediction, label_id)
    HITS1 = hits_at_k(prediction, label_id, 1)
    HITS3 = hits_at_k(prediction, label_id, 3)
    HITS5 = hits_at_k(prediction, label_id, 5)
    HITS10 = hits_at_k(prediction, label_id, 10)

    metric = {
        "eval_mrr": MRR,
        "eval_hits1": HITS1,
        "eval_hits3": HITS3,
        "eval_hits5": HITS5,
        "eval_hits10": HITS10,
    }
    return metric

class RetrieverSFT:
    def __init__(self, config):
        output_dir = get_trained_retrieval_model(config)
        model_name_or_path = config['retrieval']['model_path']
        train_data = get_retrieval_hn_mine_data_name(config)
        self.config = config['retrieval']['training']['finetune']
        self.config['model_name_or_path'] = model_name_or_path
        self.config['train_data'] = train_data
        self.config['output_dir'] = output_dir
    
    def run(self):
        parser = HfArgumentParser((ModelArguments, DataArguments, RetrieverTrainingArguments))
        model_args, data_args, training_args = parser.parse_dict(self.config)
        model_args: ModelArguments
        data_args: DataArguments
        training_args: RetrieverTrainingArguments

        if (
                os.path.exists(training_args.output_dir)
                and os.listdir(training_args.output_dir)
                and training_args.do_train
                and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

        # Set seed
        set_seed(training_args.seed)

        num_labels = 1
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=False,
        )
        model_path = None
        if training_args.resume_from_checkpoint is not None:
            model_path = training_args.resume_from_checkpoint
        else:
            model_path = model_args.model_name_or_path

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
        )
        logger.info('Config: %s', config)

        
        model = BiEncoderModel(model_name=model_path,
                            normlized=training_args.normlized,
                            sentence_pooling_method=training_args.sentence_pooling_method,
                            negatives_cross_device=training_args.negatives_cross_device,
                            temperature=training_args.temperature,
                            use_inbatch_neg=training_args.use_inbatch_neg,
                            )
        logger.info("Load model from %s", model_path)
        if training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False

        retrieval_original_dataset = load_retrieval_original_dataset(data_args, data_args.train_data)
        eval_ratio = 0.1
        train_data, eval_data = random_split(retrieval_original_dataset,
                                              [len(retrieval_original_dataset) - int(len(retrieval_original_dataset) * eval_ratio),
                                                int(len(retrieval_original_dataset) * eval_ratio)])

        train_dataset = TrainDatasetForEmbedding(
            args=data_args,
            tokenizer=tokenizer,
            dataset=train_data
        )
        eval_dataset = EvalDatasetForEmbedding(
            args=data_args,
            tokenizer=tokenizer,
            train_dataset=train_data,
            eval_dataset=eval_data
        )
        
        trainer = BiTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=EmbedCollator(
                tokenizer,
                query_max_len=data_args.query_max_len,
                passage_max_len=data_args.passage_max_len
            ),
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

        # Training
        trainer.train()
        """
        save the best model
        """
        logger.info("Saving the best model")
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
