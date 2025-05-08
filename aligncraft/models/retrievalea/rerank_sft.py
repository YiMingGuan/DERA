import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
from transformers.trainer_utils import EvalPrediction, PredictionOutput

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding
import logging
from torch.utils.data import Dataset, random_split, DataLoader
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from typing import Optional

from transformers.trainer import Trainer

from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, TrainingArguments
from transformers import (
    HfArgumentParser,
    set_seed,
)
logging.disable(logging.WARNING)
logger = logging.getLogger(__name__)
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
        default=None, metadata={"help": "Path to corpus"}
    )
    train_group_size: int = field(default=8)
    max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for input text. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class RerankerTrainingArguments(TrainingArguments):
    report_to: Optional[List[str]] = field(default=None, metadata={"help": "The list of integrations to report the results"})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Whether or not to load the best model found during training at the end of training."})
    metric_for_best_model: Optional[str] = field(default='eval_hits1', metadata={"help": "The metric to use to compare two different models."})
    greater_is_better: bool = field(default=True, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."})
    evaluation_strategy: str = field(default='steps', metadata={"help": "The evaluation strategy to adopt during training."})
    save_strategy: str = field(default='steps', metadata={"help": "The checkpoint save strategy to adopt during training."})
    save_total_limit: int = field(default=2, metadata={"help": "Limit the total amount of checkpoints."})
    eval_steps: int = field(default=10, metadata={"help": "Number of update steps between two evaluations if evaluation_strategy='steps'."})
    logging_steps: int = field(default=10, metadata={"help": "Number of update steps between two logs if logging_strategy='steps'."})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Whether or not to use gradient checkpointing to save memory at the cost of slower backward pass."})
    save_steps: int = field(default=10, metadata={"help": "Number of updates steps before two checkpoint saves if save_strategy='steps'."})

def load_rerank_original_dataset(args, data_path):
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



class TrainDatasetForCE(Dataset):
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

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        # item = self.tokenizer.encode_plus(
        #     qry_encoding,
        #     doc_encoding,
        #     truncation=True,
        #     max_length=self.args.max_len,
        #     padding=False,
        # )
        qry_tokens = self.tokenizer.tokenize(qry_encoding)
        doc_tokens = self.tokenizer.tokenize(doc_encoding)
        
        # 检查总长度是否超过max_len
        total_length = len(qry_tokens) + len(doc_tokens)
        if total_length > self.args.max_len:
            # 需要截断
            # 确定每部分应该占据的token数量
            qry_length = int(len(qry_tokens) / total_length * self.args.max_len)
            doc_length = self.args.max_len - qry_length  # 确保总和为max_len
            
            # 截断
            qry_tokens = qry_tokens[:qry_length]
            doc_tokens = doc_tokens[:doc_length]
        
        # 编码处理后的tokens
        item = self.tokenizer.encode_plus(
            self.tokenizer.convert_tokens_to_string(qry_tokens),
            self.tokenizer.convert_tokens_to_string(doc_tokens),
            truncation=True,
            max_length=self.args.max_len,
            padding=True,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        query = self.dataset[item]['query']
        pos = random.choice(self.dataset[item]['pos'])
        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)

        batch_data = []
        batch_data.append(self.create_one_example(query, pos))
        for neg in negs:
            batch_data.append(self.create_one_example(query, neg))

        return batch_data

@dataclass
class GroupCollator(DataCollatorWithPadding):
    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)
    


class CrossEncoder(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: RerankerTrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.config = self.hf_model.config
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.hf_model.gradient_checkpointing_enable(**kwargs)

    def forward(self, batch):
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits

        if self.training:
            scores = logits.view(
                self.train_args.per_device_train_batch_size,
                self.data_args.train_group_size
            )
            loss = self.cross_entropy(scores, self.target_label)

            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: RerankerTrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        state_dict = self.hf_model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.hf_model.save_pretrained(output_dir, state_dict=state_dict)




class CETrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        print(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(f'MODEL {self.model.__class__.__name__} ' f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model: CrossEncoder, inputs):
        return model(inputs)['loss']
    
    def evaluate(self, eval_dataset: Optional[Dataset]=None, ignore_keys: Optional[List[str]] = None):
        self.model.eval()
        self.eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_dataloader = DataLoader(
            self.eval_dataset,
            collate_fn=GroupCollator(self.tokenizer),
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
        )
        """
        hits@1, hits@10, mrr
        """
        ranks = []
        for step, inputs in enumerate(eval_dataloader):
            batch = {k: v.to(self.args.device) for k, v in inputs.items()}
            with torch.no_grad():
                ranker_out: SequenceClassifierOutput = self.model(batch)
                logits = ranker_out.logits
                scores = logits.view(
                    self.args.per_device_eval_batch_size,
                    self.eval_dataset.args.train_group_size
                )
                ranks.extend(
                    torch.argsort(scores, dim=1, descending=True)[:, :10].tolist()
                )
        
        eval_prediction = EvalPrediction(predictions=np.array(ranks), label_ids=None)
        metrics = self.compute_metrics(eval_prediction)
        self.log(metrics)
        return metrics

from .utils import get_trained_rerank_model, get_rerank_hn_mine_data_name

def compute_metrics(eval_output: EvalPrediction):
    """
    hits@1, hits@10, mrr
    ranker_out: SequenceClassifierOutput = self.model(batch)
                logits = ranker_out.logits
                scores = logits.view(
                    self.args.per_device_eval_batch_size,
                    self.eval_dataset.args.train_group_size
                )
                ranks.extend(
                    torch.argsort(scores, dim=1, descending=True)[:, :10].tolist()
                )
    """
    ranks = eval_output.predictions

    def hits_at_k(rank, k):
        zero_positions = np.where(rank == 0)
        return np.mean(zero_positions[1] < k)
    
    def mrr(rank):
        zero_positions = np.where(rank == 0)
        return np.mean(1 / (zero_positions[1] + 1))

    HITS_AT_1 = hits_at_k(ranks, 1)
    HITS_AT_3 = hits_at_k(ranks, 3)
    HITS_AT_5 = hits_at_k(ranks, 5)
    HITS_AT_10 = hits_at_k(ranks, 10)
    MRR = mrr(ranks)

    return {
        "eval_mrr": MRR,
        "eval_hits1": HITS_AT_1,
        "eval_hits3": HITS_AT_3,
        "eval_hits5": HITS_AT_5,
        "eval_hits10": HITS_AT_10,
    }

class RerankSFT:
    def __init__(self, config) -> None:
        output_dir = get_trained_rerank_model(config)
        model_name_or_path = config['rerank']['model_path']
        train_data = get_rerank_hn_mine_data_name(config)
        self.config = config['rerank']['training']['finetune']
        self.config['model_name_or_path'] = model_name_or_path
        self.config['train_data'] = train_data
        self.config['output_dir'] = output_dir

    def run(self):
        parser = HfArgumentParser((ModelArguments, DataArguments, RerankerTrainingArguments))
        model_args, data_args, training_args = parser.parse_dict(self.config)
        model_args: ModelArguments
        data_args: DataArguments
        training_args: RerankerTrainingArguments

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
        logger.info(
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

        set_seed(training_args.seed)

        num_labels = 1

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=False,
        )
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
        )
        _model_class = CrossEncoder

        model = _model_class.from_pretrained(
            model_args, data_args, training_args,
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

        rerank_original_dataset = load_rerank_original_dataset(data_args, data_args.train_data)

        eval_ratio = 0.1
        train_data, eval_data = random_split(rerank_original_dataset,
                                              [len(rerank_original_dataset) - int(len(rerank_original_dataset) * eval_ratio),
                                                int(len(rerank_original_dataset) * eval_ratio)])

        train_dataset = TrainDatasetForCE(
            args=data_args, 
            tokenizer=tokenizer, 
            dataset=train_data
        )

        eval_dataset = TrainDatasetForCE(
            args=data_args, 
            tokenizer=tokenizer, 
            dataset=eval_data
        )

        _trainer_class = CETrainer
        trainer = _trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=GroupCollator(tokenizer),
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
        trainer.train()
        logger.info("Saving the best model")
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
