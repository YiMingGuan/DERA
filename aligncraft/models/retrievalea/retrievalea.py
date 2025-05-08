from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification
from typing import List
from .conversation import construct_prompt
from .utils import entity_info, load_cache_file, save_cache_file
import hashlib
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import faiss
import numpy as np
import json
import logging
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import gc
from .utils import plot_calibration_curve
import copy
from .utils import get_retrieval_data_path

class RankingModel:
    def __init__(self, model_path: str, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = device
        self.model.eval()
        self.model.to(self.device)

    def rank(self, pairs: List[List[str]], batch_size=32, **params):
        try:
            scores = []
            with torch.no_grad():
                for i in range(0, len(pairs), batch_size):
                    batch = pairs[i : i + batch_size]
                    inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device) 
                    batch_scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
                    scores.extend(batch_scores.tolist())
        except Exception as e:
            return False, str(e)
        return True, scores
