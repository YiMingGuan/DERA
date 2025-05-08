from vllm import LLM, SamplingParams
from .conversation import construct_prompt
from .utils import entity_info, load_cache_file, save_cache_file, generate_entity_description, generate_entity_description_qwen
import logging
from ...data_loading.load_dataset import load_dataset
import yaml
logger = logging.getLogger(__name__)
from transformers import AutoTokenizer

from openai import OpenAI
import os
# ----

class Entity2Sequence:
    """
    Large Language Models (LLMs) for generating sequences conditioned on entities.
    """

    
    def __init__(self, config):
        assert config.get('using_vllm', False) is True, "VLLM is not loaded, please set using_vllm to True"
        # self.model = self.load_vllm_model(config['vllm']['llms_model_path'], **config)
        self.config = config
        self.eadata = self.load_eadata()
    
    def load_vllm_model(self, model_name_or_path, **kwargs):
        tensor_parallel_size = kwargs['vllm'].get('tensor_parallel_size', 1)
        swap_space = kwargs['vllm'].get('swap_space', 1)
        llm = LLM(model=model_name_or_path, tensor_parallel_size=tensor_parallel_size, swap_space=swap_space)
        logger.info(f"Successfully loaded VLLM model: {model_name_or_path}")
        return llm

    def vllm_generate(self, prompt, **kwargs):
        """
        Generate a sequence given a prompt and an entity using VLLM.
        """
        temperature = kwargs.get('temperature', 0.8)
        top_p = kwargs.get('top_p', 0.95)
        max_tokens = kwargs.get('max_tokens', 512)
        repetition_penalty = kwargs.get('repetition_penalty', 1.1)
        stop = kwargs.get('stop', None)
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop)
        logger.info(f"SamplingParams: \n{sampling_params}")
        sequences = []
        outputs = self.model.generate(prompt, sampling_params=sampling_params)
        for output in outputs:
            sequences.append(output.outputs[0].text)
        return sequences
    
    def load_eadata(self):
        data_config_path = self.config['dataset']['config_path']
        data_type = self.config['dataset']['type'].upper()
        data_config = yaml.load(open(data_config_path, 'r'), Loader=yaml.FullLoader)
        eadata = load_dataset(data_type, data_config)
        return eadata

    
    def generate_sequence(self, kg, dataset_type, kg_name, entity_info_method, url_list=None):
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

        kg1_name = self.config['dataset']['kg1']
        kg2_name = self.config['dataset']['kg2']
        prompt_type = self.config.get('prompt_type', None)
        if using_llms is False:
            prompt_type = None
        
        over_write = self.config['entity_info'].get('over_write', False)

        kg_cache_sequence_file_name = f"test_{dataset_type}-{kg1_name}-{kg2_name}_{kg_name}_using_llms{using_llms}_llms_name{llms_name}_prompt_type{prompt_type}_sequence_{method}_use_url_{use_url}_use_name_{use_name}_use_translated_url_{use_translated_url}_use_relationship_{use_relationship}_use_attributes_{use_attributes}_use_inverse_{use_inverse}_neighbors_use_attr_{neighbors_use_attr}_head_use_attr_{head_use_attr}"
        kg_cache_sequence = load_cache_file(kg_cache_sequence_file_name) if over_write is False else None
        if kg_cache_sequence is None:
            kg_cache_sequence = []

        # 这里kg_cache_sequence其实是已完成的一部分转换集合
        # logging.info(f"test information: {kg_cache_sequence[0]}")
        # logging.info(f"test information: {kg_cache_sequence[1]}")


        kg_url2sequence = {}
        for line in kg_cache_sequence:
            kg_url2sequence[line['url']] = line['sequence']
        
        # items_list = list(kg_url2sequence.items())
        # first_key, first_value = items_list[0]
        # logging.info(f"kg_url2sequence_example: First key: {first_key}, First value: {first_value}")


        entities_url = list(kg.entities.keys()) if url_list is None else url_list
        # url_list非空，entities_url[0]是http://fr.dbpedia.org/resource/Hans_A._Engelhard

        urls = set(entities_url) - set(kg_url2sequence.keys())
        # debug        原始输入三元组  -begin
        urls = set(entities_url)
        # debug        原始输入三元组  -end

        if len(urls) == 0:
            logger.info(f"Using cache file {kg_cache_sequence_file_name}")
            logger.info(f"Entities info: {kg_cache_sequence[0]}")
            return kg_cache_sequence

        entities = [kg.entities[url] for url in urls]

        dict_entities_info = [
            entity_info(
                entity,
                method=entity_info_method,
                **self.config['entity_info']
            )
            for entity in entities
        ]
        entities_info = [
            one_entity_info['mixed'] for one_entity_info in dict_entities_info
        ]

        logging.info(f"Entity info length: {len(entities_info)}")
        # logging.info("Ori entities_info")
        # logging.info(entities_info[:10])

        # logging.info("Entities:")
        # logging.info(entities[:10])
        
        """
        原始输入三元组 save_ori_entities_info
        """
        save_ori_entities_info = [
            {
                "url": entity.url, 
                "ori_triples": item
            } for entity, item in zip(entities, entities_info)
        ]

        logging.info("Save ori entities info")

        logging.info(save_ori_entities_info[0])

        # logging.info(save_ori_entities_info[10])

        """
        entities_info = [
                entity_info(
                    entity,
                    method=entity_info_method,
                    **self.config['entity_info']
                )['mixed']
            for entity in entities
        ]
        """
        """
        logging some entities info
        """
        # logger.info(f"Entities info: {entities_info[0]}")



        tokenizer = None
        # 从这里开始, 我尝试使用chenxuan在之前写的prompt保持一致性，做新的调用
        entities_info_with_prompt = [
            construct_prompt(
                self.config['prompt_type'],
                entity_info,
                tokenizer
            )
            for entity_info in entities_info
        ]


        if self.config['llms_type'] == 'qwen2-chat':
            tokenizer_path = self.config['vllm']['llms_model_path']
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f"Tokenizer loaded from {tokenizer_path}")

        if self.config.get('using_llms', False):
            assert self.config.get('using_vllm', False) is True, "VLLM is not loaded, please set using_vllm to True"
            if hasattr(self, 'model') is False:
                setattr('model', self.load_vllm_llms())
            entities_info_with_prompt = [
                construct_prompt(
                    self.config['prompt_type'],
                    entity_info,
                    tokenizer
                )
                for entity_info in entities_info
            ]
            vllm_sampling_stop = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"] if self.config['llms_type'] == 'qwen2-chat' else None
            if self.config.get('using_vllm', True):
                vllm_generation = self.config['vllm'].get('generation', {})
                if vllm_generation is None:
                    vllm_generation = {}
                sequences = self.vllm_generate(entities_info_with_prompt, stop=vllm_sampling_stop, **vllm_generation)
            else:
                raise NotImplementedError("LLMs not implemented")
            if self.config['entity_info']['use_translated_url']:
                for i, one_dict_info in enumerate(dict_entities_info):
                    sequences[i] = one_dict_info['name_or_url_info'] + "\n" + sequences[i]
            logger.info(f"Entities sequences: {sequences[0]}")
        else:
            sequences = [entity_info for entity_info in entities_info]
        
        # 这一部分是调api的代码
        # sequences = [
        #     generate_entity_description(prompt)
        #     for prompt in entities_info_with_prompt
        # ]
        # logging.info(f"multiple prompt: {entities_info_with_prompt[:10]}")
        # result_example = generate_entity_description_qwen(entities_info_with_prompt)

        # logging.info(f"entities_info_with_prompt example_qwen: {result_example[0]}")    



        new_url2sequence = [
            {
                "url": entity.url,
                "sequence": sequences[i]
            }
            for i, entity in enumerate(entities)
        ]


        if self.config['entity_info']['save_sequence']:
            logging.info(f"type1: {new_url2sequence[:2]}")
            # save_cache_file(kg_cache_sequence_file_name, new_url2sequence, True)
            save_cache_file(kg_cache_sequence_file_name, new_url2sequence + kg_cache_sequence)
        
        return new_url2sequence + kg_cache_sequence

    def get_train_sequence(self):
        kg1_url_list = [
            self.eadata.kg1.ent_ids[id1] for (id1, id2) in self.eadata.train_pairs.items()
        ]
        kg2_url_list = [
            self.eadata.kg2.ent_ids[id2] for (id1, id2) in self.eadata.train_pairs.items()
        ]
        kg1_sequence = self.generate_sequence(
            kg=self.eadata.kg1,
            dataset_type=self.config['dataset']['type'],
            kg_name=self.config['dataset']['kg1'],
            entity_info_method=self.config['entity_info_method'],
            url_list=kg1_url_list,
        )
        kg2_sequence = self.generate_sequence(
            kg=self.eadata.kg2,
            dataset_type=self.config['dataset']['type'],
            kg_name=self.config['dataset']['kg2'],
            entity_info_method=self.config['entity_info_method'],
            url_list=kg2_url_list
        )
        logger.info(f"Successfully generated sequences for training data.")
        return kg1_sequence, kg2_sequence    

    def get_test_sequence(self):
        jape_test_setting = self.config.get('jape_test_setting', True)
        kg1_url_list = None
        kg2_url_list = None
        if jape_test_setting:
            kg1_url_list = [
                self.eadata.kg1.ent_ids[id1] for (id1, id2) in self.eadata.test_pairs.items()
            ]
            kg2_url_list = [
                self.eadata.kg2.ent_ids[id2] for (id1, id2) in self.eadata.test_pairs.items()
            ]
        
        kg1_sequence = self.generate_sequence(
            kg=self.eadata.kg1,
            dataset_type=self.config['dataset']['type'],
            kg_name=self.config['dataset']['kg1'],
            entity_info_method=self.config['entity_info_method'],
            url_list=kg1_url_list,
        )
        kg2_sequence = self.generate_sequence(
            kg=self.eadata.kg2,
            dataset_type=self.config['dataset']['type'],
            kg_name=self.config['dataset']['kg2'],
            entity_info_method=self.config['entity_info_method'],
            url_list=kg2_url_list
        )
        logger.info(f"Successfully generated sequences for testing data.")
        return kg2_sequence
        return kg1_sequence, kg2_sequence

    def run(self):
        train_kg1_sequence, train_kg2_sequence = self.get_train_sequence()
        return train_kg1_sequence, train_kg2_sequence
        test_kg2_sequence = self.get_test_sequence()
        return test_kg2_sequence
        test_kg1_sequence, test_kg2_sequence = self.get_test_sequence()
        return test_kg1_sequence, test_kg2_sequence
        
        return train_kg1_sequence, train_kg2_sequence, test_kg1_sequence, test_kg2_sequence