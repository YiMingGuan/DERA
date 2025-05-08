import random
import json
import copy
from vllm import LLM, SamplingParams
import logging
logger = logging.getLogger(__name__)
import re
from openai import OpenAI
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sortedcontainers import SortedSet
from transformers import set_seed, AutoTokenizer
import requests
class RerankInstruction:
    instruction_1 = """{}Given an entity and a set of candidate entities, find the entity representing the same object from the set of candidate entities. Based only on the information provided to you. There has to be a right answer. The final answer is output as follows: The correct answer : (CHOICE).
Entity:
{}
Candidate entity:
{}
Analysis:\n"""
    instruction_2 = """{}Given an entity and a set of candidate entities, find the entity representing the same object from the set of candidate entities. Based only on the information provided to you. There has to be a right answer. After the analysis is complete, output the correct answer in this format.
Entity:
{}
Candidate entity:
{}
Analysis:
"""
    instruction_3 = """{}Given an entity and a set of candidate entities, find the entity representing the same object from the set of candidate entities. Based only on the information provided to you. There has to be a right answer. Be sure to write \"The correct answer: ()\" at the end.
Entity:
{}
Candidate entity:
{}

"""
    instruction_4 = """"""
    instruction_5 = """"""
    instruction_openai_gpt4_1 = """"""

class Demonstration:
    demonstration_1 = """Given an entity and a set of candidate entities, find the entity representing the same object from the set of candidate entities.
Entity:
哥倫比亞足球乙級聯賽
Candidate entity:
(a) Categoría Primera B
(b) Football League_Second Divisio
(c) Colombia national football team
Analysis:
(a) Categoría Primera B - This is the second division of professional football in Colombia. It's the direct translation of the given entity "哥倫比亞足球乙級聯賽" into Spanish, the official language of Colombia.
(b) Football League Second Division - This is a generic term that could refer to the second division of football in any country. Without a specific country name, it cannot be directly linked to the Colombian second division.
(c) Colombia national football team - This refers to the national football team of Colombia and not to any football league or division.
Given the above information, the correct match for "哥倫比亞足球乙級聯賽" from the set of candidate entities is (a) Categoría Primera B. This entity represents the Colombian football second division league, which is exactly what the original entity refers to.
The correct answer: (a) Categoría Primera B
Follow the pattern above to answer the following questions. Be sure to write The correct answer: () at the end."""

    demonstration_2 = """Given an entity and a set of candidate entities, find the entity representing the same object from the set of candidate entities. Based only on the information provided to you. There has to be a right answer. After the analysis is complete, output the correct answer in this format.
The correct answer: ()
Entity:
呼和浩特市
<呼和浩特市 , name, 呼和浩特市·>	<呼和浩特市 , image map, China Inner Mongolia Hohhot.svg>	<呼和浩特市 , nickname, 青城>	

Candidate entity:
(a) Inner Mongolia
<Inner Mongolia , title, Religion in Inner Mongolia>	<Inner Mongolia , image map, Inner Mongolia in China .svg>	<Inner Mongolia , latd, 44>	

(b) Hohhot
<Hohhot , image map, China Inner Mongolia Hohhot.svg>	<Hohhot , latd, 40>	<Hohhot , longd, 111>	

(c) Mongolia
<Mongolia , name, Mongolia>	<Mongolia , image map, Mongolia .svg>	<Mongolia , latd, 47>	

(d) Ulaanbaatar
<Ulaanbaatar , name, Улаанбаатар>	<Ulaanbaatar , nickname, УБ , Нийслэл , Хот>	<Ulaanbaatar , latd, 47>	

(e) Bayingolin Mongol Autonomous Prefecture
<Bayingolin Mongol Autonomous Prefecture , title, Bayingolin>	<Bayingolin Mongol Autonomous Prefecture , image map, China Xinjiang Bayingolin.svg>	<Bayingolin Mongol Autonomous Prefecture , utc offset, +8>	

(f) Nei Mongol Zhongyou F.C.
<Nei Mongol Zhongyou F.C. , name, Geng Tianming>	<Nei Mongol Zhongyou F.C. , fullname, 内蒙古中优足球俱乐部>	<Nei Mongol Zhongyou F.C. , season, 2015>	

(g) Heilongjiang
<Heilongjiang , name, Heilongjiang>	<Heilongjiang , latd, 48>	<Heilongjiang , longd, 129>	

(h) Mongols in China
<Mongols in China , name, Ethnic Mongols (蒙古族) in China>	<Mongols in China , caption, This map shows the Mongol autonomous subjects in the PRC>	<Mongols in China , group, Ethnic Mongols  in China>	

(i) Mongolian language
<Mongolian language , name, Mongolian>	<Mongolian language , title, Mongolians speaking Mongolian>	<Mongolian language , description, A three-minute video of Mongolians speaking the Khalkha dialect. At the end, a short speech is given by an older man, which is then translated into English by one of the younger Mongols.>	

(j) Ordos City
<Ordos City , image map, China Inner Mongolia Ordos.svg>	<Ordos City , latd, 39>	<Ordos City , longd, 109>

Analysis: To find the entity representing the same object as "呼和浩特市" (Hohhot) among the candidate entities, we need to analyze the provided attributes and identify which candidate closely matches or is directly related to "呼和浩特市".

Entity Analysis:

"呼和浩特市" is identified by its name, an image map ("China Inner Mongolia Hohhot.svg"), and a nickname ("青城").
Candidate Entity Analysis:

(a) Inner Mongolia: This is a broader region that includes Hohhot but is not specifically Hohhot.
(b) Hohhot: The image map matches the entity's image map ("China Inner Mongolia Hohhot.svg"), making it a strong candidate.
(c) Mongolia: A country that, while related to Inner Mongolia, is not the same as Hohhot.
(d) Ulaanbaatar: The capital of Mongolia, which is not related to the specific entity we're analyzing.
(e) Bayingolin Mongol Autonomous Prefecture: A prefecture in Xinjiang, China, not related to Hohhot.
(f) Nei Mongol Zhongyou F.C.: A football club in Inner Mongolia, not a geographic location.
(g) Heilongjiang: A province in China, not related to Hohhot.
(h) Mongols in China: Refers to the ethnic group within China, not a specific location.
(i) Mongolian language: Refers to the language, not a location.
(j) Ordos City: Another city in Inner Mongolia, not Hohhot.
Based on the analysis, the candidate that represents the same object as "呼和浩特市" is (b) Hohhot, due to the matching image map which directly links this entity to the original entity in question.

The correct answer: (b) Hohhot"""

    demonstration_3 = """Given an entity and a set of candidate entities, find the entity representing the same object from the set of candidate entities.
Entity:
哥倫比亞足球乙級聯賽
Candidate entity:
(a) Categoría Primera B
(b) Football League_Second Divisio
(c) Colombia national football team
The correct answer: (a)
"""

class EntityPool:
    def __init__(
            self,
            args,
            testTupleList_list,
            url2CandidateUrlList_dict,
            topk,
            idealTest=False,
            sample_num=None,
    ):
        self.args = args
        self.init_pool(testTupleList_list, url2CandidateUrlList_dict, topk, idealTest, sample_num)
    
    def init_pool(self, testTupleList_list, url2CandidateUrlList_dict, topk, idealTest=False, sample_num=None):
        toBeRerankedPool = set()
        unCertainPool = {}
        killedPool = set()
        candidatePool = set()

        rerankedPairsPool = set()
        logger.info(testTupleList_list[0])
        logger.info(testTupleList_list[1])
        logger.info(testTupleList_list[2])
        logger.info(self.args.test_strategy)
        logger.info(type(self.args.test_strategy))
        if self.args.test_strategy == 'only_not_at_rank_1':
            logger.info("Test strategy: Only not at rank 1")
            sampledTestTupleList_list = [(ent1, ent2) for ent1, ent2 in testTupleList_list if url2CandidateUrlList_dict[ent1][0] != ent2]
        elif self.args.test_strategy == 'only_at_rank_1':
            logger.info("Test strategy: Only at rank 1")
            sampledTestTupleList_list = [(ent1, ent2) for ent1, ent2 in testTupleList_list if url2CandidateUrlList_dict[ent1][0] == ent2]
        elif self.args.test_strategy == 'at_topk':
            logger.info("Test strategy: At topk")
            sampledTestTupleList_list = [(ent1, ent2) for ent1, ent2 in testTupleList_list if ent2 in url2CandidateUrlList_dict[ent1][: topk]]
        elif self.args.test_strategy == 'at_topk_and_not_at_rank_1':
            logger.info("Test strategy: At topk and not at rank 1")
            sampledTestTupleList_list = [(ent1, ent2) for ent1, ent2 in testTupleList_list if ent2 in url2CandidateUrlList_dict[ent1][: topk] and url2CandidateUrlList_dict[ent1][0] != ent2]
        elif self.args.test_strategy == 'same_proportion_as_test_set':
            logger.info("Test strategy: Same proportion as test set")
            if sample_num is not None:
                not_at_topk = len([(ent1, ent2) for ent1, ent2 in testTupleList_list if ent2 not in url2CandidateUrlList_dict[ent1][: topk]])
                at_top1 = len([(ent1, ent2) for ent1, ent2 in testTupleList_list if ent2 in url2CandidateUrlList_dict[ent1][: topk] and url2CandidateUrlList_dict[ent1][0] == ent2])
                at_between = len([(ent1, ent2) for ent1, ent2 in testTupleList_list if ent2 in url2CandidateUrlList_dict[ent1][: topk] and url2CandidateUrlList_dict[ent1][0] != ent2])
                assert not_at_topk + at_top1 + at_between == len(testTupleList_list), "Error in same_proportion_as_test_set"
                sample_num_not_at_topk = int(sample_num * not_at_topk / len(testTupleList_list))
                sample_num_at_top1 = int(sample_num * at_top1 / len(testTupleList_list))
                sample_num_at_between = int(sample_num * at_between / len(testTupleList_list))
                testTupleList_list_not_at_topk = [(ent1, ent2) for ent1, ent2 in testTupleList_list if ent2 not in url2CandidateUrlList_dict[ent1][: topk]]
                testTupleList_list_at_top1 = [(ent1, ent2) for ent1, ent2 in testTupleList_list if ent2 in url2CandidateUrlList_dict[ent1][: topk] and url2CandidateUrlList_dict[ent1][0] == ent2]
                testTupleList_list_at_between = [(ent1, ent2) for ent1, ent2 in testTupleList_list if ent2 in url2CandidateUrlList_dict[ent1][: topk] and url2CandidateUrlList_dict[ent1][0] != ent2]
                sampledTestTupleList_list = random.sample(testTupleList_list_not_at_topk, sample_num_not_at_topk) + random.sample(testTupleList_list_at_top1, sample_num_at_top1) + random.sample(testTupleList_list_at_between, sample_num_at_between)
        else:
            logger.info("Test strategy: All")
            sampledTestTupleList_list = copy.deepcopy(testTupleList_list)
        logger.info(f"Test tuple list: {len(sampledTestTupleList_list)}")

        if sample_num is not None:
            if sample_num > len(sampledTestTupleList_list):
                logger.info(f"Sample number {sample_num} is larger than the test tuple list {len(sampledTestTupleList_list)}, sample number is set to the length of the test tuple list")
                sample_num = len(sampledTestTupleList_list)
            sampledTestTupleList_list = random.sample(sampledTestTupleList_list, sample_num)

        setattr(self, 'testTupleList_list', sampledTestTupleList_list)
        # logger.info(sampledTestTupleList_list)
        # logger.info(self.testTupleList_list)

        for ent1, ent2 in self.testTupleList_list:
            if idealTest:
                if ent2 in url2CandidateUrlList_dict[ent1][: topk]:
                    toBeRerankedPool.add(ent1)
                    candidatePool.update(set(url2CandidateUrlList_dict[ent1][: topk]))                
            else:
                toBeRerankedPool.add(ent1)
                candidatePool.update(set(url2CandidateUrlList_dict[ent1][: topk]))
        
        setattr(self, 'toBeRerankedPool', toBeRerankedPool)
        setattr(self, 'unCertainPool', unCertainPool)
        if idealTest:
            setattr(self, 'killedPool', None)
        else:
            setattr(self, 'killedPool', killedPool)
        setattr(self, 'candidatePool', candidatePool)
        setattr(self, 'rerankedPairsPool', rerankedPairsPool)
        setattr(self, 'url2CandidateUrlList_dict', url2CandidateUrlList_dict)

        """
        statistic
        """
        logger.info(f"Number of entities to be reranked: {len(toBeRerankedPool)}")

    def getOneEntity(self):
        return self.toBeRerankedPool.pop()
    
    def getCandidateEntities(self, url):
        original_candidate_list = self.url2CandidateUrlList_dict[url]
        if self.args.candidate_filter:
            filtered_candidate_list = []
            for candidate in original_candidate_list:
                if candidate in self.candidatePool:
                    filtered_candidate_list.append(candidate)
            return filtered_candidate_list
        return original_candidate_list

    def entity_processing(self, url, pred_label, pred_aligned):
        if pred_label == 1:
            self.rerankedPairsPool.add((url, pred_aligned))
            self.candidatePool.discard(pred_aligned)
            self.unCertainPool.pop(url, None)
        else:
            if url in self.unCertainPool:
                self.unCertainPool[url] += 1
                if self.unCertainPool[url] >= self.args.maxUncertain:
                    self.killedPool.add(url)
                    self.unCertainPool.pop(url, None)
            else:
                self.unCertainPool[url] = 1
        
        self.toBeRerankedPool.discard(url)

    def fuse(self):
        self.toBeRerankedPool.update(self.unCertainPool)

    def empty(self):
        return len(self.toBeRerankedPool) == 0

    def statistic(self):
        logger.info(f"toBeRerankedPool: {len(self.toBeRerankedPool)}\n")
        logger.info(f"unCertainPool: {len(self.unCertainPool)}\n")
        logger.info(f"killedPool: {len(self.killedPool)}\n")
        logger.info(f"candidatePool: {len(self.candidatePool)}\n")
        logger.info(f"rerankedPairsPool: {len(self.rerankedPairsPool)}\n")

class RelationshipStrategy:
    def __init__(self, url2EntityObject_dict, url2EntityInfo_dict):
        self.url2EntityObject_dict = url2EntityObject_dict
        self.url2EntityInfo_dict = url2EntityInfo_dict
    
    def functionality(self, url: str, max_num=3, sampled=None, sample_strategy=None):
        assert url in self.url2EntityObject_dict, f"Error in functionality: {url}"
        if getattr(self, 'rel_functionality_dict', None) is None:
            self.rel_functionality_dict = self.calculate_rel_functionality()
        rels =  self.url2EntityInfo_dict[url]['relationships']
        sorted_rels = sorted(rels, key=lambda x: self.rel_functionality_dict[x[1]], reverse=True)
        if sampled:
            sorted_rels = self.sample(sorted_rels, max_num, sample_strategy)
        rel_str = ""
        added_num = 0
        for i, (e1, r, e2) in enumerate(sorted_rels):
            if added_num >= max_num:
                break
            rel_str += f"<{e1}, {r}, {e2}>\t"
            added_num += 1
        return rel_str

    def sample(self, rel_list, sample_num, sample_strategy='linspace'):
        if sample_strategy == 'linspace':
            return self.linspace_sample(rel_list, sample_num)
        else:
            raise ValueError(f"Unknown sample strategy: {sample_strategy}")
    
    def linspace_sample(self, rel_list, sample_num):
        min_probability = 0.05
        weights = np.linspace(start=1, stop=min_probability, num=len(rel_list))
        weights /= weights.sum()
        size = min(sample_num, len(rel_list))
        sampled_indices = np.random.choice(len(rel_list), size=size, replace=False, p=weights)
        samples = [rel_list[i] for i in sampled_indices]
        return samples


    def calculate_rel_functionality(self):
        """
        Calculate the functionality of each relationship
        """
        functionality_dict = {}
        relations = set()
        for ent_url, entity in self.url2EntityObject_dict.items():
            relationships = entity.relationships
            relations.update(set(relationships.keys()))
        
        for rel in relations:
            rel_name = rel.extract_name()
            functionality_dict[rel_name] = [0, 0]
            head_tail_dicts = rel.head_tail_dicts
            for head, tails in head_tail_dicts.items():
                functionality_dict[rel_name][0] += len(tails)
                functionality_dict[rel_name][1] += 1
        
        function_dict = {}
        for rel_name, (total, cnt) in functionality_dict.items():
            assert rel_name not in function_dict, f"Error in calculate_rel_functionality: {rel_name}"
            function_dict[rel_name] = cnt / total
        
        return function_dict

    def __call__(self, url: str, rel_strategy, max_num=5, sampled=None, sample_strategy=None, **kwargs):
        if rel_strategy == 'functionality':
            return self.functionality(url, max_num, sampled, sample_strategy)
        else:
            raise ValueError(f"Unknown relationship strategy: {rel_strategy}")


class AttributionStrategy:

    def __init__(self, url2EntityObject_dict, url2EntityInfo_dict):
        self.url2EntityObject_dict = url2EntityObject_dict
        self.url2EntityInfo_dict = url2EntityInfo_dict

    def functionality(self, url: str, max_num=3, sampled=None, sample_strategy=None):
        assert url in self.url2EntityObject_dict, f"Error in functionality: {url}"
        if getattr(self, 'attr_functionality_dict', None) is None:
            self.attr_functionality_dict = self.calculate_attr_functionality()
        attr = self.url2EntityInfo_dict[url]['attributes']
        sorted_attr = sorted(attr, key=lambda x: self.attr_functionality_dict[x[1]], reverse=True)

        if sampled:
            sorted_attr = self.sample(sorted_attr, max_num, sample_strategy)

        attr_str = ""
        added_num = 0
        for i, (e, a, v) in enumerate(sorted_attr):
            if added_num >= max_num:
                break
            attr_str += f"<{e}, {a}, {v}>\t"
            added_num += 1
        
        return attr_str

    def linspace_sample(self, attr_list, sample_num):
        min_probability = 0.05
        weights = np.linspace(start=1, stop=min_probability, num=len(attr_list))
        weights /= weights.sum()
        size = min(sample_num, len(attr_list))
        sampled_indices = np.random.choice(len(attr_list), size=size, replace=False, p=weights)
        samples = [attr_list[i] for i in sampled_indices]
        return samples
    
    def exponential_sample(self, attr_list, sample_num):
        decay_factor = 0.5
        weights = np.array([decay_factor**i for i in range(len(attr_list))])
        weights /= weights.sum()
        size = min(sample_num, len(attr_list))
        sampled_indices = np.random.choice(len(attr_list), size=size, replace=False, p=weights)
        samples = [attr_list[i] for i in sampled_indices]
        return samples

    def sample(self, attr_list, sample_num, sample_strategy='linspace'):
        if sample_strategy == 'linspace':
            return self.linspace_sample(attr_list, sample_num)
        elif sample_strategy == 'exponential':
            return self.exponential_sample(attr_list, sample_num)
        else:
            raise ValueError(f"Unknown sample strategy: {sample_strategy}")

    def statistic(self, testTupleList_list):
        if getattr(self, 'attr_functionality_dict', None) is None:
            self.attr_functionality_dict = self.calculate_attr_functionality()
        
        aligned_pair_attr_statistic = {}
        for i, (ent1, ent2) in enumerate(testTupleList_list):
            ent1_attr = self.url2EntityInfo_dict[ent1]['attributes']
            ent2_attr = self.url2EntityInfo_dict[ent2]['attributes']
            aligned_pair_attr_statistic[i] = {}
            aligned_pair_attr_statistic[i]['pair'] = [ent1, ent2]
            aligned_pair_attr_statistic[i][ent1] = {}
            aligned_pair_attr_statistic[i][ent2] = {}
            aligned_pair_attr_statistic[i][ent1]['attr'] = {}
            aligned_pair_attr_statistic[i][ent2]['attr'] = {}
            for e, a, v in ent1_attr:
                if a not in aligned_pair_attr_statistic[i][ent1]['attr']:
                    aligned_pair_attr_statistic[i][ent1]['attr'][a] = []
                aligned_pair_attr_statistic[i][ent1]['attr'][a].append(v)
            
            for e, a, v in ent2_attr:
                if a not in aligned_pair_attr_statistic[i][ent2]['attr']:
                    aligned_pair_attr_statistic[i][ent2]['attr'][a] = []
                aligned_pair_attr_statistic[i][ent2]['attr'][a].append(v)
            
            aligned_pair_attr_statistic[i][ent1]['attr_num'] = len(ent1_attr)
            aligned_pair_attr_statistic[i][ent2]['attr_num'] = len(ent2_attr)

            aligned_pair_attr_statistic[i]['same_attr'] = {}
            for e1, a1, v1 in ent1_attr:
                if a1 in aligned_pair_attr_statistic[i][ent2]['attr']:
                    if a1 not in aligned_pair_attr_statistic[i]['same_attr']:
                        aligned_pair_attr_statistic[i]['same_attr'][a1] = []

            for a in aligned_pair_attr_statistic[i]['same_attr']:
                e1_same_attr_value = list(aligned_pair_attr_statistic[i][ent1]['attr'][a])
                e2_same_attr_value = list(aligned_pair_attr_statistic[i][ent2]['attr'][a])
                aligned_pair_attr_statistic[i]['same_attr'][a].append(e1_same_attr_value)
                aligned_pair_attr_statistic[i]['same_attr'][a].append(e2_same_attr_value)

        with open('aligned_pair_attr_statistic.json', 'w', encoding='utf-8') as f:
            json.dump(aligned_pair_attr_statistic, f, indent=4, ensure_ascii=False)
        return aligned_pair_attr_statistic
    
    
    def calculate_attr_functionality(self):
        """
        Calculate the functionality of each attribute
        """
        functionality_dict = {}
        for ent_url, entity in self.url2EntityObject_dict.items():
            entity_attr_dict = entity.attributes
            value_set = set()
            for attr, value_list in entity_attr_dict.items():
                value_set.update(set(value_list))
                attr_name = attr.get_name()
                if attr_name not in functionality_dict:
                    functionality_dict[attr_name] = [0, 0]
                functionality_dict[attr_name][0] += len(set(value_list))
                functionality_dict[attr_name][1] += 1

        doc_num = len(self.url2EntityObject_dict)
        for attr, (total, cnt) in functionality_dict.items():
            functionality_dict[attr] = (cnt / total) * (cnt / doc_num)
        return functionality_dict

    def str_cooccurrence(self, url: str, max_num=3, sampled=None, sample_strategy=None, **kwargs):
        is_central = kwargs.get('is_central', False)
        is_candidate = kwargs.get('is_candidate', False)
        centity_url = kwargs.get('centity_url', None)
        candiates_list = kwargs.get('candiates_list', None)
        assert not (is_central and is_candidate), "Not both central and candidate"
        assert is_central or is_candidate, "Either central or candidate"
        if is_central:
            assert candiates_list is not None, "candiates_list cannot be None"
            centity_attr_list = []
            centity_url_attr = self.url2EntityInfo_dict[url]['attributes']
            candidate_attr_set = set()
            for candidate in candiates_list:
                candidate_attr = self.url2EntityInfo_dict[candidate]['attributes']
                for e, a, v in candidate_attr:
                    candidate_attr_set.add(a)
            
            for e, a, v in centity_url_attr:
                if a in candidate_attr_set:
                    centity_attr_list.append((e, a, v))
            
            centity_attr_str = ""

            for i, (e, a, v) in enumerate(centity_attr_list):
                centity_attr_str += f"<{e}, {a}, {v}>\t"
            if getattr(self, 'str_cooccurrence_dict', None) is None:
                self.str_cooccurrence_dict = {}
            self.str_cooccurrence_dict[url] = set([a for e, a, v in centity_attr_list])


            if len(centity_attr_list) < max_num:
                if getattr(self, 'attr_functionality_dict', None) is None:
                    self.attr_functionality_dict = self.calculate_attr_functionality()
                unused_attr = list(set(centity_url_attr) - set(centity_attr_list))
                if len(unused_attr) == 0:
                    return centity_attr_str

                sorted_attr = sorted(unused_attr, key=lambda x: self.attr_functionality_dict[x[1]], reverse=True)
                if sampled:
                    sorted_attr = self.sample(sorted_attr, max_num - len(centity_attr_list), sample_strategy)

                added_num = 0
                for i, (e, a, v) in enumerate(sorted_attr):
                    if added_num >= max_num - len(centity_attr_list):
                        break
                    centity_attr_str += f"<{e}, {a}, {v}>\t"
                    added_num += 1

            return centity_attr_str
            
        elif is_candidate:
            assert centity_url is not None, "centity_url cannot be None"
            assert centity_url in self.str_cooccurrence_dict, "centity_url not in str_cooccurrence_dict"
            centity_attr_set = self.str_cooccurrence_dict[centity_url]
            candidate_attr_list = self.url2EntityInfo_dict[url]['attributes']
            candidate_attr_str = ""
            attr_cnt = 0
            used_attr_index = []
            for i, (e, a, v) in enumerate(candidate_attr_list):
                if attr_cnt >= max_num:
                    break
                if a in centity_attr_set:
                    candidate_attr_str += f"<{e}, {a}, {v}>\t"
                    attr_cnt += 1
                    used_attr_index.append(i)

            if attr_cnt < max_num:
                if getattr(self, 'attr_functionality_dict', None) is None:
                    self.attr_functionality_dict = self.calculate_attr_functionality()
                unused_attr_index = list(set(range(len(candidate_attr_list))) - set(used_attr_index))
                ununsed_attr = [candidate_attr_list[i] for i in unused_attr_index]
                if len(ununsed_attr) == 0:
                    return candidate_attr_str
                sorted_attr = sorted(ununsed_attr, key=lambda x: self.attr_functionality_dict[x[1]], reverse=True)

                if sampled:
                    sorted_attr = self.sample(sorted_attr, max_num - attr_cnt, sample_strategy)

                added_num = 0
                for i, (e, a, v) in enumerate(sorted_attr):
                    if added_num >= max_num - attr_cnt:
                        break
                    candidate_attr_str += f"<{e}, {a}, {v}>\t"
                    added_num += 1
            
            return candidate_attr_str

        else:
            raise ValueError("Unknown type")

    def __call__(self, url: str, attr_strategy , max_num=3, sampled=None, sample_strategy=None, **kwargs):
        if attr_strategy == 'functionality':
            return self.functionality(url, max_num, sampled, sample_strategy)
        elif attr_strategy == 'str_cooccurrence':
            return self.str_cooccurrence(url, max_num, sampled, sample_strategy, **kwargs)
        else:
            raise ValueError(f"Unknown attribute strategy: {attr_strategy}")


class BootstrappingStrategy:
    
    def __init__(self) -> None:
        """
        None of the strategies need to be initialized
        """
        pass

    def shuffle_candidates(self, candidate_url_list):
        random.shuffle(candidate_url_list)
    
    def nope(self, candidate_url_list):
        pass
    
    def reverse(self, candidate_url_list):
        candidate_url_list.reverse()

    def position(self, candidate_url_list, groundtruth, pos):
        assert pos is not None, "Pos is None!"
        p_index = -1
        if groundtruth in candidate_url_list:
            p_index = candidate_url_list.index(groundtruth)
            assert p_index != -1, f"Error in position: {groundtruth} not in candidate_url_list"
            assert pos >= 1 and pos <=10
            p_url = candidate_url_list.pop(p_index)
            candidate_url_list.insert(pos, p_url)
        else:
            logger.info(f"Error in position: {groundtruth} not in candidate_url_list")

    def __call__(self, strategy, candidate_url_list, **kwargs):
        if strategy == "shuffle":
            self.shuffle_candidates(candidate_url_list)
        elif strategy == "nope":
            self.nope(candidate_url_list)
        elif strategy == "reverse":
            self.reverse(candidate_url_list)
        elif strategy == "position":
            pos = kwargs.get('position', None)
            groundtruth = kwargs.get('groundtruth', None)
            self.position(candidate_url_list, groundtruth, pos)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

def prompt_preprocess(prompts, model_type, tokenizer):
    
    if model_type == 'qwen-chat':
        if isinstance(prompts, list):
            processed_prompts = []
            for prompt in prompts:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                processed_prompts.append(text)
            return processed_prompts
        elif isinstance(prompts, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompts}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return text
    else:
        raise ValueError(f"Unknown model type: {model_type}")

class LLMsReranker:
    def __init__(self, 
                 args, 
                 url2CandidateUrlList_dict, 
                 id2CandidateIdList_dict, 
                 id2CandidateScoreList_dict, 
                 etsSequenceList_list, 
                 url2EntityInfo_dict, 
                 testTupleList_list, 
                 url2EntityObject_dict):

        """
        url2CandidateUrlList_dict: {url: [url1, url2, url3, ...] }
        id2CandidateIdList_dict: {id: [id1, id2, id3, ...] }
        id2CandidateScoreList_dict: {id: [score1, score2, score3, ...] }
        etsSequenceList_list: [sequence1, sequence2, sequence3, ...]
        url2EntityInfo_dict: {url: {'name_or_url_info': str, 'attributes': [(e, a, v), (e, a, v), ...], 'relationships': [(e, r, e), (e, r, e), ...]}, 'mixed': str}
        testTupleList_list: [(url1, url2), (url3, url4), ...]
        url2EntityObject_dict: {url: entity_object}
        """
        self.args = args
        self.url2CandidateUrlList_dict = url2CandidateUrlList_dict
        self.id2CandidateIdList_dict = id2CandidateIdList_dict
        self.id2CandidateScoreList_dict = id2CandidateScoreList_dict
        self.etsSequenceList_list = etsSequenceList_list
        self.url2EntityInfo_dict = url2EntityInfo_dict
        self.testTupleList_list = testTupleList_list
        self.url2urlTestPair_dict = {url1: url2 for url1, url2 in testTupleList_list}
        self.url2EntityObject_dict = url2EntityObject_dict

        self.bootstrapingStrategy = BootstrappingStrategy()
        self.attributeStrategy = AttributionStrategy(url2EntityObject_dict, url2EntityInfo_dict)
        self.relationshipStrategy = RelationshipStrategy(url2EntityObject_dict, url2EntityInfo_dict)
        # self.attributeStrategy.statistic(testTupleList_list)
        # self.attr_functionality_dict = self.calculate_attr_functionality()
        self.init_llms()
    

    def init_llms(self):
        assert not(self.args.openai and self.args.vllm), 'Both VLLM and OpenAI cannot be used at the same time, please specify only one of them'
        if self.args.vllm:
            llm_path = self.args.llm_path
            tensor_parallel_size = self.args.tensor_parallel_size
            swap_space = self.args.swap_space
            quantization = self.args.quantization
            self.model = LLM(model=llm_path, tensor_parallel_size=tensor_parallel_size, swap_space=swap_space, seed=self.args.seed, quantization=quantization)
            logger.info(f"Successfully loaded VLLM model: {llm_path}")
        else:
            logger.info("Using OpenAI API for reranking")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)


    def send_chat_request(self, message):
        # url = "https://fast.xeduapi.com/v1/chat/completions"
        url = self.args.openai_url
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.args.openai_key}"
        }
        data = {
            "model": self.args.openai_model,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
            "temperature": self.args.temperature
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']

    def calculate_attr_functionality(self):
        """
        Calculate the functionality of each attribute
        """
        functionality_dict = {}
        for ent_url, entity in self.url2EntityObject_dict.items():
            entity_attr_dict = entity.attributes
            value_set = set()
            for attr, value_list in entity_attr_dict.items():
                value_set.update(set(value_list))
                attr_name = attr.get_name()
                if attr_name not in functionality_dict:
                    functionality_dict[attr_name] = [0, 0]
                functionality_dict[attr_name][0] += len(set(value_list))
                functionality_dict[attr_name][1] += 1

        doc_num = len(self.url2EntityObject_dict)
        for attr, (total, cnt) in functionality_dict.items():
            functionality_dict[attr] = (cnt / total) * (cnt / doc_num)
        return functionality_dict


    def inference_1_step(self, entityPool: EntityPool):
        ent1 = entityPool.getOneEntity()
        candidate_list = entityPool.getCandidateEntities(ent1)
        pred_label, pred_aligned = self.rerank(ent1, candidate_list)
        entityPool.entity_processing(ent1, pred_label, pred_aligned)

    def rerank(self, url, candidate_url_list):
        """
        ent1: str
        Return:
            rerank_results: list
            ground_truth_choice: list
        """
        prompts, choices, candidates = self.construct_input(url, candidate_url_list)

        temperature = self.args.temperature
        top_p = self.args.top_p
        max_tokens = self.args.max_tokens
        repetition_penalty = self.args.repetition_penalty
        stop = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop, repetition_penalty=repetition_penalty)
        prompts = prompt_preprocess(prompts, "qwen-chat", self.tokenizer)
        vllm_outputs = self.model.generate(prompts, sampling_params=sampling_params)
        rerank_results = []
        for res in vllm_outputs:
            rerank_results.append(res.outputs[0].text)
        pred_choices = []
        for i, one_result in enumerate(rerank_results):
            pred_choices.append(self.parser_1(one_result))
        pred_label = -1
        pred_aligned = -1
        pred_cnt_dict = {}
        for i, pred_choice in enumerate(pred_choices):
            i_candidate = candidates[i]
            if pred_choice is not None:
                index = ord(pred_choice) - ord('a')
                if index < len(i_candidate):
                    pred_cnt_dict[i_candidate[index]] = pred_cnt_dict.get(i_candidate[index], 0) + 1
        
        if len(pred_cnt_dict) > 0:
            pred_label = 1
            pred_aligned = max(pred_cnt_dict, key=pred_cnt_dict.get)
            pred_max_cnt = pred_cnt_dict[pred_aligned]
            if self.args.bootstraping and self.args.bootstraping_num > 1 and pred_max_cnt <= self.args.bootstraping_num // 2:
                pred_label = -1
        
        if self.args.verbose:
            self.verbose_log(url, prompts, rerank_results, pred_choices, pred_cnt_dict, pred_label, pred_aligned)

        return pred_label, pred_aligned

    def verbose_log(self, url, prompts, rerank_results, pred_choices, pred_cnt_dict, pred_label, pred_aligned):
        logger.info("-" * 20)
        logger.info(f"The entity being inferred is {url}")
        for i, one_result in enumerate(rerank_results):
            logger.info(f"Prompt: {prompts[i]}")
            logger.info(f"Result: {one_result}")
            logger.info(f"Predicted choice: {pred_choices[i]}")
        
        logger.info(f"Predicted count dict: {pred_cnt_dict}")
        logger.info(f"Predicted label: {pred_label}")
        logger.info(f"Predicted aligned: {pred_aligned}")
        logger.info(f"Ground truth: {self.url2urlTestPair_dict[url]}")
        logger.info(f"Is the prediction correct?: {pred_aligned == self.url2urlTestPair_dict[url]}")
        logger.info("-" * 20)
        logger.info("\n")


    def inference(self):
        entityPool = EntityPool(
            self.args,
            self.testTupleList_list,
            self.url2CandidateUrlList_dict,
            self.args.topk,
            self.args.idealTest,
            sample_num=self.args.sample_num
        )
        if self.args.experiment == 'all_framework':
            for _ in range(self.args.max_infer_iter):
                if _ % 2 == 0:
                    self.args.use_attr = True
                    self.args.use_rel = False
                else:
                    self.args.use_attr = False
                    self.args.use_rel = True
                while not entityPool.empty():
                    self.inference_1_step(entityPool)
                
                logger.info(f"The {_} iteration is complete")
                entityPool.statistic()
                entityPool.fuse()
        
            self.metric(entityPool)
        elif self.args.experiment == 'only_attr':
            self.args.use_name = True
            self.args.use_attr = True
            self.args.use_rel = False
            while not entityPool.empty():
                self.inference_1_step(entityPool)
            entityPool.statistic()
            self.metric(entityPool)
        elif self.args.experiment == 'only_rel':
            self.args.use_name = True
            self.args.use_attr = False
            self.args.use_rel = True
            while not entityPool.empty():
                self.inference_1_step(entityPool)
            entityPool.statistic()
            self.metric(entityPool)
        elif self.args.experiment == 'only_name':
            self.args.use_attr = False
            self.args.use_rel = False
            self.args.use_name = True
            while not entityPool.empty():
                self.inference_1_step(entityPool)
            entityPool.statistic()
            self.metric(entityPool)
        elif self.args.experiment == 'without_bootstraping':
            self.args.bootstraping = True
            self.args.bootstraping_num = 1
            for _ in range(self.args.max_infer_iter):
                if _ % 2 == 0:
                    self.args.use_attr = True
                    self.args.use_rel = False
                else:
                    self.args.use_attr = False
                    self.args.use_rel = True
                while not entityPool.empty():
                    self.inference_1_step(entityPool)
                
                logger.info(f"The {_} iteration is complete")
                entityPool.statistic()
                entityPool.fuse()
        
            self.metric(entityPool)
        elif self.args.experiment == 'without_bootstraping_and_only_attr':
            self.args.use_name = True
            self.args.use_attr = True
            self.args.use_rel = False
            self.args.bootstraping = True
            self.args.bootstraping_num = 1
            while not entityPool.empty():
                self.inference_1_step(entityPool)
            entityPool.statistic()
            self.metric(entityPool)
        elif self.args.experiment == 'without_bootstraping_and_only_rel':
            self.args.use_name = True
            self.args.use_attr = False
            self.args.use_rel = True
            self.args.bootstraping = True
            self.args.bootstraping_num = 1
            while not entityPool.empty():
                self.inference_1_step(entityPool)
            entityPool.statistic()
            self.metric(entityPool)
        elif self.args.experiment == 'without_bootstraping_and_only_name':
            self.args.use_name = True
            self.args.use_attr = False
            self.args.use_rel = False
            self.args.bootstraping = True
            self.args.bootstraping_num = 1
            while not entityPool.empty():
                self.inference_1_step(entityPool)
            entityPool.statistic()
            self.metric(entityPool)
        elif self.args.experiment == 'without_bootstraping_and_only_name_and_attr':
            self.args.use_name = True
            self.args.use_attr = True
            self.args.use_rel = False
            self.args.bootstraping = True
            self.args.bootstraping_num = 1
            while not entityPool.empty():
                self.inference_1_step(entityPool)
            entityPool.statistic()
            self.metric(entityPool)
        elif self.args.experiment == 'position_bias_ablation':
            testUrlDictTmp = {}
            for url1, url2 in self.testTupleList_list:
                testUrlDictTmp[url1] = url2
            setattr(self, 'testUrlDictTmp', testUrlDictTmp)
        elif self.args.experiment == 'candidate_size_ablation':
            pass
        else:
            raise ValueError(f"Unknown experiment: {self.args.experiment}")
        
    def construct_name(self, url):
        return self.url2EntityObject_dict[url].url.split("/")[-1].replace("_", " ")

    def construct_entity(self, url, candidate_url_list=None):
        entity_name = self.construct_name(url)
        # rel = self.entity_info_dict[url]['relationships']
        entity_seq = ""
        if self.args.use_name:
            entity_seq += f"{entity_name}\n"
        
        if self.args.use_attr:
            entity_seq += "Attributes:\n"
            attr = self.attributeStrategy(
                url,
                self.args.attr_strategy,
                self.args.attr_max_num,
                self.args.attr_sampled,
                self.args.attr_sample_strategy,
                is_central=True,
                candiates_list=candidate_url_list,
            )
            entity_seq += f"{attr}\n"

        if self.args.use_rel:
            entity_seq += "Relationships:\n"
            rel = self.relationshipStrategy(
                url,
                self.args.rel_strategy,
                self.args.rel_max_num,
                self.args.rel_sampled,
                self.args.rel_sample_strategy,
            )
            entity_seq += f"{rel}\n"

        return entity_seq

    def construct_candaite_entity(self, url, centity_url):
        entity_name = self.construct_name(url)
        # rel = self.entity_info_dict[url]['relationships']
        entity_seq = ""
        if self.args.use_name:
            entity_seq += f"{entity_name}\n"
        
        if self.args.use_attr:
            entity_seq += "Attributes:\n"
            attr = self.attributeStrategy(
                url,
                self.args.attr_strategy,
                self.args.attr_max_num,
                self.args.attr_sampled,
                self.args.attr_sample_strategy,
                is_candidate=True,
                centity_url=centity_url,
            )
            entity_seq += f"{attr}\n"
        
        if self.args.use_rel:
            entity_seq += "Relationships:\n"
            rel = self.relationshipStrategy(
                url,
                self.args.rel_strategy,
                self.args.rel_max_num,
                self.args.rel_sampled,
                self.args.rel_sample_strategy,
            )
            entity_seq += f"{rel}\n"

        return entity_seq

    def construct_choices(self, url, candidate_url_list):
        candidate_ents_str = ""
        for i, one_candidate in enumerate(candidate_url_list):
            candidate_ents_str += f"({chr(97+i)}) {self.construct_candaite_entity(one_candidate, url)}\n"
        return candidate_ents_str

    def construct_one_input(self, url, candidate_url_list):
        centrality = self.construct_entity(url, candidate_url_list)
        choices = self.construct_choices(url, candidate_url_list)
        return centrality, choices

    def construct_input(self, url, candidate_url_list):
        instruction = RerankInstruction.instruction_1
        inputs = []
        choices = []
        candiates = []
        if self.args.bootstraping:
            bootstraping_num = self.args.bootstraping_num
            
            assert bootstraping_num > 0 and bootstraping_num < len(candidate_url_list), "Error in bootstraping"
            if bootstraping_num == 1:
                logger.info("=" * 20)
                logger.info(f"Bootstraping number is 1, which is just shuffling the candidate list compared to the original one")
                logger.info("=" * 20)
            for _ in range(bootstraping_num):
                """
                shuffle the candidate_url_list
                """
                copyed_candidate_url_list = copy.deepcopy(candidate_url_list)
                if self.args.experiment == "position_bias_ablation":
                    assert url in self.testUrlDictTmp, f"Error in position_bias_ablation: {url}"
                    self.bootstrapingStrategy(self.args.bootstraping_strategy, copyed_candidate_url_list, groundtruth=self.testUrlDictTmp[url], position=self.args.position)
                else:
                    self.bootstrapingStrategy(self.args.bootstraping_strategy, copyed_candidate_url_list)
                now_entitiy, candidate_ents_str = self.construct_one_input(url, copyed_candidate_url_list)
                inputs.append(instruction.format("", now_entitiy, candidate_ents_str))
                if url in copyed_candidate_url_list:
                    choices.append(copyed_candidate_url_list.index(url))
                else:
                    choices.append(-1)
                candiates.append(copy.deepcopy(copyed_candidate_url_list))
        else:
            now_entitiy, candidate_ents_str = self.construct_one_input(url, candidate_url_list)
            """
            position of url
            """
            if url in candidate_url_list:
                choices.append(candidate_url_list.index(url))
            else:
                choices.append(-1)
            inputs.append(instruction.format("", now_entitiy, candidate_ents_str))
            candiates.append(copy.deepcopy(candidate_url_list))

        return inputs, choices, candiates


    def parser_1(self, text):
        pattern = re.compile(r'(?i)(?:the\s+)?(?:correct\s+)?answer\s*(?:should\s+be)?\s*(?:is)?\s*[:=]?\s*\(([a-zA-Z])\)')
        # 搜索文本中的匹配项
        match = pattern.search(text)
        # 如果找到匹配项，则返回提取的答案
        if match:
            return match.group(1).lower()  # 将答案转换为小写
        else:
            return None

    def metric(self, entityPool: EntityPool):
        test_pair_num = len(entityPool.testTupleList_list)
        correct_num = 0
        for ent1, ent2 in entityPool.testTupleList_list:
            if (ent1, ent2) in entityPool.rerankedPairsPool:
                correct_num += 1
        
        logger.info(f"Correct number: {correct_num}")
        logger.info(f"Total number: {test_pair_num}")
        logger.info(f"Hits@1: {correct_num / test_pair_num}")

    def functional_attr_format(self, attr):
        sorted_attr = sorted(attr, key=lambda x: self.attr_functionality_dict[x[1]], reverse=True)
        attr_str = ""
        cnt = 0
        pre_functionality = -1
        pre_attr_str = ""
        for (e, a, v) in sorted_attr:
            if cnt >= self.attr_format_max_num:
                break
            if pre_functionality != self.attr_functionality_dict[a] and pre_attr_str != a:
                pre_functionality = self.attr_functionality_dict[a]
                pre_attr_str = a
                cnt += 1
                attr_str += f"<{e}, {a}, {v}>\t"
        
        return attr_str


class LLMReranker:
    def __init__(self, config, candidate_url_set, candidate_id_set, score_dict, ents, entity_info_dict, test_pair_url, entities, attr_format_max_num=3):
        """
        entity_info_dict: {url: {'mixed': str, 'relationships': str, 'attributes': str, 'mixed': str }}
        candidate_url_set: {url: [url1, url2, url3, ...] }
        """
        """
        set seed
        """

        self.config = config['llmrerank']

        assert not(self.config.get('use_vllm', False) and self.config.get('use_openai', False)), 'Both VLLM and OpenAI cannot be used at the same time, please specify only one of them'
        if self.config.get('use_vllm', False):
            assert 'vllm' in self.config, 'VLLM config is missing'
            # self.load_vllm()
        elif self.config.get('use_openai', False):
            assert 'openai' in self.config, 'OpenAI config is missing'
            # self.load_openai()
        else:
            raise ValueError('No LLM model specified')

        self.candidate_url_set = candidate_url_set
        self.candidate_id_set = candidate_id_set
        self.score_dict = score_dict
        self.ents = ents
        self.entity_info_dict = entity_info_dict
        self.instruction = RerankInstruction()
        self.test_pair_url = test_pair_url
        self.entities = entities
        self.demonstration = Demonstration()
        self.attr_functionality_dict = self.calculate_attr_functionality()
        self.attr_format_max_num = attr_format_max_num

    def load_vllm(self):
        llm_path = self.config['llm_path']
        tensor_parallel_size = self.config['vllm'].get('tensor_parallel_size', 1)
        swap_space = self.config['vllm'].get('swap_space', 4)
        llm = LLM(model=llm_path, tensor_parallel_size=tensor_parallel_size, swap_space=swap_space)
        logger.info(f"Successfully loaded VLLM model: {llm_path}")
        self.model = llm
    
    def load_openai(self):
        logger.info("Using OpenAI API for reranking")
    
    def hard_case_threshold(self):
        def hits_at_k_according_threshold(ll, rr, score_np, k, label):
            position = (score_np > ll) & (score_np < rr)
            hits_at_k = 0
            now_range_url_list = []
            queries_pos = np.where(position)[0]
            for pos in queries_pos:
                if self.test_pair_url[pos][1] in self.candidate_url_set[self.test_pair_url[pos][0]][: k]:
                    hits_at_k += 1
                now_range_url_list.append(self.test_pair_url[pos][0])
            logger.info(f"Threshold range: [{ll}, {rr}]  \nHits@{k} {label}: {hits_at_k / np.sum(position)} \nNumber of queries: {np.sum(position)} \nRatio of queries: {np.sum(position) / len(self.test_pair_url)}\n")
            return now_range_url_list

        def hits_at_k_according_cluster_labels(labels, ith_cluster, k, label):
            position = labels == ith_cluster
            hits_at_k = 0
            now_range_url_list = []
            queries_pos = np.where(position)[0]
            for pos in queries_pos:
                if self.test_pair_url[pos][1] in self.candidate_url_set[self.test_pair_url[pos][0]][: k]:
                    hits_at_k += 1
                now_range_url_list.append(self.test_pair_url[pos][0])
            logger.info(f"Cluster label: {ith_cluster} \nHits@{k} {label}: {hits_at_k / np.sum(position)} \nNumber of queries: {np.sum(position)} \nRatio of queries: {np.sum(position) / len(self.test_pair_url)}\n")
            return now_range_url_list

        retrieved_docs_scores = []
        test_ent_url2id = {}
        for i, (ent1, ent2) in enumerate(self.test_pair_url):
            test_ent_url2id[ent1] = i

        ok = False
        for ent1, ent2 in self.test_pair_url:
            retrieved_docs_scores.append(self.score_dict[test_ent_url2id[ent1]])
            if ok == False:
                logger.info(f"Case hard case threshold: {self.score_dict[test_ent_url2id[ent1]]}")
                ok = True
        retrieved_docs_scores = np.array(retrieved_docs_scores)
        sorted_scores = np.sort(retrieved_docs_scores, axis=1)[:, ::-1]
        # 获取每行的最高分和第二高分
        top_scores = sorted_scores[:, 0]
        second_top_scores = sorted_scores[:, 1]
        # 计算差值
        score_gaps = top_scores - second_top_scores
        # score_gaps = top_scores

        """
        # 使用95%分位数作为阈值
        threshold = np.quantile(score_gaps, 0.10)

        logger.info(f"Threshold: {threshold}")
        """
        score_gaps_reshaped = score_gaps.reshape(-1, 1)

        # 应用K-means聚类
        kmeans = KMeans(n_clusters=100, random_state=0).fit(score_gaps_reshaped)

        # 获取聚类中心并排序
        centers = np.sort(kmeans.cluster_centers_.flatten())
        labels = kmeans.labels_
        logger.info(centers)
        assert len(labels) == len(score_gaps), "Error in clustering"
        # 选择聚类中心之间的点作为阈值
        # hard_threshold = centers[0]
        # medium_threshold = centers[1]
        # easy_threshold = centers[2]
        candi_cnt_dict = {}
        all_candi_set = set()
        hits_at_10_cnt = 0
        hits_at_10_set = set()
        for ent1, ent2 in self.test_pair_url:
            all_candi_set.update(set(self.candidate_url_set[ent1][:10]))
            for candi in self.candidate_url_set[ent1][:10]:
                if candi not in candi_cnt_dict:
                    candi_cnt_dict[candi] = 0
                candi_cnt_dict[candi] += 1
            if ent2 in self.candidate_url_set[ent1][:10]:
                hits_at_10_cnt += 1
                hits_at_10_set.update(set(self.candidate_url_set[ent1][:10]))
        logger.info(f"Hits@10: {hits_at_10_cnt / len(self.test_pair_url)}")
        logger.info(f"Number of hits@10: {hits_at_10_cnt}")
        logger.info(f"Number of all candidates: {len(all_candi_set)}")
        logger.info(f"Number of hits@10 candidates: {len(hits_at_10_set)}")
        logger.info(f"Max candidate number: {max(candi_cnt_dict.values())}")
        logger.info(f"Min candidate number: {min(candi_cnt_dict.values())}")

        max_hits_at_1 = -1000
        min_hits_at_1 = 1000
        average_hits_at_1 = 0
        average_hits_at_1_cnt = 0
        all_hits_at_1 = 0
        for i in range(len(self.test_pair_url)):
            ent1 = self.test_pair_url[i][0]
            ent2 = self.test_pair_url[i][1]
            hits_at_1_score = sorted_scores[i][0]
            if hits_at_1_score > -1:
                all_hits_at_1 += 1
                if ent2 in self.candidate_url_set[ent1][:1]:
                    average_hits_at_1_cnt += 1
                    max_hits_at_1 = max(max_hits_at_1, hits_at_1_score)
                    min_hits_at_1 = min(min_hits_at_1, hits_at_1_score)
                    average_hits_at_1 += hits_at_1_score
                # if hits_at_1_score < 0.0:
                    # logger.info(f"Case hard case threshold: {ent1} {ent2} {hits_at_1_score}")
        logger.info(f"Max hits@1: {max_hits_at_1}")
        logger.info(f"Min hits@1: {min_hits_at_1}")
        logger.info(f"Average hits@1: {average_hits_at_1 / average_hits_at_1_cnt}")
        logger.info(f"Hits@1: {average_hits_at_1_cnt / len(self.test_pair_url)}")
        logger.info(f"Number of hits@1: {average_hits_at_1_cnt}")
        logger.info(f"Average precision: {average_hits_at_1 / all_hits_at_1}")

        # hard_position = score_gaps < hard_threshold
        # medium_position = (score_gaps >= hard_threshold) & (score_gaps < medium_threshold)
        # easy_position = score_gaps >= medium_threshold

        # assert np.sum(hard_position) + np.sum(medium_position) + np.sum(easy_position) == len(score_gaps), "Error in clustering"
        # logger.info(f"Number of hard cases: {np.sum(hard_position)}")
        # logger.info(f"Number of medium cases: {np.sum(medium_position)}")
        # logger.info(f"Number of easy cases: {np.sum(easy_position)}")


        # hard_num = np.sum(hard_position)
        # logger.info(f"Number of hard cases: {hard_num}")


        # right_num = 0
        # low_confidence_queries = np.where(hard_position)[0]
        # low_confidence_url = []
        # for pos in low_confidence_queries:
        #     if self.test_pair_url[pos][1] != self.candidate_url_set[self.test_pair_url[pos][0]][0]:
        #         right_num += 1
        #     low_confidence_url.append(self.test_pair_url[pos][0])
        
        # logger.info(f"Number of hard cases that are correct: {right_num}")
        # logger.info(f"Ratio of hard cases that are correct: {right_num / hard_num}")
        
        # hard_url_list = hits_at_k_according_threshold(0.0, hard_threshold, score_gaps, 1, "Hard")
        # medium_url_list = hits_at_k_according_threshold(hard_threshold, medium_threshold, score_gaps, 1, "Medium")
        # easy_url_list = hits_at_k_according_threshold(medium_threshold, 1.0, score_gaps, 1, "Easy")
        # hard_url_list = hits_at_k_according_cluster_labels(labels, 0, 1, "Hard")
        # medium_url_list = hits_at_k_according_cluster_labels(labels, 1, 1, "Medium")
        # easy_url_list = hits_at_k_according_cluster_labels(labels, 2, 1, "Easy")
        # for i in range(0, 100):
        #     hits_at_k_according_cluster_labels(labels, i, 1, f"Cluster {i}")
        """
        plt.hist(score_gaps, bins=30, alpha=0.7)
        plt.axvline(x=threshold, color='red', linestyle='--')
        plt.xlabel('Score Gaps')
        plt.ylabel('Frequency')
        plt.title('Distribution of Score Gaps with Threshold')
        plt.savefig("score_gaps_distribution_with_threshold.svg", format="svg")
        """
        return 0, hard_url_list

    def openai_generate(self, sequences):
        client = OpenAI(api_key=self.config['openai']['api_key'], base_url=self.config['openai']['base_url'])
        rerank_results = []
        error_cnt = 0
        for sequence in sequences:
            try:
                completion = client.chat.completions.create(
                    model= self.config['openai']['model'],
                    messages=[
                        {"role": "user", "content": sequence}
                    ],
                )
                rerank_results.append(completion.choices[0].message.content)
            except Exception as e:
                rerank_results.append("Error")
                logger.error(f"Error in OpenAI API: {e}")
                error_cnt += 1
        
        logger.info(f"Number of errors in OpenAI API: {error_cnt}")

        return rerank_results


    
    def get_not_int_rank1(self):
        """
        Return: 
            not_in_rank1: list
        """
        
        not_in_rank1 = []
        for ent1_url, ent2_url in self.test_pair_url:
            candidate_ents = list(self.candidate_url_set[ent1_url])
            rank1 = candidate_ents[0]
            if rank1 != ent2_url and rank1 in candidate_ents:
                not_in_rank1.append(ent1_url)
        
        logger.info(f"Number of not in rank1 but in candidate entities: {len(not_in_rank1)}")
        logger.info(f"Ratio of not in rank1 but in candidate entities: {len(not_in_rank1) / len(self.test_pair_url)}")
        return not_in_rank1

    def get_ground_truth_choice(self, test_ents_url):

        """
        Return: 
            ground_truth_choice: list
            in_candidate: int
        """

        ground_truth_choice = []
        in_candidate = 0

        test_pair_url_dict = {}
        for (ent1, ent2) in self.test_pair_url:
            test_pair_url_dict[ent1] = ent2

        for ent_url in test_ents_url:
            candidate_ents = self.candidate_url_set[ent_url]
            ith_choice = -1
            for i, one_candidate in enumerate(candidate_ents):
                if one_candidate == test_pair_url_dict[ent_url]:
                    assert ith_choice == -1
                    ith_choice = i

            if ith_choice == -1:
                ground_truth_choice.append("Not in candidate entities")
            else:
                ground_truth_choice.append(chr(97+ith_choice))
                in_candidate += 1
        
        logger.info(f"Number of ground truth in candidate entities: {in_candidate}")
        logger.info(f"Ratio of ground truth in candidate entities: {in_candidate / len(test_ents_url)}")
        logger.info(f"Ratio of ground truth not in candidate entities: {(len(test_ents_url) - in_candidate) / len(test_ents_url)}")

        return ground_truth_choice, in_candidate
    
    def calculate_attr_functionality(self):
        """
        Calculate the functionality of each attribute
        """
        functionality_dict = {}
        for ent_url, entity in self.entities.items():
            entity_attr_dict = entity.attributes
            value_set = set()
            for attr, value_list in entity_attr_dict.items():
                value_set.update(set(value_list))
                attr_name = attr.get_name()
                if attr_name not in functionality_dict:
                    functionality_dict[attr_name] = [0, 0]
                functionality_dict[attr_name][0] += len(set(value_list))
                functionality_dict[attr_name][1] += 1

        doc_num = len(self.entities)
        for attr, (total, cnt) in functionality_dict.items():
            functionality_dict[attr] = (cnt / total) * (cnt / doc_num)
        return functionality_dict


    def vanilla_attr_format(self, attr):
        attr_str = ""
        cnt = 0
        for (e, a, v) in attr:
            cnt += 1
            attr_str += f"<{e}, {a}, {v}>\t"
            if cnt % self.attr_format_max_num == 0:
                break
        return attr_str

    def functional_attr_format(self, attr):
        sorted_attr = sorted(attr, key=lambda x: self.attr_functionality_dict[x[1]], reverse=True)
        attr_str = ""
        cnt = 0
        pre_functionality = -1
        pre_attr_str = ""
        for (e, a, v) in sorted_attr:
            if cnt >= self.attr_format_max_num:
                break
            if pre_functionality != self.attr_functionality_dict[a] and pre_attr_str != a:
                pre_functionality = self.attr_functionality_dict[a]
                pre_attr_str = a
                cnt += 1
                attr_str += f"<{e}, {a}, {v}>\t"
        
        return attr_str
    
    def get_candidates_attr_format(self, url_list):
        candiates_attr = {}
        for url in url_list:
            candiates_attr[url] = self.functional_attr_format(self.entity_info_dict[url]['attributes'])
        return candiates_attr

    def rel_format(self, rel):
        """
        TODO
        """
        return rel[0][0]

    def construct_choices(self, url):
        candidate_ents = self.candidate_url_set[url]
        candidate_ents_str = ""
        for i, one_candidate in enumerate(candidate_ents):
            candidate_ents_str += f"({chr(97+i)}) {self.construct_centrality(one_candidate)}\n"
        return candidate_ents_str

    def construct_centrality(self, url):
        entity_name = self.entities[url].url.split("/")[-1].replace("_", " ")
        attr = self.entity_info_dict[url]['attributes']
        rel = self.entity_info_dict[url]['relationships']
        entity_seq = ""
        if self.config['information']['name']:
            entity_seq += f"{entity_name}\n"
        if self.config['information']['attr']:
            entity_seq += f"{self.functional_attr_format(attr)}\n"
        if self.config['information']['rel']:
            entity_seq += f"{self.rel_format(rel)}\n"
        return entity_seq

    def construct_one_input(self, ent_url):
        centrality = self.construct_centrality(ent_url)
        choices = self.construct_choices(ent_url)
        return centrality, choices

    def construct_input(self, test_ents_url):
        instruction = getattr(self.instruction, f"instruction_{self.config['instruction']}")
        if self.config.get('use_demonstration', False):
            demonstration = getattr(self.demonstration, f"demonstration_{self.config['demonstration']}")

        inputs = []
        for ent_url in test_ents_url:
            now_entitiy, candidate_ents_str = self.construct_one_input(ent_url)
            if self.config.get('use_demonstration', False):
                sequence = instruction.format(demonstration + "\n", now_entitiy, candidate_ents_str)
            else:
                sequence = instruction.format("", now_entitiy, candidate_ents_str)
            inputs.append(sequence)
        
        return inputs

    def vllm_generate(self, sequences):
        if hasattr(self, 'model') is False:
            self.load_vllm()
        temperature = self.config['vllm'].get('temperature', 0.8)
        top_p = self.config.get('top_p', 0.95)
        max_tokens = self.config.get('max_tokens', 1024)
        repetition_penalty = self.config.get('repetition_penalty', 1.1)
        stop = self.config.get('stop', None)
        stop = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop, repetition_penalty=repetition_penalty)
        rerank_results = self.model.generate(sequences, sampling_params=sampling_params)
        rerank_output = []
        for res in rerank_results:
            rerank_output.append(res.outputs[0].text)
        
        logger.info(f"Case Rerank results: {rerank_output[0]}")
        return rerank_output

    def rerank(self, test_ents_url):
        ground_truth_choice, in_candidate = self.get_ground_truth_choice(test_ents_url)
        sequences = self.construct_input(test_ents_url)

        logger.info(f"Case vllm rerank: {sequences[0]}")
        
        if self.config.get('use_vllm', False):
            rerank_output = self.vllm_generate(sequences)
        elif self.config.get('use_openai', False):
            rerank_output = self.openai_generate(sequences)
        else:
            raise ValueError('No LLM model specified')
        
        return rerank_output, ground_truth_choice

    def parser_1(self, text):
        pattern = re.compile(r'(?i)(?:the\s+)?(?:correct\s+)?answer\s*(?:should\s+be)?\s*(?:is)?\s*[:=]?\s*\(([a-zA-Z])\)')
        # 搜索文本中的匹配项
        match = pattern.search(text)
        # 如果找到匹配项，则返回提取的答案
        if match:
            return match.group(1).lower()  # 将答案转换为小写
        else:
            return "None"  # 如果没有找到匹配项，返回None

    def metrics(self, analysis_json):
        """
        Hits@1: The percentage of cases where the correct entity is ranked first.
        """
        correct = 0
        for one_case in analysis_json:
            for ent, ent_dict in one_case.items():
                if ent_dict['correct']:
                    correct += 1
        hits_at_1 = correct / len(analysis_json)
        logger.info(f"Hits@1: {hits_at_1}")
        return {
            "hits@1": hits_at_1
        }

    def candidate_split(self):
        """
        Split the candidate entities into n parts
        """
        strategy = self.config.get('candidate_split_strategy', 'emh')
        if strategy == 'emh':
            threshold, low_confidence_url = self.hard_case_threshold()
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        return threshold, low_confidence_url

    def parse_rerank_results(self, rerank_results):
        pred_choice = []
        for one_result in rerank_results:
            parsed_result = self.parser_1(one_result)
            pred_choice.append(parsed_result)
        return pred_choice


    def compute_metrics(self, pred_choice, ground_truth_choice, low_confidence_url):
        """
        Compute the metrics
        """
        high_confidence_url = list(set(self.candidate_url_set.keys()) - set(low_confidence_url))
        high_confidence_ground_truth_choice, _ = self.get_ground_truth_choice(high_confidence_url)

        high_confidence_true_num = np.sum([1 if one_choice == 'a' else 0 for one_choice in high_confidence_ground_truth_choice])
        low_confidence_true_num = 0
        assert len(pred_choice) == len(ground_truth_choice), "Len Error...."
        for i in range(len(pred_choice)):
            if pred_choice[i] == ground_truth_choice[i]:
                low_confidence_true_num += 1
        
        hits1 = (low_confidence_true_num + high_confidence_true_num) / len(self.test_pair_url)
        low_confidence_hits1 = low_confidence_true_num / len(low_confidence_url)
        high_confidence_hits1 = high_confidence_true_num / len(high_confidence_url)
        return {
            "Hits@1": hits1,
            "Low confidence Hits@1": low_confidence_hits1,
            "High confidence Hits@1": high_confidence_hits1,
        }

    def analysis(self, pred_choice, ground_truth_choice, rerank_results, sample_ents):
        all_analysis_json_list = []
        ground_truth_dict = {}
        for (ent1, ent2) in self.test_pair_url:
            ground_truth_dict[ent1] = ent2

        pred_choice = []
        for one_result in rerank_results:
            parsed_result = self.parser_1(one_result)
            pred_choice.append(parsed_result)

        for i, ent in enumerate(sample_ents):
            analysis_json= {}
            analysis_json[ent] = {}
            analysis_json[ent]['attributes'] = self.functional_attr_format(self.entity_info_dict[ent]['attributes'])
            analysis_json[ent]['ground_truth'] = ground_truth_dict[ent]
            analysis_json[ent]['candidate_entities'] = list(self.candidate_url_set[ent])

            analysis_json[ent]['candidate_attrs'] = self.get_candidates_attr_format(list(self.candidate_url_set[ent]))

            analysis_json[ent]['ground_truth_in_candidate'] = ground_truth_dict[ent] in self.candidate_url_set[ent]
            analysis_json[ent]['rerank_result'] = rerank_results[i]
            analysis_json[ent]['parsed_result'] = pred_choice[i]
            analysis_json[ent]['correct'] = pred_choice[i] == ground_truth_choice[i]
            all_analysis_json_list.append(analysis_json) 
        
        return all_analysis_json_list

    def oto_mapping(self):
        threshold, low_confidence_url = self.candidate_split()
        test_pair_url_dict = {}
        for (ent1, ent2) in self.test_pair_url:
            test_pair_url_dict[ent1] = ent2
        
        high_confidence_url = list(set(self.candidate_url_set.keys()) - set(low_confidence_url))
        

        solved_entities_set = set()
        for url in high_confidence_url:
            high_confidence_candidate_ents = self.candidate_url_set[url]
            solved_entities_set.add(high_confidence_candidate_ents[0])
        
        min_candidate_num = 100
        max_candidate_num = 0
        all_candidate_num = 0
        for url in low_confidence_url:
            low_confidence_candidate_ents = self.candidate_url_set[url]
            cnt = 0
            for can_url in low_confidence_candidate_ents:
                if can_url in solved_entities_set:
                    cnt += 1
            
            cnt = len(low_confidence_candidate_ents) - cnt
            min_candidate_num = min(min_candidate_num, cnt)
            max_candidate_num = max(max_candidate_num, cnt)
            all_candidate_num += cnt
        
        logger.info(f"Min candidate num: {min_candidate_num}")
        logger.info(f"Max candidate num: {max_candidate_num}")
        logger.info(f"Average candidate num: {all_candidate_num / len(low_confidence_url)}")
        




    def run(self):
        threshold, low_confidence_url = self.candidate_split()
        rerank_results, ground_truth_choice = self.rerank(low_confidence_url)
        pred_choices = self.parse_rerank_results(rerank_results)
        metric = self.compute_metrics(pred_choices, ground_truth_choice, low_confidence_url)
        logger.info(metric)
        analysis_result = self.analysis(
            pred_choice=pred_choices,
            ground_truth_choice=ground_truth_choice,
            rerank_results=rerank_results,
            sample_ents=low_confidence_url,
        )
        return analysis_result

    def pilot_test(self, sample_num=10, seed=0, strategy=None):
        candidate_url_list = list(self.candidate_url_set.keys())
        if strategy is None:
            if sample_num >= len(candidate_url_list):
                sample_num = len(candidate_url_list)
            sample_ents = random.sample(candidate_url_list, k=sample_num)
            logger.info(f"Sampled {sample_num} entities for pilot test")
            logger.info(f"Sampled entities case: {sample_ents[0]}")
        elif strategy == "not_in_rank1":
            sample_ents = self.get_not_int_rank1()
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
    
        rerank_results, ground_truth_choice = self.rerank(sample_ents)

        all_analysis_json_list = []
        ground_truth_dict = {}
        for (ent1, ent2) in self.test_pair_url:
            ground_truth_dict[ent1] = ent2

        pred_choice = []
        for one_result in rerank_results:
            parsed_result = self.parser_1(one_result)
            pred_choice.append(parsed_result)

        for i, ent in enumerate(sample_ents):
            analysis_json= {}
            analysis_json[ent] = {}
            analysis_json[ent]['attributes'] = self.functional_attr_format(self.entity_info_dict[ent]['attributes'])
            analysis_json[ent]['ground_truth'] = ground_truth_dict[ent]
            analysis_json[ent]['candidate_entities'] = list(self.candidate_url_set[ent])

            analysis_json[ent]['candidate_attrs'] = self.get_candidates_attr_format(list(self.candidate_url_set[ent]))

            analysis_json[ent]['ground_truth_in_candidate'] = ground_truth_dict[ent] in self.candidate_url_set[ent]
            analysis_json[ent]['rerank_result'] = rerank_results[i]
            analysis_json[ent]['parsed_result'] = pred_choice[i]
            analysis_json[ent]['correct'] = pred_choice[i] == ground_truth_choice[i]
            all_analysis_json_list.append(analysis_json)
        
        self.metrics(all_analysis_json_list)

        return all_analysis_json_list

