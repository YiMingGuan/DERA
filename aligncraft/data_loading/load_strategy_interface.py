from collections import deque
import yaml
import numpy as np
import copy
import logging
from abc import ABC, abstractclassmethod

logger = logging.getLogger(__name__)
import re
from abc import ABC, abstractclassmethod

NAME_VARIETAS = ["name", "label", "title", 'nick', 'nickname', 'alias', 'aka', 'also known as', 'alternative name', 'alternative names', 'other name', 'other names', 'othername', 'othernames', 'other', 'others', 'altname', 'altnames', 'alt', 'alsoknownas', 'alsoknown', 'alsoknowns', 'alsoknownsa', 'givenname', 'givennames', 'given', 'firstname', 'firstnames', 'first', 'forename', 'forenames', 'fore', 'surname', 'surnames', 'sur', 'lastname', 'lastnames', 'last', 'familyname', 'familynames', 'family', 'maidenname', 'maidennames', 'maiden', 'birthname', 'birthnames', 'birth', 'originalname', 'originalnames', 'original', 'realname', 'realnames', 'real', 'truenames', 'truename', 'true', 'legalname', 'legalnames', 'legal', 'officialname', 'officialnames', 'official', 'formalname', 'formalnames', 'formal', 'commonname', 'commonnames', 'common', 'nickname', 'nicknames', 'nick', 'pseudonym', 'pseudonyms', 'pseudo', 'penname', 'pennames', 'pen', 'st']

class Relation(ABC):
    def __init__(self, url, is_inv=False, is_attr=False):
        self.url = url
        self.is_inv = is_inv
        self.is_attr = is_attr

        self.name = self.extract_name()
            
        self.head_tail_pairs = []
        self.head_tail_dicts = {}
    
    def add_head_tail_pair(self, head, tail):
        self.head_tail_pairs.append((head, tail))
    
    def add_head_tail_dict(self, head, tail):
        if self.head_tail_dicts.get(head) is None:
            self.head_tail_dicts[head] = set()
        self.head_tail_dicts[head].add(tail)
    
    @abstractclassmethod
    def extract_name(self):
        pass
    
    @abstractclassmethod
    def get_name(self):
        pass
 
    def __str__(self):
        return f"Prediction URL: {self.url}"


class Entity(ABC):
    def __init__(self, url):
        self.url = url
        self.relationships = {}
        self.attributes = {}
        self.name = []
        self.ontology: str = None
        self.literal_info = None
         
    def add_relationship(self, prediction: Relation , tail):
        if self.relationships.get(prediction) is None:
            self.relationships[prediction] = []
        self.relationships[prediction].append(tail)
        prediction.add_head_tail_dict(self.url, tail.url)
    
    def add_attribute(self, attribute, value):
        if self.attributes.get(attribute) is None:
            self.attributes[attribute] = []
        self.attributes[attribute].append(value)

        #extract entity name from attribute
        attribute_name = attribute.get_name()
        if attribute_name.lower() in NAME_VARIETAS:
            self.name.append(value.lower())
  
    def __str__(self):
        return f"Entity URL: {self.url}\n"



class KG(ABC):
    def __init__(self, config, kg_id, verbose=True):

        """
        {url: Entity}
        {url: Prediction}
        {url: Attribute}
        """
        self.entities = {}
        self.predictions = {}
        self.attributes = {}

        """
        {entity_url: entity_id}
        {prediction_url: prediction_id}
        {attribute_url: attribute_id}
        """
        self.config = config
        self.kg_id = kg_id
        self.kg_name = config['name']
        self.init_kg()
        if verbose:
            self.kg_details()

    @abstractclassmethod
    def get_entity_class(self):
        """
        Return the class of entity
        """
        pass

    @abstractclassmethod
    def get_relation_class(self):
        """
        Return the class of relation
        """
        pass

    @abstractclassmethod
    def preprocess_data(self):
        pass
    
    def postprocess_data(self):
        pass

    def init_kg(self):
        self.preprocess_data()
        self.build_kg()
        self.postprocess_data()
    
    def get_entity(self, url):
        if self.entities.get(url, None) is None:
            _entity_class = self.get_entity_class()
            self.entities[url] = _entity_class(url)
        return self.entities[url]
    
    def get_attribute(self, url):
        if self.attributes.get(url, None) is None:
            _relation_class = self.get_relation_class()
            self.attributes[url] = _relation_class(url, is_attr=True)
        return self.attributes[url]
    
    def get_prediction(self, url, inv=False):
        trans_url = url + "_(inv)" if inv else url
        if self.predictions.get(trans_url, None) is None:
            _relation_class = self.get_relation_class()
            self.predictions[trans_url] = _relation_class(trans_url, is_inv=inv)
        return self.predictions[trans_url]


    def build_kg(self):
        
        def build_relation_graph(triples):
            for triple in triples:
                head, prediction, tail = triple
                head_entity = self.get_entity(head)
                tail_entity = self.get_entity(tail)
                prediciton_instance = self.get_prediction(prediction)
                inv_prediction_instance = self.get_prediction(prediction, inv=True)        
                head_entity.add_relationship(prediciton_instance, tail_entity)
                tail_entity.add_relationship(inv_prediction_instance, head_entity)
        
        def build_attribute_graph(triples):
            for triple in triples:
                head, attr, value = triple
                head_entity = self.get_entity(head)
                attr_instance = self.get_attribute(attr)
                head_entity.add_attribute(attr_instance, value)

        attr_triples = self.load_triples(is_attr=True)
        rel_triples = self.load_triples()

        build_relation_graph(rel_triples)
        build_attribute_graph(attr_triples)

    @abstractclassmethod
    def load_triples(self, is_attr=False):
        pass
    
    def kg_details(self):
        logger.info("Number of entities: {}".format(len(self.entities)))
        logger.info("Number of relations: {}".format(len(self.predictions)))
        logger.info("Number of attributes: {}".format(len(self.attributes)))


class KGs(ABC):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.init_kgs()
    
    @abstractclassmethod
    def _get_kg_class(self):
        pass

    def init_kgs(self):
        _kg_class = self._get_kg_class()
        self.kg1 = _kg_class(self.config, 1)
        self.kg2 = _kg_class(self.config, 2)
        self.build_dataset()
    
    @abstractclassmethod
    def build_dataset(self):
        pass