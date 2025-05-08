import logging
from ..load_strategy_interface import KG, Entity, Relation, KGs
from bidict import bidict
from isodate.isoerror import ISO8601Error
import re, os
import hashlib
import pickle
logger = logging.getLogger(__name__)
class DBP15KEntity(Entity):
    def __init__(self, url):
        super().__init__(url)
        self.translated_url = None
     
    def parse_url(self):
        name = self.url.split('/')[-1].replace("_", " ")
        return name


class DBP15KRelation(Relation):
    def __init__(self, url, is_inv=False, is_attr=False):
        super().__init__(url, is_inv, is_attr)
    
    def extract_name(self):
        """
        case: http://dbpedia.org/ontology/releaseDate
        """
        dirty_name = self.url.split("/")[-1]
        if self.is_inv:
            assert dirty_name.endswith("_(inv)")
            dirty_name = dirty_name[:-6]
        name = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', dirty_name).replace("_", " ").lower()
        return name
    
    def get_name(self):
        return self.name  

class DBP15KKG(KG):
    def __init__(self, config, kg_id, verbose=True):
        super().__init__(config, kg_id, verbose)

    def get_entity_class(self):
        return DBP15KEntity
    
    def get_relation_class(self):
        return DBP15KRelation
    
    def load_triples(self, is_attr=False):
        """
        DPB15K triples are ids triplets.
        (head_id, relation_id, tail_id)
        """

        def id2url(mapping, id):
            return mapping[id]

        triples = []
        if is_attr:
            file = self.config[f'attr_triples_{str(self.kg_id)}']
        else:
            file = self.config[f'rel_triples_{str(self.kg_id)}']
        
        if is_attr:
            from rdflib import Graph
            g = Graph()
            home_directory = os.path.expanduser('~')
            attr_triple_file = f"{self.kg_name}_attr_triples_{str(self.kg_id)}"
            cache_file_path = os.path.join(home_directory, 'aligncraft', 'benchmark', '.cache')
            if not os.path.exists(cache_file_path):
                os.makedirs(cache_file_path)
            hasher = hashlib.md5()
            hasher.update(attr_triple_file.encode('utf-8'))
            hash_attr_file = hasher.hexdigest()
            cache_file = os.path.join(cache_file_path, hash_attr_file)

            load_cache = self.config.get('load_cache', False)
            if load_cache and os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    triples = pickle.load(f)
            else:
                with open(file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        try:
                            g.parse(data=line, format="n3")
                        except Exception as e:
                            if isinstance(e, ISO8601Error):
                                continue
                            else:
                                logger.warning(f"Error in loading attribute triple: {line} in file: {file}")
                for entity, attribute, value in g:
                    triples.append((str(entity), str(attribute), str(value)))

                save_cache = self.config.get('save_cache', True)
                if save_cache:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(triples, f)
            
            return triples
        else:
            with open(file, 'r') as f:
                for line in f:
                    try:
                        head, relation, tail = line.strip().split("\t")
                        head = id2url(self.ent_ids, int(head))
                        tail = id2url(self.ent_ids, int(tail))
                        relation = id2url(self.rel_ids, int(relation))
                        triples.append((head, relation, tail))
                    except:
                        logger.warning(f"Error in loading relation triple: {line} in file: {file}")
            return triples

    def preprocess_data(self):

        ent_ids = self.load_ent_ids(self.config[f'ent_ids_{str(self.kg_id)}'])
        rel_ids = self.load_rel_ids(self.config[f'rel_ids_{str(self.kg_id)}'])

        setattr(self, 'ent_ids', ent_ids)
        setattr(self, 'rel_ids', rel_ids)

    def load_ent_ids(self, file):
        """
        Load entity ids from file
        """
        ent_id2url = bidict()
        with open(file, 'r') as f:
            for line in f:
                id, url = line.strip().split("\t")
                ent_id2url[int(id)] = url
        return ent_id2url
    
    def load_rel_ids(self, file):
        """
        Load relation ids from file
        """
        rel_id2url = bidict()
        with open(file, 'r') as f:
            for line in f:
                id, url = line.strip().split("\t")
                rel_id2url[int(id)] = url
        return rel_id2url
    
    def postprocess_data(self):
        pass

class EADataDBP15K(KGs):
    def __init__(self, config) -> None:
        super().__init__(config)
        
    def load_data(self):
        return self
    
    def _get_kg_class(self):
        return DBP15KKG
    
    def set_translated_urls(self):
        """
        Load translated urls from file
        """
        file_path = self.config['trans_url']
        trans_url_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                trans_url_list.append(line.strip())
        
        ent_ids_path = self.config['ent_ids_1']
        ent_ids_list = []
        with open(ent_ids_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                ent_ids_list.append(line.strip().split("\t")[1])
        
        for idx, url in enumerate(ent_ids_list):
            self.kg1.entities[url].translated_url = trans_url_list[idx]
        
        for id in self.kg2.ent_ids.keys():
            entity = self.kg2.entities[self.kg2.ent_ids[id]]
            entity.translated_url = entity.parse_url()
        
        logger.info(f"Loaded {len(trans_url_list)} translated urls.")
        """
        one translated case
        """
        kg1_ent_case = self.kg1.entities[list(self.kg1.ent_ids.values())[0]]
        kg2_ent_case = self.kg2.entities[list(self.kg2.ent_ids.values())[0]]
        logger.info(f"Translated url example: {kg1_ent_case.url} -> {kg1_ent_case.translated_url}")
        logger.info(f"Translated url example: {kg2_ent_case.url} -> {kg2_ent_case.translated_url}")
        logger.info(f"Successfully set translated urls.")


    def build_dataset(self):
        def load_links(file):
            links = bidict()
            with open(file, 'r') as f:
                for line in f:
                    id1, id2 = line.strip().split("\t")
                    links[int(id1)] = int(id2)
            return links

        def load_hard_setting_links(file):
            links = bidict()
            with open(file, 'r') as f:
                for line in f:
                    url1, url2 = line.strip().split("\t")
                    id1 = self.kg1.ent_ids.inv[url1]
                    id2 = self.kg2.ent_ids.inv[url2]
                    links[int(id1)] = int(id2)
            return links

        all_pairs = load_links(self.config['all_pairs'])

        is_hard_test_setting = self.config.get('hard_test_setting', False)
        if is_hard_test_setting:
            train_pairs = load_hard_setting_links(self.config['hard_test_setting_train_pairs'])
            valid_pairs = load_hard_setting_links(self.config['hard_test_setting_valid_pairs'])
            test_pairs = load_hard_setting_links(self.config['hard_test_setting_test_pairs'])
            train_pairs = {**train_pairs, **valid_pairs}
            logger.info("Using hard test setting.")
        else:
            train_pairs = load_links(self.config['train_pairs'])
            test_pairs = load_links(self.config['test_pairs'])
            logger.info("Using normal test setting.")

        setattr(self, 'all_pairs', all_pairs)
        setattr(self, 'train_pairs', train_pairs)
        setattr(self, 'test_pairs', test_pairs)

        logger.info(f"Loaded {len(all_pairs)} all pairs")
        logger.info(f"Loaded {len(train_pairs)} train pairs")
        logger.info(f"Loaded {len(test_pairs)} test pairs")
        """
        train valid test case:
        """
        logger.info(f"Train pair example: {list(train_pairs.items())[0]}")
        logger.info(f"Test pair example: {list(test_pairs.items())[0]}")
        logger.info(f"Successfully built dataset.")
        self.set_translated_urls()


    