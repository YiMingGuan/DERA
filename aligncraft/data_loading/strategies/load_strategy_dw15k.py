import logging
from ..load_strategy_interface import KG, Entity, Relation, KGs
from bidict import bidict
from isodate.isoerror import ISO8601Error
import re, os
import hashlib
import pickle
logger = logging.getLogger(__name__)
class DW15KEntity(Entity):
    def __init__(self, url):
        super().__init__(url)
        self.translated_url = None
     
    def parse_url(self):
        name = self.url.split('/')[-1].replace("_", " ")
        return name


class DW15KRelation(Relation):
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

class DW15KKG(KG):
    def __init__(self, config, kg_id, verbose=True, rel_id_start=0):
        self.rel_id_start = rel_id_start
        super().__init__(config, kg_id, verbose)
    def get_entity_class(self):
        return DW15KEntity
    
    def get_relation_class(self):
        return DW15KRelation
    

    def load_triples(self, is_attr=False):
        """
        DPB15K triples are ids triplets.
        (head_id, relation_id, tail_id)
        """

        def id2url(mapping, id):
            return mapping[id]

        def parse_attr_line(one_line):
            """
            Parse attribute triple line
            """
            unparsed_h, unparsed_a, unparsed_v = one_line.strip().split("\t")
            unparsed_h = "<" + unparsed_h + ">"
            unparsed_a = "<" + unparsed_a + ">"
            unparsed_v = re.sub(r'\\\"|\\', '', unparsed_v)
            """
            clean after str :^^
            """
            if "^^" in unparsed_v:
                unparsed_v = unparsed_v.split("^^")[0]
                assert unparsed_v.startswith("\"") and unparsed_v.endswith("\"")
                unparsed_v = unparsed_v[1:-1]
            
            if "\"@" in unparsed_v:
                unparsed_v = unparsed_v.split("\"@")[0]

            if "XMLSchema" in unparsed_v:
                unparsed_v = unparsed_v + " ."
            else:
                unparsed_v = "\"" + unparsed_v + "\" ."
            
            rdf_line = " ".join([unparsed_h, unparsed_a, unparsed_v])
            return rdf_line

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
                        parsed_line = parse_attr_line(line)
                        # print('---------------------')
                        # print(parsed_line)
                        try:
                            g.parse(data=parsed_line)
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
            rel_knt = self.rel_id_start
            with open(file, 'r') as f:
                for line in f:
                    try:
                        head, relation, tail = line.strip().split("\t")
                        if relation not in self.rel_ids.inv:
                            self.rel_ids[rel_knt] = relation
                            rel_knt += 1
                        triples.append((head, relation, tail))
                    except:
                        logger.warning(f"Error in loading relation triple: {line} in file: {file}")
            logger.info(f"Loaded {len(triples)} triples from {file}")
            logger.info(f"Loaded {len(self.rel_ids)} relations")
            logger.info(f"rel ids case: {list(self.rel_ids.items())[0]}")
            return triples

    def preprocess_data(self):
        ent_links: bidict = self.load_ent_links(self.config['ent_links'])
        setattr(self, 'ent_links', ent_links)
        self.set_ent_ids()
        setattr(self, 'rel_ids', bidict())


    def load_ent_links(self, file):
        """
        Load entity links from file
        case:
        http://dbpedia.org/resource/Dave_Grusin	http://www.wikidata.org/entity/Q502923
        """
        ent_links = bidict()
        with open(file, 'r') as f:
            for line in f:
                ent1, ent2 = line.strip().split("\t")
                ent_links[ent1] = ent2
        return ent_links

    def set_ent_ids(self):
        """
        set entity ids
        """
        num_ent = len(self.ent_links)
        k_ent = 0
        ent_ids_1 = bidict()
        ent_ids_2 = bidict()
        for ent1, ent2 in self.ent_links.items():
            ent_ids_1[k_ent] = ent1
            ent_ids_2[k_ent + num_ent] = ent2
            k_ent += 1  
        
        setattr(self, 'ent_ids', ent_ids_1 if self.kg_id == 1 else ent_ids_2)
    
   
    def postprocess_data(self):
        pass

class EADataDW15K(KGs):
    def __init__(self, config) -> None:
        super().__init__(config)
        
    def load_data(self):
        return self
    
    def _get_kg_class(self):
        return DW15KKG
    
    def init_kgs(self):
        _kg_class = self._get_kg_class()
        self.kg1 = _kg_class(self.config, 1)
        self.kg2 = _kg_class(self.config, 2, rel_id_start=len(self.kg1.rel_ids))
        self.build_dataset()

    def build_dataset(self):
        def load_links(ent_links):
            links = bidict()
            links_list = []
            for link in list(ent_links.items()):
                try:
                    url1, url2 = link
                except:
                    logger.info(f"Error in loading link: {link}")
                id1 = self.kg1.ent_ids.inv[url1]
                id2 = self.kg2.ent_ids.inv[url2]
                links[int(id1)] = int(id2)
                links_list.append((int(id1), int(id2)))
            return links, links_list

        def split_train_test(pairs_list, train_ratio=0.3):
            num_train = int(len(pairs_list) * train_ratio)
            train_pairs = pairs_list[:num_train]
            test_pairs = pairs_list[num_train:]
            return train_pairs, test_pairs

        all_pairs, all_pairs_list = load_links(self.kg1.ent_links)
        train_pairs, test_pairs = split_train_test(all_pairs_list, train_ratio=0.3)
        train_pairs = bidict(train_pairs)
        test_pairs = bidict(test_pairs)

        setattr(self, 'all_pairs', all_pairs)
        setattr(self, 'train_pairs', train_pairs)
        setattr(self, 'test_pairs', test_pairs)

        logger.info(f"Loaded {len(all_pairs)} all pairs")
        logger.info(f"Loaded {len(train_pairs)} train pairs")
        logger.info(f"Loaded {len(test_pairs)} test pairs")


    