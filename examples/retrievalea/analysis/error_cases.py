from aligncraft.models.retrievalea.retrieval import Retrieval
import argparse
import yaml
import logging
import hashlib
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    return parser.parse_args()

def main(config):
    retriever = Retrieval(config)
    url2seq = retriever.get_kgs_all_url2sequence()
    url2info = retriever.get_kgs_all_url2entitiesinfo()
    test_pair_url2url = retriever.get_test_pair_url2url_dict()    

    url_list = [
        'http://ja.dbpedia.org/resource/アリアンツ・リヴィエラ',
        'http://ja.dbpedia.org/resource/北琉球方言',
        'http://ja.dbpedia.org/resource/江戸川区',
        'http://ja.dbpedia.org/resource/諫早市',
        'http://ja.dbpedia.org/resource/シェキラ!_サウンドトラック'
    ]
    candidates = {
        "candidate_attrs": {
                "http://dbpedia.org/resource/Kurashiki,_Okayama": "",
                "http://dbpedia.org/resource/Kasugai,_Aichi": "",
                "http://dbpedia.org/resource/Mitaka,_Tokyo": "",
                "http://dbpedia.org/resource/Shimada,_Shizuoka": "",
                "http://dbpedia.org/resource/Katori,_Chiba": "",
                "http://dbpedia.org/resource/Ishioka,_Ibaraki": "",
                "http://dbpedia.org/resource/Urayasu": "",
                "http://dbpedia.org/resource/Funabashi,_Chiba": "",
                "http://dbpedia.org/resource/Kinokawa,_Wakayama": "",
                "http://dbpedia.org/resource/Asao-ku,_Kawasaki": ""
            }
    }
    for url in url_list:
        ground_truth = test_pair_url2url[url]
        print('url:', url)
        print('ground_truth:', ground_truth)
        print('sequence:', url2seq[url])
        print('ground_truth sequence:', url2seq[ground_truth])

        print('info: ', url2info[url])
        print('ground_truth info:', url2info[ground_truth])
        print('-' * 20)

    for can in candidates['candidate_attrs']:
        print('candidate:', can)
        print('sequence:', url2seq[can])
        print('info:', url2info[can])
        print('===' * 20)

if __name__ == "__main__":
    original_file_name = 'dbp15k-ja-en_en_using_llmsTrue_llms_nameMistral-7B-Instruct-v0.2_sequence_vanilla_use_url_True_use_name_False_use_translated_url_False_use_relationship_True_use_attributes_False_use_inverse_False_neighbors_use_attr_False_head_use_attr_False'
    hasher = hashlib.md5()
    hasher.update(original_file_name.encode('utf-8'))
    hash_file = hasher.hexdigest()
    print(hash_file)
    args = get_args()
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    main(config)