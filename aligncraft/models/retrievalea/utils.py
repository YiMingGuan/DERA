
import random
import os
import hashlib
import pickle
import logging
from openai import OpenAI
logger = logging.getLogger(__name__)
def vanilla_entity_info(entity, 
                        use_url: bool = True, 
                        use_name: bool = True, 
                        use_translated_url = True, 
                        use_relationship: bool = True, 
                        use_attributes: bool = True, 
                        use_inverse: bool = True, 
                        neighbors_use_attr: bool = False,
                        head_use_attr: bool = False,
                        ):
    def serialize_entity(head,
                        use_name,
                        use_url,
                        use_translated_url,
                        add_attribute
                        ):
        name_info = get_name_url_info(head, use_name, use_url, use_translated_url, separator=' ')
        if name_info == "":
            name_info = "An Entity"
        if add_attribute:
            neighbors_str = "" 
            for attribute, values in head.attributes.items():
                for value in values:
                    neighbors_str += f" {attribute.get_name()}:{value};"
            name_info = f"{name_info} [{neighbors_str}]"
        
        return name_info

    
    def get_entity_name(head):
        if head.name == []:
            return None
        else:
            return random.choice(head.name)

    def get_name_url_info(head, use_name, use_url, use_translated_url, separator=' '):
        info = ""
        if use_name:
            name = get_entity_name(head)
            if name is not None:
                info += f"{name}{separator}"
        if use_url:
            if use_translated_url:
                assert head.translated_url is not None, "Translated url is None. Please set it before using it."
                info += f"{head.translated_url}{separator}"
            else:
                info += f"{head.parse_url()}{separator}" 
        return info

    name_or_url_info = get_name_url_info(entity, use_name, use_url, use_translated_url)
    
    mixed = f"{name_or_url_info}"

    serialized_relationships = []
    if use_relationship:
        for relation, tails in entity.relationships.items():
            if not use_inverse and relation.is_inv:
                continue
            for tail in tails:
                if relation.is_inv:
                    serialized_relationships.append((
                        serialize_entity(tail, use_url=use_url, use_name=use_name, use_translated_url=use_translated_url, add_attribute=neighbors_use_attr), 
                        relation.get_name(), 
                        serialize_entity(entity, use_url=use_url, use_name=use_name, use_translated_url=use_translated_url, add_attribute=head_use_attr)
                    ))
                else:
                    serialized_relationships.append((
                        serialize_entity(entity, use_url=use_url, use_name=use_name, use_translated_url=use_translated_url, add_attribute=head_use_attr), 
                        relation.get_name(), 
                        serialize_entity(tail, use_url=use_url, use_name=use_name, use_translated_url=use_translated_url, add_attribute=neighbors_use_attr)
                    ))
                mixed += f"\n{serialized_relationships[-1][0]} {serialized_relationships[-1][1]} {serialized_relationships[-1][2]}"
    
    serialized_attributes = [] 
    if use_attributes:
        for attribute, values in entity.attributes.items():
            for value in values:
                serialized_attributes.append((
                    serialize_entity(entity, use_url=use_url, use_name=use_name, use_translated_url=use_translated_url, add_attribute=head_use_attr), 
                    attribute.get_name(), 
                    value
                ))
                mixed += f"\n{serialized_attributes[-1][1]} {serialized_attributes[-1][2]}"
    

    return {
        "name_or_url_info": name_or_url_info,
        "relationships": serialized_relationships,
        "attributes": serialized_attributes,
        "mixed": mixed,
    } 
     

def entity_info(
        entity,
        method="vallina",
        use_url: bool = True,
        use_name: bool = True,
        use_translated_url = True,
        use_relationship: bool = True,
        use_attributes: bool = True,
        use_inverse: bool = True,
        neighbors_use_attr: bool = False,
        head_use_attr: bool = False,
        **kwargs,
    ):
    config = {
        "use_url": use_url,
        "use_name": use_name,
        "use_translated_url": use_translated_url,
        "use_relationship": use_relationship,
        "use_attributes": use_attributes,
        "use_inverse": use_inverse,
        "neighbors_use_attr": neighbors_use_attr,
        "head_use_attr": head_use_attr
    }
    if method == "vanilla":
        return vanilla_entity_info(entity, **config)
    else:
        raise ValueError(f"Method {method} not recognized")


def load_cache_file(original_file_name):
    home_directory = os.path.expanduser('~')
    file_name = os.path.join(home_directory, '.cache', 'aligncraft')
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    hasher = hashlib.md5()
    hasher.update(original_file_name.encode('utf-8'))
    hash_file = hasher.hexdigest()
    logger.info(f"Loading cache file from {hash_file}")
    cache_file = os.path.join(file_name, hash_file)
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_cache_file(original_file_name, data, overwrite=False):
    home_directory = os.path.expanduser('~')
    file_name = os.path.join(home_directory, '.cache', 'aligncraft')
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    hasher = hashlib.md5()
    hasher.update(original_file_name.encode('utf-8'))
    hash_file = hasher.hexdigest()
    cache_file = os.path.join(file_name, hash_file)

    if os.path.exists(cache_file) and not overwrite:
        logger.info(f"Cache file already exists at {cache_file} and overwrite is set to False")
    else:
        logger.info(f"Saving cache file to {hash_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    return cache_file

import matplotlib.pyplot as plt
import numpy as np
def plot_calibration_curve(y_true, y_prob, n_bins=20, pic_path=None):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_true_means = []
    bin_prob_means = []
    bin_counts = []

    for i in range(n_bins):
        bin_mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        bin_true = y_true[bin_mask]
        bin_prob = y_prob[bin_mask]

        if len(bin_true) > 0:
            bin_true_means.append(np.mean(bin_true))
            bin_prob_means.append(np.mean(bin_prob))
        else:
            bin_true_means.append(None)
            bin_prob_means.append(None)

        bin_counts.append(len(bin_true))

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 绘制校准曲线
    ax1.plot(bin_prob_means, bin_true_means, marker='o', label='Calibration Curve')
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Calibration Curve with Sample Counts')
    ax1.grid(True)

    # 绘制样本数量的柱状图
    ax2 = ax1.twinx()
    ax2.bar(bins[:-1] + 0.025, bin_counts, width=0.05, alpha=0.3, color='green', label='Sample Count')
    ax2.set_ylabel('Sample Count')
    ax2.set_ylim(0, max(bin_counts) * 1.1)

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.savefig(f'{pic_path}.png')  # 保存图片
    # plt.show()

def get_llmrerank_data_ori_name(config):
    assert 'retrieval' in config, "retrieval model is not set in the config file"
    using_llms = config.get('using_llms', False)
    method = config.get('entity_info_method') if using_llms else "None"
    use_url = config['entity_info'].get('use_url', False)
    use_name = config['entity_info'].get('use_name', False)
    use_translated_url = config['entity_info'].get('use_translated_url', False)
    use_relationship = config['entity_info'].get('use_relationship', False)
    use_attributes = config['entity_info'].get('use_attributes', False)
    use_inverse = config['entity_info'].get('use_inverse', False)
    neighbors_use_attr = config['entity_info'].get('neighbors_use_attr', False)
    head_use_attr = config['entity_info'].get('head_use_attr', False)
    llms_name = config.get('llms_name', None)

    if using_llms is False:
        llms_name = None

    prompt_type = config.get('prompt_type', None)
    if using_llms is False:
        prompt_type = None

    dataset_type = f"{config['dataset']['type']}-{config['dataset']['kg1']}-{config['dataset']['kg2']}"
    retriever_name = config['retrieval']['retriever_name']
    llmrerank_input_data_name = f"llmrerank_input_data_retriever_trained_{retriever_name}_{dataset_type}_using_llms{using_llms}_llms_name{llms_name}_prompt_type{prompt_type}_sequence_{method}_use_url_{use_url}_use_name_{use_name}_use_translated_url_{use_translated_url}_use_relationship_{use_relationship}_use_attributes_{use_attributes}_use_inverse_{use_inverse}_neighbors_use_attr_{neighbors_use_attr}_head_use_attr_{head_use_attr}"
    
    if config.get('hard_test_setting', False) is True:
        llmrerank_input_data_name += "_hard_test_setting"
    
    return llmrerank_input_data_name

def get_retrieval_data_path(config):
    using_llms = config.get('using_llms', False)
    method = config.get('entity_info_method') if using_llms else "None"
    use_url = config['entity_info'].get('use_url', False)
    use_name = config['entity_info'].get('use_name', False)
    use_translated_url = config['entity_info'].get('use_translated_url', False)
    use_relationship = config['entity_info'].get('use_relationship', False)
    use_attributes = config['entity_info'].get('use_attributes', False)
    use_inverse = config['entity_info'].get('use_inverse', False)
    neighbors_use_attr = config['entity_info'].get('neighbors_use_attr', False)
    head_use_attr = config['entity_info'].get('head_use_attr', False)
    llms_name = config.get('llms_name', None)

    if using_llms is False:
        llms_name = None

    prompt_type = config.get('prompt_type', None)
    if using_llms is False:
        prompt_type = None
    
    dataset_type = f"{config['dataset']['type']}-{config['dataset']['kg1']}-{config['dataset']['kg2']}"
    retrieval_data_name = f"retrieval_data_{dataset_type}_using_llms{using_llms}_llms_name{llms_name}_prompt_type{prompt_type}_sequence_{method}_use_url_{use_url}_use_name_{use_name}_use_translated_url_{use_translated_url}_use_relationship_{use_relationship}_use_attributes_{use_attributes}_use_inverse_{use_inverse}_neighbors_use_attr_{neighbors_use_attr}_head_use_attr_{head_use_attr}"

    if config.get('hard_test_setting', False) is True:
        retrieval_data_name += "_hard_test_setting"

    home_directory = os.path.expanduser('~')
    file_name = os.path.join(home_directory, '.cache', 'aligncraft')
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    hasher = hashlib.md5()
    hasher.update(retrieval_data_name.encode('utf-8'))
    hash_file = hasher.hexdigest()
    final_path = os.path.join(file_name, hash_file)
    return final_path

def get_retrieval_hn_mine_data_name(config):
    using_llms = config.get('using_llms', False)
    method = config.get('entity_info_method') if using_llms else "None"
    use_url = config['entity_info'].get('use_url', False)
    use_name = config['entity_info'].get('use_name', False)
    use_translated_url = config['entity_info'].get('use_translated_url', False)
    use_relationship = config['entity_info'].get('use_relationship', False)
    use_attributes = config['entity_info'].get('use_attributes', False)
    use_inverse = config['entity_info'].get('use_inverse', False)
    neighbors_use_attr = config['entity_info'].get('neighbors_use_attr', False)
    head_use_attr = config['entity_info'].get('head_use_attr', False)
    llms_name = config.get('llms_name', None)

    if using_llms is False:
        llms_name = None

    prompt_type = config.get('prompt_type', None)
    if using_llms is False:
        prompt_type = None

    dataset_type = f"{config['dataset']['type']}-{config['dataset']['kg1']}-{config['dataset']['kg2']}"
    retrieval_data_name = f"retrieval_hn_mine_data_{dataset_type}_using_llms{using_llms}_llms_name{llms_name}_prompt_type{prompt_type}_sequence_{method}_use_url_{use_url}_use_name_{use_name}_use_translated_url_{use_translated_url}_use_relationship_{use_relationship}_use_attributes_{use_attributes}_use_inverse_{use_inverse}_neighbors_use_attr_{neighbors_use_attr}_head_use_attr_{head_use_attr}"
    
    if config.get('hard_test_setting', False) is True:
        retrieval_data_name += "_hard_test_setting"

    home_directory = os.path.expanduser('~')
    file_name = os.path.join(home_directory, '.cache', 'aligncraft')
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    hasher = hashlib.md5()
    hasher.update(retrieval_data_name.encode('utf-8'))
    hash_file = hasher.hexdigest()
    final_path = os.path.join(file_name, hash_file)
    return final_path

def get_trained_retrieval_model(config):
    assert 'retrieval' in config, "retrieval model is not set in the config file"
    using_llms = config.get('using_llms', False)
    method = config.get('entity_info_method') if using_llms else "None"
    use_url = config['entity_info'].get('use_url', False)
    use_name = config['entity_info'].get('use_name', False)
    use_translated_url = config['entity_info'].get('use_translated_url', False)
    use_relationship = config['entity_info'].get('use_relationship', False)
    use_attributes = config['entity_info'].get('use_attributes', False)
    use_inverse = config['entity_info'].get('use_inverse', False)
    neighbors_use_attr = config['entity_info'].get('neighbors_use_attr', False)
    head_use_attr = config['entity_info'].get('head_use_attr', False)
    llms_name = config.get('llms_name', None)

    if using_llms is False:
        llms_name = None

    prompt_type = config.get('prompt_type', None)
    if using_llms is False:
        prompt_type = None

    dataset_type = f"{config['dataset']['type']}-{config['dataset']['kg1']}-{config['dataset']['kg2']}"
    retriever_name = config['retrieval']['retriever_name']
    retriever = f"retriever_trained_{retriever_name}_{dataset_type}_using_llms{using_llms}_llms_name{llms_name}_prompt_type{prompt_type}_sequence_{method}_use_url_{use_url}_use_name_{use_name}_use_translated_url_{use_translated_url}_use_relationship_{use_relationship}_use_attributes_{use_attributes}_use_inverse_{use_inverse}_neighbors_use_attr_{neighbors_use_attr}_head_use_attr_{head_use_attr}"
    
    if config.get('hard_test_setting', False) is True:
        retriever += "_hard_test_setting"

    home_directory = os.path.expanduser('~')
    file_name = os.path.join(home_directory, '.cache', 'aligncraft')
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    hasher = hashlib.md5()
    hasher.update(retriever.encode('utf-8'))
    hash_file = hasher.hexdigest()
    final_path = os.path.join(file_name, hash_file)
    return final_path

def get_rerank_data_path(config):
    return get_retrieval_data_path(config)

def get_rerank_hn_mine_data_name(config):
    using_llms = config.get('using_llms', False)
    method = config.get('entity_info_method') if using_llms else "None"
    use_url = config['entity_info'].get('use_url', False)
    use_name = config['entity_info'].get('use_name', False)
    use_translated_url = config['entity_info'].get('use_translated_url', False)
    use_relationship = config['entity_info'].get('use_relationship', False)
    use_attributes = config['entity_info'].get('use_attributes', False)
    use_inverse = config['entity_info'].get('use_inverse', False)
    neighbors_use_attr = config['entity_info'].get('neighbors_use_attr', False)
    head_use_attr = config['entity_info'].get('head_use_attr', False)
    llms_name = config.get('llms_name', None)

    if using_llms is False:
        llms_name = None

    prompt_type = config.get('prompt_type', None)
    if using_llms is False:
        prompt_type = None

    dataset_type = f"{config['dataset']['type']}-{config['dataset']['kg1']}-{config['dataset']['kg2']}"
    rerank_data_name = f"rerank_hn_mine_data_{dataset_type}_using_llms{using_llms}_llms_name{llms_name}_prompt_type{prompt_type}_sequence_{method}_use_url_{use_url}_use_name_{use_name}_use_translated_url_{use_translated_url}_use_relationship_{use_relationship}_use_attributes_{use_attributes}_use_inverse_{use_inverse}_neighbors_use_attr_{neighbors_use_attr}_head_use_attr_{head_use_attr}"
    
    if config.get('hard_test_setting', False) is True:
        rerank_data_name += "_hard_test_setting"

    home_directory = os.path.expanduser('~')
    file_name = os.path.join(home_directory, '.cache', 'aligncraft')
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    hasher = hashlib.md5()
    hasher.update(rerank_data_name.encode('utf-8'))
    hash_file = hasher.hexdigest()
    final_path = os.path.join(file_name, hash_file)
    return final_path

def get_trained_rerank_model(config):
    assert 'rerank' in config, "rerank model is not set in the config file"
    using_llms = config.get('using_llms', False)
    method = config.get('entity_info_method') if using_llms else "None"
    use_url = config['entity_info'].get('use_url', False)
    use_name = config['entity_info'].get('use_name', False)
    use_translated_url = config['entity_info'].get('use_translated_url', False)
    use_relationship = config['entity_info'].get('use_relationship', False)
    use_attributes = config['entity_info'].get('use_attributes', False)
    use_inverse = config['entity_info'].get('use_inverse', False)
    neighbors_use_attr = config['entity_info'].get('neighbors_use_attr', False)
    head_use_attr = config['entity_info'].get('head_use_attr', False)
    llms_name = config.get('llms_name', None)

    if using_llms is False:
        llms_name = None

    prompt_type = config.get('prompt_type', None)
    if using_llms is False:
        prompt_type = None

    dataset_type = f"{config['dataset']['type']}-{config['dataset']['kg1']}-{config['dataset']['kg2']}"
    reranker_name = config['rerank']['reranker_name']
    reranker = f"retriever_trained_{reranker_name}_{dataset_type}_using_llms{using_llms}_llms_name{llms_name}_prompt_type{prompt_type}_sequence_{method}_use_url_{use_url}_use_name_{use_name}_use_translated_url_{use_translated_url}_use_relationship_{use_relationship}_use_attributes_{use_attributes}_use_inverse_{use_inverse}_neighbors_use_attr_{neighbors_use_attr}_head_use_attr_{head_use_attr}"
    
    if config.get('hard_test_setting', False) is True:
        reranker += "_hard_test_setting"

    home_directory = os.path.expanduser('~')
    file_name = os.path.join(home_directory, '.cache', 'aligncraft')
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    hasher = hashlib.md5()
    hasher.update(reranker.encode('utf-8'))
    hash_file = hasher.hexdigest()
    final_path = os.path.join(file_name, hash_file)
    return final_path

from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
def sim(embed1, embed2, metric='inner', normalize=False, csls_k=0):
    """
    Compute pairwise similarity between the two collections of embeddings.

    Parameters
    ----------
    embed1 : matrix_like
        An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    embed2 : matrix_like
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    metric : str, optional, inner default.
        The distance metric to use. It can be 'cosine', 'euclidean', 'inner'.
    normalize : bool, optional, default false.
        Whether to normalize the input embeddings.
    csls_k : int, optional, 0 by default.
        K value for csls. If k > 0, enhance the similarity by csls.

    Returns
    -------
    sim_mat : An similarity matrix of size n1*n2.
    """
    if normalize:
        embed1 = preprocessing.normalize(embed1)
        embed2 = preprocessing.normalize(embed2)
    if metric == 'inner':
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'cosine' and normalize:
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'euclidean':
        sim_mat = 1 - euclidean_distances(embed1, embed2)
        # print(type(sim_mat), sim_mat.dtype)
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'cosine':
        sim_mat = 1 - cdist(embed1, embed2, metric='cosine')  # numpy.ndarray, float64
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'manhattan':
        sim_mat = 1 - cdist(embed1, embed2, metric='cityblock')
        sim_mat = sim_mat.astype(np.float32)
    else:
        sim_mat = 1 - cdist(embed1, embed2, metric=metric)
        sim_mat = sim_mat.astype(np.float32)
    if csls_k > 0:
        sim_mat = csls_sim(sim_mat, csls_k)
    return sim_mat


def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.

    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.

    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """
    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat - nearest_values1 - nearest_values2.T
    return csls_sim_mat

def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1, keepdims=True)

# 这部分是我写的调用函数, 调用chatgpt版本
def generate_entity_description(prompt):

    client = OpenAI(
        api_key="your api key",  
        base_url="https://api.chatfire.cn/v1"  
    )
    
    if not isinstance(prompt, str) or len(prompt.strip()) == 0:
        raise ValueError("输入的 prompt 必须是非空字符串。")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", 
                 "content": (
                    "You are an expert in knowledge graphs and creating concise descriptions. "
                    "Follow the provided instructions and generate a coherent response based on the input prompt."
                 )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,  
            temperature=0.3,  
        )
        
        # 提取生成的描述
        description = response.choices[0].message.content.strip()
        return description

    except Exception as e:
        return f"生成描述时发生错误: {e}"

    except Exception as e:
        return f"生成描述时发生错误: {e}"


## 这部分是调用通义千问版本
# def generate_entity_description_qwen(prompts, model="qwen-turbo-1101", batch_size=300):
#     client = OpenAI(
#         api_key="your api-key",  
#         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  
#     )

#     if not isinstance(prompts, list) or not all(isinstance(prompt, str) and len(prompt.strip()) > 0 for prompt in prompts):
#         raise ValueError("输入的 prompts 必须是非空字符串的列表。")

#     results = []
#     successful_calls = 0  

#     for idx, prompt in enumerate(prompts, start=1):
#         try:
#             completion = client.chat.completions.create(
#                 model=model,  
#                 messages=[
#                     {"role": "system", 
#                      "content": (
#                         "You are an expert in knowledge graphs and creating concise descriptions. "
#                         "Follow the provided instruction and generate a coherent response based on the input prompt."
#                      )
#                     },
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=300,  
#                 temperature=0.3,  
#             )
#             description = completion.choices[0].message.content.strip()
#             results.append(description)
#             successful_calls += 1

#             # 每 batch_size 次成功调用后输出成功信息
#             if successful_calls % batch_size == 0:
#                 logging.info(f"成功完成了 {successful_calls} 次 API 调用.")

#         except Exception as e:
#             results.append(f"生成描述时发生错误: {e}")

#     return results



def generate_entity_description_qwen(prompt, model="qwen-turbo-0211"):
    client = OpenAI(
        api_key="your api-key",  
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
    )

    if not isinstance(prompt, str) or len(prompt.strip()) == 0:
        raise ValueError("输入的 prompt 必须是非空字符串。")

    try:
        # 创建对话完成请求
        completion = client.chat.completions.create(
            model=model,  
            messages=[
                {"role": "system", 
                 "content": (
                    "You are an expert in knowledge graphs and creating concise descriptions. "
                    "Follow the provided instruction and generate a coherent response based on the input prompt."
                 )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,  
            temperature=0.3,  
        )
        
        # 提取生成的描述
        description = completion.choices[0].message.content.strip()
        return description

    except Exception as e:
        return f"生成描述时发生错误: {e}"

