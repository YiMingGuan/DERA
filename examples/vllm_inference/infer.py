from transformers import AutoTokenizer
from aligncraft.data_loading.load_dataset import load_dataset
import yaml
import argparse
from vllm import LLM, SamplingParams
import logging
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='VLLM Inference')
    parser.add_argument('--data_config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--data_type', type=str, required=True, help='Type of the data')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top p for sampling')
    parser.add_argument('--max_tokens', type=int, default=512, help='Max tokens for sampling')
    parser.add_argument('--repetition_penalty', type=float, default=1.1, help='Repetition penalty for sampling')
    return parser.parse_args()

def load_eadata(args):
    data_config_path = args.data_config
    data_type = args.data_type.upper()
    data_config = yaml.load(open(data_config_path, 'r'), Loader=yaml.FullLoader)
    eadata = load_dataset(data_type, data_config)
    return eadata


def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer

def prepreocess(tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    print(messages)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

    
def load_vllm_model(args):
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=1, swap_space=4)
    logger.info(f"Successfully loaded VLLM model: {args.model_name_or_path}")
    return llm

def vllm_generate(model, prompt, **kwargs):
    """
    Generate a sequence given a prompt and an entity using VLLM.
    """
    temperature = kwargs.get('temperature', 0.8)
    top_p = kwargs.get('top_p', 0.95)
    max_tokens = kwargs.get('max_tokens', 512)
    repetition_penalty = kwargs.get('repetition_penalty', 1.1)
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, repetition_penalty=repetition_penalty)
    logger.info(f"SamplingParams: \n{sampling_params}")
    sequences = []
    outputs = model.generate(prompt, sampling_params=sampling_params)
    for output in outputs:
        sequences.append(output.outputs[0].text)
    return sequences

template = """Now that you are an encyclopedia, please give a brief overview to the entities provided to you.
Entity Attributes: {}

Attention:
1.All answers are in English.
2.Refer to the information provided to you and combine your knowledge to describe the entity.
3.The information provided to you may contain errors or redundancies, and you need to refer to it with caution.
4.Do not print any information other than the brief overview of the entity.
"""

template2 = """There is a central entity, along with its neighbors and its properties. Generate a short, ontological description of the entity based on its triples in English. Here is the triplet of that entity: {} \nAnswer in English!"""

if __name__ == '__main__':
    args = parse_args()
    # eadata = load_eadata(args)
    vllm = load_vllm_model(args)
    tokenizer = load_tokenizer(args.model_name_or_path)
    prompt = """operator 龐巴迪運輸（Bombardier Transportation）
operator (（坎登線與布朗斯威克線）)
operator （與馬里蘭運輸局簽約）
operator （賓州線）
stations 42
began operation 1984
caption 下圖：賓州線巴爾的摩－華盛頓國際機場站的馬里蘭-{區域}-通勤鐵路列車。
caption 上圖：布朗斯威克線狄克森車站的馬里蘭-{區域}-通勤鐵路列車
marks MARC
owner 馬里蘭運輸局（Maryland Transit Administration）
imagesize 310
imagesize 200
transit type -{區域}-／通勤鐵路
image MARC Dickerson station.jpg
image 7.0
name 馬里蘭-{區域}-通勤鐵路（Maryland Area Regional Commuter Rail， MARC）
ridership 33696
el 賓州線為交流電，25 Hz
lines 3
chief executive John Hovatter
map state collapsed"""
    prompt = prepreocess(tokenizer, template2.format(prompt))
    seq = vllm_generate(vllm, prompt)
    for s in seq:
        print(s)
        print('-' * 20)
