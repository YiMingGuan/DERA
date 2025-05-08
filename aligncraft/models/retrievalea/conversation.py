
"""
Data processing and prompt for Retrieval
"""

entity_info = "<Mount Everest, located in, Himalayas [name:Himalayas; country:Nepal; China;]>\n<Mount Everest, first ascended by, Edmund Hillary [nationality:New Zealander; year:1953;]>\n<Mount Everest, first ascended by, Tenzing Norgay [nationality:Nepalese; year:1953;]>\n<Mount Everest, height, 8848.86 meters [measurement unit:Meters;]>\n<Mount Everest, prominence, 8848 meters [measurement unit:Meters;]>\n<Sagarmatha National Park [name:Sagarmatha National Park; area:1148 sq km;], contains, Mount Everest>\n<K2 [name:K2; height:8611 meters;], second highest, Mount Everest>\n<Mount Everest, coordinates, 27.9881° N, 86.9250° E [measurement unit:Degree;]>\n<Mount Everest, also known as, Sagarmatha [language:Nepalese;]>\n<Mount Everest, also known as, Qomolangma [language:Tibetan;]>"
llama2_chat_template = """[INST]
{} [/INST]"""


mistral_instruct_template = """[INST]There is a central entity, along with its neighbors and its properties. Generate a short, ontological description of the entity based on its triples in English. Here is the triplet of that entity:
{} \bAnswer in English![/INST]"""

qwen2_chat_template = """There is a central entity, along with its neighbors and its properties. Generate a short, ontological description of the entity based on its triples in English. Here is the triplet of that entity: {} \nAnswer in English!"""

qwen2_chat_template_2 = """[INST]You generate a strongly distinguishable description of the entity based on the name of the entity provided to you. These include the ontology type of the entity, the information entropy of the entity is high information. All answers are in English (if non-English words are encountered, the answer is translated into English).\n{}[/INST]"""

qwen2_chat_chinese_translation_template = """请根据提供给您的实体名称，三元组关系生成该实体的总结。请全部使用英文回答。Please generate a strongly distinguishable and brief summary of the entity based on the name of the entity provided to you, the triplet relationship. All answers will be in English.\n{}"""

qwen2_chat_med_bbk_9k_vanilla_template = """你是一名医学领域专家，现在给你一个医学名词或者相关信息，请你根据提供的信息对该医学名词生成一段详细描述。\n注意：\n1.有些医学名词没有具体的名称，只有代号（比如a348921），这时候你需要根据图谱中它的邻居信息和属性信息进行推断。\n2.不要输出推断信息，只输出最后的描述即可。\n3.请使用中文回答。\n{}"""

mistral_instruct_template_only_name_translation = """[INST]You generate a strongly distinguishable description of the entity based on the name of the entity provided to you. The entity name is surrounded by <\t>.\n<\t>{}<\t>\nIf the name of the entity is not in English, translate it into English.[/INST]"""

qwen2_chat_template_only_name_translation = """Now that you are an encyclopedia, ask you to explain and describe the entities provided to you. 
Entity: {}
Attention: 
1. All answers are in English. If the entity name provided to you is not in English, please first translate it into English. 
2. Don't output inferences, just the final description. 
3. Use your knowledge to describe the entity."""

qwen2_chat_template_only_name_1 = """Now that you are an encyclopedia, please give a brief introduction to the entities provided to you.
Entity: {}

Attention:
1.All answers are in English. If the entity name provided to you is not in English, please first translate it into English.
2.Do not output inferences, only the final introduction.
3.Introduce the English name (or translated name), type and other proprietary information of the entity.
4.The output format is :
Type: 
Proprietary information: """

qwen2_chat_template_only_name_2 = """Now that you are an encyclopedia, please give a brief introduction to the entities provided to you.
Entity and Triples: {}

Attention:
1.All answers are in English. If the entity name provided to you is not in English, please first translate it into English.
2.Do not output inferences, only the final introduction.
3.Introduce the English name (or translated name), type and other proprietary information of the entity.
4.The output format is :
Type: 
Proprietary information: """

mistral_instruct_template_only_name_2 = """[INST]Now that you are an encyclopedia, please give a brief introduction to the entities provided to you.
Entity and Triples: {}

Attention:
1.All answers are in English. If the entity name provided to you is not in English, please first translate it into English.
2.Do not output inferences, only the final introduction.
3.Introduce the English name (or translated name), type and other proprietary information of the entity.
4.The output format is :
Type: 
Proprietary information: [/INST]"""

qwen2_0_point_5_chat_template = """Now that you are an encyclopedia, please give a brief introduction to the entities provided to you.
Entity and Triples: {}

Attention:
1.All answers are in English. If the entity name provided to you is not in English, please first translate it into English.
2.Do not output inferences, only the final introduction.
3.Introduce the English name (or translated name), type and other proprietary information of the entity.
4.The output format is :
Type: 
Proprietary information: """

def construct_prompt(prompt_type, input_message, tokenizer=None):
    # 根据模型类型构造特定的prompt
    if prompt_type == "llama2-chat":
        return f"{llama2_chat_template.format(input_message)}"
    elif prompt_type == 'mistral-instruct':
        return f"{mistral_instruct_template.format(input_message)}"
    elif prompt_type == 'qwen2-chat':
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": qwen2_chat_template.format(input_message)}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text
    elif prompt_type == 'qwen2-chat-translation':
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": qwen2_chat_template_2.format(input_message)}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text
    elif prompt_type == "qwen2_chat_chinese_translation_template":
         return f"{qwen2_chat_chinese_translation_template.format(input_message)}"
        # messages = [
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": qwen2_chat_chinese_translation_template.format(input_message)}
        # ]
        # text = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # return text
    elif prompt_type == "qwen2_chat_med_bbk_9k_vanilla_template":
        messages = [
            {"role": "system", "content": "You are a medical expert."},
            {"role": "user", "content": qwen2_chat_med_bbk_9k_vanilla_template.format(input_message)}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text
    elif prompt_type == "mistral_instruct_template_only_name_translation":
        return f"{mistral_instruct_template_only_name_translation.format(input_message)}"
    elif prompt_type == "qwen2_chat_template_only_name_translation":
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": qwen2_chat_template_only_name_translation.format(input_message)}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text
    elif prompt_type == "qwen2_chat_template_only_name_1":
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": qwen2_chat_template_only_name_1.format(input_message)}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text
    elif prompt_type == "qwen2_chat_template_only_name_2":
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": qwen2_chat_template_only_name_2.format(input_message)}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text
    elif prompt_type == "mistral_instruct_template_only_name_2":
        return f"{mistral_instruct_template_only_name_2.format(input_message)}"
    else:
        raise ValueError(f"Model type {model_type} not supported")
    