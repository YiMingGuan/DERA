import json
import re


def parser(text):
    pattern = re.compile(r'(?i)(?:the\s+)?(?:correct\s+)?answer\s*(?:should\s+be)?\s*(?:is)?\s*[:=]?\s*\(([a-zA-Z])\)')
    # 搜索文本中的匹配项
    match = pattern.search(text)
    # 如果找到匹配项，则返回提取的答案
    if match:
        return match.group(1).lower()  # 将答案转换为小写
    else:
        return "None"  # 如果没有找到匹配项，返回None

if __name__ == '__main__':
    path = 'pilot_result.json'
    with open(path, 'r') as file:
        data = json.load(file)
    
    parsed_data = []
    for line in data:
        url = list(line.keys())
        assert len(url) == 1
        entity_info = line[url[0]]
        parsed_line = parser(entity_info['rerank_result'])
        parsed_data.append(parsed_line)

    print(parsed_data)