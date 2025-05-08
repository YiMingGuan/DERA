
"""
test DBP15K build kg
"""
import yaml


config_path = '../benchmark/config/MEDBBK9K.yaml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

from aligncraft.data_loading.load_dataset import load_dataset

eadata = load_dataset('MEDBBK9K', config)

print(eadata)

print(list(eadata.kg1.predictions.keys())[: 5])
print(list(eadata.kg2.predictions.keys())[: 5])

print(list(eadata.kg1.attributes.values())[: 5])
print(list(eadata.kg2.attributes.values())[: 5])

print(list(eadata.kg1.attributes.keys())[: 5])
print(list(eadata.kg2.attributes.keys())[: 5])

print(list(eadata.kg1.ent_links.items())[: 5])

one_attr = list(eadata.kg1.attributes.values())[0]

entity = list(eadata.kg1.entities.values())[0]

for attribute, values in entity.attributes.items():
    for value in values:
        print(f"<{attribute.get_name()}>: {value}")