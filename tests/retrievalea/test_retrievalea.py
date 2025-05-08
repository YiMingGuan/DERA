#!/usr/bin/env python

"""Tests for `retrievalea` package."""


import unittest
import itertools
from aligncraft.data_loading.load_dataset import load_dataset
import yaml
import random
from aligncraft.models.retrievalea.utils import entity_info
import os, json

class TestRetrieval(unittest.TestCase):
    """Tests for `retrievalea` package."""

    def setUp(self):
        config_path = '../benchmark/config/DBP15K_FR_EN.yaml'
        config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.data = load_dataset('DBP15K', config)
        self.log_path = 'logs/retrievalea'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_entityinfo(self):
        random_entity = random.choice(list(self.data.kg1.entities.values()))
        parameters = [True, False]
        outputs = []
        for combination in itertools.product(parameters, repeat=7):
            use_name, use_url, use_relationship, use_attributes, use_inverse, neighbors_use_attr, head_use_attr = combination
            info = entity_info(
                random_entity,
                use_name=use_name,
                use_url=use_url,
                use_translated_url=False,
                use_relationship=use_relationship,
                use_attributes=use_attributes,
                use_inverse=use_inverse,
                neighbors_use_attr=neighbors_use_attr,
                head_use_attr=head_use_attr
                )

            test_entity_info_log = os.path.join(self.log_path, 'test_entity_info.jsonl')
            output = {
                "use_name": use_name,
                "use_url": use_url,
                "use_relationship": use_relationship,
                "use_attributes": use_attributes,
                "use_inverse": use_inverse,
                "neighbors_use_attr": neighbors_use_attr,
                "head_use_attr": head_use_attr,
                "info": info
            }
            outputs.append(output)
        with open(test_entity_info_log, 'w', encoding='utf-8') as f:
            for output in outputs:
                json.dump(output, f, ensure_ascii=False)
                f.write('\n')

            

if __name__ == '__main__':
    unittest.main()

