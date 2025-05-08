from .strategies.load_strategy_dbp15k import EADataDBP15K
from .strategies.load_strategy_dw15k import EADataDW15K 
from .strategies.load_strategy_medbbk9k import EADataMEDBBK9K
class LoadStrategyFactory:
    strategies = {
        "DBP15K": EADataDBP15K,
        "DW15KV2": EADataDW15K,
        "MEDBBK9K": EADataMEDBBK9K
    }

    @staticmethod
    def get_strategy(dataset_name):
        if dataset_name in LoadStrategyFactory.strategies:
            return LoadStrategyFactory.strategies[dataset_name.upper()]
        else:
            raise ValueError(f"Load strategy for dataset '{dataset_name}' is not defined.")
