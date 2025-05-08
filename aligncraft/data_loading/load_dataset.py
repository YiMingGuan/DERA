
from .load_strategy_factory import LoadStrategyFactory

def load_dataset(name, config):
    strategy = LoadStrategyFactory.get_strategy(name)
    data = strategy(config)
    return data
