
from aligncraft.models.retrievalea.retrievalea import RetrievalEA
from aligncraft.data_loading.load_dataset import load_dataset
import yaml

config_path = 'config/dbp15k_fr_en.yaml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

data_config_path = '/public/home/chenxuan/aligncraft/benchmark/config/DBP15K_FR_EN.yaml'
data_config = yaml.load(open(data_config_path, 'r'), Loader=yaml.FullLoader)

eadata = load_dataset('DBP15K', data_config)
retrieval = RetrievalEA(eadata, config)

retrieval.run()