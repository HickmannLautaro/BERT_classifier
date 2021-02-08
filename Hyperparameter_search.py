import yaml
from Train_BERT import run_experiment



conf = "./Configs/amazon_config_lvl1_bert-base-uncased.yaml"
with open(conf) as f:
    arguments = yaml.load(f, Loader=yaml.FullLoader)
for i in range(arguments['repetitions']):
    run_experiment(arguments, True)
