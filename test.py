import yaml
import os

BASE_PATH = os.getcwd()
config_path = os.path.join(BASE_PATH, "config\config.yml")

with open(config_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg["prompt_template"])