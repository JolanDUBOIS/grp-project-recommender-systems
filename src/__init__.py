import logging.config
from pathlib import Path

import yaml


config_file = Path('config') / 'logging.yml'
with open(config_file, 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
