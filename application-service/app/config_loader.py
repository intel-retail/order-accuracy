import yaml
import os

_CONFIG = None

def load_config(path=None):
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    path = path or os.getenv("APP_CONFIG", "/config/application.yaml")
    with open(path, "r") as f:
        _CONFIG = yaml.safe_load(f)

    return _CONFIG
