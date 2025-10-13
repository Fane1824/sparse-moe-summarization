import os
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


def load_env(env_path: str | None = None) -> None:
    if env_path and os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        # Try project root .env
        root_env = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
        if os.path.exists(root_env):
            load_dotenv(root_env)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

