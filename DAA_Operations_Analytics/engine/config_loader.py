import json
from pathlib import Path
from typing import Dict, Any

def load_config(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return json.load(f)

# Optional helper
def get_sys_config(filepath: str = r'data\inputs\config.json') -> Dict[str, Any]:
    return load_config(filepath)
