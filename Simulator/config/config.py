import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def get_project_root(search_file=".git"):
    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if (parent / search_file).exists():
            return parent
    return None


ROOT_PATH = os.path.join(get_project_root(), "Simulator")
DATA_PATH = os.path.join(ROOT_PATH, "data", "tasks.json")

# --- openai ---
API_KEY = os.getenv("API_KEY")
API_BASE = "https://api.deepseek.com/v1"

TEMPERATURE = 0.7
MODEL_NAME = "deepseek-chat"
