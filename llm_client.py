"""
Единая точка подключения к LLM через .env.
Переменные окружения (см. .env):
    LLM_ENDPOINT   — например http://localhost:1234/v1
    LLM_TOKEN      — например lm-studio
    LLM_CANDIDATES — через запятую, первая = основная модель
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Не задана переменная окружения {name}. "
            f"Проверьте файл .env в корне проекта."
        )
    return value.strip()


# Endpoint и токен — обязательные
ENDPOINT = _require("LLM_ENDPOINT")
TOKEN = _require("LLM_TOKEN")

# Список моделей: строка "model1, model2, ..." → ["model1", "model2", ...]
CANDIDATES_RAW = _require("LLM_CANDIDATES")
CANDIDATES = [m.strip() for m in CANDIDATES_RAW.split(",") if m.strip()]

if not CANDIDATES:
    raise RuntimeError("LLM_CANDIDATES пуст — укажите хотя бы одну модель.")

# Основная модель — первая в списке
MODEL = CANDIDATES[0]

# Готовый клиент
client = OpenAI(api_key=TOKEN, base_url=ENDPOINT)