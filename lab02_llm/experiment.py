"""
Лабораторная работа № 2.
Параметры генерации и конфигурация больших языковых моделей.

Реализация выполнена в виде небольшого фреймворка:
    - ExperimentRunner управляет жизненным циклом прогонов;
    - Case описывает один тестовый сценарий;
    - Recorder накапливает и сохраняет результаты.

Подключение к LLM — через общий модуль llm_client (читает .env).
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# --- Единая точка подключения ---
from llm_client import client as llm_client, MODEL, ENDPOINT, CANDIDATES


# ---------------------------------------------------------------------------
# Конфигурация эксперимента
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Основная модель берётся из llm_client (первая в LLM_CANDIDATES)
DEFAULT_MODEL = MODEL

# Резервные модели — остальные из списка (если понадобится сравнение)
FALLBACK_MODELS = CANDIDATES[1:]

NEUTRAL_SYSTEM = "Отвечай точно и по существу на русском языке."


# ---------------------------------------------------------------------------
# Модель данных эксперимента
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """Один вызов модели со всеми фактическими параметрами."""
    model: str
    stage: str
    case_id: str
    prompt: str
    system_id: str | None = None
    system_text: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    seed: int | None = None
    attempt: int = 1
    latency_s: float | None = None
    answer: str | None = None
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class Case:
    """Описание тестового сценария."""
    case_id: str
    prompt: str
    system_id: str | None = None
    system_text: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


# ---------------------------------------------------------------------------
# Журнал эксперимента
# ---------------------------------------------------------------------------

CSV_COLUMNS = list(Trial.__dataclass_fields__.keys())


@dataclass
class Recorder:
    trials: list[Trial] = field(default_factory=list)

    def add(self, trial: Trial) -> None:
        self.trials.append(trial)

    def select(self, stage: str) -> list[Trial]:
        return [t for t in self.trials if t.stage == stage]

    def save(self) -> tuple[Path, Path]:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = OUTPUT_DIR / f"log_{stamp}.json"
        csv_path = OUTPUT_DIR / f"log_{stamp}.csv"

        payload = [asdict(t) for t in self.trials]

        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(payload)

        return json_path, csv_path


# ---------------------------------------------------------------------------
# Клиент и исполнитель запросов
# ---------------------------------------------------------------------------

class ModelClient:
    """
    Обёртка над общим llm_client: один вызов — один Trial.
    Использует уже готовый OpenAI-клиент из llm_client, не создаёт новый.
    """

    def __init__(self, model: str):
        self.model = model
        self._client = llm_client          # <-- общий клиент из llm_client
        self._unsupported: set[str] = set()

    def call(self, case: Case, attempt: int = 1) -> Trial:
        trial = Trial(
            model=self.model,
            stage="",
            case_id=case.case_id,
            prompt=case.prompt,
            system_id=case.system_id,
            system_text=case.system_text,
            temperature=case.temperature,
            max_tokens=case.max_tokens,
            attempt=attempt,
        )

        messages: list[dict[str, str]] = []
        if case.system_text:
            messages.append({"role": "system", "content": case.system_text})
        messages.append({"role": "user", "content": case.prompt})

        payload: dict[str, Any] = {"model": self.model, "messages": messages}

        if case.temperature is not None and "temperature" not in self._unsupported:
            payload["temperature"] = case.temperature
        if case.max_tokens is not None and "max_tokens" not in self._unsupported:
            payload["max_tokens"] = case.max_tokens

        try:
            t0 = time.perf_counter()
            response = self._client.chat.completions.create(**payload)
            trial.latency_s = time.perf_counter() - t0

            trial.answer = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            if usage:
                trial.prompt_tokens = usage.prompt_tokens
                trial.completion_tokens = usage.completion_tokens
                trial.total_tokens = usage.total_tokens
        except Exception as exc:
            message = str(exc).lower()
            for param in ("temperature", "max_tokens"):
                if param in message and param not in self._unsupported:
                    self._unsupported.add(param)
                    trial.error = f"unsupported parameter: {param}"
                    return trial
            trial.error = str(exc)

        return trial

    def warm(self) -> None:
        try:
            self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Серии эксперимента (без изменений)
# ---------------------------------------------------------------------------

STAGE1_CASES = [
    Case(
        case_id="hash_vs_cipher",
        prompt=(
            "Объясни студенту 4 курса разницу между хешированием и "
            "симметричным шифрованием. Не более 120 слов."
        ),
        system_id="neutral",
        system_text=NEUTRAL_SYSTEM,
    ),
    Case(
        case_id="extract_json",
        prompt=(
            "Извлеки из текста сущности и верни JSON с полями ip, user, "
            "timestamp, event_type. Текст: В 03:17 12.05.2026 с адреса "
            "192.168.10.15 пользователь admin выполнил вход в систему. "
            "Тип события: authentication."
        ),
        system_id="neutral",
        system_text=NEUTRAL_SYSTEM,
    ),
    Case(
        case_id="sha256_check",
        prompt=(
            "Напиши Python-функцию проверки SHA-256 хеша строки "
            "(на входе строка и ожидаемый хеш, на выходе True/False)."
        ),
        system_id="neutral",
        system_text=NEUTRAL_SYSTEM,
    ),
]

INCIDENT_QUERY = (
    "Событие: в 03:17 зафиксировано 25 неудачных попыток входа в учётную "
    "запись admin с одного IP, после чего произошёл успешный вход. "
    "Проанализируй ситуацию."
)

STAGE2_SYSTEMS = {
    "plain": "Отвечай на запрос пользователя.",
    "role_5_points": (
        "Ты ассистент аналитика ИБ. Отвечай не более чем в 5 пунктах. "
        "Для каждого пункта укажи риск и рекомендуемое действие."
    ),
    "facts_hypotheses_actions": (
        "Ты ассистент аналитика ИБ. Не выдумывай недостающие данные. "
        "Явно отделяй факты от предположений. Ответ верни в формате: "
        "Факты / Гипотезы / Действия."
    ),
}

STAGE3_PROMPT = (
    "Предложи пять названий сервиса для автоматического анализа журналов "
    "событий информационной безопасности. Для каждого названия дай "
    "пояснение в одном предложении."
)
STAGE3_TEMPS = [0.0, 0.3, 0.7, 1.0]
STAGE3_REPEATS = 5

STAGE4_PROMPT = (
    "Объясни студенту 4 курса архитектуру RAG-системы: ingestion, "
    "chunking, embeddings, vector database, retrieval, prompt construction "
    "и generation. Для каждого этапа укажи его назначение и одну "
    "типичную ошибку."
)
STAGE4_LIMITS = [100, 400, 1500]

STAGE5_CASE = Case(
    case_id="incident_recommended",
    prompt=INCIDENT_QUERY,
    system_id="facts_hypotheses_actions",
    system_text=STAGE2_SYSTEMS["facts_hypotheses_actions"],
    temperature=0.0,
    max_tokens=800,
)


def run_stage1(client: ModelClient, recorder: Recorder) -> None:
    print("\n[Этап 1] Базовая конфигурация")
    for case in STAGE1_CASES:
        trial = client.call(case)
        trial.stage = "baseline"
        recorder.add(trial)
        _print_trial(trial)


def run_stage2(client: ModelClient, recorder: Recorder) -> None:
    print("\n[Этап 2] Влияние system prompt")
    for system_id, system_text in STAGE2_SYSTEMS.items():
        case = Case(
            case_id=f"incident__{system_id}",
            prompt=INCIDENT_QUERY,
            system_id=system_id,
            system_text=system_text,
        )
        trial = client.call(case)
        trial.stage = "system_prompt"
        recorder.add(trial)
        _print_trial(trial)


def run_stage3(client: ModelClient, recorder: Recorder) -> None:
    print("\n[Этап 3] Влияние temperature")
    for temp in STAGE3_TEMPS:
        for attempt in range(1, STAGE3_REPEATS + 1):
            case = Case(
                case_id=f"naming__t{temp}",
                prompt=STAGE3_PROMPT,
                temperature=temp,
            )
            trial = client.call(case, attempt=attempt)
            trial.stage = "temperature"
            recorder.add(trial)
    _summarize_temperature(recorder)


def run_stage4(client: ModelClient, recorder: Recorder) -> None:
    print("\n[Этап 4] Ограничение длины ответа")
    for limit in STAGE4_LIMITS:
        case = Case(
            case_id=f"rag__max{limit}",
            prompt=STAGE4_PROMPT,
            max_tokens=limit,
        )
        trial = client.call(case)
        trial.stage = "max_tokens"
        recorder.add(trial)
        _print_trial(trial)


def run_stage5(client: ModelClient, recorder: Recorder) -> None:
    print("\n[Этап 5] Проверка рекомендуемой конфигурации")
    trial = client.call(STAGE5_CASE)
    trial.stage = "recommended"
    recorder.add(trial)
    _print_trial(trial)


# ---------------------------------------------------------------------------
# Печать и сводки (без изменений)
# ---------------------------------------------------------------------------

def _print_trial(trial: Trial) -> None:
    if trial.error:
        print(f"  {trial.case_id}: ошибка — {trial.error}")
        return
    answer = trial.answer or ""
    latency = trial.latency_s or 0.0
    tokens = trial.completion_tokens if trial.completion_tokens is not None else "?"
    print(f"  {trial.case_id}: символов={len(answer)}, "
          f"время={latency:.3f} с, токенов={tokens}")


def _summarize_temperature(recorder: Recorder) -> None:
    print("\n  Сводка по temperature:")
    for temp in STAGE3_TEMPS:
        rows = [
            t for t in recorder.select("temperature")
            if t.temperature == temp
        ]
        answers = [t.answer or "" for t in rows]
        unique = len(set(answers))
        latencies = [t.latency_s for t in rows if t.latency_s is not None]
        avg = statistics.mean(latencies) if latencies else 0.0
        print(f"    t={temp}: попыток={len(rows)}, "
              f"уникальных={unique}, среднее время={avg:.3f} с")


def summarize_all(recorder: Recorder) -> None:
    print("\n=== Итоговые сводки ===")

    print("\nПо system prompt:")
    for t in recorder.select("system_prompt"):
        if t.answer:
            print(f"  {t.system_id}: символов={len(t.answer)}, "
                  f"время={t.latency_s:.3f} с")

    print("\nПо max_tokens:")
    for t in recorder.select("max_tokens"):
        if t.answer:
            print(f"  max_tokens={t.max_tokens}: символов={len(t.answer)}, "
                  f"время={t.latency_s:.3f} с, "
                  f"токенов={t.completion_tokens}")

    print("\nРекомендованная конфигурация:")
    for t in recorder.select("recommended"):
        if t.answer:
            print(f"  символов={len(t.answer)}, "
                  f"время={t.latency_s:.3f} с, "
                  f"токенов={t.completion_tokens}")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Модель:   {DEFAULT_MODEL}")
    print(f"Endpoint: {ENDPOINT}")
    if FALLBACK_MODELS:
        print(f"Резерв:   {FALLBACK_MODELS}")

    client = ModelClient(DEFAULT_MODEL)
    recorder = Recorder()

    client.warm()

    run_stage1(client, recorder)
    run_stage2(client, recorder)
    run_stage3(client, recorder)
    run_stage4(client, recorder)
    run_stage5(client, recorder)

    json_path, csv_path = recorder.save()
    print(f"\nЖурнал сохранён:\n  {json_path}\n  {csv_path}")

    summarize_all(recorder)


if __name__ == "__main__":
    main()