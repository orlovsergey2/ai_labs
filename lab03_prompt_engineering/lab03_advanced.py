"""
Лабораторная работа №3 — дополнительное задание повышенной сложности.
Автоматизированный прогон тестов по нескольким версиям промпта,
сохранение в CSV/JSON, проверка JSON-схемы, LLM-as-a-Judge, сводная таблица.
"""

import os
import csv
import json
import time
import re
from pathlib import Path
from statistics import mean
from dotenv import load_dotenv
from openai import OpenAI
from jsonschema import validate, ValidationError

from llm_client import client, MODEL, ENDPOINT

PROMPTS_DIR = Path("prompts")
TESTS_FILE = Path("tests.json")
JUDGE_PROMPT_FILE = Path("judge_prompt.txt")
RESULTS_CSV = Path("results.csv")
RESULTS_JSON = Path("results.json")
JUDGE_CSV = Path("judge_results.csv")
SUMMARY_CSV = Path("summary.csv")

# JSON-схема для валидации ответов v3 (и финального промпта)
ANSWER_SCHEMA = {
    "type": "object",
    "required": ["category", "risk", "evidence", "missing_data", "next_action", "status"],
    "properties": {
        "category": {"type": "string"},
        "risk": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "missing_data": {"type": "array", "items": {"type": "string"}},
        "next_action": {"type": "string"},
        "status": {"type": "string", "enum": ["ok", "insufficient_data"]},
    },
    "additionalProperties": False,
}


# ---------- Базовые утилиты ----------

def ask(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> tuple[str, float]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    elapsed = time.perf_counter() - started
    return (response.choices[0].message.content or ""), elapsed


def extract_json(text: str):
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", t, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


def load_prompts() -> dict:
    return {p.stem: p.read_text(encoding="utf-8")
            for p in sorted(PROMPTS_DIR.glob("prompt_v*.txt"))}


def load_tests() -> list:
    return json.loads(TESTS_FILE.read_text(encoding="utf-8"))


# ---------- Проверка JSON по схеме ----------

def validate_answer(raw: str) -> tuple[bool, str]:
    """Возвращает (валиден ли JSON, причина)."""
    parsed = extract_json(raw)
    if parsed is None:
        return False, "invalid_json"
    try:
        validate(instance=parsed, schema=ANSWER_SCHEMA)
        return True, "ok"
    except ValidationError as e:
        return False, f"schema_error: {e.message[:120]}"


# ---------- Прогон ----------

def run_all(prompts: dict, tests: list) -> list:
    rows = []
    for version, system_prompt in prompts.items():
        print(f"\n=== {version} ===")
        for test in tests:
            try:
                text, sec = ask(system_prompt, test["input"])
            except Exception as e:
                text, sec = f"ERROR: {e}", 0.0

            is_valid, reason = validate_answer(text)
            rows.append({
                "test_id": test["id"],
                "test_type": test["type"],
                "version": version,
                "time_sec": round(sec, 2),
                "json_valid": is_valid,
                "json_reason": reason,
                "response": text.replace("\n", " ")[:2000],
            })
            print(f"  {test['id']:>3} ({test['type']:<18}) "
                  f"json_valid={is_valid}  time={sec:.2f}s")
    return rows


def save_results(rows: list):
    # CSV
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # JSON
    RESULTS_JSON.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------- LLM-as-a-Judge ----------

def judge_one(input_text: str, response_text: str) -> dict | None:
    judge_template = JUDGE_PROMPT_FILE.read_text(encoding="utf-8")
    prompt = (judge_template
              .replace("{{INPUT}}", input_text)
              .replace("{{RESPONSE}}", response_text))
    try:
        text, _ = ask("Ты беспристрастный оценщик.", prompt, temperature=0.0)
    except Exception:
        return None
    parsed = extract_json(text)
    if not parsed:
        return None
    return parsed


def run_judge(rows: list, tests: list) -> list:
    test_by_id = {t["id"]: t for t in tests}
    judged = []
    total = len(rows)
    for i, row in enumerate(rows, 1):
        test = test_by_id[row["test_id"]]
        print(f"[judge {i}/{total}] {row['version']} / {row['test_id']}")
        verdict = judge_one(test["input"], row["response"])
        if verdict is None:
            judged.append({
                "test_id": row["test_id"],
                "version": row["version"],
                "task": None, "factual": None, "format": None,
                "completeness": None, "uncertainty": None,
                "judge_total": None, "comment": "judge_failed",
            })
            continue
        total_score = sum(verdict.get(k, 0) or 0
                          for k in ("task", "factual", "format", "completeness", "uncertainty"))
        judged.append({
            "test_id": row["test_id"],
            "version": row["version"],
            "task": verdict.get("task"),
            "factual": verdict.get("factual"),
            "format": verdict.get("format"),
            "completeness": verdict.get("completeness"),
            "uncertainty": verdict.get("uncertainty"),
            "judge_total": total_score,
            "comment": (verdict.get("comment") or "")[:200],
        })
    return judged


def save_judge(judged: list):
    if not judged:
        return
    with JUDGE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=judged[0].keys())
        writer.writeheader()
        writer.writerows(judged)


# ---------- Сводная таблица ----------

def build_summary(rows: list, judged: list) -> list:
    versions = sorted({r["version"] for r in rows})
    judge_by = {(j["version"], j["test_id"]): j for j in judged}

    summary = []
    for v in versions:
        v_rows = [r for r in rows if r["version"] == v]
        v_judge = [judge_by[(v, r["test_id"])] for r in v_rows if (v, r["test_id"]) in judge_by]

        json_valid_rate = mean(1 if r["json_valid"] else 0 for r in v_rows)
        avg_time = mean(r["time_sec"] for r in v_rows)

        judge_totals = [j["judge_total"] for j in v_judge if j["judge_total"] is not None]
        avg_judge = mean(judge_totals) if judge_totals else None

        summary.append({
            "version": v,
            "tests": len(v_rows),
            "json_valid_rate": round(json_valid_rate, 3),
            "avg_time_sec": round(avg_time, 2),
            "avg_judge_total": round(avg_judge, 2) if avg_judge is not None else "",
        })

    return summary


def save_summary(summary: list):
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


# ---------- main ----------

def main():
    print(f"Endpoint: {ENDPOINT}")
    print(f"Model:    {MODEL}")

    prompts = load_prompts()
    tests = load_tests()
    print(f"Prompts:  {list(prompts)}")
    print(f"Tests:    {len(tests)}")

    # 1. Прогон
    rows = run_all(prompts, tests)
    save_results(rows)
    print(f"\nСырые ответы: {RESULTS_CSV} и {RESULTS_JSON}")

    # 2. LLM-as-a-Judge
    print("\n=== LLM-as-a-Judge ===")
    judged = run_judge(rows, tests)
    save_judge(judged)
    print(f"Оценки судьи: {JUDGE_CSV}")

    # 3. Сводная
    summary = build_summary(rows, judged)
    save_summary(summary)
    print(f"\n=== Сводка ===")
    for s in summary:
        print(f"  {s['version']}: json_valid={s['json_valid_rate']}, "
              f"avg_time={s['avg_time_sec']}s, judge_total={s['avg_judge_total']}")
    print(f"\nСводная таблица: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()