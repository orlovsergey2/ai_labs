"""
Лабораторная работа №4.
Оценка качества LLM. Сравнение двух конфигураций.
"""

import os
import csv
import json
import time
import re
from pathlib import Path
from statistics import mean, median
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

from llm_client import client, MODEL, ENDPOINT

PROMPTS_DIR = Path("prompts")
TESTS_FILE = Path("tests.csv")


def ask(system_prompt: str, user_prompt: str, temperature: float = 0.2):
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
    usage = getattr(response, "usage", None)
    return response.choices[0].message.content or "", elapsed, usage


def extract_json(text: str):
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def load_tests() -> list:
    with TESTS_FILE.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_model(name: str, system_prompt: str, temperature: float, tests: list):
    rows = []
    for t in tests:
        raw, latency, usage = ask(system_prompt, t["input"], temperature)
        parsed = extract_json(raw)
        predicted = parsed.get("label") if parsed else "parse_error"
        confidence = parsed.get("confidence") if parsed else None
        reason = parsed.get("reason") if parsed else None

        in_tok = getattr(usage, "prompt_tokens", None) if usage else None
        out_tok = getattr(usage, "completion_tokens", None) if usage else None

        rows.append({
            "id": t["id"],
            "input": t["input"],
            "expected": t["expected_label"],
            "predicted": predicted,
            "confidence": confidence,
            "reason": reason,
            "format_ok": parsed is not None,
            "latency_s": round(latency, 2),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "raw": raw.replace("\n", " ")[:1000],
        })

    out_path = Path(f"results_{name}.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"[{name}] сохранено в {out_path}")
    return rows


def compute_metrics(rows: list) -> dict:
    y_true = [r["expected"] for r in rows]
    y_pred = [r["predicted"] for r in rows]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="suspicious", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="suspicious", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="suspicious", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=["normal", "suspicious"]).tolist()

    format_ok = sum(1 for r in rows if r["format_ok"]) / len(rows)
    lat = [r["latency_s"] for r in rows]
    p95 = sorted(lat)[int(0.95 * len(lat)) - 1]
    in_tok = sum(r["input_tokens"] or 0 for r in rows)
    out_tok = sum(r["output_tokens"] or 0 for r in rows)

    return {
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "confusion_matrix": cm,
        "format_compliance": round(format_ok, 3),
        "mean_latency": round(mean(lat), 2),
        "median_latency": round(median(lat), 2),
        "p95_latency": round(p95, 2),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }


def main():
    print(f"Endpoint: {ENDPOINT}")
    print(f"Model:    {MODEL}")

    tests = load_tests()
    print(f"Tests:    {len(tests)}")

    prompt_a = (PROMPTS_DIR / "prompt_simple.txt").read_text(encoding="utf-8")
    prompt_b = (PROMPTS_DIR / "prompt_json.txt").read_text(encoding="utf-8")

    print("\n=== Model A (simple, temp=0.7) ===")
    rows_a = run_model("model_a", prompt_a, 0.7, tests)

    print("\n=== Model B (strict JSON, temp=0.2) ===")
    rows_b = run_model("model_b", prompt_b, 0.2, tests)

    m_a = compute_metrics(rows_a)
    m_b = compute_metrics(rows_b)

    print("\n=== Метрики ===")
    for name, m in (("A", m_a), ("B", m_b)):
        print(f"\nModel {name}:")
        for k, v in m.items():
            print(f"  {k}: {v}")

    summary = []
    for name, m in (("model_a", m_a), ("model_b", m_b)):
        row = {"model": name}
        row.update({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in m.items()})
        summary.append(row)

    with Path("summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    print("\nСводка сохранена в summary.csv")


if __name__ == "__main__":
    main()