"""
Запуск полного пайплайна одной командой:
    python run_eval.py --tag v1_2026-10-07
Сохраняет всё в results/<tag>/.
"""
import argparse
import json
import subprocess
import shutil
import csv as _csv
from datetime import datetime
from pathlib import Path

from llm_client import ENDPOINT, MODEL, CANDIDATES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=datetime.now().strftime("run_%Y%m%d_%H%M"))
    args = parser.parse_args()

    out_dir = Path("results") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Конфигурация берётся напрямую из llm_client, а не хардкодится
    config = {
        "tag": args.tag,
        "date": datetime.now().isoformat(),
        "endpoint": ENDPOINT,
        "model": MODEL,
        "candidates": CANDIDATES,
        "model_a": {"prompt": "prompt_simple.txt", "temperature": 0.7},
        "model_b": {"prompt": "prompt_json.txt", "temperature": 0.2},
        "tests": "tests.csv",
        "n_tests": 30,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))

    shutil.copy("tests.csv", out_dir / "tests.csv")
    shutil.copytree("prompts", out_dir / "prompts", dirs_exist_ok=True)

    subprocess.run(["python", "lab04.py"], check=True)

    for f in ["results_model_a.csv", "results_model_b.csv", "summary.csv"]:
        shutil.copy(f, out_dir / f)

    summary = list(_csv.DictReader(open("summary.csv", encoding="utf-8")))
    md = [f"# Отчёт: {args.tag}", "", f"Дата: {config['date']}", ""]
    md.append("## Метрики")
    md.append("")
    headers = list(summary[0].keys())
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in summary:
        md.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Готово: {out_dir}")


if __name__ == "__main__":
    main()