# -*- coding: utf-8 -*-

import json
import os
import re
import time
from typing import Any, Dict, List

from openai import OpenAI


INPUT_FILE = "../difference_types_SMILES/aromatic_hydrocarbon.json"
OUTPUT_FILE = "../data/aromatic_hydrocarbon_outputs/results.json"
PROMPT_FILE = "./prompts/aromatic_hydrocarbon_prompt.txt"

MODEL = "gpt-4.1"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = [16384, 32768, 65536]

ID_MIN = 1
ID_MAX = 1957
START = 0

SAVE_EVERY = 1
SLEEP_SECONDS = 0.0
USE_JSON_MODE = False
SKIP_EXISTING = False

API_KEY = os.environ.get("OPENAI_API_KEY")


def load_json_records(path: str) -> List[Dict[str, Any]]:
    text = open(path, "r", encoding="utf-8").read().strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ["data", "records", "items", "results"]:
                if isinstance(obj.get(key), list):
                    return obj[key]
            return [obj]
    except json.JSONDecodeError:
        pass
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def save_json(records: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
    raise ValueError("Cannot parse JSON output.")


def normalize_spectrum(data: Dict[str, Any]):
    spectrum = data.get("spectrum", {})
    if isinstance(spectrum, dict):
        return {str(k): v for k, v in spectrum.items()}
    return {}


def build_prompt(data: Dict[str, Any]):
    info = {
        "source_smiles": data.get("smiles", ""),
        "name": data.get("name", ""),
        "formula": data.get("formula", ""),
        "mw": data.get("mw", ""),
        "compound_class": data.get("compound_class", ""),
        "spectrum": normalize_spectrum(data),
    }
    return (
        "Please perform EI (70 eV) fragmentation decision annotation "
        "for the following molecule.\n"
        "Output strict JSON only.\n\n"
        + json.dumps(info, ensure_ascii=False, indent=2)
    )


def call_model(client, messages):
    last_error = None
    for token in MAX_OUTPUT_TOKENS:
        try:
            response = client.responses.create(
                model=MODEL,
                input=messages,
                temperature=TEMPERATURE,
                max_output_tokens=token,
            )
            return extract_json(response.output_text), token, None
        except Exception as e:
            last_error = str(e)
            time.sleep(2)
    return None, None, last_error


def main():
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY is missing.")

    client = OpenAI(api_key=API_KEY)
    prompt = load_prompt(PROMPT_FILE)

    records = load_json_records(INPUT_FILE)

    selected = []
    for item in records:
        rid = item.get("id")
        if isinstance(rid, int) and ID_MIN <= rid <= ID_MAX:
            selected.append(item)

    selected = selected[START:]

    results = []

    for i, data in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] id={data.get('id')} name={data.get('name')}")

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": build_prompt(data)},
        ]

        output, token, error = call_model(client, messages)

        results.append(
            {
                "id": data.get("id"),
                "name": data.get("name"),
                "smiles": data.get("smiles"),
                "formula": data.get("formula"),
                "mw": data.get("mw"),
                "compound_class": data.get("compound_class", ""),
                "input_spectrum": normalize_spectrum(data),
                "model_output": output,
                "parse_ok": output is not None,
                "error": error,
                "used_max_output_tokens": token,
            }
        )

        if i % SAVE_EVERY == 0:
            save_json(results, OUTPUT_FILE)

        if SLEEP_SECONDS:
            time.sleep(SLEEP_SECONDS)

    save_json(results, OUTPUT_FILE)
    print(f"Saved {len(results)} results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
