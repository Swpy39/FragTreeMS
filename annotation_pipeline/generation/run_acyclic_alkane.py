import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


INPUT_FILE = "./difference_types_SMILES/acyclic_alkane.json"
OUTPUT_FILE = "./outputs/openai_gpt41_test_results.json"
PROMPT_FILE = "./prompt.txt"

MODEL = "gpt-4.1"
TEMPERATURE = 0.0
TOKEN_ATTEMPTS = [2048, 4096, 8192]

START = 0
LIMIT = None
SLEEP_SECONDS = 0.0
SAVE_EVERY = 1
RETRY_SLEEP = 3.0
USE_JSON_MODE = False
SKIP_EXISTING = False

API_KEY = os.environ.get("OPENAI_API_KEY")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_prompt(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompt


def safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def sort_spectrum_desc(spectrum: Dict[str, Any]) -> Tuple[Dict[str, Any], List[int]]:
    pairs = []
    for mz, intensity in spectrum.items():
        mz_int = safe_int(mz)
        if mz_int is not None:
            pairs.append((mz_int, intensity))

    pairs.sort(key=lambda x: x[0], reverse=True)
    sorted_spectrum = {str(mz): intensity for mz, intensity in pairs}
    mz_list_desc = [mz for mz, _ in pairs]
    return sorted_spectrum, mz_list_desc


def build_user_prompt(data: Dict[str, Any]) -> str:
    spectrum = data.get("spectrum", {})
    sorted_spectrum, mz_list_desc = sort_spectrum_desc(spectrum)

    molecular_info = {
        "source_smiles": data.get("smiles", ""),
        "name": data.get("name", ""),
        "formula": data.get("formula", ""),
        "mw": data.get("mw", ""),
        "compound_class": data.get("compound_class", ""),
        "spectrum": sorted_spectrum,
        "spectrum_list": mz_list_desc,
    }

    return (
        "Please perform EI (70 eV) fragmentation decision annotation for the following molecule.\n"
        "You must output strict JSON only, following the schema specified in the system prompt.\n\n"
        "[Molecular information]\n"
        + json.dumps(molecular_info, ensure_ascii=False, indent=2)
    )


def extract_json_from_text(text: str) -> Dict[str, Any]:
    if text is None:
        raise ValueError("Empty response content.")

    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(raw[start:end + 1])

    raise ValueError(f"Cannot parse JSON from response:\n{raw[:1000]}")


def validate_basic_output(parsed: Dict[str, Any], input_data: Dict[str, Any]) -> List[str]:
    warnings = []
    required_top_keys = ["smiles", "formula", "mw", "mass_spectrum", "triples"]

    for key in required_top_keys:
        if key not in parsed:
            warnings.append(f"Missing top-level key: {key}")

    triples = parsed.get("triples", {})
    if not isinstance(triples, dict):
        warnings.append("Field 'triples' is not a dict.")
        return warnings

    input_mz_set = {str(k) for k in input_data.get("spectrum", {}).keys()}

    for mz, item in triples.items():
        if str(mz) not in input_mz_set:
            warnings.append(f"Output contains m/z not found in input spectrum: {mz}")

        if not isinstance(item, dict):
            warnings.append(f"m/z {mz}: item is not a dict.")
            continue

        for key in ["decision", "origin_type", "core_motif", "triplet_status", "triplet"]:
            if key not in item:
                warnings.append(f"m/z {mz}: missing key '{key}'")

        status = item.get("triplet_status")
        triplet = item.get("triplet")

        if status == "ok":
            if not triplet:
                warnings.append(f"m/z {mz}: triplet_status is ok but triplet is empty.")
        elif status == "invalid":
            if triplet != []:
                warnings.append(f"m/z {mz}: triplet_status is invalid but triplet is not [].")
            if item.get("invalid_reason") in [None, ""]:
                warnings.append(f"m/z {mz}: invalid_reason is missing.")
        else:
            warnings.append(f"m/z {mz}: unknown triplet_status: {status}")

    return warnings


def call_api_once(
    client: OpenAI,
    messages: List[Dict[str, str]],
    max_tokens: int,
    use_json_mode: bool,
) -> str:
    kwargs = {
        "model": MODEL,
        "input": messages,
        "temperature": TEMPERATURE,
        "max_output_tokens": max_tokens,
    }

    if use_json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}

    response = client.responses.create(**kwargs)

    if getattr(response, "output_text", None):
        return response.output_text

    parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)

    if parts:
        return "\n".join(parts)

    return str(response)


def call_api_with_retry(
    client: OpenAI,
    messages: List[Dict[str, str]],
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    last_error = None
    raw_text = ""

    for attempt, current_max_tokens in enumerate(TOKEN_ATTEMPTS, start=1):
        try:
            print(f"[Attempt {attempt}/{len(TOKEN_ATTEMPTS)}] max_tokens={current_max_tokens}")
            raw_text = call_api_once(client, messages, current_max_tokens, USE_JSON_MODE)
            return extract_json_from_text(raw_text), raw_text, None
        except Exception as e:
            last_error = str(e)
            print(f"[Attempt {attempt}/{len(TOKEN_ATTEMPTS)}] failed: {last_error}")

            if USE_JSON_MODE and ("response_format" in last_error or "json" in last_error.lower()):
                try:
                    print(f"[Fallback without JSON mode] max_tokens={current_max_tokens}")
                    raw_text = call_api_once(client, messages, current_max_tokens, False)
                    return extract_json_from_text(raw_text), raw_text, None
                except Exception as fallback_error:
                    last_error = str(fallback_error)
                    print(f"[Fallback failed] {last_error}")

            if attempt < len(TOKEN_ATTEMPTS):
                time.sleep(RETRY_SLEEP)

    return None, raw_text, last_error


def main() -> None:
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")

    system_prompt = load_prompt(PROMPT_FILE)
    client = OpenAI(api_key=API_KEY)

    datas = load_json(INPUT_FILE)
    if not isinstance(datas, list):
        raise ValueError("Input JSON must be a list of molecular records.")

    if LIMIT is None:
        selected_datas = datas[START:]
    else:
        selected_datas = datas[START:START + LIMIT]

    results: List[Dict[str, Any]] = []
    processed_ids = set()

    if SKIP_EXISTING and os.path.exists(OUTPUT_FILE):
        try:
            results = load_json(OUTPUT_FILE)
            processed_ids = {item["id"] for item in results if "id" in item}
            print(f"Loaded {len(results)} existing results; {len(processed_ids)} ids will be skipped.")
        except Exception as e:
            print(f"Failed to load existing output file: {e}")
            results = []
            processed_ids = set()

    total = len(selected_datas)
    print(f"Total selected samples: {total}")
    print(f"Model: {MODEL}")
    print(f"Prompt file: {PROMPT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")

    for idx, data in enumerate(selected_datas, start=1):
        data_id = data.get("id")

        if SKIP_EXISTING and data_id in processed_ids:
            print(f"[{idx}/{total}] skip id={data_id}")
            continue

        name = data.get("name", "")
        smiles = data.get("smiles", "")
        formula = data.get("formula", "")
        mw = data.get("mw", "")

        print("=" * 80)
        print(f"[{idx}/{total}] id={data_id}, name={name}, smiles={smiles}, formula={formula}, mw={mw}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(data)},
        ]

        parsed, raw_text, error = call_api_with_retry(client, messages)

        if parsed is not None:
            warnings = validate_basic_output(parsed, data)
            result_item = {
                "id": data_id,
                "name": name,
                "smiles": smiles,
                "formula": formula,
                "mw": mw,
                "compound_class": data.get("compound_class", ""),
                "input_spectrum": data.get("spectrum", {}),
                "model_output": parsed,
                "raw_response": raw_text,
                "parse_ok": True,
                "error": None,
                "warnings": warnings,
            }
            print(f"Success. warnings={len(warnings)}")
            for warning in warnings[:10]:
                print(f"  [Warning] {warning}")
        else:
            result_item = {
                "id": data_id,
                "name": name,
                "smiles": smiles,
                "formula": formula,
                "mw": mw,
                "compound_class": data.get("compound_class", ""),
                "input_spectrum": data.get("spectrum", {}),
                "model_output": None,
                "raw_response": raw_text,
                "parse_ok": False,
                "error": error,
                "warnings": [],
            }
            print(f"Failed. error={error}")

        results.append(result_item)

        if SAVE_EVERY > 0 and len(results) % SAVE_EVERY == 0:
            save_json(results, OUTPUT_FILE)
            print(f"Saved intermediate results to: {OUTPUT_FILE}")

        if SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)

    save_json(results, OUTPUT_FILE)
    print("=" * 80)
    print(f"All done. Saved {len(results)} results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
