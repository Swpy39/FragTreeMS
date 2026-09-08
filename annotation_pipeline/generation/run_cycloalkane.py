# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


INPUT_FILE = "./difference_types_SMILES/cycloalkane.json"
OUTPUT_FILE = "./outputs/results.json"
PROMPT_FILE = "./prompts/cycloalkane_prompt.txt"
MODEL = "gpt-4.1"
TEMPERATURE = 0.0
API_KEY = os.environ.get("OPENAI_API_KEY")


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ============================================================
# 1. System prompt
#    仅增强环烷烃分子片段生成与离子对齐提示；其余接口与流程不变
# ============================================================

# ============================================================
# 2. Argument parser
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help=(
            "Input JSON file. Each item should contain id, smiles/SMILES, "
            "formula, mw, spectrum, spectrum_list, compound_class, etc."
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON file for model responses. The output is always a JSON list.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="xx",
        help="OpenAI API key. If not provided, OPENAI_API_KEY will be used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Recommended to use 0.0 for deterministic structured annotation.",
    )

    parser.add_argument(
        "--id_min",
        type=int,
        default=None,
        help="Only process records with id >= id_min.",
    )
    parser.add_argument(
        "--id_max",
        type=int,
        default=None,
        help="Only process records with id <= id_max.",
    )
    parser.add_argument(
        "--id_list",
        type=str,
        default=None,
        help="Optional comma-separated ids to process, e.g. 1,3,8. If set, it has priority over id_min/id_max.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index after id filtering.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N items after id filtering and start. Useful for testing.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=TEMPERATURE,
        help="Sleep seconds between requests.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save intermediate results every N processed samples.",
    )
    parser.add_argument(
        "--use_json_mode",
        action="store_true",
        help="Use text.format={'type':'json_object'} if the OpenAI model supports it.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="If output_file already exists, skip records whose id has already been processed.",
    )

    return parser.parse_args()


# ============================================================
# 3. JSON / record utilities
# ============================================================

def load_json(path: str) -> List[Dict[str, Any]]:
    """
    Load an input JSON file.

    Supported input formats:
    1. JSON list:
       [{...}, {...}]
    2. JSON object wrapper:
       {"data": [{...}]} / {"records": [...]} / {"items": [...]} / {"results": [...]}
    3. Single JSON object:
       {...}

    JSONL is intentionally not supported here, because this script is required to
    keep both input and output as JSON files.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict):
        records = None
        for key in ["data", "records", "items", "results"]:
            if isinstance(obj.get(key), list):
                records = obj[key]
                break
        if records is None:
            records = [obj]
    else:
        raise ValueError("Input JSON must be a list, a wrapper object, or a single object.")

    for i, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Input item {i} is not a JSON object.")

    return records


def save_json(obj: Any, path: str) -> None:
    """
    Save output as a JSON file.

    The output format is always a JSON list with indentation, never JSONL.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def record_id_as_int(record: Dict[str, Any]) -> Optional[int]:
    return safe_int(record.get("id"))


def parse_id_list(id_list: Optional[str]) -> Optional[set]:
    if not id_list:
        return None

    output = set()
    for part in str(id_list).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            output.add(int(part))
        except ValueError as e:
            raise ValueError(f"Invalid id in --id_list: {part}") from e

    return output


def filter_records_by_id(records: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = []
    id_set = parse_id_list(args.id_list)

    for record in records:
        rid = record_id_as_int(record)

        if id_set is not None:
            if rid not in id_set:
                continue
        else:
            if args.id_min is not None and rid is not None and rid < args.id_min:
                continue
            if args.id_max is not None and rid is not None and rid > args.id_max:
                continue

        selected.append(record)

    if args.limit is None:
        return selected[args.start:]
    return selected[args.start:args.start + args.limit]


def get_record_key(record: Dict[str, Any]) -> Any:
    """
    Priority key for skip_existing.
    Prefer id; otherwise use smiles + formula + mw.
    """
    if "id" in record:
        return record["id"]
    smiles = record.get("smiles", record.get("SMILES", ""))
    return f"{smiles}||{record.get('formula', '')}||{record.get('mw', '')}"


def normalize_spectrum_from_record(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize spectrum into a dict with string m/z keys.

    Priority:
    1. spectrum dict: {"83": 100, "84": 8}
    2. spectrum list: [[83, 100], {"mz": 84, "intensity": 8}]
    3. spectrum_list: [83, 84, ...], intensity set to None
    """
    spectrum = data.get("spectrum", {})

    if isinstance(spectrum, dict):
        return {str(k): v for k, v in spectrum.items()}

    normalized: Dict[str, Any] = {}

    if isinstance(spectrum, list):
        for item in spectrum:
            if isinstance(item, dict):
                mz = item.get("mz", item.get("m/z", item.get("mass", None)))
                intensity = item.get(
                    "intensity",
                    item.get("relative_intensity", item.get("rel_intensity", None)),
                )
                if mz is not None:
                    normalized[str(mz)] = intensity
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                normalized[str(item[0])] = item[1]

    if normalized:
        return normalized

    spectrum_list = data.get("spectrum_list", [])
    if isinstance(spectrum_list, list):
        for mz in spectrum_list:
            mz_int = safe_int(mz)
            if mz_int is not None:
                normalized[str(mz_int)] = None

    return normalized


def sort_spectrum_desc(spectrum: Dict[str, Any]) -> Tuple[Dict[str, Any], List[int]]:
    """
    Sort spectrum peaks by m/z from high to low.

    Returns:
        sorted_spectrum: dict with string m/z keys, sorted descending
        mz_list_desc: list of integer m/z values, sorted descending
    """
    pairs = []
    for mz, intensity in spectrum.items():
        mz_int = safe_int(mz)
        if mz_int is not None:
            pairs.append((mz_int, intensity))

    pairs.sort(key=lambda x: x[0], reverse=True)
    sorted_spectrum = {str(mz): intensity for mz, intensity in pairs}
    mz_list_desc = [mz for mz, _ in pairs]
    return sorted_spectrum, mz_list_desc


# ============================================================
# 4. Prompt / response utilities
# ============================================================

def build_user_prompt(data: Dict[str, Any]) -> str:
    """
    Build the user message for one cycloalkane / alicyclic hydrocarbon molecule.

    The model receives only molecular information here.
    All EI decision rules are placed in SYSTEM_PROMPT.
    """
    smiles = data.get("smiles", data.get("SMILES", ""))
    name = data.get("name", "")
    formula = data.get("formula", "")
    mw = data.get("mw", "")
    compound_class = data.get("compound_class", "")
    spectrum = normalize_spectrum_from_record(data)

    sorted_spectrum, mz_list_desc = sort_spectrum_desc(spectrum)

    molecular_info = {
        "source_smiles": smiles,
        "name": name,
        "formula": formula,
        "mw": mw,
        "compound_class": compound_class,
        "spectrum": sorted_spectrum,
        "spectrum_list": mz_list_desc,
    }

    return (
        "Please perform EI (70 eV) fragmentation decision annotation for the following cycloalkane molecule.\n"
        "You must output strict JSON only, following the schema specified in the system prompt.\n"
        "Apply the mandatory fragment-grounding audit independently to every m/z before producing the final JSON.\n"
        "For each structure-driven triplet, first compare the parent carbon count with the product-ion carbon count, "
        "then copy an exact, connected, mechanism-compatible fragment literally from source_smiles.\n"
        "When the product contains fewer carbons than the parent, never use the complete source_smiles as the fragment.\n"
        "Do not invent or canonicalize a chemically equivalent fragment that is not an exact source_smiles substring.\n"
        "Do not generate C/CC/CCC by carbon counting alone; such a fragment is valid only when it is a real retained "
        "source-derived skeleton for the corresponding C1/C2/C3 ion.\n"
        "Preserve relevant branching and ring-derived topology, and make the fragment, mechanism, core_motif, "
        "origin_type, decision fields, ion formula, and m/z mutually consistent.\n"
        "If no valid structure-driven fragment exists, use a supported ion-evolution precursor only when all arithmetic, "
        "spectrum-presence, same-carbon, higher-hydrogen, and intensity-direction constraints pass; otherwise mark invalid.\n"
        "Perform all checks internally and output no reasoning or explanatory text outside the required JSON.\n\n"
        "[Molecular information]\n"
        f"{json.dumps(molecular_info, ensure_ascii=False, indent=2)}"
    )


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robustly extract JSON object from model output.

    Handles:
    1. Pure JSON
    2. ```json ... ``` fenced JSON
    3. Extra text before/after JSON object
    """
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
        candidate = raw[start : end + 1]
        return json.loads(candidate)

    raise ValueError(f"Cannot parse JSON from response:\n{raw[:1000]}")


def validate_basic_output(parsed: Dict[str, Any], input_data: Dict[str, Any]) -> List[str]:
    """
    Lightweight validation for the returned JSON.
    This does not replace chemical validation; it only checks schema-level issues.
    """
    warnings = []

    required_top_keys = ["smiles", "formula", "mw", "mass_spectrum", "triples"]
    for key in required_top_keys:
        if key not in parsed:
            warnings.append(f"Missing top-level key: {key}")

    triples = parsed.get("triples", {})
    if not isinstance(triples, dict):
        warnings.append("Field 'triples' is not a dict.")
        return warnings

    spectrum = normalize_spectrum_from_record(input_data)
    input_mz_set = {str(safe_int(k)) for k in spectrum.keys() if safe_int(k) is not None}

    for mz, item in triples.items():
        mz_key = str(safe_int(mz)) if safe_int(mz) is not None else str(mz)
        if input_mz_set and mz_key not in input_mz_set:
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
            if "invalid_reason" not in item or item.get("invalid_reason") in [None, ""]:
                warnings.append(f"m/z {mz}: invalid_reason is missing.")
        else:
            warnings.append(f"m/z {mz}: unknown triplet_status: {status}")

    return warnings


# ============================================================
# 5. API caller
# ============================================================

def call_api_once(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_output_tokens: int,
    use_json_mode: bool = False,
) -> str:
    """
    Official OpenAI Responses API call.
    """
    kwargs = {
        "model": model,
        "input": messages,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    if use_json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}

    response = client.responses.create(**kwargs)

    if hasattr(response, "output_text") and response.output_text:
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


def call_api_token_sequence(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    use_json_mode: bool,
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    """
    Try output token limits in fixed order: 2048 -> 4096 -> 8192.

    Returns:
        parsed_json, raw_text, error_message
    """
    last_error = None
    raw_text = ""
    token_attempts = [4096, 8192, 16384]

    for attempt, current_tokens in enumerate(token_attempts, start=1):
        try:
            print(
                f"[Token Attempt {attempt}/{len(token_attempts)}] "
                f"max_output_tokens={current_tokens}"
            )

            raw_text = call_api_once(
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_output_tokens=current_tokens,
                use_json_mode=use_json_mode,
            )
            parsed = extract_json_from_text(raw_text)
            return parsed, raw_text, None

        except Exception as e:
            last_error = str(e)
            print(
                f"[Token Attempt {attempt}/{len(token_attempts)}] "
                f"API call or JSON extraction failed: {last_error}"
            )

            if use_json_mode and ("response_format" in last_error or "json" in last_error.lower()):
                try:
                    print(
                        f"[Fallback without json mode] "
                        f"max_output_tokens={current_tokens}"
                    )
                    raw_text = call_api_once(
                        client=client,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_output_tokens=current_tokens,
                        use_json_mode=False,
                    )
                    parsed = extract_json_from_text(raw_text)
                    return parsed, raw_text, None
                except Exception as e2:
                    last_error = str(e2)
                    print(f"[Fallback without json mode failed] {last_error}")

            if attempt < len(token_attempts):
                time.sleep(3)

    return None, raw_text, last_error


# ============================================================
# 6. Main process
# ============================================================

def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is missing. Please pass --api_key or set OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)

    records = load_json(args.input_file)

    selected_records = filter_records_by_id(records, args)

    results: List[Dict[str, Any]] = []
    processed_keys = set()

    if args.skip_existing and os.path.exists(args.output_file):
        try:
            results = load_json(args.output_file)
            for item in results:
                if isinstance(item, dict):
                    processed_keys.add(get_record_key(item))
            print(f"Loaded {len(results)} existing results. Will skip {len(processed_keys)} processed records.")
        except Exception as e:
            print(f"Failed to load existing output file. Start from empty results. Error: {e}")
            results = []
            processed_keys = set()

    total = len(selected_records)
    print(f"Total selected cycloalkane / alicyclic hydrocarbon samples: {total}")
    print(f"id_min={args.id_min}, id_max={args.id_max}, id_list={args.id_list}, start={args.start}, limit={args.limit}")
    print(f"Model: {args.model}")
    print("API: OpenAI official Responses API")
    print("Token attempts: 2048 -> 4096 -> 8192")
    print(f"Output file: {args.output_file}")

    for idx, data in enumerate(selected_records, start=1):
        data_key = get_record_key(data)

        if args.skip_existing and data_key in processed_keys:
            print(f"[{idx}/{total}] Skip existing record={data_key}")
            continue

        data_id = data.get("id", None)
        smiles = data.get("smiles", data.get("SMILES", ""))
        name = data.get("name", "")
        formula = data.get("formula", "")
        mw = data.get("mw", "")

        print("=" * 80)
        print(f"[{idx}/{total}] id={data_id}, name={name}, smiles={smiles}, formula={formula}, mw={mw}")

        user_prompt = build_user_prompt(data)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        parsed, raw_text, error = call_api_token_sequence(
            client=client,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            use_json_mode=args.use_json_mode,
        )

        if parsed is not None:
            warnings = validate_basic_output(parsed, data)
            result_item = {
                "id": data_id,
                "name": name,
                "smiles": smiles,
                "formula": formula,
                "mw": mw,
                "compound_class": data.get("compound_class", ""),
                "input_spectrum": normalize_spectrum_from_record(data),
                "model_output": parsed,
                "raw_response": raw_text,
                "parse_ok": True,
                "error": None,
                "warnings": warnings,
            }
            print(f"Success. warnings={len(warnings)}")
            if warnings:
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
                "input_spectrum": normalize_spectrum_from_record(data),
                "model_output": None,
                "raw_response": raw_text,
                "parse_ok": False,
                "error": error,
                "warnings": [],
            }
            print(f"Failed. error={error}")

        results.append(result_item)

        if args.save_every > 0 and len(results) % args.save_every == 0:
            save_json(results, args.output_file)
            print(f"Intermediate results saved to: {args.output_file}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    save_json(results, args.output_file)
    print("=" * 80)
    print(f"All done. Saved {len(results)} results to: {args.output_file}")


if __name__ == "__main__":
    main()


"""
Example:

python run_openai_cycloalkane.py --input_file ../difference_types_SMILES/cycloalkane_5_or_6_ring.json --output_file ./cycloalkane_outputs/openai_gpt41_test_results_558_558.json --model gpt-4.1 --temperature 0.0 --id_min 558 --id_max 558 --sleep 0

"""

