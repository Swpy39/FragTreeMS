import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

INPUT_FILE = "./difference_types_SMILES/alkene.json"
OUTPUT_FILE = "../data/alkene_data/openai_gpt41_test_results.json"
PROMPT_FILE = "./prompts/alkene_prompt.txt"

MODEL = "gpt-4.1"
TEMPERATURE = 0.0
MAX_TOKENS = 30000
TOKEN_ATTEMPTS = [30000, 50000, 80000]
API_KEY = os.environ.get("OPENAI_API_KEY")

START = 0
LIMIT = None
ID_MIN = 1
ID_MAX = 666
SLEEP_SECONDS = 0.0
SAVE_EVERY = 1
RETRY = 3
USE_JSON_MODE = False

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompt

def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

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

def build_user_prompt(data: Dict[str, Any], target_mz: int) -> str:
    """
    Build a single-ion prompt.

    The complete spectrum is still supplied as context, but the model is asked to
    return exactly one entry in ``triples``: the currently missing target m/z.
    Existing ions are never sent for regeneration and are never overwritten.
    """
    smiles = data.get("smiles", "")
    name = data.get("name", "")
    formula = data.get("formula", "")
    mw = data.get("mw", "")
    compound_class = data.get("compound_class", "")
    spectrum = data.get("spectrum", {})

    sorted_spectrum, mz_list_desc = sort_spectrum_desc(spectrum)
    target_intensity = sorted_spectrum.get(str(int(target_mz)))

    molecular_info = {
        "source_smiles": smiles,
        "name": name,
        "formula": formula,
        "mw": mw,
        "compound_class": compound_class,
        "spectrum": sorted_spectrum,
        "spectrum_list": mz_list_desc,
        "target_mz": int(target_mz),
        "target_intensity": target_intensity,
    }

    return (
        "Please perform EI (70 eV) fragmentation decision annotation for exactly "
        f"one target ion: m/z {int(target_mz)}.\n"
        "Use the complete spectrum as context, but do not regenerate any other ion.\n"
        "Return the same strict top-level JSON schema required by the system prompt. "
        "The triples object must contain exactly one entry, whose key is the target "
        f"m/z string '{int(target_mz)}'.\n"
        "If a compliant triplet can be produced, triplet_status must be 'ok' and "
        "triplet must be non-empty. Output strict JSON only.\n\n"
        "[Molecular information and single target]\n"
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
    for k in required_top_keys:
        if k not in parsed:
            warnings.append(f"Missing top-level key: {k}")

    triples = parsed.get("triples", {})
    if not isinstance(triples, dict):
        warnings.append("Field 'triples' is not a dict.")
        return warnings

    spectrum = input_data.get("spectrum", {})
    input_mz_set = {str(k) for k in spectrum.keys()}

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
            if "invalid_reason" not in item or item.get("invalid_reason") in [None, ""]:
                warnings.append(f"m/z {mz}: invalid_reason is missing.")
        else:
            warnings.append(f"m/z {mz}: unknown triplet_status: {status}")

    return warnings

ALLOWED_MECHANISMS = {
    "Molecular ion",
    "Isotopic peak",
    "Alpha-cleavage",
    "Sigma-bond cleavage",
    "Benzylic cleavage",
    "Allylic cleavage",
    "McLafferty rearrangement",
    "Neutral loss",
    "Retro-Diels–Alder fragmentation",
    "Hydrogen transfer",
    "Radical-ion rearrangement",
    "Dehydrogenation / Sequential dehydrogenation",
    "Ring cleavage / Ring rearrangement",
}

def canonical_id(value: Any) -> str:
    """Normalize int/string ids so 12, '12', and 12.0 share one key."""
    try:
        return str(int(float(value)))
    except Exception:
        return str(value).strip()

def parse_mz_value(value: Any) -> Optional[int]:
    """Parse an m/z integer from a numeric key or an explicit ``m/z N`` string."""
    if value is None:
        return None
    try:
        return int(float(value))
    except Exception:
        text = str(value)
        match = re.search(r"m/z\s*([0-9]+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        stripped = text.strip()
        return int(stripped) if re.fullmatch(r"[0-9]+", stripped) else None

def normalize_triplet_entries(value: Any) -> List[List[str]]:
    """Accept either one direct triplet or a list of triplets."""
    if isinstance(value, list) and len(value) == 3 and all(
        isinstance(x, (str, int, float)) for x in value
    ):
        return [[str(x) for x in value]]

    output: List[List[str]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                output.append([str(item[0]), str(item[1]), str(item[2])])
    return output

def get_model_output(result_item: Any) -> Dict[str, Any]:
    if not isinstance(result_item, dict):
        return {}
    model_output = result_item.get("model_output")
    return model_output if isinstance(model_output, dict) else {}

def get_triples_dict(result_item: Any) -> Dict[str, Any]:
    model_output = get_model_output(result_item)
    triples = model_output.get("triples")
    return triples if isinstance(triples, dict) else {}

def find_mz_block(triples: Dict[str, Any], target_mz: int) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    for key, block in triples.items():
        if parse_mz_value(key) == int(target_mz) and isinstance(block, dict):
            return str(key), block
    return None, None

def block_has_valid_triplet(block: Any, target_mz: int) -> bool:
    """
    An ion counts as completed only when at least one valid three-element triplet
    exists and its product ion points to the requested target m/z.

    Empty/invalid blocks are deliberately treated as missing and will be generated.
    """
    if not isinstance(block, dict):
        return False
    entries = normalize_triplet_entries(block.get("triplet"))
    if not entries:
        return False

    for triplet in entries:
        mechanism = str(triplet[1]).strip()
        product_mz = parse_mz_value(triplet[2])
        if mechanism in ALLOWED_MECHANISMS and product_mz == int(target_mz):
            return True
    return False

def extract_target_block(parsed: Dict[str, Any], target_mz: int) -> Optional[Dict[str, Any]]:
    triples = parsed.get("triples") if isinstance(parsed, dict) else None
    if not isinstance(triples, dict):
        return None
    _, block = find_mz_block(triples, target_mz)
    return block

def validate_target_output(
    parsed: Dict[str, Any],
    input_data: Dict[str, Any],
    target_mz: int,
) -> List[str]:
    """Validate only the newly generated target ion; existing ions are untouched."""
    warnings = validate_basic_output(parsed, input_data)
    block = extract_target_block(parsed, target_mz)
    if block is None:
        warnings.append(f"Target m/z {target_mz}: missing from returned triples.")
        return warnings
    if not block_has_valid_triplet(block, target_mz):
        warnings.append(
            f"Target m/z {target_mz}: no valid non-empty triplet matching the target ion."
        )
    return warnings

def insert_triple_block_descending(
    triples: Dict[str, Any],
    target_mz: int,
    block: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Insert one new ion by descending numeric m/z while preserving all existing
    block objects unchanged. Any duplicate key for the same numeric m/z is removed.
    """
    numeric_items: List[Tuple[int, str, Any]] = []
    nonnumeric_items: List[Tuple[str, Any]] = []

    for key, value in triples.items():
        mz = parse_mz_value(key)
        if mz is None:
            nonnumeric_items.append((str(key), value))
        elif mz != int(target_mz):
            numeric_items.append((mz, str(key), value))

    numeric_items.append((int(target_mz), str(int(target_mz)), block))
    numeric_items.sort(key=lambda item: item[0], reverse=True)

    merged: Dict[str, Any] = {}
    for _, key, value in numeric_items:
        merged[key] = value
    for key, value in nonnumeric_items:
        if key not in merged:
            merged[key] = value
    return merged

def make_base_result(data: Dict[str, Any]) -> Dict[str, Any]:
    sorted_spectrum, mz_list_desc = sort_spectrum_desc(data.get("spectrum", {}))
    return {
        "id": data.get("id"),
        "name": data.get("name", ""),
        "smiles": data.get("smiles", ""),
        "formula": data.get("formula", ""),
        "mw": data.get("mw", ""),
        "compound_class": data.get("compound_class", ""),
        "input_spectrum": data.get("spectrum", {}),
        "model_output": {
            "smiles": data.get("smiles", ""),
            "formula": data.get("formula", ""),
            "mw": data.get("mw", ""),
            "mass_spectrum": mz_list_desc,
            "triples": {},
        },
        "raw_response": "",
        "parse_ok": True,
        "error": None,
        "warnings": [],
    }

def ensure_result_model_output(result_item: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    model_output = result_item.get("model_output")
    if not isinstance(model_output, dict):
        model_output = {}
        result_item["model_output"] = model_output

    _, mz_list_desc = sort_spectrum_desc(data.get("spectrum", {}))
    model_output.setdefault("smiles", data.get("smiles", ""))
    model_output.setdefault("formula", data.get("formula", ""))
    model_output.setdefault("mw", data.get("mw", ""))
    model_output["mass_spectrum"] = mz_list_desc
    if not isinstance(model_output.get("triples"), dict):
        model_output["triples"] = {}
    return model_output

def build_result_index(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Map each molecule id to its first existing result position."""
    index: Dict[str, int] = {}
    for position, item in enumerate(results):
        if isinstance(item, dict) and "id" in item:
            index.setdefault(canonical_id(item.get("id")), position)
    return index

def insert_result_by_id(results: List[Dict[str, Any]], result_item: Dict[str, Any]) -> int:
    """Insert a previously absent molecule before the first larger numeric id."""
    new_id = parse_mz_value(result_item.get("id"))
    if new_id is None:
        results.append(result_item)
        return len(results) - 1

    for position, existing in enumerate(results):
        existing_id = parse_mz_value(existing.get("id")) if isinstance(existing, dict) else None
        if existing_id is not None and existing_id > new_id:
            results.insert(position, result_item)
            return position
    results.append(result_item)
    return len(results) - 1

def update_result_with_generated_ion(
    result_item: Dict[str, Any],
    data: Dict[str, Any],
    parsed: Dict[str, Any],
    target_mz: int,
    raw_text: str,
    warnings: List[str],
) -> None:
    """Merge only the requested ion into the molecule result."""
    block = extract_target_block(parsed, target_mz)
    if block is None:
        raise ValueError(f"Returned JSON does not contain target m/z {target_mz}.")

    model_output = ensure_result_model_output(result_item, data)
    model_output["triples"] = insert_triple_block_descending(
        model_output.get("triples", {}), target_mz, block
    )

    result_item.setdefault("raw_response_by_mz", {})[str(target_mz)] = raw_text
    result_item.setdefault("warnings_by_mz", {})[str(target_mz)] = warnings
    result_item.setdefault("parse_ok_by_mz", {})[str(target_mz)] = True
    result_item.setdefault("error_by_mz", {})[str(target_mz)] = None
    result_item["parse_ok"] = True
    result_item["error"] = None

def completed_mz_set(result_item: Dict[str, Any], input_mz: List[int]) -> set:
    triples = get_triples_dict(result_item)
    return {
        mz for mz in input_mz
        if block_has_valid_triplet(find_mz_block(triples, mz)[1], mz)
    }

def call_api_once(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    use_json_mode: bool = False,
) -> str:
    """
    Official OpenAI API call using the Responses API.

    The command-line argument is still named --max_tokens to keep your original
    running commands unchanged, but Responses API uses max_output_tokens.
    """
    kwargs = {
        "model": model,
        "input": messages,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
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

def call_api_with_retry(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    retry: int,
    use_json_mode: bool,
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    """
    Returns:
        parsed_json, raw_text, error_message

    Retry strategy:
    - For each SMILES, try max_tokens in the order: 2048 -> 4096 -> 8192.
    - If one attempt fails because of API error or JSON parsing error, wait 3 seconds
      and then retry with the next larger max_tokens.
    - If 8192 still fails, return parsed_json=None. The main process will mark
      the current SMILES as "parse_ok": false and continue to the next SMILES.
    """
    last_error = None
    raw_text = ""

    token_attempts = TOKEN_ATTEMPTS

    for attempt, current_max_tokens in enumerate(token_attempts, start=1):
        try:
            print(
                f"[Token Attempt {attempt}/{len(token_attempts)}] "
                f"max_tokens={current_max_tokens}"
            )

            raw_text = call_api_once(
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=current_max_tokens,
                use_json_mode=use_json_mode,
            )
            parsed = extract_json_from_text(raw_text)
            return parsed, raw_text, None

        except Exception as e:
            last_error = str(e)
            print(
                f"[Token Attempt {attempt}/{len(token_attempts)}] "
                f"API or JSON parsing failed with max_tokens={current_max_tokens}: {last_error}"
            )

            if use_json_mode and ("response_format" in last_error or "json" in last_error.lower()):
                try:
                    print(
                        f"[Fallback without json mode] "
                        f"max_tokens={current_max_tokens}"
                    )
                    raw_text = call_api_once(
                        client=client,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=current_max_tokens,
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

def main() -> None:
    api_key = API_KEY
    system_prompt = load_prompt(PROMPT_FILE)
    if not api_key:
        raise ValueError("API key is missing. Please pass --api_key or set OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)

    datas = load_json(INPUT_FILE)
    if not isinstance(datas, list):
        raise ValueError("Input JSON must be a list of molecular records.")

    if ID_MIN is not None and ID_MAX is not None and ID_MIN > ID_MAX:
        raise ValueError("--id_min must be less than or equal to --id_max.")

    id_filtered_datas: List[Dict[str, Any]] = []
    for data in datas:
        if not isinstance(data, dict):
            continue
        data_id_int = safe_int(data.get("id"))
        if ID_MIN is not None and (data_id_int is None or data_id_int < ID_MIN):
            continue
        if ID_MAX is not None and (data_id_int is None or data_id_int > ID_MAX):
            continue
        id_filtered_datas.append(data)

    if LIMIT is None:
        selected_datas = id_filtered_datas[START:]
    else:
        selected_datas = id_filtered_datas[START:START + LIMIT]

    results: List[Dict[str, Any]] = []
    if os.path.exists(OUTPUT_FILE):
        try:
            loaded = load_json(OUTPUT_FILE)
            if isinstance(loaded, list):
                results = loaded
                print(f"Loaded {len(results)} existing molecule results for per-ion resume.")
            else:
                print("Existing output is not a list; starting from empty results.")
        except Exception as exc:
            print(f"Failed to load existing output; starting empty. Error: {exc}")

    result_index = build_result_index(results)

    total_input_ions = 0
    total_existing_ions = 0
    total_missing_ions = 0
    per_molecule_plan: Dict[str, List[int]] = {}

    for data in selected_datas:
        _, mz_list_desc = sort_spectrum_desc(data.get("spectrum", {}))
        total_input_ions += len(mz_list_desc)
        key = canonical_id(data.get("id"))
        position = result_index.get(key)
        existing_result = results[position] if position is not None else {}
        completed = completed_mz_set(existing_result, mz_list_desc)
        missing = [mz for mz in mz_list_desc if mz not in completed]
        per_molecule_plan[key] = missing
        total_existing_ions += len(completed)
        total_missing_ions += len(missing)

    print("=" * 88)
    print(f"Selected molecules:             {len(selected_datas)}")
    print(f"Input ions in selected range:   {total_input_ions}")
    print(f"Existing valid triplets:        {total_existing_ions}")
    print(f"Missing ions to generate:       {total_missing_ions}")
    print("Resume rule: molecule id + target m/z + non-empty valid triplet")
    print("Generation mode: one API request per missing m/z; existing ion blocks are untouched")
    print(f"Model:                          {MODEL}")
    print(f"Output file:                    {OUTPUT_FILE}")
    print("=" * 88)

    generated_since_save = 0
    generated_total = 0

    for molecule_no, data in enumerate(selected_datas, start=1):
        data_id = data.get("id")
        key = canonical_id(data_id)
        smiles = data.get("smiles", "")
        name = data.get("name", "")
        formula = data.get("formula", "")
        mw = data.get("mw", "")
        _, mz_list_desc = sort_spectrum_desc(data.get("spectrum", {}))

        position = result_index.get(key)
        if position is None:
            result_item = make_base_result(data)
            position = insert_result_by_id(results, result_item)
            result_index = build_result_index(results)
            position = result_index[key]
        result_item = results[position]
        ensure_result_model_output(result_item, data)

        triples = get_triples_dict(result_item)
        missing_mz = [
            mz for mz in mz_list_desc
            if not block_has_valid_triplet(find_mz_block(triples, mz)[1], mz)
        ]

        print("=" * 88)
        print(
            f"[Molecule {molecule_no}/{len(selected_datas)}] id={data_id}, name={name}, "
            f"smiles={smiles}, formula={formula}, mw={mw}, "
            f"ions={len(mz_list_desc)}, missing={len(missing_mz)}"
        )

        if not missing_mz:
            print("[Skip molecule] Every spectrum ion already has a valid triplet.")
            continue

        for target_mz in mz_list_desc:
            triples = get_triples_dict(result_item)
            _, existing_block = find_mz_block(triples, target_mz)
            if block_has_valid_triplet(existing_block, target_mz):
                print(f"[Skip existing triplet] id={data_id}, m/z={target_mz}")
                continue

            generated_total += 1
            print("-" * 88)
            print(
                f"[Generate missing ion {generated_total}/{total_missing_ions}] "
                f"id={data_id}, m/z={target_mz}"
            )

            user_prompt = build_user_prompt(data, target_mz)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            parsed, raw_text, error = call_api_with_retry(
                client=client,
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                retry=RETRY,
                use_json_mode=USE_JSON_MODE,
            )

            if parsed is None:
                result_item.setdefault("raw_response_by_mz", {})[str(target_mz)] = raw_text
                result_item.setdefault("parse_ok_by_mz", {})[str(target_mz)] = False
                result_item.setdefault("error_by_mz", {})[str(target_mz)] = error
                print(f"[Failed] id={data_id}, m/z={target_mz}, error={error}")
                continue

            warnings = validate_target_output(parsed, data, target_mz)
            target_block = extract_target_block(parsed, target_mz)
            if target_block is None or not block_has_valid_triplet(target_block, target_mz):
                result_item.setdefault("raw_response_by_mz", {})[str(target_mz)] = raw_text
                result_item.setdefault("warnings_by_mz", {})[str(target_mz)] = warnings
                result_item.setdefault("parse_ok_by_mz", {})[str(target_mz)] = False
                result_item.setdefault("error_by_mz", {})[str(target_mz)] = (
                    "Returned target block has no valid non-empty triplet."
                )
                print(
                    f"[Rejected output] id={data_id}, m/z={target_mz}: "
                    "no valid target triplet was returned; it remains missing."
                )
                for warning in warnings[:10]:
                    print(f"  [Warning] {warning}")
                continue

            update_result_with_generated_ion(
                result_item=result_item,
                data=data,
                parsed=parsed,
                target_mz=target_mz,
                raw_text=raw_text,
                warnings=warnings,
            )
            generated_since_save += 1

            print(
                f"[Inserted] id={data_id}, m/z={target_mz}; "
                f"warnings={len(warnings)}; existing later ions were not regenerated."
            )

            if SAVE_EVERY > 0 and generated_since_save >= SAVE_EVERY:
                save_json(results, OUTPUT_FILE)
                print(f"[Saved] {OUTPUT_FILE}")
                generated_since_save = 0

            if SLEEP_SECONDS > 0:
                time.sleep(SLEEP_SECONDS)

    save_json(results, OUTPUT_FILE)

    remaining = 0
    completed = 0
    result_index = build_result_index(results)
    for data in selected_datas:
        _, mz_list_desc = sort_spectrum_desc(data.get("spectrum", {}))
        pos = result_index.get(canonical_id(data.get("id")))
        item = results[pos] if pos is not None else {}
        done = completed_mz_set(item, mz_list_desc)
        completed += len(done)
        remaining += len(mz_list_desc) - len(done)

    print("=" * 88)
    print("All done.")
    print(f"Valid triplets now present: {completed}")
    print(f"Ions still missing:         {remaining}")
    print(f"Saved results to:           {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
