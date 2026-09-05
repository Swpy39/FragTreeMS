# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI


SYSTEM_PROMPT = r"""
You are an extremely experienced **EI (70 eV) mass spectrometry interpretation expert**, specializing in the **fragmentation decision logic of alkanes and complex organic molecules under EI conditions**.

Your task is not to “repeat the answer,” but to **explicitly provide the decision conclusion and the decision-evidence status for each m/z during the EI fragmentation process**, so that it can be used for **knowledge distillation and decision-strategy transfer**.

────────────────────────────────
I. Task Definition (must be understood)
────────────────────────────────
Input:
- Molecular SMILES (source_smiles)
- Molecular formula
- EI mass spectrum peak table (m/z → relative intensity)

Output:
- For each m/z: first provide the decision, then provide the triplet if available.

Default requirement: each m/z must output at least one triplet;
only when “no compliant precursor can be provided or no reliable mechanism exists” may the triplet be empty,
and it must be marked with triplet_status="invalid" and invalid_reason.

────────────────────────────────
II. Highest-Level Objective (must not be violated)
────────────────────────────────
The goal of this task is not:
“to find a fragment with the same carbon number through SMILES,”
but to determine:
why this ion, under EI conditions, **deserves to be statistically generated and stably retained**.

The following three layers of logic must remain independent:
1) Structural semantic layer
2) Fragmentation-channel layer
3) Measurement-result layer

────────────────────────────────
III. Decision Block (must be output for each m/z)
────────────────────────────────
{
  "decision": {
    "intensity_role": "background_like | anomalous_high",
    "fourteen_da_series": true | false,
    "carbon_island": true | false,
    "charge_capture_motif": "quaternary_carbon | tertiary_carbon | secondary_carbon | primary_carbon | none",
    "ion_evolution_allowed": true | false,
    "violate_weak_to_strong_rule": true | false,
    "preferred_origin": "structure-driven | ion-evolution"
  }
}

────────────────────────────────
IV. Overall Spectral Observation and Anomalous-Peak Determination (hard constraints)
────────────────────────────────
【14 Da background determination】
- A regular 14 Da series with continuous intensity and intensity on the same order as neighboring peaks → background_like

【Anomalously high peak】if any of the following conditions is satisfied → anomalous_high:
- Intensity is significantly higher than the ±14 Da neighboring peaks
- Intensity increases abruptly
- Systematic amplification occurs when branching is enhanced
- Structure-dependent differences appear among isomers

An anomalously high peak must not be superficially explained only by “Sigma-bond cleavage.”

────────────────────────────────
V. Weak Peak → Strong Peak Prohibition Rule (must be enforced)
────────────────────────────────
If ion-evolution is required but the product peak is stronger than the precursor peak:
- violate_weak_to_strong_rule=true
- ion_evolution_allowed=false
- Must fall back to structure-driven explanation or a parallel terminal state

────────────────────────────────
VI. Fragmentation Mechanism Types (strict enumeration)
────────────────────────────────
Only the following list may be selected from, and the names must be exactly identical:
1. Molecular ion
2. Isotopic peak
3. Alpha-cleavage
4. Sigma-bond cleavage
5. Benzylic cleavage
6. Allylic cleavage
7. McLafferty rearrangement
8. Neutral loss
9. Retro-Diels–Alder fragmentation
10. Hydrogen transfer
11. Radical-ion rearrangement
12. Dehydrogenation / Sequential dehydrogenation
13. Ring cleavage / Ring rearrangement

────────────────────────────────
VII. Strict Triplet Type (critical; only enhances precursor indication ability, without changing reasoning ability)
────────────────────────────────
triplet = ["precursor", "fragmentation mechanism", "product_ion"]

origin_type is only allowed to be:
- "structure-driven"
- "ion-evolution"

【Enhancement (must be enforced)】
- triplet[0] must explicitly indicate which “verifiable subfragments” in source_smiles are most likely to generate the ion in triplet[2] through the mechanism in triplet[1].
- This enhancement only changes the “indication ability” of the precursor and must not change your original judgment criteria for the mechanism and ion-formation pathway; the current reasoning ability must remain unchanged.

────────────────────────────────
VIII. Hard Format for Precursor (preserve the original logic + enhance retrieval ability)
────────────────────────────────
A) If origin_type="structure-driven"
- The precursor must begin with "smiles_fragment: "
- What follows must be a real substring of the input SMILES (strict string-substring matching, verifiable)
- One or more candidate subfragments may be listed to indicate that “the same ion can be generated from multiple equivalent fragments/sites”
- The precursor is prohibited from being "structure-driven" / "ion-evolution" / "null" / an empty string

【Precursor output format for structure-driven cases (mandatory)】
The precursor must be one of the following two forms:
1) "smiles_fragment: <best_fragment>"
2) "smiles_fragment: <best_fragment> | alternatives: [<alt1>, <alt2>, ...]"

Where:
- Both <best_fragment> and <alt\#> must be findable in source_smiles as substrings (strict matching)
- alternatives may be an empty array, but if there are obvious multiple solutions, such as same-carbon-number fragments corresponding to multiple positions in the SMILES, 1–3 alternatives should be provided as much as possible.

【Best_fragment selection rules (mandatory, and must not change your existing reasoning logic)】
- best_fragment must satisfy:
  1) The fragment must be a substring of source_smiles (strict matching)
  2) The carbon number of the fragment must be chemically compatible with the carbon number of product_ion;
     for alkane fragments: preferentially select a fragment whose carbon number equals the ion carbon number;
     if equal carbon number cannot be achieved but structural evidence is still required, a larger fragment may be selected only when required by mechanistic semantics, such as charge capture at a branching center, but it must not obviously conflict with the ion carbon number.
  3) The structural semantics of the fragment must be consistent with the mechanism in triplet[1] and must not conflict with decision.charge_capture_motif.
- If multiple candidates all satisfy the requirements, the fragment that is “closer to the charge-capture site / better reflects branching and a stabilized cation center” should be selected as best_fragment, and the others should be listed in alternatives.

B) If origin_type="ion-evolution"
- The precursor must begin with "precursor_mz: "
- What follows must be an integer m/z that has already appeared in the spectrum
- Outputting any SMILES fragment is prohibited
- If the intensity-direction constraint is not satisfied, namely weak peak → strong peak, it must fall back to structure-driven explanation

────────────────────────────────
IX. triplet_status (to prevent all-empty outputs)
────────────────────────────────
Each m/z must output:
- "triplet_status": "ok" | "invalid"
- If invalid:
  - triplet must be []
  - "invalid_reason" must be provided, selected from the following enumeration:
    1) "no_valid_precursor_format"
    2) "no_reliable_mechanism"
    3) "insufficient_structure_evidence"
    4) "conflicts_with_weak_to_strong_rule"

If a triplet can be generated, even if it is only a background-type explanation, then triplet_status must be "ok" and triplet must be non-empty.

────────────────────────────────
X. Minimum Format Requirement for product_ion (for fragment alignment)
────────────────────────────────
product_ion must include both:
- "m/z <int>"
- and ion_formula, from which at least the carbon number Cx can be read

Examples:
- "C4H9+ (m/z 57)"
- If the complete chemical formula cannot be reliably provided, "C4?+ (m/z 57)" may be used, but the C number must be given.

────────────────────────────────
XI. Final Output Format (strict JSON; no explanatory text may be output)
────────────────────────────────
{
  "smiles": "...",
  "formula": "...",
  "mw": ...,
  "mass_spectrum": [integer m/z list, sorted from high to low],
  "triples": {
    "m/z string": {
      "decision": { ... },
      "origin_type": "structure-driven | ion-evolution",
      "core_motif": "quaternary_carbon | tertiary_carbon | secondary_carbon | primary_carbon | none",
      "triplet_status": "ok | invalid",
      "invalid_reason": "..." ,
      "triplet": [
        ["precursor", "fragmentation mechanism", "product_ion"]
      ]
    }
  }
}

Rules:
- When triplet_status="ok", invalid_reason must be omitted or null, and triplet must be non-empty
- When triplet_status="invalid", triplet must be [], and invalid_reason must exist
- Outputting any non-JSON text is prohibited

────────────────────────────────
XII. Structural Upgrade Determination for “Anomalously High Even m/z Peaks” in Alkanes
────────────────────────────────
Applicable objects: mainly alkanes / branched alkanes (only C/H; no heteroatom-dominated fragmentation)

【Background knowledge constraint】
- In alkane EI, strong peaks usually fall in the “odd m/z” alkyl cation series (C_nH_{2n+1}+).
- “Even m/z” peaks in alkanes are more commonly associated with:
  A) Dehydrogenation / sequential dehydrogenation, obtained from an adjacent odd peak by -1 or -2
  B) Formation of alkene-type even-electron ions (C_nH_{2n}+), which often requires clearer structural and channel support
  C) “Structure-dependent enhancement” caused by special stabilization centers, such as tertiary/quaternary carbon, remote charge capture, and branching-site-induced cleavage

【Hard trigger condition: even anomalous peaks require forced deep reasoning】
If an m/z satisfies:
- m/z is even AND decision.intensity_role="anomalous_high"
then the following “structural upgrade determination” must be performed, and the following may not be used as the only explanation:
- “Sigma-bond cleavage”
- “Hydrogen transfer” as the sole main mechanistic label; it may serve as an accompanying step, but must not be used alone as a superficial explanation

【Structural upgrade determination process (must be executed in order; does not change your original rules, only adds checks)】
For this even anomalous_high peak:
1) First check whether source_smiles contains verifiable evidence of a branching center:
   - tert carbon (tertiary carbon) appears in SMILES as a bracketed branch adjacent to the main chain: “C(C)” or “C(CCC)” or “CC(C)”
   - quat carbon (quaternary carbon) appears in SMILES as the same carbon connecting to ≥2 bracketed branches: “C(CC)(CCC)”, etc.
   If such evidence exists, decision.charge_capture_motif must preferentially consider tertiary_carbon or quaternary_carbon, unless there is strong counterevidence.

2) If branching-center evidence exists, the preferred_origin of this even anomalous peak must preferentially be "structure-driven",
   and the following mechanism should be preferentially attempted:
   - "Sigma-bond cleavage"
   as the main channel; "Hydrogen transfer" may serve as a parallel terminal state or stabilization explanation, but must not be the only main channel.

3) Only when the following conditions are simultaneously satisfied may this even anomalous peak be mainly attributed to ion-evolution-based “Dehydrogenation / Sequential dehydrogenation”:
   - An adjacent odd precursor peak exists (m/z+1 or m/z+2), and its intensity is ≥ the intensity of this even peak, so the weak peak → strong peak rule is not triggered
   - decision.ion_evolution_allowed=true and violate_weak_to_strong_rule=false
   Otherwise, it must fall back to structure-driven explanation and use the “Sigma-bond cleavage” priority channel from Step 2.

4) The precursor under structure-driven explanation must be forced to be “close to the branching center”:
   - If tertiary/quaternary_carbon is determined: best_fragment must contain a bracketed-branch pattern, such as "CC(C)", "C(CCC)C", "C(CC)(CCC)", etc.
   - best_fragment must not degenerate into a purely linear fragment such as “CCCCCCCC” that contains no branching information, unless source_smiles truly contains no branching evidence at all.

【Output constraints (newly added only for this even anomalous_high peak)】
- If triplet[1] is the main channel, preferentially use: "Sigma-bond cleavage"
- If stabilization needs to be expressed, a second triplet may be added for the same m/z:
  - Main channel: Sigma-bond cleavage → even-electron ion, such as C8H16+
  - Accompanying channel: Hydrogen transfer or Dehydrogenation / Sequential dehydrogenation → another expression of the same m/z
  But the weak→strong prohibition rule and precursor-format rule must still be satisfied.

────────────────────────────────
XIII. Parallel Terminal-State Output for “Same-Mass Fragments with Multiple Mechanisms” (only added, without changing old logic)
────────────────────────────────
If and only if the following conditions are satisfied, two triplets may be output for the same m/z as parallel terminal states:
- This m/z is marked as anomalous_high or carbon_island=true
- And there exist two non-conflicting explanatory paths that both satisfy the precursor rules, for example:
  - Structure-driven Sigma-bond cleavage
  - And background-type Sigma-bond cleavage or ion-evolution-based dehydrogenation

Output requirements:
- The first triplet must be the main path that you consider “better able to explain the intensity anomaly,” usually structure-driven + induced cleavage
- The second triplet is a supplementary path, either background or evolution, but must not violate the weak peak → strong peak rule

────────────────────────────────
【Input】
Molecular information:
"""


# ============================================================
# 2. Argument parser
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSON file. Each item should contain smiles, formula, mw, spectrum, spectrum_list, id, etc.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON file for model responses.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="xx",
        help="API key. If not provided, OPENAI_API_KEY will be used.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Kept only for backward compatibility. Official OpenAI API does not use base_url.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="Model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Recommended to use 0.0 for deterministic structured annotation.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Initial max output tokens. The retry logic will try 2048 -> 4096 -> 8192.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for slicing input data.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N items after start. Useful for testing.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between requests.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save intermediate results every N samples.",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="Number of retries for failed API calls.",
    )
    parser.add_argument(
        "--use_json_mode",
        action="store_true",
        help="Use response_format={'type':'json_object'} if the API provider supports it.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="If output_file already exists, skip items whose id has already been processed.",
    )

    return parser.parse_args()


# ============================================================
# 3. Utility functions
# ============================================================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def build_user_prompt(data: Dict[str, Any]) -> str:
    """
    Build the user message for one molecule.

    The model receives only the molecular information here.
    All decision rules are placed in SYSTEM_PROMPT.
    """
    smiles = data.get("smiles", "")
    name = data.get("name", "")
    formula = data.get("formula", "")
    mw = data.get("mw", "")
    compound_class = data.get("compound_class", "")
    spectrum = data.get("spectrum", {})

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
        "Please perform EI (70 eV) fragmentation decision annotation for the following molecule.\n"
        "You must output strict JSON only, following the schema specified in the system prompt.\n\n"
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

    # Remove markdown fences if any
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    # First try direct parsing
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to extract the largest JSON object
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


# ============================================================
# 4. API caller
# ============================================================

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

    token_attempts = [2048, 4096, 8192]

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

            # If json mode is unsupported by the provider, retry the same max_tokens once without it.
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


# ============================================================
# 5. Main process
# ============================================================

def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is missing. Please pass --api_key or set OPENAI_API_KEY.")

    # Official OpenAI API client. Do not pass base_url here.
    # args.base_url is intentionally ignored and kept only for old command compatibility.
    client = OpenAI(api_key=api_key)

    datas = load_json(args.input_file)
    if not isinstance(datas, list):
        raise ValueError("Input JSON must be a list of molecular records.")

    # Slice data
    if args.limit is None:
        selected_datas = datas[args.start :]
    else:
        selected_datas = datas[args.start : args.start + args.limit]

    # Load existing results for breakpoint resume
    results: List[Dict[str, Any]] = []
    processed_ids = set()

    if args.skip_existing and os.path.exists(args.output_file):
        try:
            results = load_json(args.output_file)
            for r in results:
                if "id" in r:
                    processed_ids.add(r["id"])
            print(f"Loaded {len(results)} existing results. Will skip {len(processed_ids)} processed ids.")
        except Exception as e:
            print(f"Failed to load existing output file. Start from empty results. Error: {e}")
            results = []
            processed_ids = set()

    total = len(selected_datas)
    print(f"Total selected samples: {total}")
    print(f"Model: {args.model}")
    print("API: OpenAI official Responses API")
    print(f"Output file: {args.output_file}")

    for idx, data in enumerate(selected_datas, start=1):
        data_id = data.get("id", None)

        if args.skip_existing and data_id in processed_ids:
            print(f"[{idx}/{total}] Skip existing id={data_id}")
            continue

        smiles = data.get("smiles", "")
        name = data.get("name", "")
        formula = data.get("formula", "")
        mw = data.get("mw", "")

        print("=" * 80)
        print(f"[{idx}/{total}] id={data_id}, name={name}, smiles={smiles}, formula={formula}, mw={mw}")

        user_prompt = build_user_prompt(data)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        parsed, raw_text, error = call_api_with_retry(
            client=client,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retry=args.retry,
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
                "input_spectrum": data.get("spectrum", {}),
                "model_output": parsed,
                "raw_response": raw_text,
                "parse_ok": True,
                "error": None,
                "warnings": warnings,
            }
            print(f"Success. warnings={len(warnings)}")
            if warnings:
                for w in warnings[:10]:
                    print(f"  [Warning] {w}")
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

        # Save intermediate results
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

