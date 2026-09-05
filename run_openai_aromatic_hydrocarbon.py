# -*- coding: utf-8 -*-
"""
python run_openai_aromatic_hydrocarbon.py ^
  --input_file ../difference_types_SMILES/non-process/aromatic_hydrocarbon.jsonl ^
  --output_file ./aromatic_hydrocarbon_non_process_outputs/openai_gpt41_test_results.jsonl ^
  --model gpt-4.1 ^
  --temperature 0.0 ^
  --id_min 800 ^
  --id_max 805

"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# ============================================================
# 1. System prompt
#    注意：这里保持用户原始 prompt 内容不变
# ============================================================

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

For aromatic and PAH spectra, an anomalously high peak should not be explained by a generic mechanism label alone. The explanation must specify whether the ion is a molecular ion, isotopic peak, benzylic ion, aromatic-ring-retained terminal ion, neutral-loss product, or ring-cleavage/rearrangement product.

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
VI-0. Mandatory Arithmetic and Formula Consistency Check
────────────────────────────────

Before assigning any mechanism, perform the following checks:

1. Isotopic peak is allowed only when:
   product_mz > precursor_mz
   and product_mz - precursor_mz = 1 or 2.

   If product_mz < precursor_mz, it must never be assigned as "Isotopic peak".

2. Molecular ion must match the nominal molecular mass, including isotope labels if present.

3. Dehydrogenation / Sequential dehydrogenation must conserve carbon number.

4. Neutral loss requires precursor_mz > product_mz and a chemically meaningful mass loss.

If any of these checks fails, do not assign that mechanism.

────────────────────────────────
VI-A. Aromatic and PAH Fragmentation Guidance
(NEW)
────────────────────────────────

Applicable objects:

- aromatic hydrocarbons
- alkyl-substituted aromatic hydrocarbons
- fused-ring aromatics
- polycyclic aromatic hydrocarbons (PAHs)

Examples:

benzene
alkylbenzene
indane
tetralin
naphthalene derivatives
phenanthrene derivatives
pyrene derivatives

Rules:

1.

Strong molecular ions are common in aromatic and PAH spectra.

When the molecular ion is present with significant intensity,
prefer:

"Molecular ion"

rather than forcing fragmentation explanations.

2.

For M+1 peaks:

prefer:

"Isotopic peak"

when the peak is exactly M+1 and its intensity is
consistent with isotopic abundance.

3.

For alkyl-substituted aromatic hydrocarbons:

strong M−15 ions should preferentially be interpreted as:

loss of CH3·

using:

["precursor_mz: M",
 "Neutral loss",
 "<M−15 product ion>"]

unless strong structural evidence supports another pathway.

4.

For aromatic and PAH fragment ions:

if carbon number changes between precursor and product,

do NOT use:

"Dehydrogenation / Sequential dehydrogenation"

For fused-ring aromatics and PAHs, do NOT assign:

"Neutral loss"

merely because the mass difference can be arithmetically expressed as a small hydrocarbon formula.

Use "Neutral loss" only when the lost neutral fragment is chemically meaningful and structurally plausible, such as:

- loss of CH3· from an alkyl substituent
- loss of C2H2 from an aromatic or alkyne-related system
- loss of C2H4 or C3H6 from a partially saturated side chain or hydroaromatic ring
- loss of H2 from dehydrogenation/aromatization-related processes

If the product ion is a smaller aromatic, fused-ring aromatic, or highly unsaturated PAH-like ion and no clear neutral fragment can be structurally assigned, prefer:

"Ring cleavage / Ring rearrangement"

5.

Use:

"Benzylic cleavage"

only when cleavage occurs at a benzylic position
and directly explains formation of a benzyl-type,
tropylium-type, or benzylic-stabilized ion.

Do not assign:

"Benzylic cleavage"

to a simple M−15 ion unless the resulting ion
is clearly benzylic in nature.

6.

For fused-ring aromatics and PAHs:

prefer:

"Ring cleavage / Ring rearrangement"

when the product ion corresponds to a smaller aromatic
or fused-ring aromatic ion.

6-A.

For aromatic and PAH neutral-loss assignments:

If the neutral loss is from the molecular ion and the molecular ion peak is present in the spectrum, the precursor should preferentially be:

"precursor_mz: <M>"

rather than the full molecular "smiles_fragment".

Example:

Preferred:
["precursor_mz: 260", "Neutral loss", "C18H16+ (m/z 232)"]

Avoid:
["smiles_fragment: <full molecule>", "Neutral loss", "C18H16+ (m/z 232)"]

Use a structure-driven smiles_fragment for "Neutral loss" only when the local structural fragment clearly identifies the neutral-loss site.

7.


For aromatic ions that retain an intact aromatic ring, do not assign
"Ring cleavage / Ring rearrangement" merely because the ion is aromatic.

Examples:
- C6H6+ (m/z 78) retains an intact benzene ring.
- C6H5+ (m/z 77) retains a phenyl ring.

Use "Ring cleavage / Ring rearrangement" mainly when the aromatic ring system is broken,
contracted, rearranged, or converted to a smaller aromatic ion such as C5H5+, C4H3+, or C3H3+.

For alkylbenzenes, m/z 77 may be treated as an aromatic-ring-retained terminal ion unless
there is direct evidence for a stronger precursor-product ion-evolution path.
Do not force 78 → 77 or 91 → 77.

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
X-A. Formula–m/z Consistency Rule
────────────────────────────────

Before outputting any product_ion, verify that the ion formula and reported m/z are internally consistent.

1. Calculate the nominal mass from the product ion formula.
2. The nominal mass must equal the reported m/z.
3. If formula and m/z are inconsistent, regenerate the product ion formula before output.
4. Do not output impossible formula/m/z combinations such as C17H17+ (m/z 229).
5. For isotope peaks, use a parser-friendly product label such as:
   C20H20+ [M+1] (m/z 261)
   rather than C20H20+1 (m/z 261).

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
【Input】

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
        help="Input JSON/JSONL file. Each item should contain smiles, formula, mw, spectrum, spectrum_list, id, etc.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON/JSONL file for aromatic hydrocarbon EI annotation results. The suffix decides the format.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key. If not provided, OPENAI_API_KEY will be used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Recommended to use 0.0 for deterministic structured annotation.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for slicing input data.",
    )
    parser.add_argument(
        "--id_min",
        "--idmin",
        dest="id_min",
        type=int,
        default=None,
        help=(
            "Minimum molecule id to process, inclusive. "
            "For example, --id_min 1 --id_max 100 processes records with 1 <= id <= 100."
        ),
    )
    parser.add_argument(
        "--id_max",
        "--idmax",
        dest="id_max",
        type=int,
        default=None,
        help=(
            "Maximum molecule id to process, inclusive. "
            "If only --id_min is set, process id >= id_min; "
            "if only --id_max is set, process id <= id_max."
        ),
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
        "--use_json_mode",
        action="store_true",
        help="Use text.format={'type':'json_object'} if the OpenAI model supports it.",
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

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load JSON / JSONL molecular records.

    Supported input formats:
    1. JSON list:
       [{...}, {...}]
    2. JSON object wrapper:
       {"data": [{...}]} / {"records": [...]} / {"items": [...]} / {"results": [...]}
    3. JSONL:
       one molecular record per line
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            for key in ["data", "records", "items", "results"]:
                if isinstance(obj.get(key), list):
                    return [x for x in obj[key] if isinstance(x, dict)]
            return [obj]
    except json.JSONDecodeError:
        pass

    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Bad JSON at {path}:{line_no}: {e}\n{s[:500]}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"JSONL line {line_no} is not a JSON object.")
        records.append(obj)

    return records


def save_records(records: List[Dict[str, Any]], path: str) -> None:
    """
    Save records.
    - .jsonl suffix: one JSON object per line
    - other suffix: JSON list with indentation
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if path.lower().endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def normalize_spectrum_from_record(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize spectrum into a dict with string m/z keys.

    Priority:
    1. spectrum dict: {"91": 100, "92": 8}
    2. spectrum list: [[91, 100], {"mz": 92, "intensity": 8}]
    3. spectrum_list: [91, 92, ...], intensity set to None
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


def build_user_prompt(data: Dict[str, Any]) -> str:
    """
    Build the user message for one aromatic hydrocarbon molecule.

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
# 4. API caller
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
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str], Optional[int]]:
    """
    Try output token limits in fixed order: 2048 -> 4096 -> 8192.

    If an API call fails or JSON extraction fails, the next larger token limit
    will be used. If 8192 also fails, return parsed_json=None.
    """
    last_error = None
    raw_text = ""
    token_attempts = [16384, 32768, 65536]

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
            return parsed, raw_text, None, current_tokens

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
                    return parsed, raw_text, None, current_tokens
                except Exception as e2:
                    last_error = str(e2)
                    print(f"[Fallback without json mode failed] {last_error}")

            if attempt < len(token_attempts):
                time.sleep(3)

    return None, raw_text, last_error, None


# ============================================================
# 5. Main process
# ============================================================

def get_record_key(record: Dict[str, Any]) -> Any:
    """
    Priority key for skip_existing.
    Prefer id; otherwise use smiles + formula + mw.
    """
    if "id" in record:
        return record["id"]
    smiles = record.get("smiles", record.get("SMILES", ""))
    return f"{smiles}||{record.get('formula', '')}||{record.get('mw', '')}"


def get_record_id_as_int(record: Dict[str, Any]) -> Optional[int]:
    """
    Convert record id to int for id-range filtering.

    Supported examples:
      {"id": 8}
      {"id": "8"}
      {"id": 8.0}

    If the record has no valid numeric id, return None.
    """
    rid = record.get("id", record.get("ID", None))
    return safe_int(rid)


def select_records_by_id_range(
    records: List[Dict[str, Any]],
    *,
    id_min: Optional[int],
    id_max: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Select records by inclusive id range.

    - If id_min and id_max are both None, return all records.
    - If id_min is set, keep id >= id_min.
    - If id_max is set, keep id <= id_max.
    - Records without a valid numeric id are skipped when id range is used.
    """
    if id_min is None and id_max is None:
        return records[:]

    selected: List[Dict[str, Any]] = []
    skipped_no_valid_id = 0

    for rec in records:
        rid = get_record_id_as_int(rec)
        if rid is None:
            skipped_no_valid_id += 1
            continue

        if id_min is not None and rid < id_min:
            continue
        if id_max is not None and rid > id_max:
            continue

        selected.append(rec)

    if skipped_no_valid_id > 0:
        print(
            f"[WARN] Skipped {skipped_no_valid_id} records without valid numeric id "
            f"while applying id range."
        )

    return selected


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is missing. Please pass --api_key or set OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)

    datas = load_json_or_jsonl(args.input_file)
    if not isinstance(datas, list):
        raise ValueError("Input file must contain a list of molecular records.")
    for i, item in enumerate(datas):
        if not isinstance(item, dict):
            raise ValueError(f"Input item {i} is not a JSON object.")

    # Select records by molecule id instead of using --limit.
    # The id range is inclusive: id_min <= id <= id_max.
    selected_datas = select_records_by_id_range(
        datas,
        id_min=args.id_min,
        id_max=args.id_max,
    )

    # Keep the old --start behavior for optional index-based resume/debugging,
    # but it is applied after id-range selection.
    if args.start > 0:
        selected_datas = selected_datas[args.start :]

    results: List[Dict[str, Any]] = []
    processed_keys = set()

    if args.skip_existing and os.path.exists(args.output_file):
        try:
            results = load_json_or_jsonl(args.output_file)
            for r in results:
                if isinstance(r, dict):
                    processed_keys.add(get_record_key(r))
            print(f"Loaded {len(results)} existing results. Will skip {len(processed_keys)} processed records.")
        except Exception as e:
            print(f"Failed to load existing output file. Start from empty results. Error: {e}")
            results = []
            processed_keys = set()

    total = len(selected_datas)
    print(f"Total selected aromatic hydrocarbon samples: {total}")
    print(f"Model: {args.model}")
    print("API: OpenAI official Responses API")
    print(f"Output file: {args.output_file}")
    print(f"ID range: id_min={args.id_min}, id_max={args.id_max}")
    if args.start > 0:
        print(f"Start offset after id filtering: {args.start}")

    for idx, data in enumerate(selected_datas, start=1):
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        parsed, raw_text, error, used_tokens = call_api_token_sequence(
            client=client,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            use_json_mode=args.use_json_mode,
        )

        common_fields = {
            "id": data_id,
            "name": name,
            "smiles": smiles,
            "formula": formula,
            "mw": mw,
            "compound_class": data.get("compound_class", ""),
            "input_spectrum": normalize_spectrum_from_record(data),
        }

        if parsed is not None:
            warnings = validate_basic_output(parsed, data)
            result_item = {
                **common_fields,
                "model_output": parsed,
                "raw_response": raw_text,
                "parse_ok": True,
                "error": None,
                "warnings": warnings,
                "used_max_output_tokens": used_tokens,
            }
            print(f"Success. used_max_output_tokens={used_tokens}, warnings={len(warnings)}")
            if warnings:
                for w in warnings[:10]:
                    print(f"  [Warning] {w}")
        else:
            result_item = {
                **common_fields,
                "model_output": None,
                "raw_response": raw_text,
                "parse_ok": False,
                "error": error,
                "warnings": [],
                "used_max_output_tokens": used_tokens,
            }
            print(f"Failed. error={error}")

        results.append(result_item)

        if args.save_every > 0 and len(results) % args.save_every == 0:
            save_records(results, args.output_file)
            print(f"Intermediate results saved to: {args.output_file}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    save_records(results, args.output_file)
    print("=" * 80)
    print(f"All done. Saved {len(results)} results to: {args.output_file}")


if __name__ == "__main__":
    main()


"""

python run_openai_aromatic_hydrocarbon.py --input_file ../difference_types_SMILES/non-process/aromatic_hydrocarbon.json --output_file ./aromatic_hydrocarbon_non_process_outputs/openai_gpt41_test_results.json ^ --model gpt-4.1 --temperature 0.0 --id_min 1 --id_max 1957 --api_key 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

"""

