# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import stat
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

try:
    from rdkit import Chem
except Exception:
    Chem = None


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


────────────────────────────────
XIV. Mandatory non-empty repair mode (highest priority)
────────────────────────────────

The current request is a repair request for exactly one target m/z whose old
triplet was empty. This section overrides every earlier permission to output an
invalid or empty triplet.

For the requested target m/z:

1. triplet_status MUST be exactly "ok".
2. triplet MUST contain at least one three-string triplet.
3. triplet=[] is forbidden.
4. triplet_status="invalid" is forbidden.
5. invalid_reason MUST be null or omitted.
6. You MUST choose the single most reasonable explanation supported by the
   molecule, target intensity, complete spectrum, the current output result,
   and already completed higher-m/z triplets.
7. If an ion-evolution route violates weak-to-strong intensity direction, choose
   the most defensible structure-driven route instead; do not return empty.
8. Return only the requested target m/z block inside the unchanged top-level
   output schema. Do not regenerate or modify any other ion.
"""


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

DECISION_DEFAULTS: Dict[str, Any] = {
    "intensity_role": "background_like",
    "fourteen_da_series": False,
    "carbon_island": False,
    "charge_capture_motif": "none",
    "ion_evolution_allowed": False,
    "violate_weak_to_strong_rule": False,
    "preferred_origin": "structure-driven",
}


# ============================================================
# 2. Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Nuwa to repair only empty triplets in an existing EI-MS result "
            "file, writing a separate merged output file."
        )
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Original molecular JSON containing id, smiles, formula, mw and spectrum.",
    )
    parser.add_argument(
        "--current_result_file",
        required=True,
        help="Existing old result containing empty triplet blocks. This file is read-only.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="New merged result path. Old and repaired blocks are written here.",
    )
    parser.add_argument(
        "--failure_log_file",
        default=None,
        help=(
            "Separate JSON repair log. Defaults to <output_file>.repair_failures.json. "
            "The main result schema is not changed."
        ),
    )
    parser.add_argument("--api_key", required=True)
    parser.add_argument(
        "--base_url",
        default="https://api.nuwaapi.com/v1",
        help="Nuwa OpenAI-compatible base URL.",
    )
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument(
        "--repair_attempts",
        type=int,
        default=3,
        help="Maximum semantic repair rounds for one empty ion.",
    )
    parser.add_argument(
        "--api_retries",
        type=int,
        default=3,
        help="API/JSON retries inside each semantic repair round.",
    )
    parser.add_argument("--use_json_mode", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--id_min", type=int, default=None)
    parser.add_argument("--id_max", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--disable_code_fallback",
        action="store_true",
        help=(
            "Do not insert a deterministic non-empty fallback after all model "
            "attempts fail. By default the fallback is enabled to guarantee a "
            "non-empty repaired block."
        ),
    )
    return parser.parse_args()


# ============================================================
# 3. JSON and file safety
# ============================================================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(obj: Any, path: str, retries: int = 30) -> None:
    """Windows-safe full-file rewrite with a unique temporary file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        if target.exists():
            try:
                os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
            except Exception:
                pass

        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                os.replace(temp_path, target)
                return
            except PermissionError as exc:
                last_error = exc
                if attempt >= retries:
                    break
                wait = min(0.25 * attempt, 3.0)
                print(
                    f"[SAVE RETRY {attempt}/{retries}] {target} is locked; "
                    f"retrying in {wait:.2f}s..."
                )
                time.sleep(wait)

        pending = target.with_name(
            target.name
            + "."
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".pending.json"
        )
        os.replace(temp_path, pending)
        raise PermissionError(
            f"Could not replace locked file {target}. The complete new data was "
            f"preserved at {pending}. Last error: {last_error}"
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def canonical_id(value: Any) -> str:
    try:
        return str(int(float(value)))
    except Exception:
        return str(value).strip()


def safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def parse_mz(value: Any) -> Optional[int]:
    if value is None:
        return None
    direct = safe_int(value)
    if direct is not None:
        return direct
    match = re.search(r"m/z\s*([0-9]+)", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def sort_spectrum_desc(spectrum: Any) -> Tuple[Dict[str, Any], List[int]]:
    if not isinstance(spectrum, dict):
        return {}, []
    pairs: List[Tuple[int, Any]] = []
    for mz_raw, intensity in spectrum.items():
        mz = safe_int(mz_raw)
        if mz is not None:
            pairs.append((mz, intensity))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return ({str(mz): intensity for mz, intensity in pairs}, [mz for mz, _ in pairs])


# ============================================================
# 4. Result navigation and empty-triplet detection
# ============================================================

def get_model_output(result_item: Any) -> Dict[str, Any]:
    if not isinstance(result_item, dict):
        return {}
    value = result_item.get("model_output")
    return value if isinstance(value, dict) else {}


def get_triples(result_item: Any) -> Dict[str, Any]:
    value = get_model_output(result_item).get("triples")
    return value if isinstance(value, dict) else {}


def normalize_triplet_entries(value: Any) -> List[List[str]]:
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(
        isinstance(x, (str, int, float)) for x in value
    ):
        return [[str(value[0]), str(value[1]), str(value[2])]]

    output: List[List[str]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                output.append([str(item[0]), str(item[1]), str(item[2])])
    return output


def block_triplet_is_empty(block: Any) -> bool:
    """Repair only blocks with no usable triplet entries."""
    if not isinstance(block, dict):
        return False
    return len(normalize_triplet_entries(block.get("triplet"))) == 0


def block_has_nonempty_target_triplet(block: Any, target_mz: int) -> bool:
    if not isinstance(block, dict):
        return False
    if block.get("triplet_status") == "invalid":
        return False
    for triplet in normalize_triplet_entries(block.get("triplet")):
        if (
            triplet[0].strip()
            and triplet[1].strip() in ALLOWED_MECHANISMS
            and parse_mz(triplet[2]) == int(target_mz)
        ):
            return True
    return False


def find_target_key(triples: Dict[str, Any], target_mz: int) -> Optional[str]:
    for key in triples.keys():
        if parse_mz(key) == int(target_mz):
            return str(key)
    return None


def build_result_index(results: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    index: Dict[str, int] = {}
    for position, item in enumerate(results):
        if isinstance(item, dict) and "id" in item:
            index.setdefault(canonical_id(item.get("id")), position)
    return index


def build_source_index(datas: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for item in datas:
        if isinstance(item, dict) and "id" in item:
            index[canonical_id(item.get("id"))] = item
    return index


def source_data_for_result(
    result_item: Dict[str, Any],
    source_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    key = canonical_id(result_item.get("id"))
    data = copy.deepcopy(source_index.get(key, {}))
    data.setdefault("id", result_item.get("id"))
    data.setdefault("name", result_item.get("name", ""))
    data.setdefault("smiles", result_item.get("smiles", ""))
    data.setdefault("formula", result_item.get("formula", ""))
    data.setdefault("mw", result_item.get("mw", ""))
    data.setdefault("compound_class", result_item.get("compound_class", ""))
    if not isinstance(data.get("spectrum"), dict) or not data.get("spectrum"):
        spectrum = result_item.get("input_spectrum", {})
        data["spectrum"] = spectrum if isinstance(spectrum, dict) else {}
    return data


# ============================================================
# 5. Higher-m/z context, intensity and formulas
# ============================================================

def target_intensity_context(data: Dict[str, Any], target_mz: int) -> Dict[str, Any]:
    spectrum, _ = sort_spectrum_desc(data.get("spectrum", {}))
    numeric = {int(k): safe_float(v) for k, v in spectrum.items()}
    current = numeric.get(int(target_mz))
    known = [(mz, val) for mz, val in numeric.items() if val is not None]
    base = max(known, key=lambda item: item[1]) if known else (None, None)
    neighbors: Dict[str, Any] = {}
    for delta in (-14, -2, -1, 1, 2, 14):
        mz = int(target_mz) + delta
        if mz in numeric and numeric[mz] is not None:
            neighbors[str(mz)] = numeric[mz]
    return {
        "target_mz": int(target_mz),
        "target_intensity": current,
        "base_peak_mz": base[0],
        "base_peak_intensity": base[1],
        "neighboring_peaks": neighbors,
    }


def valid_higher_mz_context(
    result_item: Dict[str, Any],
    data: Dict[str, Any],
    target_mz: int,
) -> List[Dict[str, Any]]:
    spectrum, _ = sort_spectrum_desc(data.get("spectrum", {}))
    target_intensity = safe_float(spectrum.get(str(int(target_mz))))
    context: List[Dict[str, Any]] = []
    for key, block in get_triples(result_item).items():
        mz = parse_mz(key)
        if mz is None or mz <= int(target_mz):
            continue
        entries = normalize_triplet_entries(block.get("triplet") if isinstance(block, dict) else None)
        valid_entries = [
            triplet
            for triplet in entries
            if triplet[1].strip() in ALLOWED_MECHANISMS
            and parse_mz(triplet[2]) == mz
        ]
        if not valid_entries:
            continue
        precursor_intensity = safe_float(spectrum.get(str(mz)))
        allows = None
        if precursor_intensity is not None and target_intensity is not None:
            allows = precursor_intensity >= target_intensity
        context.append(
            {
                "mz": mz,
                "intensity": precursor_intensity,
                "target_intensity": target_intensity,
                "intensity_allows_downward_evolution": allows,
                "origin_type": block.get("origin_type") if isinstance(block, dict) else None,
                "triplet_status": block.get("triplet_status") if isinstance(block, dict) else None,
                "verified_triplets": valid_entries,
            }
        )
    context.sort(key=lambda item: int(item["mz"]), reverse=True)
    return context


def parent_carbon_count(formula: Any, smiles: str) -> int:
    match = re.search(r"C(\d*)", str(formula or ""))
    if match:
        return int(match.group(1) or "1")
    if Chem is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    return max(1, len(re.findall(r"C(?!l)", smiles)))


def infer_product_formula(data: Dict[str, Any], target_mz: int) -> Tuple[str, int, int]:
    max_c = max(1, parent_carbon_count(data.get("formula"), str(data.get("smiles", ""))))
    candidates: List[Tuple[int, int, int]] = []
    preferred_h_offset = 1 if int(target_mz) % 2 == 1 else 0
    for carbon in range(1, max_c + 1):
        hydrogen = int(target_mz) - 12 * carbon
        if 0 <= hydrogen <= 2 * carbon + 2:
            preferred = 2 * carbon + preferred_h_offset
            candidates.append((abs(hydrogen - preferred), -carbon, hydrogen))
    if not candidates:
        carbon = min(max_c, max(1, int(target_mz) // 12))
        hydrogen = max(0, int(target_mz) - 12 * carbon)
    else:
        _, neg_c, hydrogen = min(candidates)
        carbon = -neg_c
    formula = f"C{carbon if carbon != 1 else ''}H{hydrogen if hydrogen != 1 else ''}+"
    return formula, carbon, hydrogen


def formula_nominal_mass(formula: Any) -> Optional[int]:
    text = str(formula or "")
    c = re.search(r"C(\d*)", text)
    h = re.search(r"H(\d*)", text)
    if not c:
        return None
    carbon = int(c.group(1) or "1")
    hydrogen = int(h.group(1) or "1") if h else 0
    return 12 * carbon + hydrogen


# ============================================================
# 6. Prompt construction
# ============================================================

def build_repair_prompt(
    data: Dict[str, Any],
    result_item: Dict[str, Any],
    target_key: str,
    target_mz: int,
    failure_count: int,
    previous_bad_response: Optional[str] = None,
    previous_errors: Optional[Sequence[str]] = None,
) -> str:
    spectrum, mz_list = sort_spectrum_desc(data.get("spectrum", {}))
    triples = get_triples(result_item)
    old_block = copy.deepcopy(triples.get(target_key))
    higher_context = valid_higher_mz_context(result_item, data, target_mz)

    current_output_context = {
        "id": result_item.get("id"),
        "name": result_item.get("name", ""),
        "smiles": result_item.get("smiles", data.get("smiles", "")),
        "formula": result_item.get("formula", data.get("formula", "")),
        "mw": result_item.get("mw", data.get("mw", "")),
        "model_output": copy.deepcopy(get_model_output(result_item)),
    }

    payload: Dict[str, Any] = {
        "source_smiles": data.get("smiles", ""),
        "name": data.get("name", ""),
        "formula": data.get("formula", ""),
        "mw": data.get("mw", ""),
        "compound_class": data.get("compound_class", ""),
        "complete_spectrum": spectrum,
        "spectrum_list": mz_list,
        "target_mz": int(target_mz),
        "target_intensity_context": target_intensity_context(data, target_mz),
        "old_empty_target_block": old_block,
        "current_output_result_before_repair": current_output_context,
        "all_completed_higher_mz_triplets": higher_context,
        "current_failure_count": int(failure_count),
    }
    if previous_bad_response:
        payload["previous_bad_response"] = previous_bad_response[-12000:]
    if previous_errors:
        payload["previous_validation_errors"] = list(previous_errors)

    return (
        f"Repair exactly one empty EI-MS triplet block: m/z {int(target_mz)}.\n"
        f"The triples object in your answer must contain exactly one key: '{int(target_mz)}'.\n"
        "The old block is empty and must be replaced by the single most reasonable "
        "non-empty triplet. triplet_status must be 'ok'; triplet=[] and invalid are forbidden.\n"
        "Use the complete spectrum, current output result, target intensity, and all "
        "completed higher-m/z triplets. A higher-m/z ion may be selected as precursor_mz "
        "only when its chemistry is compatible and its intensity is not lower than the "
        "target intensity. Otherwise choose the best structure-driven explanation.\n"
        "Do not change any other ion. Return strict JSON only in the original top-level format.\n\n"
        "[Repair input]\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


# ============================================================
# 7. Nuwa OpenAI-compatible API
# ============================================================

def extract_json_from_text(text: str) -> Dict[str, Any]:
    if text is None:
        raise ValueError("Empty response content.")
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(raw[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Cannot parse JSON from response: {raw[:1000]}")


def call_nuwa_once(
    client: OpenAI,
    args: argparse.Namespace,
    messages: List[Dict[str, str]],
    max_tokens: int,
    use_json_mode: bool,
) -> str:
    kwargs: Dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": max_tokens,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content or ""


def call_nuwa_with_retry(
    client: OpenAI,
    args: argparse.Namespace,
    messages: List[Dict[str, str]],
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    base = max(1024, int(args.max_tokens))
    token_attempts: List[int] = []
    for value in (base, max(4096, base * 2), max(8192, base * 4)):
        if value not in token_attempts:
            token_attempts.append(value)
    token_attempts = token_attempts[: max(1, int(args.api_retries))]

    raw_text = ""
    last_error: Optional[str] = None
    for attempt, current_tokens in enumerate(token_attempts, start=1):
        try:
            print(
                f"[Nuwa Token Attempt {attempt}/{len(token_attempts)}] "
                f"max_tokens={current_tokens}"
            )
            raw_text = call_nuwa_once(
                client=client,
                args=args,
                messages=messages,
                max_tokens=current_tokens,
                use_json_mode=args.use_json_mode,
            )
            return extract_json_from_text(raw_text), raw_text, None
        except Exception as exc:
            last_error = str(exc)
            print(f"[Nuwa attempt failed] {last_error}")
            if args.use_json_mode and (
                "response_format" in last_error or "json" in last_error.lower()
            ):
                try:
                    raw_text = call_nuwa_once(
                        client=client,
                        args=args,
                        messages=messages,
                        max_tokens=current_tokens,
                        use_json_mode=False,
                    )
                    return extract_json_from_text(raw_text), raw_text, None
                except Exception as fallback_exc:
                    last_error = str(fallback_exc)
            if attempt < len(token_attempts):
                time.sleep(3)
    return None, raw_text, last_error


# ============================================================
# 8. Validation and block normalization
# ============================================================

def extract_returned_target_block(
    parsed: Dict[str, Any], target_mz: int
) -> Optional[Dict[str, Any]]:
    triples = parsed.get("triples") if isinstance(parsed, dict) else None
    if not isinstance(triples, dict):
        return None
    for key, block in triples.items():
        if parse_mz(key) == int(target_mz) and isinstance(block, dict):
            return block
    return None


def validate_model_block(block: Any, target_mz: int) -> Tuple[List[str], List[List[str]]]:
    errors: List[str] = []
    valid_triplets: List[List[str]] = []
    if not isinstance(block, dict):
        return ["Target block is missing or not an object."], []
    if block.get("triplet_status") != "ok":
        errors.append("triplet_status must be exactly 'ok'.")

    entries = normalize_triplet_entries(block.get("triplet"))
    if not entries:
        errors.append("triplet is empty; empty output is forbidden.")

    for triplet in entries:
        precursor = triplet[0].strip()
        mechanism = triplet[1].strip()
        product = triplet[2].strip()
        local_errors: List[str] = []
        if not precursor:
            local_errors.append("precursor is empty")
        if not (
            precursor.startswith("smiles_fragment: ")
            or precursor.startswith("precursor_mz: ")
        ):
            local_errors.append("precursor prefix is invalid")
        if mechanism not in ALLOWED_MECHANISMS:
            local_errors.append("mechanism is outside the strict enumeration")
        if parse_mz(product) != int(target_mz):
            local_errors.append("product m/z does not match target")
        if not local_errors:
            valid_triplets.append([precursor, mechanism, product])
        else:
            errors.append("; ".join(local_errors) + f": {triplet}")

    if not valid_triplets:
        errors.append("No code-valid non-empty target triplet remains.")
    return errors, valid_triplets


def normalize_repaired_block(
    block: Dict[str, Any], valid_triplets: List[List[str]]
) -> Dict[str, Any]:
    normalized = copy.deepcopy(block)
    decision = normalized.get("decision")
    if not isinstance(decision, dict):
        decision = {}
    for key, value in DECISION_DEFAULTS.items():
        decision.setdefault(key, value)
    normalized["decision"] = decision

    origin_type = normalized.get("origin_type")
    first_precursor = valid_triplets[0][0]
    if origin_type not in {"structure-driven", "ion-evolution"}:
        origin_type = (
            "ion-evolution"
            if first_precursor.startswith("precursor_mz: ")
            else "structure-driven"
        )
    normalized["origin_type"] = origin_type
    normalized.setdefault("core_motif", decision.get("charge_capture_motif", "none"))
    normalized["triplet_status"] = "ok"
    normalized["invalid_reason"] = None
    normalized["triplet"] = valid_triplets
    return normalized


# ============================================================
# 9. Failure log
# ============================================================

def default_failure_log(output_file: str) -> str:
    return output_file + ".repair_failures.json"


def load_failure_state(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            loaded = load_json(path)
            if isinstance(loaded, dict) and isinstance(loaded.get("records"), dict):
                return loaded
        except Exception as exc:
            print(f"[WARN] Could not load failure log: {exc}")
    return {"records": {}}


def failure_key(data_id: Any, mz: int) -> str:
    return f"{canonical_id(data_id)}|{int(mz)}"


def record_failure(
    state: Dict[str, Any],
    path: str,
    data_id: Any,
    mz: int,
    error: str,
    raw_response: str,
) -> int:
    records = state.setdefault("records", {})
    key = failure_key(data_id, mz)
    item = records.setdefault(
        key,
        {
            "id": data_id,
            "mz": int(mz),
            "failure_count": 0,
            "resolved": False,
            "fallback_used": False,
            "history": [],
        },
    )
    item["failure_count"] = int(item.get("failure_count", 0)) + 1
    item["resolved"] = False
    item["last_error"] = error
    item["last_raw_response"] = raw_response[-12000:]
    item["updated_at"] = datetime.now().isoformat(timespec="seconds")
    item.setdefault("history", []).append(
        {
            "failure_no": item["failure_count"],
            "error": error,
            "timestamp": item["updated_at"],
        }
    )
    save_json_atomic(state, path)
    return int(item["failure_count"])


def record_success(
    state: Dict[str, Any],
    path: str,
    data_id: Any,
    mz: int,
    fallback_used: bool,
) -> None:
    records = state.setdefault("records", {})
    key = failure_key(data_id, mz)
    item = records.setdefault(
        key,
        {
            "id": data_id,
            "mz": int(mz),
            "failure_count": 0,
            "history": [],
        },
    )
    item["resolved"] = True
    item["fallback_used"] = bool(fallback_used)
    item["resolved_at"] = datetime.now().isoformat(timespec="seconds")
    save_json_atomic(state, path)


# ============================================================
# 10. Guaranteed last-resort fallback
# ============================================================

def connected_fragment_candidate(smiles: str, target_c: int) -> Optional[str]:
    if not smiles or target_c <= 0:
        return None
    if Chem is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            carbon_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
            if target_c <= len(carbon_atoms):
                adjacency: Dict[int, List[int]] = {}
                carbon_set = set(carbon_atoms)
                for idx in carbon_atoms:
                    adjacency[idx] = [
                        n.GetIdx()
                        for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                        if n.GetIdx() in carbon_set
                    ]
                seen: set = set()
                stack = [frozenset([idx]) for idx in carbon_atoms]
                candidates: List[str] = []
                while stack and len(seen) < 10000 and len(candidates) < 100:
                    subset = stack.pop()
                    if subset in seen:
                        continue
                    seen.add(subset)
                    if len(subset) == target_c:
                        try:
                            frag = Chem.MolFragmentToSmiles(
                                mol,
                                atomsToUse=sorted(subset),
                                canonical=True,
                                isomericSmiles=True,
                            )
                        except Exception:
                            continue
                        if frag and "." not in frag:
                            candidates.append(frag)
                        continue
                    if len(subset) > target_c:
                        continue
                    frontier = set()
                    for atom_idx in subset:
                        frontier.update(adjacency.get(atom_idx, []))
                    frontier.difference_update(subset)
                    for nxt in frontier:
                        new_subset = frozenset(set(subset) | {nxt})
                        if len(new_subset) <= target_c:
                            stack.append(new_subset)
                if candidates:
                    candidates = sorted(
                        set(candidates),
                        key=lambda x: ("=" not in x, len(x), x),
                    )
                    return candidates[0]

    # Conservative text fallback: use the whole molecule when carbon counts match.
    if len(re.findall(r"C(?!l)", smiles)) == target_c:
        return smiles
    return None


def make_nonempty_fallback_block(
    data: Dict[str, Any],
    result_item: Dict[str, Any],
    target_mz: int,
) -> Dict[str, Any]:
    spectrum, _ = sort_spectrum_desc(data.get("spectrum", {}))
    target_intensity = safe_float(spectrum.get(str(int(target_mz))))
    formula, carbon, hydrogen = infer_product_formula(data, target_mz)
    product = f"{formula} (m/z {int(target_mz)})"
    source_smiles = str(data.get("smiles", "")).strip() or "C"

    parent_mass = formula_nominal_mass(data.get("formula"))
    mw = safe_float(data.get("mw"))
    molecular_mass_match = (
        parent_mass == int(target_mz)
        or (mw is not None and int(round(mw)) == int(target_mz))
    )

    decision = copy.deepcopy(DECISION_DEFAULTS)
    if molecular_mass_match:
        triplet = [f"smiles_fragment: {source_smiles}", "Molecular ion", product]
    else:
        higher = valid_higher_mz_context(result_item, data, target_mz)
        eligible = [
            item
            for item in higher
            if item.get("intensity_allows_downward_evolution") is True
        ]
        selected = min(eligible, key=lambda item: int(item["mz"])) if eligible else None

        use_higher_route = False
        if selected is not None:
            precursor_mz = int(selected["mz"])
            precursor_formula: Optional[Dict[str, Any]] = None
            verified = selected.get("verified_triplets") or []
            if verified:
                text = str(verified[0][2])
                match = re.search(r"C(\d*)H(\d*)", text)
                if match:
                    precursor_formula = {
                        "carbon": int(match.group(1) or "1"),
                        "hydrogen": int(match.group(2) or "1"),
                    }

            if (
                precursor_formula
                and precursor_formula["carbon"] == carbon
                and precursor_formula["hydrogen"] > hydrogen
            ):
                mechanism = "Dehydrogenation / Sequential dehydrogenation"
                use_higher_route = True
            elif precursor_formula and precursor_formula["carbon"] > carbon:
                # Neutral loss is used only when the mass difference can be a
                # plausible closed-shell hydrocarbon, not for arbitrary radical
                # losses such as 13 or 15 Da.
                mass_loss = precursor_mz - int(target_mz)
                carbon_loss = precursor_formula["carbon"] - carbon
                hydrogen_loss = mass_loss - 12 * carbon_loss
                if (
                    carbon_loss > 0
                    and hydrogen_loss >= 0
                    and hydrogen_loss <= 2 * carbon_loss + 2
                    and hydrogen_loss % 2 == 0
                ):
                    mechanism = "Neutral loss"
                    use_higher_route = True

            if use_higher_route:
                triplet = [f"precursor_mz: {precursor_mz}", mechanism, product]
                decision["ion_evolution_allowed"] = True
                decision["preferred_origin"] = "ion-evolution"

        if not use_higher_route:
            fragment = connected_fragment_candidate(source_smiles, carbon)
            if fragment is None:
                # Absolute final fallback. It remains a valid non-empty schema and
                # is explicitly tracked in the separate failure log.
                fragment = source_smiles
            mechanism = "Allylic cleavage" if "=" in fragment else "Sigma-bond cleavage"
            triplet = [f"smiles_fragment: {fragment}", mechanism, product]

    return {
        "decision": decision,
        "origin_type": decision["preferred_origin"],
        "core_motif": decision["charge_capture_motif"],
        "triplet_status": "ok",
        "invalid_reason": None,
        "triplet": [triplet],
    }


# ============================================================
# 11. Main repair process
# ============================================================

def main() -> None:
    args = parse_args()

    if args.repair_attempts < 1:
        raise ValueError("--repair_attempts must be >= 1")
    if args.id_min is not None and args.id_max is not None and args.id_min > args.id_max:
        raise ValueError("--id_min must be <= --id_max")

    old_path = os.path.abspath(args.current_result_file)
    new_path = os.path.abspath(args.output_file)
    if old_path == new_path:
        raise ValueError(
            "--current_result_file and --output_file must be different. "
            "The old file is intentionally kept unchanged."
        )

    api_key = args.api_key or os.environ.get("NUWA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key is missing. Pass --api_key or set NUWA_API_KEY/OPENAI_API_KEY."
        )

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    source_datas = load_json(args.input_file)
    if not isinstance(source_datas, list):
        raise ValueError("--input_file must contain a JSON list of molecular records.")
    source_index = build_source_index(
        [item for item in source_datas if isinstance(item, dict)]
    )

    old_results = load_json(args.current_result_file)
    if not isinstance(old_results, list):
        raise ValueError("--current_result_file must contain a JSON list.")

    # Resume from the new merged output when it already exists; otherwise make a
    # deep in-memory copy of the old result. The old file is never written.
    if os.path.exists(args.output_file):
        working_results = load_json(args.output_file)
        if not isinstance(working_results, list):
            raise ValueError("Existing --output_file must contain a JSON list.")
        print(f"[RESUME] Loaded merged output: {args.output_file}")
    else:
        working_results = copy.deepcopy(old_results)
        save_json_atomic(working_results, args.output_file)
        print(f"[INIT] Copied old result to new merged output: {args.output_file}")

    failure_log_path = args.failure_log_file or default_failure_log(args.output_file)
    failure_state = load_failure_state(failure_log_path)
    failure_state["current_result_file"] = args.current_result_file
    failure_state["output_file"] = args.output_file
    failure_state["base_url"] = args.base_url
    failure_state["model"] = args.model
    save_json_atomic(failure_state, failure_log_path)

    repair_targets: List[Tuple[int, str, int, str]] = []
    for result_pos, result_item in enumerate(working_results):
        if not isinstance(result_item, dict):
            continue
        data_id_int = safe_int(result_item.get("id"))
        if args.id_min is not None and (data_id_int is None or data_id_int < args.id_min):
            continue
        if args.id_max is not None and (data_id_int is None or data_id_int > args.id_max):
            continue
        triples = get_triples(result_item)
        for key, block in triples.items():
            mz = parse_mz(key)
            if mz is not None and block_triplet_is_empty(block):
                repair_targets.append((result_pos, str(key), mz, canonical_id(result_item.get("id"))))

    repair_targets.sort(
        key=lambda item: (
            safe_int(working_results[item[0]].get("id"))
            if safe_int(working_results[item[0]].get("id")) is not None
            else 10**18,
            -item[2],
        )
    )
    if args.start:
        repair_targets = repair_targets[args.start :]
    if args.limit is not None:
        repair_targets = repair_targets[: args.limit]

    print("=" * 92)
    print(f"Nuwa base URL:                  {args.base_url}")
    print(f"Model:                          {args.model}")
    print(f"Old result (read-only):         {args.current_result_file}")
    print(f"New merged output:              {args.output_file}")
    print(f"Failure log:                    {failure_log_path}")
    print(f"Empty triplet blocks to repair: {len(repair_targets)}")
    print("Mode: single-ion repair; replace only the existing empty block; rewrite full output after each success")
    print("=" * 92)

    repaired = 0
    unresolved = 0
    fallback_count = 0

    for order, (result_pos, target_key, target_mz, _) in enumerate(repair_targets, start=1):
        result_item = working_results[result_pos]
        data_id = result_item.get("id")
        triples = get_triples(result_item)
        current_block = triples.get(target_key)

        # The block may already have been repaired during a previous run.
        if block_has_nonempty_target_triplet(current_block, target_mz):
            print(f"[{order}/{len(repair_targets)}] Skip repaired id={data_id}, m/z={target_mz}")
            continue

        data = source_data_for_result(result_item, source_index)
        log_item = failure_state.get("records", {}).get(
            failure_key(data_id, target_mz), {}
        )
        failure_count = int(log_item.get("failure_count", 0))
        previous_bad = None
        previous_errors: List[str] = []
        repaired_block: Optional[Dict[str, Any]] = None
        last_raw = ""

        print("-" * 92)
        print(
            f"[{order}/{len(repair_targets)}] Repair id={data_id}, "
            f"m/z={target_mz}, prior_failures={failure_count}"
        )

        for semantic_attempt in range(1, args.repair_attempts + 1):
            prompt = build_repair_prompt(
                data=data,
                result_item=result_item,
                target_key=target_key,
                target_mz=target_mz,
                failure_count=failure_count,
                previous_bad_response=previous_bad,
                previous_errors=previous_errors,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            print(
                f"[Semantic repair {semantic_attempt}/{args.repair_attempts}] "
                f"id={data_id}, m/z={target_mz}"
            )
            parsed, raw_text, api_error = call_nuwa_with_retry(client, args, messages)
            last_raw = raw_text

            if parsed is None:
                previous_errors = [api_error or "Nuwa API/JSON parsing failed."]
                failure_count = record_failure(
                    failure_state,
                    failure_log_path,
                    data_id,
                    target_mz,
                    previous_errors[0],
                    raw_text,
                )
                previous_bad = raw_text
                continue

            returned_block = extract_returned_target_block(parsed, target_mz)
            errors, valid_triplets = validate_model_block(returned_block, target_mz)
            if errors:
                previous_errors = errors
                failure_count = record_failure(
                    failure_state,
                    failure_log_path,
                    data_id,
                    target_mz,
                    " | ".join(errors),
                    raw_text,
                )
                previous_bad = raw_text
                print(f"[Rejected] {' | '.join(errors[:5])}")
                continue

            repaired_block = normalize_repaired_block(returned_block, valid_triplets)
            break

        used_fallback = False
        if repaired_block is None:
            if args.disable_code_fallback:
                unresolved += 1
                print(
                    f"[UNRESOLVED] id={data_id}, m/z={target_mz}; "
                    "old empty block remains in the new output."
                )
                continue
            repaired_block = make_nonempty_fallback_block(
                data=data,
                result_item=result_item,
                target_mz=target_mz,
            )
            used_fallback = True
            fallback_count += 1
            print(
                f"[CODE FALLBACK] id={data_id}, m/z={target_mz}; "
                "inserted a non-empty last-resort triplet and recorded it in the failure log."
            )

        # Replacing an existing dict key preserves its original position.
        triples[target_key] = repaired_block
        repaired += 1
        save_json_atomic(working_results, args.output_file)
        record_success(
            failure_state,
            failure_log_path,
            data_id,
            target_mz,
            fallback_used=used_fallback,
        )
        print(
            f"[INSERTED + FULL REWRITE] id={data_id}, m/z={target_mz}, "
            f"fallback={used_fallback}, output={args.output_file}"
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

    remaining_empty = 0
    for item in working_results:
        for block in get_triples(item).values() if isinstance(item, dict) else []:
            if block_triplet_is_empty(block):
                remaining_empty += 1

    failure_state["summary"] = {
        "targets_scanned": len(repair_targets),
        "repaired": repaired,
        "fallback_used": fallback_count,
        "unresolved": unresolved,
        "remaining_empty_blocks_in_output": remaining_empty,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json_atomic(failure_state, failure_log_path)
    save_json_atomic(working_results, args.output_file)

    print("=" * 92)
    print("Repair completed.")
    print(f"Successfully filled blocks:     {repaired}")
    print(f"Code fallbacks used:            {fallback_count}")
    print(f"Unresolved without fallback:    {unresolved}")
    print(f"Empty blocks still in output:   {remaining_empty}")
    print(f"Old result remains unchanged:   {args.current_result_file}")
    print(f"Merged repaired output:         {args.output_file}")
    print(f"Failure/attempt log:             {failure_log_path}")


if __name__ == "__main__":
    main()
