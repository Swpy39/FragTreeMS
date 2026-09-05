# -*- coding: utf-8 -*-
"""

python run_openai_cycloalkane_json.py \
  --input_file ./difference_types_SMILES/cycloalkane.json \
  --output_file ./outputs/cycloalkane_openai_gpt41_results.json \
  --model gpt-4.1 \
  --temperature 0.0 \
  --id_min 1 \
  --id_max 50 \
  --sleep 5

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
#    仅增强环烷烃分子片段生成与离子对齐提示；其余接口与流程不变
# ============================================================

SYSTEM_PROMPT = r"""You are an extremely experienced **EI (70 eV) mass spectrometry interpretation expert**, specializing in the **fragmentation decision logic of alkanes and complex organic molecules under EI conditions**.

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
VI-C. Cycloalkane and Alicyclic Fragmentation Guidance
(NEW)
────────────────────────────────

Applicable objects:

- monocyclic cycloalkanes
- alkyl-substituted cycloalkanes
- bicyclic cycloalkanes
- spiro cycloalkanes
- bridged cycloalkanes
- fused saturated ring systems
- strained polycyclic alicyclic hydrocarbons

Rules:

1.

For cycloalkanes and alicyclic hydrocarbons, major fragment ions are often produced by:

"Ring cleavage / Ring rearrangement"

rather than ordinary linear-chain "Sigma-bond cleavage".

2.

For strong or anomalously enhanced cyclic hydrocarbon ions such as:

C3H5+  (m/z 41)
C4H7+  (m/z 55)
C5H7+  (m/z 67)
C5H9+  (m/z 69)
C6H9+  (m/z 81)
C6H11+ (m/z 83)
C7H11+ (m/z 95)
C7H13+ (m/z 97)

prefer:

"Ring cleavage / Ring rearrangement"

when the structure contains cyclic, fused-ring, bridged-ring, or spiro-ring motifs.

[primary cycloalkane ion-series safeguard]
For cycloalkanes and alicyclic hydrocarbons, the following CnH2n−1+ ions are primary ring-cleavage / ring-rearrangement ions when structurally plausible:

C3H5+  (m/z 41)
C4H7+  (m/z 55)
C5H9+  (m/z 69)
C6H11+ (m/z 83)
C7H13+ (m/z 97)

Do not assign these ions to "Dehydrogenation / Sequential dehydrogenation" merely because CnH2n+ or CnH2n+1+ ions are also present.
The same-carbon higher-hydrogen precursor rule is subordinate to this cycloalkane primary-ion rule.

3.

For even-m/z cyclic ions such as:

C2H4+  (m/z 28)
C3H6+  (m/z 42)
C4H6+  (m/z 54)
C4H8+  (m/z 56)
C5H8+  (m/z 68)
C5H10+ (m/z 70)
C6H10+ (m/z 82)

do not automatically assign them to dehydrogenation.

If the ion is strong or anomalously enhanced and the molecule contains a ring system, prefer structure-driven:

"Ring cleavage / Ring rearrangement"

unless a stronger same-carbon precursor clearly supports ion evolution.

4.

"Dehydrogenation / Sequential dehydrogenation" may be used for same-carbon ion evolution only when:

- the precursor ion exists in the spectrum;
- the precursor has the same carbon number;
- the precursor has higher hydrogen count;
- the intensity direction does not violate the weak-peak-to-strong-peak rule.

5.

For alkyl side-chain fragments from alkyl-substituted cycloalkanes, ordinary:

"Sigma-bond cleavage"

is still allowed, especially for alkyl cations such as:

C2H5+  (m/z 29)
C3H7+  (m/z 43)
C4H9+  (m/z 57)
C5H11+ (m/z 71)

6.

For strained small rings, spiro systems, bridged systems, and fused saturated rings, strong molecular ions may occur.

When the molecular ion is present with significant intensity, retain:

"Molecular ion"

instead of forcing a fragmentation explanation.

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

[smiles_fragment semantic definition]
"smiles_fragment" denotes the minimum parent-derived carbon skeleton with the same carbon number as the product ion whenever possible.
It is a structural source marker only.
It is not a real neutral precursor, not a charged precursor, and not required to match the hydrogen count, charge state, or radical state of the product ion.
Mechanism selection must be based on EI fragmentation logic, not on forcing the fragment hydrogen count to match the product ion.

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
VIII-A. Mandatory Fragment Grounding and Product-Ion Alignment Protocol
(HIGHEST PRIORITY FOR EVERY structure-driven TRIPLET)
────────────────────────────────

For every triplet whose origin_type is "structure-driven", perform the following
reasoning silently before writing the final JSON. Do not output this reasoning.

Step 1: determine the two carbon counts
- C_parent: the number of carbon atoms in source_smiles / molecular formula.
- C_target: the carbon number explicitly written in product_ion.
- Count carbon atoms chemically. "Cl" is chlorine and must not be counted as carbon.

Step 2: identify the actually retained parent-derived skeleton
- Determine which connected atoms of the parent molecule can remain in the charged
  product after the mechanism in triplet[1].
- The selected fragment must represent that retained connected carbon skeleton,
  not merely a nearby part of the molecule and not merely a string having the same
  number of letter "C" characters.
- Preserve the relevant branching, ring membership, substitution site, and
  charge-stabilizing environment whenever these features explain formation or
  retention of the ion.

Step 3: encode the skeleton by grounding it in source_smiles
- Copy <best_fragment> directly from source_smiles as an exact literal substring.
- Do not canonicalize, reorder, redraw, simplify, or invent another equivalent SMILES.
- Do not add branch parentheses, ring digits, atoms, or bonds that are absent from
  the copied source_smiles span.
- A chemically imaginable motif is still invalid here if its text is not literally
  verifiable in source_smiles.

Step 4: enforce exact fragment-product carbon alignment
- Normally C_fragment must equal C_target exactly.
- A fragment with fewer carbons than product_ion is forbidden.
- A fragment with more carbons than product_ion is forbidden merely because it is
  closer to the charge-capture site or contains the relevant ring.
- For carbon-losing fragmentation, the retained fragment must show the carbon
  skeleton of the product ion, not the complete parent skeleton before carbon loss.

[whole-parent SMILES prohibition]
If C_target < C_parent:
- <best_fragment> must not equal source_smiles;
- alternatives must not equal source_smiles;
- the fragment must not contain all parent carbon atoms;
- "the ion comes from this molecule", "the ring participates", or "ring cleavage
  occurs somewhere in the parent" is not sufficient justification for using the
  complete source_smiles.

The complete source_smiles may be used as smiles_fragment only when:
1) the mechanism is "Molecular ion"; or
2) the product retains the complete parent carbon skeleton (C_target = C_parent)
   and the selected carbon-conserving mechanism genuinely requires that skeleton.

Step 5: enforce mechanism-fragment compatibility
- For "Sigma-bond cleavage", select the actual retained alkyl/branched skeleton on
  one side of the cleaved bond. Do not use the entire ring-containing parent when
  the product is a smaller alkyl cation.
- For "Ring cleavage / Ring rearrangement", select the ring-derived or opened-ring
  retained skeleton that has C_target carbons. Do not use the complete parent solely
  because the parent is cyclic.
- For a branched alkyl ion, preserve a branch-aware fragment when that branching is
  present and relevant. Do not flatten a branched retained skeleton into a generic
  linear chain merely to match the carbon number.
- Do not invent a smaller ring such as "C1CC1" unless that exact ring-coded substring
  occurs in source_smiles and genuinely represents the retained parent substructure.

Step 6: distinguish real small fragments from placeholders
- "C", "CC", or "CCC" is not automatically wrong.
- It is allowed only when it is copied literally from source_smiles, represents the
  actual connected retained skeleton, and C_target is respectively 1, 2, or 3.
- Never generate "C", "CC", "CCC", or "CCCC" simply by repeating C according to
  the product carbon number.
- For example, "CC(C)C" may be used for a C4 ion only if "CC(C)C" is an exact
  substring of source_smiles and that branched C4 skeleton is the actual plausible
  retained fragment. Chemical equivalence without literal grounding is insufficient.

Step 7: choose among multiple valid fragments
- First discard every candidate that is not an exact substring, is disconnected,
  has the wrong carbon count, or conflicts with the mechanism.
- Among the remaining exact-C_target candidates, choose as best_fragment the one
  that best preserves the cleavage site, branching environment, ring origin, and
  stabilized charge-retention motif.
- alternatives may contain only candidates that independently pass all the same
  checks. Do not put a full-parent SMILES or a wrong-carbon fragment in alternatives.

Step 8: no fabricated fallback
If no exact, connected, mechanism-compatible structure-driven fragment can be
reliably grounded:
- do not fall back to the complete source_smiles for a smaller product ion;
- do not fabricate a linear C/CC/CCC-style placeholder;
- use origin_type="ion-evolution" only if a spectrum-supported precursor satisfies
  all ion-evolution constraints;
- otherwise set triplet_status="invalid" with the appropriate invalid_reason.

[mandatory examples of invalid fragment use]
For a C10 parent molecule:
- using the full C10 source_smiles for C5H9+ is invalid;
- using the full C10 source_smiles for C4H7+ is invalid;
- using a C3 fragment for C4H9+ is invalid;
- using an invented canonical fragment that is not a literal source_smiles substring
  is invalid even if it has four carbons.

[cross-field consistency]
- origin_type="structure-driven" requires a valid "smiles_fragment: ..." precursor.
- origin_type="ion-evolution" requires a valid "precursor_mz: ..." precursor.
- decision.preferred_origin, origin_type, triplet[0], triplet[1], core_motif, and
  product_ion must describe the same pathway and must not contradict one another.

────────────────────────────────
VIII-B. Mandatory Ion Formula, m/z, and Precursor Arithmetic Audit
────────────────────────────────

Before finalizing every m/z entry, silently verify:

1. product_ion formula arithmetic
- For a hydrocarbon CxHy+, nominal m/z must equal 12*x + y.
- Never write an unchanged formula at an incompatible m/z, such as C10H20+ at m/z 125.
- The product_ion m/z must equal the dictionary key being annotated.

2. molecular ion
- "Molecular ion" is allowed only when product formula and product m/z match the
  intact input molecule and its nominal molecular mass.

3. isotopic peak
- "Isotopic peak" requires a valid lower-mass precursor and product_mz = precursor_mz + 1
  or +2. A peak below the proposed precursor can never be an isotopic peak.

4. dehydrogenation
- Precursor and product must have the same carbon number.
- The precursor must have more hydrogen and a correspondingly larger m/z.
- precursor_mz must occur in the supplied spectrum and must not violate the
  weak-peak-to-strong-peak rule.
- Do not cite an absent or hypothetical precursor_mz.

5. fragment choice versus ion evolution
- Do not use smiles_fragment merely to avoid proving a precursor_mz.
- Do not use precursor_mz when a structure-driven terminal ring/alkyl ion is the
  chemically justified path.

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
XII. Parallel Terminal-State Output for “Same-Mass Fragments with Multiple Mechanisms” (only added, without changing old logic)
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
        default=0.0,
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
            {"role": "system", "content": SYSTEM_PROMPT},
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
