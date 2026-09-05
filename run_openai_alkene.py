# -*- coding: utf-8 -*-
"""
python run_openai_alkene.py ^
  --input_file ../difference_types_SMILES/alkene.json ^
  --output_file ./alkene_outputs/openai_gpt41_test_results.json ^
  --model gpt-4.1 ^
  --max_tokens 30000 ^
  --temperature 0.0 ^
  --id_min 1 ^
  --id_max 666 ^
  --save_every 1
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
# ============================================================

SYSTEM_PROMPT = r"""
You are an extremely experienced **EI (70 eV) mass spectrometry interpretation expert**, specializing in the **fragmentation decision logic of acyclic alkenes and polyenes under EI conditions**.

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
VI-A. Mandatory Alkene and Polyene Fragmentation Guidance
────────────────────────────────

Precedence rule: this section overrides all alkane-derived odd/even-ion heuristics elsewhere in this prompt whenever source_smiles contains C=C. The strict 13-label enumeration, arithmetic checks, precursor format, and weak-to-strong rule remain active.

Applicable structures in this dataset:

- acyclic monoalkenes, including terminal, internal, 1,1-disubstituted, tri-substituted, and tetra-substituted C=C bonds;
- isolated, conjugated, and cumulated dienes;
- trienes and higher acyclic polyenes;
- branched isoprenoid/terpenoid polyenes and long conjugated polyenes.

The source molecules contain only C and H. They contain no ring, aromatic atom, triple bond, or heteroatom. Therefore, do not use Alpha-cleavage, Ring cleavage / Ring rearrangement, or heteroatom-specific neutral-loss explanations for this dataset.

Before assigning a mechanism, identify from source_smiles:

1. total C=C count;
2. terminal versus internal C=C;
3. substitution degree of each C=C;
4. allylic and bis-allylic C-C bonds;
5. isolated pattern C=C-C-C-C=C;
6. conjugated pattern C=C-C=C;
7. cumulated pattern C=C=C;
8. branching adjacent to a C=C;
9. the smallest source-connected carbon fragment capable of producing the ion.

Mechanism priority for acyclic alkenes:

1. Retain "Molecular ion" when product_mz equals the nominal molecular mass. Molecular-ion intensity may increase with conjugation; do not force a fragmentation explanation.

2. For cleavage of a C-C bond adjacent to C=C that retains charge on an allylic or conjugated fragment, use exactly "Allylic cleavage". This is the principal structure-driven alkene rule. These molecules contain no aromatic ring, so "Benzylic cleavage" is forbidden. Combined labels such as "Benzylic / Allylic cleavage" and "Benzylic cleavage / Allylic cleavage" are also forbidden because they are not members of the strict mechanism enumeration.

3. Use "Sigma-bond cleavage" for ordinary non-allylic C-C cleavage, including formation of saturated alkyl ions when no allylic stabilization is involved.

4. Allylic cleavage must be structurally verifiable. The retained product framework must be connected in source_smiles and must preserve or plausibly rearrange into an allylic/conjugated cation. Do not invoke allylic cleavage solely because the product formula is CnH(2n-1)+.

5. Common allylic ion families such as C3H5+ (m/z 41), C4H7+ (55), C5H9+ (69), C6H11+ (83), and their substituted homologues may be formed directly by "Allylic cleavage" when a matching same-carbon allylic fragment exists in source_smiles. They must not automatically be forced to arise by dehydrogenation from CnH(2n+1)+.

5a. Vinyl and highly unsaturated ions require separate treatment. C2H3+ (27), C3H3+ (39), and related ions may arise by direct cleavage of an alkene/polyene framework, molecular-ion H/H2 loss, or supported same-carbon ion evolution. Do not call them allylic unless the retained connectivity is actually allylic. For ethylene, allene, and very small molecules where no C-C cleavage can retain the stated carbon count, prefer molecular-ion-derived dehydrogenation/H loss over an invented smiles_fragment cleavage.

6. For terminal alkenes, explicitly examine the allylic bond and the possibility of charge retention on either the alkene-containing fragment or the complementary alkyl fragment. Do not assume every terminal alkene gives only a terminal-vinyl ion.

7. For internal and highly substituted alkenes, preferentially examine cleavage on both sides of C=C. Prefer the pathway producing the more substituted, allylic, or conjugated cation when this is consistent with the observed intensity, but do not invent a fragment absent from source_smiles.

8. For branched alkenes, cleavage adjacent to an allylic tertiary or highly substituted center may explain an anomalously intense peak. best_fragment must preserve the relevant branch and C=C environment; it must not collapse to an unrelated linear substring.

Polyene-specific rules:

9. For conjugated dienes/polyenes, preserve the longest relevant conjugated segment when assigning a strong product ion. Prefer "Allylic cleavage" for side-chain or terminal cleavage that leaves a conjugated cation. Use "Hydrogen transfer" or "Radical-ion rearrangement" only when bond connectivity and hydrogen balance require it.

10. For isolated polyenes, treat each alkene/allylic region locally. Do not claim whole-chain conjugation across two or more intervening single bonds.

11. For cumulated systems C=C=C, do not treat the central carbon as an ordinary saturated carbon. Cleavage or rearrangement that converts an allenic ion into a more stable allylic/propargylic-like hydrocarbon ion may use exactly "Radical-ion rearrangement" when direct allylic cleavage is insufficient, but the product formula and carbon skeleton must remain valid.

12. For this strictly acyclic dataset, do not use "Retro-Diels-Alder". A conjugated acyclic diene by itself is insufficient evidence for RDA; use allylic cleavage, Sigma-bond cleavage, neutral hydrocarbon loss, or rearrangement as appropriate.

13. "Hydrogen transfer" must not be used as a vague substitute for allylic cleavage. Use it only when explicit hydrogen relocation is needed to reconcile precursor and product formulas.

14. "Neutral loss" may be used only for loss of a clearly defined closed-shell neutral hydrocarbon from a verified precursor_mz. The precursor must therefore use "precursor_mz: <int>", not source_smiles or a full-molecule smiles_fragment. Calculate the exact mass difference and state a chemically meaningful neutral formula internally before output. Loss of CH3 radical (15 Da), C2H5 radical (29 Da), or another alkyl radical from the molecular radical cation is a bond-cleavage event and must be assigned to "Sigma-bond cleavage" or "Allylic cleavage", not "Neutral loss". Never use Neutral loss as a generic explanation for M to every smaller peak.

Direct structure formation versus ion evolution:

15. If a product ion has a source-connected allylic or conjugated structural origin, a structure-driven triplet is valid even when a same-carbon higher-hydrogen peak exists. Do not automatically replace direct alkene cleavage with dehydrogenation.

16. Use "Dehydrogenation / Sequential dehydrogenation" only for genuine same-carbon ion evolution. Its precursor must be formatted as "precursor_mz: <int>" and must be an observed, previous verified ion (including a verified molecular ion). The precursor and product must have the same carbon number, the precursor must have a higher hydrogen count, and its intensity must not be lower than the product intensity. Do not use smiles_fragment directly with this mechanism.

17. For an unsaturated ion, apply this order:
   a. first test a direct allylic/conjugated source fragment;
   b. then test supported same-carbon dehydrogenation;
   c. then consider rearrangement or hydrogen transfer;
   d. otherwise mark insufficient evidence.

18. The weak-peak-to-strong-peak prohibition applies to ion evolution only. It does not prohibit a strong product ion formed independently by structure-driven allylic cleavage.

19. If a peak can reasonably have both a direct allylic-cleavage origin and a supported same-carbon ion-evolution origin, parallel terminal states are allowed only under the existing anomalous_high or carbon_island condition. Output the direct structure-driven route first when it better explains the intensity.

Fragment representation rules for alkenes:

20. smiles_fragment must be the smallest connected source-derived fragment that preserves the decisive C=C/allylic/branching motif. HARD EQUALITY: count every carbon token in smiles_fragment, including carbons inside parentheses, and require fragment_C_count == product_ion_C_count. If the counts differ, the triplet is invalid and must not be output. A larger contextual fragment is never allowed as a substitute for a same-carbon fragment.

21. Never output a fragment with more carbon atoms than the product ion. Never use a disconnected combination of substrings. Never invent a ring, aromatic bond, triple bond, or heteroatom.

22. For a saturated alkyl product ion, the source fragment may contain a C=C only if hydrogen transfer/rearrangement is explicitly required and mass balance is valid. Otherwise select the saturated connected alkyl portion and use Sigma-bond cleavage.

23. For E/Z names whose SMILES lacks / or \\ stereobonds, do not infer stereospecific fragmentation. Treat stereochemistry as unavailable unless explicitly encoded.

24. Nominal-mass formula check for hydrocarbon ions is mandatory: m/z = 12*C + H for singly charged CcHh+. Never assign more hydrogen than allowed by the proposed hydrocarbon skeleton. For precursor_mz evolution, state or infer a compatible precursor formula before accepting the path.

25. A strong M-1 or M-2 peak may be "Dehydrogenation / Sequential dehydrogenation" from the molecular ion only if M is present or the molecular ion is otherwise explicitly supported. An isotope label is forbidden for M-1 or M-2.

26. Before outputting each triplet, execute this mandatory gate in order:
   a. parse product formula CxHy+ and verify m/z = 12*x + y;
   b. for this acyclic hydrocarbon dataset require 0 <= y <= 2*x+2; formulas such as C2H7+ are forbidden;
   c. if precursor is smiles_fragment, count all C tokens and require exactly x carbons;
   d. verify the fragment is a valid connected molecular substructure of source_smiles; raw text substring identity is not required, but molecular graph connectivity is required;
   e. verify the exact mechanism label belongs to the 13-label enumeration;
   f. verify the mechanism is compatible with the precursor type;
   g. for precursor_mz ion evolution, verify the precursor peak exists and does not violate weak-to-strong.
If any item fails, do not improvise. Set triplet_status="invalid" with the applicable invalid_reason.

27. Anti-cheating rule for smiles_fragment:
   - Strings such as C5H10, C4H7, C3H5, C2H3, or any pattern matching C<number>H<number> are molecular formulas, not SMILES, and are absolutely forbidden after "smiles_fragment:".
   - A valid fragment must explicitly encode atom connectivity using SMILES syntax, for example CC, C=C, C=CC, C=C(C)C, or another graph-valid connected structure.
   - Do not copy product_ion formula into precursor.

28. Required decision order for low-mass unsaturated ions:
   a. C3H5+ (41), C4H7+ (55), C5H9+ (69), C6H11+ (83): first test a direct same-carbon allylic substructure. If valid, "Allylic cleavage" is allowed. If not, test a verified stronger same-carbon higher-H precursor.
   b. C3H3+ (39), C4H5+ (53), C5H7+ (67), and analogous more-unsaturated ions: first test a verified same-carbon ion such as 41, 55, or 69 and prefer "Dehydrogenation / Sequential dehydrogenation" when the precursor exists and is at least as intense. Do not label these ions Allylic cleavage merely from their formula.
   c. C2H3+ (27) is a vinyl ion, not an allyl ion. "Allylic cleavage" is forbidden. Use a valid C2 structural cleavage or supported same-carbon ion evolution.
   d. If neither a valid same-carbon structural fragment nor a supported precursor exists, output invalid. Do not invent a conjugated fragment such as C=CC=C when the source contains only isolated double bonds.

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
- What follows must be a valid SMILES for one connected molecular substructure of the input molecule. It must be graph-verifiable from source_smiles. A raw string substring match is not required because branches may be serialized differently.
- The precursor is prohibited from being "structure-driven" / "ion-evolution" / "null" / an empty string

【Precursor output format for structure-driven cases (mandatory)】
The only allowed form is:
"smiles_fragment: <best_fragment>"

Do not append alternatives, comments, lists, pipes, or explanatory text inside triplet[0]. If several equivalent sites exist, select only the single best verifiable fragment. In particular, the following is forbidden:
"smiles_fragment: <fragment> | alternatives: [...]"

【Best_fragment selection rules (mandatory, and must not change your existing reasoning logic)】
- best_fragment must satisfy:
  1) The fragment must be valid SMILES and a connected molecular substructure of source_smiles. It must not be a molecular formula, ion formula, disconnected expression, or invented bond pattern.
  2) Count every uppercase C token in the fragment, including all branch carbons in parentheses. The count must be exactly equal to the carbon number of product_ion. "Chemically compatible", "approximately equal", or a larger contextual fragment is not sufficient.
  3) The structural semantics of the fragment must be consistent with the mechanism in triplet[1] and must not conflict with decision.charge_capture_motif.
- If multiple candidates satisfy the requirements, output only the fragment that best preserves the charge-capture site, C=C environment, branching, and stabilized cation motif.

B) If origin_type="ion-evolution"
- The precursor must begin with "precursor_mz: "
- What follows must be an integer m/z that has already appeared in the spectrum
- Outputting any SMILES fragment is prohibited
- If the intensity-direction constraint is not satisfied, namely weak peak → strong peak, it must fall back to structure-driven explanation

Mechanism/precursor compatibility is mandatory:
- "Molecular ion", "Sigma-bond cleavage", and "Allylic cleavage" may use smiles_fragment when structurally valid.
- "Dehydrogenation / Sequential dehydrogenation", "Isotopic peak", and "Neutral loss" must use precursor_mz.
- A smiles_fragment precursor with Dehydrogenation / Sequential dehydrogenation or Neutral loss is forbidden.

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
    5) "formula_or_carbon_count_mismatch"
    6) "invalid_mechanism_label"
    7) "unsupported_ion_evolution"

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
XII. Structural Upgrade Determination for Anomalously High Peaks in Acyclic Alkenes
────────────────────────────────
Applicable objects: acyclic alkenes and polyenes containing only C/H.

This alkene-specific section completely replaces alkane odd/even heuristics. Do not infer mechanism from m/z parity alone.

For every anomalous_high or carbon_island peak, execute:

1. Test a direct same-carbon structure origin first:
   - allylic cleavage adjacent to C=C;
   - cleavage retaining a conjugated fragment;
   - ordinary Sigma-bond cleavage producing a saturated alkyl ion;
   - branch-assisted charge stabilization.

2. If the product is CnH(2n-1)+, do not automatically call it dehydrogenation. Use exactly "Allylic cleavage" when a connected same-carbon allylic fragment exists. Use dehydrogenation only when a stronger verified same-carbon higher-H precursor exists.

3. For CnH2n+ and other even-m/z hydrocarbon ions, do not automatically call them Sigma-bond cleavage or dehydrogenation. Select the mechanism from verified structure and precursor evidence.

4. When branching explains the anomaly, best_fragment must contain the relevant branch and C=C environment, but it must still contain exactly the same carbon count as product_ion. Never enlarge the fragment merely to show context.

5. The main path must be the valid path that best explains the intensity. A second path is allowed only under Section XIII and must independently pass every formula, carbon-count, precursor-format, and intensity check.

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
{{MOLECULAR_INFORMATION_JSON}}

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
        "--id_min",
        type=int,
        default=None,
        help="Minimum molecule id to process, inclusive.",
    )
    parser.add_argument(
        "--id_max",
        type=int,
        default=None,
        help="Maximum molecule id to process, inclusive.",
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
# 3-A. Per-ion resume, validation, and ordered insertion
# ============================================================

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

    token_attempts = [30000, 50000, 80000]

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

    client = OpenAI(api_key=api_key)

    datas = load_json(args.input_file)
    if not isinstance(datas, list):
        raise ValueError("Input JSON must be a list of molecular records.")

    if args.id_min is not None and args.id_max is not None and args.id_min > args.id_max:
        raise ValueError("--id_min must be less than or equal to --id_max.")

    id_filtered_datas: List[Dict[str, Any]] = []
    for data in datas:
        if not isinstance(data, dict):
            continue
        data_id_int = safe_int(data.get("id"))
        if args.id_min is not None and (data_id_int is None or data_id_int < args.id_min):
            continue
        if args.id_max is not None and (data_id_int is None or data_id_int > args.id_max):
            continue
        id_filtered_datas.append(data)

    if args.limit is None:
        selected_datas = id_filtered_datas[args.start:]
    else:
        selected_datas = id_filtered_datas[args.start:args.start + args.limit]

    # Per-ion resume is automatic whenever output_file exists. --skip_existing is
    # retained for command compatibility but is no longer limited to whole ids.
    results: List[Dict[str, Any]] = []
    if os.path.exists(args.output_file):
        try:
            loaded = load_json(args.output_file)
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
    print(f"Model:                          {args.model}")
    print(f"Output file:                    {args.output_file}")
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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
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

            if parsed is None:
                result_item.setdefault("raw_response_by_mz", {})[str(target_mz)] = raw_text
                result_item.setdefault("parse_ok_by_mz", {})[str(target_mz)] = False
                result_item.setdefault("error_by_mz", {})[str(target_mz)] = error
                print(f"[Failed] id={data_id}, m/z={target_mz}, error={error}")
                # Do not insert an empty/invalid block. A later run will retry it.
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

            if args.save_every > 0 and generated_since_save >= args.save_every:
                save_json(results, args.output_file)
                print(f"[Saved] {args.output_file}")
                generated_since_save = 0

            if args.sleep > 0:
                time.sleep(args.sleep)

    save_json(results, args.output_file)

    # Recount after completion.
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
    print(f"Saved results to:           {args.output_file}")



if __name__ == "__main__":
    main()
