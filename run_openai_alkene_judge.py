# -*- coding: utf-8 -*-
"""
python run_openai_alkene_judge.py ^
  --input_file ./alkene_outputs/openai_gpt41_test_results.json ^
  --output_file ./alkene_outputs/openai_gpt41_test_results_judge_all.json ^
  --corrected_output_file ./alkene_outputs/openai_gpt41_test_results_judge_extract.json ^
  --api_key xx --model gpt-4.1 --temperature 0.0 ^
  --id_min 1 --id_max 666

"""

import argparse
import copy
import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from openai import OpenAI

try:
    from rdkit import Chem
except Exception:
    Chem = None


# ============================================================
# 1. Combined alkene Judge prompt with complete intensity evidence
# ============================================================

SYSTEM_PROMPT = r"""
You are a professional review-style Judge for EI-MS fragmentation triplets of ACYCLIC ALKENES AND POLYENES.

Your responsibility is to verify whether one existing triplet is chemically reasonable and, when it is not reasonable, revise it into a chemically reasonable triplet that still produces the same target product ion.

You are not a free generator. You are not required to find the uniquely correct mechanism. EI fragmentation may admit multiple chemically reasonable pathways.

Your task is:

- accept chemically reasonable triplets;
- revise clearly incorrect, structurally impossible, formula-inconsistent, intensity-inconsistent, or EI-incompatible triplets;
- preserve the same observed target product ion unless its formula and m/z are internally inconsistent;
- make only the minimum necessary correction.

────────────────────────
I. No-delete principle
────────────────────────

The verdict "delete" is forbidden.

You must output only:

- "accept"
- "revise"

corrected_triplet must always be a non-empty list containing exactly three strings.

If the original triplet is unreasonable, revise the precursor and/or mechanism instead of deleting it.

If the original triplet is chemically reasonable, accept it even when another pathway is more common, more canonical, or more preferred.

A preferred alternative is not sufficient reason for revision.

────────────────────────
II. Strict mechanism enumeration
────────────────────────

corrected_triplet[1] must exactly equal one of the following mechanism labels:

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

Do not output combined, abbreviated, renamed, or invented mechanism labels.

For the present acyclic C/H-only alkene dataset:

- Allylic cleavage is applicable when the molecular connectivity supports it.
- Benzylic cleavage is incompatible because no aromatic ring is present.
- Alpha-cleavage is normally incompatible because no heteroatom is present.
- Ring cleavage / Ring rearrangement is incompatible because the molecules are acyclic.
- Retro-Diels–Alder fragmentation must not be assigned merely because an acyclic conjugated diene exists.
- McLafferty rearrangement should not be used without a chemically valid rearrangement framework.

────────────────────────
III. Input evidence and exact field names
────────────────────────

The user payload may contain:

- id
- name
- smiles
- formula
- mw
- compound_class
- complete_spectrum
- spectrum_summary
- current_product_context
- original_triplet
- original_fragment_validation
- original_precursor_status
- all_original_triplets_for_this_molecule
- previous_verified_corrected_triplets_for_this_molecule
- previous_reliable_precursor_triplets_for_this_molecule
- same_carbon_higher_h_precursor_candidates
- structure_fragment_candidates_for_product

Use the actual field name "smiles". Do not assume that a separate source_smiles field exists.

complete_spectrum is the complete EI peak table for the molecule:

{
  "m/z string": relative_intensity
}

current_product_context contains the exact target m/z and its measured relative intensity whenever available.

Each same-carbon higher-hydrogen precursor candidate may contain:

{
  "mz": integer,
  "formula": "CxHy+",
  "intensity": number or null,
  "product_intensity": number or null,
  "same_carbon": true or false,
  "higher_hydrogen": true or false,
  "previously_verified": true or false,
  "violate_weak_to_strong_rule": true or false,
  "is_intensity_eligible": true or false or null,
  "source_triplet": [...]
}

Candidate lists and graph-validation fields are evidence, not unquestionable ground truth. Independently check formula, carbon number, connectivity, mechanism, and intensity.

Never invent a peak intensity or a molecular connection that is not supplied.

────────────────────────
IV. Revision priority
────────────────────────

Judge in this order:

1. Product-ion formula and m/z consistency.
2. Product-ion carbon number versus the molecule.
3. Precursor syntax and precursor support.
4. Structural validity of a smiles_fragment.
5. Mechanism compatibility with the structure and formula change.
6. Precursor/product intensity direction for precursor_mz ion evolution.
7. Only after the above checks, consider whether an alternative explanation is preferable.

Revision is justified only when one of items 1-6 has a clear problem.

Do not revise a valid triplet merely because item 7 suggests a more common pathway.

────────────────────────
V. Product formula and nominal mass
────────────────────────

For a hydrocarbon ion CxHy+, verify:

m/z = 12*x + y

For the current acyclic hydrocarbon dataset, a proposed product formula should satisfy a chemically plausible hydrogen range. A formula such as C2H7+ is invalid.

If the original formula and m/z disagree, preserve the observed target m/z and revise the formula only when a unique chemically reasonable hydrocarbon formula can be determined.

Do not change an internally consistent target product ion merely because another formula is imaginable.

────────────────────────
VI. Precursor formats
────────────────────────

The precursor must use one of these formats:

1. Structure-driven:

"smiles_fragment: <fragment>"

or

"smiles_fragment: <best_fragment> | alternatives: [<alt1>, <alt2>, ...]"

2. Ion evolution:

"precursor_mz: <integer>"

A molecular formula such as C5H10, C4H7, C3H5, or C2H3 is not a SMILES fragment.

For smiles_fragment:

- the best fragment must be a valid connected molecular graph;
- it must be a genuine substructure of the input smiles;
- it must preserve relevant bond orders and branching required by the proposed mechanism;
- it must not invent a ring, triple bond, heteroatom, or conjugation absent from the molecule;
- alternatives, when present, must satisfy the same requirements.

Carbon-count rule:

- A fragment with fewer carbon atoms than the product ion is invalid.
- For direct retained-fragment mechanisms such as Sigma-bond cleavage and Allylic cleavage, the retained fragment should normally have the same carbon number as the product ion.
- A larger connected source fragment may be accepted for Hydrogen transfer, Radical-ion rearrangement, McLafferty rearrangement, or another mechanism only when the larger fragment is mechanistically necessary and the retained product carbon skeleton remains explainable.
- Do not revise a reasonable fragment solely because another exact-carbon serialization exists.

Raw text substring identity is not required when graph validation confirms the same connected substructure with an equivalent SMILES serialization.

────────────────────────
VII. Mechanism and precursor compatibility
────────────────────────

The following mechanisms normally require precursor_mz because they describe evolution from an observed ion:

- Isotopic peak
- Neutral loss
- Dehydrogenation / Sequential dehydrogenation

The following normally use smiles_fragment for direct structure-driven formation:

- Molecular ion
- Sigma-bond cleavage
- Allylic cleavage
- Hydrogen transfer
- Radical-ion rearrangement

A smiles_fragment combined with Dehydrogenation / Sequential dehydrogenation is normally invalid because dehydrogenation should identify the actual precursor ion.

Neutral loss must identify a defined closed-shell neutral loss and a chemically supported precursor_mz. Generic C-C cleavage or loss of an alkyl radical is not Neutral loss.

────────────────────────
VIII. Complete-spectrum intensity rule
────────────────────────

Use complete_spectrum and current_product_context to compare exact measured intensities.

For a precursor_mz pathway, check:

- the precursor peak exists in complete_spectrum;
- the product peak exists in complete_spectrum;
- the precursor and product intensities are known;
- the precursor formula is compatible with the current product;
- the mechanism is chemically valid.

Weak-peak-to-strong-peak prohibition:

If precursor_intensity < product_intensity, do not use that precursor_mz as the parent of the stronger product peak.

This restriction applies to precursor_mz-based ion evolution. It does not prohibit an independently formed strong structure-driven ion.

A previous corrected triplet, a same-carbon relationship, or a higher hydrogen count does not override an explicit weak-to-strong violation.

For a NEW precursor_mz pathway proposed during revision, exact precursor and product intensities must be available and precursor_intensity must be at least product_intensity.

For an EXISTING precursor_mz pathway:

- apply the same intensity rule when both intensities are supplied;
- if an explicit weak-to-strong violation exists, revise it;
- if intensity is genuinely unavailable after checking complete_spectrum and the supplied context, do not invent it and do not create another unsupported precursor_mz pathway.

────────────────────────
IX. Same-carbon dehydrogenation
────────────────────────

Use Dehydrogenation / Sequential dehydrogenation only when:

- precursor and product preserve carbon number;
- the precursor has more hydrogen than the product;
- the precursor peak exists and is chemically supported;
- the intensity direction is allowed.

Multi-step hydrogen loss within the same carbon family may be reasonable.

Do not force the highest-hydrogen precursor when the original precursor is already valid.

Do not automatically reinterpret every CnH(2n-1)+ ion as a dehydrogenation product. In alkene spectra, C3H5+, C4H7+, C5H9+, C6H11+, and higher homologues may form directly by Allylic cleavage.

A directly verifiable allylic/conjugated formation pathway must not be replaced by dehydrogenation merely because a same-carbon higher-hydrogen ion exists.

────────────────────────
X. Alkene and polyene structural rules
────────────────────────

First inspect the input smiles for:

- terminal, internal, or substituted C=C;
- isolated double bonds;
- conjugated C=C-C=C regions;
- cumulated C=C=C regions;
- allylic C-C bonds;
- branching adjacent to unsaturation;
- connected conjugated or resonance-stabilized retained fragments.

Allylic cleavage:

Use exactly "Allylic cleavage" when cleavage occurs at a bond allylic to C=C and the retained carbon skeleton can support the stated allylic or conjugated ion.

Common direct allylic products include:

- C3H5+ (m/z 41)
- C4H7+ (m/z 55)
- C5H9+ (m/z 69)
- C6H11+ (m/z 83)

If the original Allylic cleavage is structurally valid, accept it even when another ion-evolution explanation is possible.

If the original mechanism is Sigma-bond cleavage but the same valid fragment clearly describes a specific allylic cleavage, revise minimally by changing only the mechanism.

Do not call a saturated fragment allylic merely because the product formula resembles an allylic series.

C2H3+ (m/z 27) is a vinyl ion, not an allylic ion.

Highly unsaturated ions such as C3H3+, C4H5+, and C5H7+ must not be called allylic solely from formula. Require valid connectivity or use a supported ion-evolution/rearrangement pathway.

Sigma-bond cleavage:

Use Sigma-bond cleavage for ordinary non-allylic C-C cleavage and formation of a structurally supported hydrocarbon cation when allylic stabilization is not the defining feature.

Conjugated polyenes:

Accept a cleavage that retains a connected conjugated cation. Do not claim conjugation across two or more intervening saturated single bonds.

Isolated polyenes:

Evaluate each allylic region locally rather than treating the entire molecule as one continuous conjugated system.

Cumulated systems:

For C=C=C systems, do not treat the central carbon as an ordinary saturated carbon. Radical-ion rearrangement may be reasonable only when a genuine allenic reorganization is needed and formula/connectivity support it.

Branched alkenes:

When branching is essential to stabilization, the selected fragment should preserve the decisive branch and alkene environment.

Stereochemistry:

If E/Z information appears only in the name but is not encoded by / or \\ in smiles, do not invent stereospecific fragmentation.

────────────────────────
XI. Low-mass hierarchy without excessive dehydrogenation bias
────────────────────────

For C3H3+ (39), C4H5+ (53), C5H7+ (67), and analogous highly unsaturated ions:

- inspect all verified same-carbon higher-hydrogen candidates;
- apply the exact intensity gate;
- do not use a weaker precursor;
- when no eligible precursor exists, inspect a valid structure-driven or rearrangement pathway.

For C3H5+ (41), C4H7+ (55), C5H9+ (69), C6H11+ (83), and higher allylic homologues:

- retain a valid direct Allylic cleavage pathway;
- do not replace it with dehydrogenation when the proposed precursor is weaker than the product;
- do not replace it merely because a same-carbon higher-H ion is listed.

For C2H3+ (27):

- do not label it Allylic cleavage;
- use a valid C2 structure-driven explanation, supported same-carbon ion evolution, or a necessary rearrangement based on the actual evidence.

────────────────────────
XII. Context anti-poisoning
────────────────────────

Previous corrected triplets are evidence, not unquestionable ground truth.

Before using a previous product ion as precursor_mz evidence, verify:

- its m/z matches the claimed precursor_mz;
- its formula is compatible with the current product;
- carbon number is conserved when required;
- its measured intensity does not violate the weak-to-strong rule;
- the previous triplet was not an automatic fallback or otherwise marked unreliable.

Do not propagate an erroneous or intensity-ineligible precursor into later decisions.

────────────────────────
XIII. Minimal revision rule
────────────────────────

If only the fragment is wrong, change only corrected_triplet[0].

If only the mechanism is wrong, change only corrected_triplet[1].

If the precursor_mz is wrong, replace it with a valid precursor or a valid structure-driven path while preserving the target product ion.

If two explanations are chemically reasonable, accept the original reasonable explanation.

Use Hydrogen transfer or Radical-ion rearrangement only when direct cleavage and eligible ion evolution are insufficient and the formula/connectivity genuinely require the more complex mechanism.

────────────────────────
XIV. Required reason content
────────────────────────

The reason field should state, when relevant:

- the product formula/mass check;
- the fragment carbon count and graph-validation result;
- the exact precursor and product intensities;
- whether weak-to-strong is violated;
- why the selected mechanism matches the alkene structure.

Do not claim that intensity is unavailable before checking complete_spectrum.

Do not claim that a fragment is absent based only on candidate-list absence.

────────────────────────
XV. Strict output format
────────────────────────

Output strict JSON only, with no markdown and no explanatory text outside the JSON object.

The JSON must follow exactly this structure:

{
  "id": ...,
  "name": "...",
  "smiles": "...",
  "formula": "...",
  "mw": "...",
  "original_triplet": ["...", "...", "..."],
  "verdict": "accept | revise",
  "reason": "...",
  "corrected_triplet": ["...", "...", "..."],
  "overall_quality": "...",
  "major_errors": [],
  "recommended_strategy": "..."
}

Rules:

- verdict must be only "accept" or "revise";
- if verdict="accept", corrected_triplet must exactly equal original_triplet;
- if verdict="revise", corrected_triplet must contain the minimally revised valid triplet;
- corrected_triplet must contain exactly three strings;
- corrected_triplet must preserve the target product ion unless its formula/mz is internally inconsistent;
- corrected_triplet[1] must exactly match one mechanism from the strict enumeration;
- major_errors must be an array; use [] when no major error exists;
- never output verdict="delete";
- never output corrected_triplet=[].
"""


ALLOWED_MECHANISMS: Set[str] = {
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

ION_RE = re.compile(
    r"([A-Z][A-Za-z0-9?]*(?:\+|＋)?(?:[•·])?)\s*\(m/z\s*(\d+)\)"
)
FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
FORMULA_LIKE_FRAGMENT_RE = re.compile(r"^C\d*H\d+$", flags=re.IGNORECASE)


# ============================================================
# 2. Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Full generator JSON or extracted-triplet JSON."
    )
    parser.add_argument(
        "--spectrum_source_file", type=str, default=None,
        help=(
            "Optional full generator JSON used to restore input_spectrum by id "
            "when input_file is an extracted-triplet file."
        ),
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Full output JSON containing one Judge result per triplet."
    )
    parser.add_argument(
        "--corrected_output_file", type=str, required=True,
        help="Grouped output JSON containing final corrected_triplet lists."
    )

    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max_tokens", type=int, default=4096,
        help="Initial output-token budget; retry attempts increase this budget."
    )

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--id_min", type=int, default=None)
    parser.add_argument("--id_max", type=int, default=None)

    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--semantic_retry", type=int, default=2)
    parser.add_argument("--use_json_mode", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--include_unsafe_in_corrected", action="store_true",
        help="Retained for command compatibility; no-delete output always keeps valid triplets."
    )

    return parser.parse_args()


# ============================================================
# 3. Basic JSON, formula, ion, and spectrum utilities
# ============================================================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def normalize_triplet(triplet: Any) -> Optional[List[str]]:
    if isinstance(triplet, list) and len(triplet) == 3:
        return [str(triplet[0]), str(triplet[1]), str(triplet[2])]
    return None


def parse_formula_counts(formula: str) -> Dict[str, int]:
    if not isinstance(formula, str):
        return {}

    cleaned = (
        formula.replace("+", "")
        .replace("＋", "")
        .replace("•", "")
        .replace("·", "")
        .replace("?", "")
    )

    counts: Dict[str, int] = defaultdict(int)
    for element, number in FORMULA_RE.findall(cleaned):
        counts[element] += int(number) if number else 1
    return dict(counts)


def nominal_mass(counts: Dict[str, int]) -> int:
    masses = {
        "H": 1, "C": 12, "N": 14, "O": 16, "F": 19,
        "P": 31, "S": 32, "Cl": 35, "Br": 79, "I": 127,
    }
    return sum(masses.get(element, 0) * count for element, count in counts.items())


def parse_ion(product_ion: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(product_ion, str):
        return None

    match = ION_RE.search(product_ion)
    if not match:
        return None

    formula = match.group(1)
    mz = int(match.group(2))
    counts = parse_formula_counts(formula)
    return {
        "formula": formula,
        "mz": mz,
        "counts": counts,
        "carbon": counts.get("C", 0),
        "hydrogen": counts.get("H", 0),
        "nominal_mass": nominal_mass(counts),
        "has_unknown": "?" in formula,
    }


def parse_precursor_mz(precursor: Any) -> Optional[int]:
    if not isinstance(precursor, str):
        return None
    match = re.fullmatch(r"\s*precursor_mz:\s*(\d+)\s*", precursor)
    return int(match.group(1)) if match else None


def normalize_spectrum(spectrum: Any) -> Dict[str, float]:
    if not isinstance(spectrum, dict):
        return {}

    normalized: Dict[str, float] = {}
    for mz_raw, intensity_raw in spectrum.items():
        mz = safe_int(mz_raw)
        intensity = safe_float(intensity_raw)
        if mz is None or intensity is None:
            continue
        normalized[str(mz)] = intensity

    return dict(sorted(normalized.items(), key=lambda kv: int(kv[0])))


def get_record_spectrum(record: Dict[str, Any]) -> Dict[str, float]:
    for key in ("input_spectrum", "spectrum"):
        spectrum = normalize_spectrum(record.get(key))
        if spectrum:
            return spectrum
    return {}


def get_peak_intensity(spectrum: Dict[str, float], mz: Optional[int]) -> Optional[float]:
    if mz is None:
        return None
    return spectrum.get(str(int(mz)))


def format_number(value: Optional[float]) -> Optional[Any]:
    if value is None:
        return None
    if float(value).is_integer():
        return int(value)
    return value


def build_spectrum_summary(spectrum: Dict[str, float]) -> Dict[str, Any]:
    if not spectrum:
        return {
            "peak_count": 0,
            "base_peak_mz": None,
            "base_peak_intensity": None,
            "min_mz": None,
            "max_mz": None,
        }

    pairs = [(int(mz), intensity) for mz, intensity in spectrum.items()]
    base_mz, base_intensity = max(pairs, key=lambda item: item[1])
    return {
        "peak_count": len(pairs),
        "base_peak_mz": base_mz,
        "base_peak_intensity": format_number(base_intensity),
        "min_mz": min(mz for mz, _ in pairs),
        "max_mz": max(mz for mz, _ in pairs),
    }


def build_product_context(product_ion: str, spectrum: Dict[str, float]) -> Dict[str, Any]:
    ion = parse_ion(product_ion)
    if ion is None:
        return {
            "product_ion": product_ion,
            "product_mz": None,
            "product_formula": None,
            "product_intensity": None,
            "intensity_rank": None,
            "is_base_peak": None,
            "relative_to_base_peak": None,
            "neighboring_peaks": {},
        }

    product_mz = ion["mz"]
    product_intensity = get_peak_intensity(spectrum, product_mz)

    sorted_by_intensity = sorted(
        ((int(mz), intensity) for mz, intensity in spectrum.items()),
        key=lambda item: (-item[1], item[0]),
    )
    rank = None
    if product_intensity is not None:
        for idx, (mz, _) in enumerate(sorted_by_intensity, start=1):
            if mz == product_mz:
                rank = idx
                break

    summary = build_spectrum_summary(spectrum)
    base_intensity = safe_float(summary.get("base_peak_intensity"))
    relative_to_base = None
    if product_intensity is not None and base_intensity not in (None, 0):
        relative_to_base = product_intensity / base_intensity

    neighbor_mzs = [
        product_mz - 14, product_mz - 2, product_mz - 1,
        product_mz + 1, product_mz + 2, product_mz + 14,
    ]
    neighbors: Dict[str, Any] = {}
    for mz in neighbor_mzs:
        intensity = get_peak_intensity(spectrum, mz)
        if intensity is not None:
            neighbors[str(mz)] = format_number(intensity)

    return {
        "product_ion": product_ion,
        "product_mz": product_mz,
        "product_formula": ion["formula"],
        "product_carbon": ion["carbon"],
        "product_hydrogen": ion["hydrogen"],
        "nominal_mass_from_formula": ion["nominal_mass"],
        "formula_mz_consistent": (
            None if ion["has_unknown"] else ion["nominal_mass"] == product_mz
        ),
        "product_intensity": format_number(product_intensity),
        "intensity_rank": rank,
        "is_base_peak": (
            None if product_intensity is None
            else product_mz == summary.get("base_peak_mz")
        ),
        "relative_to_base_peak": relative_to_base,
        "neighboring_peaks": neighbors,
    }


# ============================================================
# 4. Spectrum restoration from full generator output
# ============================================================

def build_record_index(records: Any) -> Dict[Any, Dict[str, Any]]:
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        return {}

    index: Dict[Any, Dict[str, Any]] = {}
    for record in records:
        if isinstance(record, dict) and "id" in record:
            index[record.get("id")] = record
            try:
                index[int(record.get("id"))] = record
            except Exception:
                pass
    return index


def merge_spectrum_source(
    records: List[Dict[str, Any]],
    spectrum_source_records: Optional[Any],
) -> None:
    if spectrum_source_records is None:
        return

    source_index = build_record_index(spectrum_source_records)
    for record in records:
        if not isinstance(record, dict):
            continue
        if get_record_spectrum(record):
            continue

        source = source_index.get(record.get("id"))
        if source is None:
            try:
                source = source_index.get(int(record.get("id")))
            except Exception:
                source = None
        if not isinstance(source, dict):
            continue

        source_spectrum = get_record_spectrum(source)
        if source_spectrum:
            record["input_spectrum"] = {
                mz: format_number(intensity)
                for mz, intensity in source_spectrum.items()
            }

        for key in ("compound_class",):
            if key not in record and key in source:
                record[key] = source.get(key)


# ============================================================
# 5. SMILES fragment parsing and optional RDKit graph validation
# ============================================================

def count_c_in_smiles_text(smiles: str) -> int:
    if not isinstance(smiles, str):
        return 0
    return len(re.findall(r"C(?!l)", smiles))


def parse_smiles_fragment_precursor(precursor: Any) -> Tuple[Optional[str], List[str]]:
    if not isinstance(precursor, str) or not precursor.startswith("smiles_fragment:"):
        return None, []

    body = precursor.split("smiles_fragment:", 1)[1].strip()
    parts = body.split("|", 1)
    best = parts[0].strip() if parts else None
    alternatives: List[str] = []

    if len(parts) == 2:
        match = re.search(r"alternatives:\s*\[(.*?)\]", parts[1])
        if match:
            alternatives = [
                value.strip().strip("'\"")
                for value in match.group(1).split(",")
                if value.strip()
            ]

    return best or None, alternatives


def validate_fragment_graph(
    parent_smiles: str,
    precursor: str,
    product_ion: str,
) -> Dict[str, Any]:
    best, alternatives = parse_smiles_fragment_precursor(precursor)
    fragments = [fragment for fragment in [best] + alternatives if fragment]
    product = parse_ion(product_ion)
    target_c = product["carbon"] if product else None

    if not fragments:
        return {
            "status": "not_applicable" if not precursor.startswith("smiles_fragment:") else "invalid",
            "reason": "No smiles_fragment was supplied.",
            "target_product_carbon": target_c,
            "fragments": [],
        }

    if Chem is None:
        details = []
        for fragment in fragments:
            carbon_count = count_c_in_smiles_text(fragment)
            details.append({
                "fragment": fragment,
                "parseable": None,
                "connected": None,
                "is_parent_substructure": None,
                "carbon_count": carbon_count,
                "exact_carbon_match": (
                    None if target_c is None else carbon_count == target_c
                ),
                "has_at_least_product_carbon": (
                    None if target_c is None else carbon_count >= target_c
                ),
                "formula_like_not_smiles": bool(FORMULA_LIKE_FRAGMENT_RE.fullmatch(fragment)),
            })
        return {
            "status": "unknown",
            "reason": "RDKit is unavailable; only text-level carbon counting was performed.",
            "target_product_carbon": target_c,
            "fragments": details,
        }

    parent_mol = Chem.MolFromSmiles(parent_smiles) if parent_smiles else None
    if parent_mol is None:
        return {
            "status": "unknown",
            "reason": "The parent smiles could not be parsed by RDKit.",
            "target_product_carbon": target_c,
            "fragments": [],
        }

    details: List[Dict[str, Any]] = []
    for fragment in fragments:
        formula_like = bool(FORMULA_LIKE_FRAGMENT_RE.fullmatch(fragment))
        fragment_mol = None if formula_like else Chem.MolFromSmiles(fragment)
        parseable = fragment_mol is not None
        connected = None
        substructure = None
        carbon_count = count_c_in_smiles_text(fragment)
        contains_double_bond = None

        if fragment_mol is not None:
            connected = len(Chem.GetMolFrags(fragment_mol)) == 1
            carbon_count = sum(
                1 for atom in fragment_mol.GetAtoms() if atom.GetAtomicNum() == 6
            )
            contains_double_bond = any(
                bond.GetBondType() == Chem.BondType.DOUBLE
                for bond in fragment_mol.GetBonds()
            )
            substructure = bool(connected and parent_mol.HasSubstructMatch(fragment_mol))

        details.append({
            "fragment": fragment,
            "parseable": parseable,
            "connected": connected,
            "is_parent_substructure": substructure,
            "carbon_count": carbon_count,
            "exact_carbon_match": (
                None if target_c is None else carbon_count == target_c
            ),
            "has_at_least_product_carbon": (
                None if target_c is None else carbon_count >= target_c
            ),
            "contains_double_bond": contains_double_bond,
            "formula_like_not_smiles": formula_like,
        })

    best_detail = details[0]
    best_valid = (
        best_detail.get("parseable") is True
        and best_detail.get("connected") is True
        and best_detail.get("is_parent_substructure") is True
        and not best_detail.get("formula_like_not_smiles")
        and (
            target_c is None
            or int(best_detail.get("carbon_count", 0)) >= int(target_c)
        )
    )

    return {
        "status": "valid" if best_valid else "invalid",
        "reason": (
            "The best fragment is a connected RDKit substructure with sufficient carbon count."
            if best_valid
            else "The best fragment failed parsing, connectivity, parent-substructure, formula-like, or carbon-count validation."
        ),
        "target_product_carbon": target_c,
        "fragments": details,
    }


def enumerate_connected_carbon_subgraphs(
    parent_smiles: str,
    target_c: int,
    max_candidates: int = 12,
    max_states: int = 10000,
) -> List[Dict[str, Any]]:
    if not parent_smiles or target_c <= 0:
        return []

    if Chem is None:
        candidates: List[Dict[str, Any]] = []
        n = len(parent_smiles)
        seen: Set[str] = set()
        for start in range(n):
            for end in range(start + 1, n + 1):
                fragment = parent_smiles[start:end]
                if fragment in seen:
                    continue
                if count_c_in_smiles_text(fragment) != target_c:
                    continue
                if not fragment.startswith("C"):
                    continue
                if fragment.count("(") != fragment.count(")"):
                    continue
                seen.add(fragment)
                candidates.append({
                    "smiles": fragment,
                    "carbon_count": target_c,
                    "contains_double_bond": "=" in fragment,
                    "graph_validation": "text_candidate_only",
                })
                if len(candidates) >= max_candidates:
                    return candidates
        return candidates

    mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None:
        return []

    carbon_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
    if target_c > len(carbon_atoms):
        return []

    carbon_set = set(carbon_atoms)
    adjacency: Dict[int, Set[int]] = {}
    for atom_idx in carbon_atoms:
        adjacency[atom_idx] = {
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            if neighbor.GetIdx() in carbon_set
        }

    completed: Set[frozenset] = set()
    visited: Set[frozenset] = set()
    stack: List[frozenset] = [frozenset([idx]) for idx in carbon_atoms]
    states = 0

    while stack and states < max_states and len(completed) < max_candidates * 20:
        subset = stack.pop()
        states += 1
        if subset in visited:
            continue
        visited.add(subset)

        if len(subset) == target_c:
            completed.add(subset)
            continue
        if len(subset) > target_c:
            continue

        frontier: Set[int] = set()
        for atom_idx in subset:
            frontier.update(adjacency.get(atom_idx, set()))
        frontier.difference_update(subset)

        for next_atom in sorted(frontier, reverse=True):
            new_subset = frozenset(set(subset) | {next_atom})
            if len(new_subset) <= target_c and new_subset not in visited:
                stack.append(new_subset)

    candidates_by_smiles: Dict[str, Dict[str, Any]] = {}
    for subset in completed:
        try:
            fragment_smiles = Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=sorted(subset),
                canonical=True,
                isomericSmiles=True,
            )
        except Exception:
            continue
        if not fragment_smiles or "." in fragment_smiles:
            continue

        fragment_mol = Chem.MolFromSmiles(fragment_smiles)
        if fragment_mol is None:
            continue
        contains_double = any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            for bond in fragment_mol.GetBonds()
        )
        branch_score = sum(
            1 for atom in fragment_mol.GetAtoms()
            if atom.GetAtomicNum() == 6 and atom.GetDegree() >= 3
        )
        candidates_by_smiles[fragment_smiles] = {
            "smiles": fragment_smiles,
            "carbon_count": target_c,
            "contains_double_bond": contains_double,
            "branch_score": branch_score,
            "graph_validation": "rdkit_connected_parent_subgraph",
        }

    candidates = list(candidates_by_smiles.values())
    candidates.sort(
        key=lambda item: (
            -int(bool(item.get("contains_double_bond"))),
            -int(item.get("branch_score", 0)),
            len(str(item.get("smiles", ""))),
            str(item.get("smiles", "")),
        )
    )
    return candidates[:max_candidates]


def get_structure_fragment_candidates(parent_smiles: str, product_ion: str) -> List[Dict[str, Any]]:
    product = parse_ion(product_ion)
    if product is None or product["carbon"] <= 0:
        return []
    return enumerate_connected_carbon_subgraphs(parent_smiles, product["carbon"])


# ============================================================
# 6. Previous-ion and intensity-aware candidate context
# ============================================================

def find_previous_ions_by_mz(
    previous_triplets: Sequence[List[str]],
    mz: int,
) -> List[Dict[str, Any]]:
    ions: List[Dict[str, Any]] = []
    for triplet in previous_triplets:
        normalized = normalize_triplet(triplet)
        if normalized is None:
            continue
        ion = parse_ion(normalized[2])
        if ion is not None and ion["mz"] == mz:
            ions.append({**ion, "source_triplet": normalized})
    return ions


def enrich_previous_triplets(
    previous_triplets: Sequence[List[str]],
    spectrum: Dict[str, float],
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for triplet in previous_triplets:
        normalized = normalize_triplet(triplet)
        if normalized is None:
            continue
        ion = parse_ion(normalized[2])
        enriched.append({
            "triplet": normalized,
            "product_mz": None if ion is None else ion["mz"],
            "product_formula": None if ion is None else ion["formula"],
            "product_carbon": None if ion is None else ion["carbon"],
            "product_hydrogen": None if ion is None else ion["hydrogen"],
            "product_intensity": (
                None if ion is None
                else format_number(get_peak_intensity(spectrum, ion["mz"]))
            ),
        })
    return enriched


def get_same_carbon_higher_h_precursor_candidates(
    previous_triplets: Sequence[List[str]],
    product_ion: str,
    spectrum: Dict[str, float],
) -> List[Dict[str, Any]]:
    product = parse_ion(product_ion)
    if product is None:
        return []

    product_intensity = get_peak_intensity(spectrum, product["mz"])
    candidates: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, str]] = set()

    for triplet in previous_triplets:
        normalized = normalize_triplet(triplet)
        if normalized is None:
            continue
        ion = parse_ion(normalized[2])
        if ion is None:
            continue
        if ion["carbon"] != product["carbon"] or ion["hydrogen"] <= product["hydrogen"]:
            continue

        key = (ion["mz"], ion["formula"])
        if key in seen:
            continue
        seen.add(key)

        precursor_intensity = get_peak_intensity(spectrum, ion["mz"])
        weak_to_strong = None
        eligible = None
        if precursor_intensity is not None and product_intensity is not None:
            weak_to_strong = precursor_intensity < product_intensity
            eligible = not weak_to_strong

        candidates.append({
            "mz": ion["mz"],
            "formula": ion["formula"],
            "carbon": ion["carbon"],
            "hydrogen": ion["hydrogen"],
            "hydrogen_difference_to_product": ion["hydrogen"] - product["hydrogen"],
            "intensity": format_number(precursor_intensity),
            "product_mz": product["mz"],
            "product_formula": product["formula"],
            "product_intensity": format_number(product_intensity),
            "same_carbon": True,
            "higher_hydrogen": True,
            "previously_verified": True,
            "violate_weak_to_strong_rule": weak_to_strong,
            "is_intensity_eligible": eligible,
            "source_triplet": normalized,
        })

    candidates.sort(
        key=lambda item: (
            int(item["hydrogen_difference_to_product"]),
            int(item["mz"]),
        )
    )
    return candidates


def get_original_precursor_status(
    triplet: List[str],
    previous_reliable_triplets: Sequence[List[str]],
    spectrum: Dict[str, float],
) -> Dict[str, Any]:
    normalized = normalize_triplet(triplet)
    if normalized is None:
        return {"is_supported": False, "reason": "Invalid original triplet format."}

    precursor, mechanism, product_ion = normalized
    precursor_mz = parse_precursor_mz(precursor)
    product = parse_ion(product_ion)

    if precursor_mz is None or product is None:
        return {
            "is_supported": False,
            "original_precursor_mz": precursor_mz,
            "mechanism": mechanism,
            "product_ion": product_ion,
            "precursor_intensity": None,
            "product_intensity": (
                None if product is None
                else format_number(get_peak_intensity(spectrum, product["mz"]))
            ),
            "violate_weak_to_strong_rule": None,
            "supporting_previous_ions": [],
        }

    previous_ions = find_previous_ions_by_mz(previous_reliable_triplets, precursor_mz)
    compatible_ions = [
        ion for ion in previous_ions
        if ion["carbon"] == product["carbon"]
        and ion["hydrogen"] > product["hydrogen"]
    ]

    precursor_intensity = get_peak_intensity(spectrum, precursor_mz)
    product_intensity = get_peak_intensity(spectrum, product["mz"])
    weak_to_strong = None
    if precursor_intensity is not None and product_intensity is not None:
        weak_to_strong = precursor_intensity < product_intensity

    supported = bool(compatible_ions)
    intensity_eligible = None if weak_to_strong is None else not weak_to_strong

    return {
        "is_supported": supported,
        "original_precursor_mz": precursor_mz,
        "precursor_peak_exists": str(precursor_mz) in spectrum,
        "precursor_intensity": format_number(precursor_intensity),
        "product_mz": product["mz"],
        "product_intensity": format_number(product_intensity),
        "same_carbon_higher_hydrogen_supported": bool(compatible_ions),
        "violate_weak_to_strong_rule": weak_to_strong,
        "is_intensity_eligible": intensity_eligible,
        "mechanism": mechanism,
        "product_ion": product_ion,
        "supporting_previous_ions": compatible_ions,
    }


# ============================================================
# 7. Prompt construction
# ============================================================

def build_user_prompt(
    record: Dict[str, Any],
    triplet: List[str],
    triplet_index: int,
    previous_verified_triplets: Sequence[List[str]],
    previous_reliable_triplets: Sequence[List[str]],
) -> str:
    product_ion = triplet[2]
    spectrum = get_record_spectrum(record)
    product_context = build_product_context(product_ion, spectrum)

    payload = {
        "id": record.get("id"),
        "name": record.get("name", ""),
        "smiles": record.get("smiles", ""),
        "formula": record.get("formula", ""),
        "mw": record.get("mw", ""),
        "compound_class": record.get("compound_class", "alkene"),
        "triplet_index": triplet_index,
        "original_triplet": triplet,
        "complete_spectrum": {
            mz: format_number(intensity) for mz, intensity in spectrum.items()
        },
        "spectrum_summary": build_spectrum_summary(spectrum),
        "current_product_context": product_context,
        "original_fragment_validation": validate_fragment_graph(
            record.get("smiles", ""), triplet[0], product_ion
        ),
        "original_precursor_status": get_original_precursor_status(
            triplet, previous_reliable_triplets, spectrum
        ),
        "all_original_triplets_for_this_molecule": record.get(
            "_judge_triplets_flat", record.get("triplets", [])
        ),
        "previous_verified_corrected_triplets_for_this_molecule": enrich_previous_triplets(
            previous_verified_triplets, spectrum
        ),
        "previous_reliable_precursor_triplets_for_this_molecule": enrich_previous_triplets(
            previous_reliable_triplets, spectrum
        ),
        "same_carbon_higher_h_precursor_candidates": get_same_carbon_higher_h_precursor_candidates(
            previous_reliable_triplets, product_ion, spectrum
        ),
        "structure_fragment_candidates_for_product": get_structure_fragment_candidates(
            record.get("smiles", ""), product_ion
        ),
    }

    return (
        "Judge the following existing EI-MS fragmentation triplet for the supplied "
        "acyclic alkene/polyene molecule.\n"
        "Use the complete measured spectrum and the exact current product intensity. "
        "For every precursor_mz pathway, explicitly compare precursor and product "
        "intensities and enforce the weak-peak-to-strong-peak prohibition.\n"
        "Accept any chemically reasonable original pathway. Revise only a clear "
        "formula, precursor, fragment, intensity-direction, or mechanism error.\n"
        "Return strict JSON only using the exact schema from the system prompt. "
        "verdict must be accept or revise; delete is forbidden; corrected_triplet "
        "must be a non-empty 3-element list.\n\n"
        "[Triplet judgment input]\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


# ============================================================
# 8. API and strict output validation
# ============================================================

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
    if start >= 0 and end > start:
        return json.loads(raw[start:end + 1])

    raise ValueError(f"Cannot parse JSON from response:\n{raw[:1000]}")


def is_product_ion_mass_consistent(product_ion: Any) -> bool:
    ion = parse_ion(product_ion)
    return bool(
        ion is not None
        and not ion.get("has_unknown")
        and ion["nominal_mass"] == ion["mz"]
    )


def product_ion_change_allowed(
    original_product: Any,
    corrected_product: Any,
    molecule_formula: str = "",
) -> bool:
    original = parse_ion(original_product)
    corrected = parse_ion(corrected_product)
    if original is None or corrected is None:
        return False
    if original["mz"] != corrected["mz"]:
        return False
    if is_product_ion_mass_consistent(original_product):
        return False
    if not is_product_ion_mass_consistent(corrected_product):
        return False

    molecule_c = parse_formula_counts(molecule_formula).get("C", 0)
    return not (molecule_c > 0 and corrected["carbon"] > molecule_c)


def enforce_current_record_metadata(
    parsed: Dict[str, Any],
    record: Dict[str, Any],
    original_triplet: List[str],
) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        return parsed
    parsed["id"] = record.get("id")
    parsed["name"] = record.get("name", "")
    parsed["smiles"] = record.get("smiles", "")
    parsed["formula"] = record.get("formula", "")
    parsed["mw"] = record.get("mw", "")
    parsed["original_triplet"] = original_triplet
    return parsed


def validate_judge_output(parsed: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    required_keys = [
        "id", "name", "smiles", "formula", "mw", "original_triplet",
        "verdict", "reason", "corrected_triplet", "overall_quality",
        "major_errors", "recommended_strategy",
    ]

    for key in required_keys:
        if key not in parsed:
            warnings.append(f"Missing key: {key}")

    verdict = parsed.get("verdict")
    if verdict not in {"accept", "revise"}:
        warnings.append(f"Invalid or forbidden verdict: {verdict}")

    original = normalize_triplet(parsed.get("original_triplet"))
    corrected = normalize_triplet(parsed.get("corrected_triplet"))
    if original is None:
        warnings.append("Judge original_triplet is not a valid 3-element EI-MS triplet list.")
    if corrected is None:
        warnings.append("Judge corrected_triplet must be a non-empty valid 3-element EI-MS triplet list.")

    if corrected is not None and corrected[1] not in ALLOWED_MECHANISMS:
        warnings.append(f"Corrected mechanism is outside the strict enumeration: {corrected[1]}")

    if not isinstance(parsed.get("major_errors"), list):
        warnings.append("Judge major_errors is not a list.")

    if not isinstance(parsed.get("reason"), str) or not parsed.get("reason", "").strip():
        warnings.append("Judge reason must be a non-empty string.")
    if not isinstance(parsed.get("overall_quality"), str):
        warnings.append("Judge overall_quality must be a string.")
    if not isinstance(parsed.get("recommended_strategy"), str):
        warnings.append("Judge recommended_strategy must be a string.")

    if original is not None and corrected is not None:
        if verdict == "accept" and corrected != original:
            warnings.append("verdict=accept requires corrected_triplet to exactly equal original_triplet.")
        if original[2] != corrected[2]:
            if not product_ion_change_allowed(
                original[2], corrected[2], str(parsed.get("formula", ""))
            ):
                warnings.append(
                    "Judge corrected_triplet changed the target product ion without the allowed formula/mz-mismatch exception."
                )

    return warnings


def has_hard_format_error(warnings: Sequence[str]) -> bool:
    hard_patterns = (
        "Missing key",
        "Invalid or forbidden verdict",
        "corrected_triplet must be a non-empty",
        "original_triplet is not a valid",
        "outside the strict enumeration",
        "changed the target product ion",
        "verdict=accept requires",
        "reason must be a non-empty",
        "major_errors is not a list",
    )
    return any(any(pattern in warning for pattern in hard_patterns) for warning in warnings)


def normalize_revise_without_change(parsed: Dict[str, Any]) -> Dict[str, Any]:
    original = normalize_triplet(parsed.get("original_triplet"))
    corrected = normalize_triplet(parsed.get("corrected_triplet"))
    if parsed.get("verdict") == "revise" and original is not None and corrected == original:
        parsed["verdict"] = "accept"
    return parsed


def call_api_once(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    use_json_mode: bool,
) -> str:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content


def call_api_with_token_retry(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    base_max_tokens: int,
    use_json_mode: bool,
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    token_attempts: List[int] = []
    for value in (
        max(1024, base_max_tokens),
        max(8192, base_max_tokens * 2),
        max(16384, base_max_tokens * 4),
    ):
        if value not in token_attempts:
            token_attempts.append(value)

    raw_text = ""
    last_error: Optional[str] = None

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
            return extract_json_from_text(raw_text), raw_text, None

        except Exception as exc:
            last_error = str(exc)
            print(
                f"[Token Attempt {attempt}/{len(token_attempts)}] failed: {last_error}"
            )

            if use_json_mode and (
                "response_format" in last_error or "json" in last_error.lower()
            ):
                try:
                    print(f"[Fallback without json mode] max_tokens={current_max_tokens}")
                    raw_text = call_api_once(
                        client=client,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=current_max_tokens,
                        use_json_mode=False,
                    )
                    return extract_json_from_text(raw_text), raw_text, None
                except Exception as fallback_exc:
                    last_error = str(fallback_exc)
                    print(f"[Fallback without json mode failed] {last_error}")

            if attempt < len(token_attempts):
                time.sleep(3)

    return None, raw_text, last_error


def semantic_retry_if_needed(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    parsed: Optional[Dict[str, Any]],
    raw_text: str,
    temperature: float,
    base_max_tokens: int,
    use_json_mode: bool,
    max_retries: int,
    record: Dict[str, Any],
    original_triplet: List[str],
) -> Tuple[Optional[Dict[str, Any]], str, List[str], bool]:
    if parsed is None:
        return None, raw_text, ["Initial parsed output is None."], False

    parsed = enforce_current_record_metadata(parsed, record, original_triplet)
    warnings = validate_judge_output(parsed)
    if not has_hard_format_error(warnings):
        return parsed, raw_text, warnings, False

    current_parsed = parsed
    current_raw = raw_text
    current_warnings = warnings

    retry_schema = {
        "id": record.get("id"),
        "name": record.get("name", ""),
        "smiles": record.get("smiles", ""),
        "formula": record.get("formula", ""),
        "mw": record.get("mw", ""),
        "original_triplet": original_triplet,
        "verdict": "accept or revise",
        "reason": "non-empty string",
        "corrected_triplet": ["precursor", "allowed mechanism", "same product ion"],
        "overall_quality": "string",
        "major_errors": [],
        "recommended_strategy": "string",
    }

    for retry_index in range(1, max_retries + 1):
        print(
            f"[Semantic Retry {retry_index}/{max_retries}] "
            f"hard warnings: {current_warnings}"
        )
        retry_messages = messages + [
            {"role": "assistant", "content": current_raw},
            {
                "role": "user",
                "content": (
                    "Your previous answer violates the required strict JSON schema. "
                    "Return one JSON object only. Do not omit fields. verdict must be "
                    "accept or revise; delete is forbidden; corrected_triplet must contain "
                    "exactly three non-empty strings; its mechanism must exactly match the "
                    "allowed enumeration; preserve the target product ion unless its formula "
                    "and m/z are inconsistent. Use this exact schema:\n"
                    + json.dumps(retry_schema, ensure_ascii=False, indent=2)
                ),
            },
        ]

        retry_parsed, retry_raw, retry_error = call_api_with_token_retry(
            client=client,
            model=model,
            messages=retry_messages,
            temperature=temperature,
            base_max_tokens=base_max_tokens,
            use_json_mode=use_json_mode,
        )
        if retry_parsed is None:
            current_warnings = [f"Semantic retry failed: {retry_error}"]
            continue

        retry_parsed = enforce_current_record_metadata(
            retry_parsed, record, original_triplet
        )
        retry_warnings = validate_judge_output(retry_parsed)
        current_parsed = retry_parsed
        current_raw = retry_raw
        current_warnings = retry_warnings

        if not has_hard_format_error(retry_warnings):
            return current_parsed, current_raw, current_warnings, True

    return current_parsed, current_raw, current_warnings, True


# ============================================================
# 9. Fallback and context-safety checks
# ============================================================

def fallback_corrected_triplet(
    record: Dict[str, Any],
    original_triplet: List[str],
) -> List[str]:
    normalized = normalize_triplet(original_triplet)
    if normalized is not None:
        return normalized

    smiles = str(record.get("smiles", "")).strip() or "C"
    return [f"smiles_fragment: {smiles}", "Sigma-bond cleavage", "C?+ (m/z 0)"]


def make_fallback_judge_output(
    record: Dict[str, Any],
    original_triplet: List[str],
    reason_prefix: str,
    model_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_output = model_output or {}
    corrected = fallback_corrected_triplet(record, original_triplet)

    return {
        "id": record.get("id"),
        "name": record.get("name", ""),
        "smiles": record.get("smiles", ""),
        "formula": record.get("formula", ""),
        "mw": record.get("mw", ""),
        "original_triplet": original_triplet,
        "verdict": "revise",
        "reason": str(model_output.get("reason") or (
            reason_prefix + " The no-delete fallback preserved the original triplet."
        )),
        "corrected_triplet": corrected,
        "overall_quality": str(model_output.get("overall_quality") or (
            "The model response was unusable; the original triplet was preserved as a no-delete fallback."
        )),
        "major_errors": (
            model_output.get("major_errors")
            if isinstance(model_output.get("major_errors"), list)
            else ["Invalid or incomplete Judge response; automatic fallback used."]
        ),
        "recommended_strategy": str(model_output.get("recommended_strategy") or (
            "Review this fallback result manually because it was not accepted as a reliable Judge decision."
        )),
    }


def is_context_safe_triplet(
    triplet: List[str],
    record: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    normalized = normalize_triplet(triplet)
    if normalized is None:
        return False, ["Triplet is not a valid three-element list."]

    precursor, mechanism, product_ion = normalized
    product = parse_ion(product_ion)
    spectrum = get_record_spectrum(record)

    if mechanism not in ALLOWED_MECHANISMS:
        reasons.append("Mechanism is outside the strict enumeration.")
    if product is None:
        reasons.append("Product ion cannot be parsed.")
    elif not product.get("has_unknown") and product["nominal_mass"] != product["mz"]:
        reasons.append("Product-ion formula and m/z are inconsistent.")

    if precursor.startswith("smiles_fragment:"):
        validation = validate_fragment_graph(
            record.get("smiles", ""), precursor, product_ion
        )
        if validation.get("status") == "invalid":
            reasons.append("The selected smiles_fragment failed graph validation.")
    elif precursor.startswith("precursor_mz:"):
        precursor_mz = parse_precursor_mz(precursor)
        if precursor_mz is None:
            reasons.append("precursor_mz is malformed.")
        elif str(precursor_mz) not in spectrum:
            reasons.append("precursor_mz peak is absent from the supplied spectrum.")
        elif product is not None:
            precursor_intensity = get_peak_intensity(spectrum, precursor_mz)
            product_intensity = get_peak_intensity(spectrum, product["mz"])
            if (
                precursor_intensity is not None
                and product_intensity is not None
                and precursor_intensity < product_intensity
            ):
                reasons.append("precursor_mz violates the weak-peak-to-strong-peak rule.")
    else:
        reasons.append("Precursor format is invalid.")

    return len(reasons) == 0, reasons


def is_reliable_for_precursor_context(
    triplet: List[str],
    record: Dict[str, Any],
) -> bool:
    safe, _ = is_context_safe_triplet(triplet, record)
    if not safe:
        return False
    normalized = normalize_triplet(triplet)
    if normalized is None:
        return False
    product = parse_ion(normalized[2])
    if product is None:
        return False
    return str(product["mz"]) in get_record_spectrum(record)


# ============================================================
# 10. Input triplet extraction and output grouping
# ============================================================

def get_judge_triplets_from_record(record: Dict[str, Any]) -> List[List[str]]:
    raw = record.get("triplets")
    if isinstance(raw, list) and raw:
        triplets = [
            normalized for item in raw
            if (normalized := normalize_triplet(item)) is not None
        ]
        if triplets:
            return triplets

    raw = record.get("corrected_triplet")
    if isinstance(raw, list) and raw:
        triplets = [
            normalized for item in raw
            if (normalized := normalize_triplet(item)) is not None
        ]
        if triplets:
            return triplets

    triples_obj = None
    if isinstance(record.get("model_output"), dict):
        triples_obj = record["model_output"].get("triples")
    if triples_obj is None and isinstance(record.get("triples"), dict):
        triples_obj = record.get("triples")

    output: List[List[str]] = []
    if isinstance(triples_obj, dict):
        def sort_key(value: Any) -> Tuple[float, str]:
            try:
                return float(value), str(value)
            except Exception:
                return float("inf"), str(value)

        for mz_key in sorted(triples_obj.keys(), key=sort_key, reverse=True):
            block = triples_obj.get(mz_key)
            if not isinstance(block, dict):
                continue
            raw_triplet = block.get("triplet", [])
            direct = normalize_triplet(raw_triplet)
            if direct is not None:
                output.append(direct)
                continue
            if isinstance(raw_triplet, list):
                for item in raw_triplet:
                    normalized = normalize_triplet(item)
                    if normalized is not None:
                        output.append(normalized)

    deduplicated: List[List[str]] = []
    seen: Set[str] = set()
    for triplet in output:
        key = json.dumps(triplet, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            deduplicated.append(triplet)
    return deduplicated


def build_corrected_outputs(
    input_records: List[Dict[str, Any]],
    full_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Any, Dict[str, Any]] = {}

    for record in input_records:
        data_id = record.get("id")
        grouped[data_id] = {
            "id": data_id,
            "name": record.get("name", ""),
            "smiles": record.get("smiles", ""),
            "formula": record.get("formula", ""),
            "mw": record.get("mw", ""),
            "compound_class": record.get("compound_class", "alkene"),
            "input_spectrum": {
                mz: format_number(intensity)
                for mz, intensity in get_record_spectrum(record).items()
            },
            "corrected_triplet": [],
        }

    for result in full_results:
        data_id = result.get("id")
        if data_id not in grouped:
            grouped[data_id] = {
                "id": data_id,
                "name": result.get("name", ""),
                "smiles": result.get("smiles", ""),
                "formula": result.get("formula", ""),
                "mw": result.get("mw", ""),
                "compound_class": result.get("compound_class", "alkene"),
                "input_spectrum": result.get("input_spectrum", {}),
                "corrected_triplet": [],
            }

        final_triplet = normalize_triplet(result.get("final_corrected_triplet"))
        if final_triplet is not None:
            grouped[data_id]["corrected_triplet"].append(final_triplet)

    return list(grouped.values())


# ============================================================
# 11. Record filtering
# ============================================================

def filter_records(
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []

    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = safe_int(record.get("id"))

        if args.id_min is not None and (record_id is None or record_id < args.id_min):
            continue
        if args.id_max is not None and (record_id is None or record_id > args.id_max):
            continue
        selected.append(record)

    if args.limit is None:
        return selected[args.start:]
    return selected[args.start:args.start + args.limit]


# ============================================================
# 12. Main
# ============================================================

# ============================================================
# 12. Missing-ion resume and ordered insertion utilities
# ============================================================

def canonical_id(value: Any) -> Any:
    """Normalize numeric/string ids so 9, "9", and 9.0 share one key."""
    parsed = safe_int(value)
    return parsed if parsed is not None else str(value)


def id_sort_key(value: Any) -> Tuple[int, Any]:
    parsed = safe_int(value)
    if parsed is not None:
        return (0, parsed)
    return (1, str(value))


def triplet_json_key(triplet: Any) -> Optional[str]:
    normalized = normalize_triplet(triplet)
    if normalized is None:
        return None
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def product_mz_from_triplet(triplet: Any) -> Optional[int]:
    normalized = normalize_triplet(triplet)
    if normalized is None:
        return None
    ion = parse_ion(normalized[2])
    return None if ion is None else safe_int(ion.get("mz"))


def product_mz_from_result(result: Dict[str, Any]) -> Optional[int]:
    """
    Recover the judged target ion m/z from a detailed judge_all result.

    Priority is given to the original target, because the no-delete Judge should
    preserve the observed target m/z even when it repairs the formula.
    """
    context = result.get("current_product_context")
    if isinstance(context, dict):
        mz = safe_int(context.get("product_mz"))
        if mz is not None:
            return mz

    for key in ("original_triplet", "final_corrected_triplet"):
        mz = product_mz_from_triplet(result.get(key))
        if mz is not None:
            return mz

    judge_output = result.get("judge_output")
    if isinstance(judge_output, dict):
        for key in ("original_triplet", "corrected_triplet"):
            mz = product_mz_from_triplet(judge_output.get(key))
            if mz is not None:
                return mz
    return None


def existing_result_is_usable(result: Dict[str, Any]) -> bool:
    """A result counts as generated only when it contains a valid final triplet."""
    return normalize_triplet(result.get("final_corrected_triplet")) is not None


def match_existing_results_for_record(
    record: Dict[str, Any],
    triplets: Sequence[List[str]],
    existing_results: Sequence[Dict[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Match existing judge_all entries to the current input ions without assuming
    that triplet_index is continuous.

    Matching is deliberately conservative and consumes each existing result once:
      1. exact original_triplet match;
      2. stored triplet_index, provided the target m/z is compatible;
      3. target product m/z match in the original triplet order.

    This identifies holes in the middle of a molecule even when later ions and
    later molecule ids have already been judged.
    """
    usable = [item for item in existing_results if isinstance(item, dict) and existing_result_is_usable(item)]
    matched: Dict[int, Dict[str, Any]] = {}
    used: Set[int] = set()

    input_keys = [triplet_json_key(t) for t in triplets]
    input_mzs = [product_mz_from_triplet(t) for t in triplets]

    # Pass 1: exact original triplet.
    for input_index, expected_key in enumerate(input_keys):
        if expected_key is None:
            continue
        for result_index, result in enumerate(usable):
            if result_index in used:
                continue
            if triplet_json_key(result.get("original_triplet")) == expected_key:
                matched[input_index] = result
                used.add(result_index)
                break

    # Pass 2: explicit triplet_index, but require compatible target m/z when known.
    for result_index, result in enumerate(usable):
        if result_index in used:
            continue
        stored_index = safe_int(result.get("triplet_index"))
        if stored_index is None or stored_index < 0 or stored_index >= len(triplets):
            continue
        if stored_index in matched:
            continue
        expected_mz = input_mzs[stored_index]
        actual_mz = product_mz_from_result(result)
        if expected_mz is not None and actual_mz is not None and expected_mz != actual_mz:
            continue
        matched[stored_index] = result
        used.add(result_index)

    # Pass 3: same target ion m/z, consuming duplicates one by one in input order.
    for input_index, expected_mz in enumerate(input_mzs):
        if input_index in matched or expected_mz is None:
            continue
        for result_index, result in enumerate(usable):
            if result_index in used:
                continue
            if product_mz_from_result(result) == expected_mz:
                matched[input_index] = result
                used.add(result_index)
                break

    unmatched_existing = [
        result for result_index, result in enumerate(usable)
        if result_index not in used
    ]
    return matched, unmatched_existing


def build_results_by_id(full_results: Sequence[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for item in full_results:
        if isinstance(item, dict):
            grouped[canonical_id(item.get("id"))].append(item)
    return grouped


def build_input_order_map(
    input_records: Sequence[Dict[str, Any]],
    full_results: Sequence[Dict[str, Any]],
) -> Dict[int, Tuple[Tuple[int, Any], int, int]]:
    """
    Return an order key for every result object id(). Existing object contents are
    never edited. The key is molecule id, original ion position, then old position.
    """
    records_by_id = {
        canonical_id(record.get("id")): record
        for record in input_records if isinstance(record, dict)
    }
    results_by_id = build_results_by_id(full_results)
    object_order: Dict[int, Tuple[Tuple[int, Any], int, int]] = {}
    original_position = {id(item): pos for pos, item in enumerate(full_results)}

    all_ids = set(results_by_id)
    for cid in all_ids:
        record = records_by_id.get(cid)
        existing = results_by_id[cid]
        if record is None:
            for local_pos, item in enumerate(existing):
                stored = safe_int(item.get("triplet_index"))
                rank = stored if stored is not None else 10**9 + local_pos
                object_order[id(item)] = (id_sort_key(item.get("id")), rank, original_position[id(item)])
            continue

        triplets = get_judge_triplets_from_record(record)
        matched, unmatched = match_existing_results_for_record(record, triplets, existing)
        matched_object_to_index = {id(item): idx for idx, item in matched.items()}
        for local_pos, item in enumerate(existing):
            rank = matched_object_to_index.get(id(item))
            if rank is None:
                stored = safe_int(item.get("triplet_index"))
                rank = stored if stored is not None else 10**9 + local_pos
            object_order[id(item)] = (id_sort_key(item.get("id")), rank, original_position[id(item)])

    return object_order


def order_full_results_without_changing_items(
    full_results: List[Dict[str, Any]],
    input_records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Order the list while preserving every existing result dictionary unchanged."""
    order_map = build_input_order_map(input_records, full_results)
    return sorted(
        full_results,
        key=lambda item: order_map.get(
            id(item),
            (id_sort_key(item.get("id")), 10**9, 10**9),
        ),
    )


def save_json_atomic(obj: Any, path: str) -> None:
    """Write through a temporary file so interruption cannot leave half a JSON file."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)


def rebuild_corrected_outputs_in_result_order(
    input_records: List[Dict[str, Any]],
    ordered_full_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # Only include molecules that already have at least one detailed Judge result.
    # This avoids creating empty corrected_triplet records for untouched future ids.
    present_ids = {
        canonical_id(item.get("id"))
        for item in ordered_full_results
        if isinstance(item, dict)
    }
    present_records = [
        record for record in input_records
        if canonical_id(record.get("id")) in present_ids
    ]
    corrected = build_corrected_outputs(present_records, ordered_full_results)
    corrected.sort(key=lambda item: id_sort_key(item.get("id")))
    return corrected


def restore_existing_context(
    existing: Dict[str, Any],
    record: Dict[str, Any],
    previous_verified_triplets: List[List[str]],
    previous_reliable_triplets: List[List[str]],
) -> None:
    """Use an already judged earlier ion as context for a missing later ion."""
    if existing.get("auto_fallback_used", False):
        return
    final_triplet = normalize_triplet(existing.get("final_corrected_triplet"))
    if final_triplet is None:
        return

    context_safe = existing.get("context_safe")
    if not isinstance(context_safe, bool):
        context_safe, _ = is_context_safe_triplet(final_triplet, record)
    if not context_safe:
        return

    previous_verified_triplets.append(final_triplet)
    context_reliable = existing.get("context_reliable_for_precursor")
    if not isinstance(context_reliable, bool):
        context_reliable = is_reliable_for_precursor_context(final_triplet, record)
    if context_reliable:
        previous_reliable_triplets.append(final_triplet)


# ============================================================
# 13. Main: single-thread fill only missing ions
# ============================================================

def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is missing. Pass --api_key or set OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)

    records = load_json(args.input_file)
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list or a single dict.")

    spectrum_source_records = None
    if args.spectrum_source_file:
        spectrum_source_records = load_json(args.spectrum_source_file)
    merge_spectrum_source(records, spectrum_source_records)

    # Keep the complete input record list for global ordering and corrected output.
    all_records = [record for record in records if isinstance(record, dict)]
    selected_records = sorted(
        filter_records(all_records, args),
        key=lambda record: id_sort_key(record.get("id")),
    )

    # The final judge_all file is the only resume source.
    full_results: List[Dict[str, Any]] = []
    if os.path.exists(args.output_file):
        existing_results = load_json(args.output_file)
        if not isinstance(existing_results, list):
            raise ValueError(
                f"Existing --output_file must be a JSON list: {args.output_file}"
            )
        full_results = [item for item in existing_results if isinstance(item, dict)]
        print(f"Loaded {len(full_results)} existing detailed Judge results from final output.")
    else:
        print("Final output does not exist; all selected ions will be judged.")

    full_results = order_full_results_without_changing_items(full_results, all_records)
    results_by_id = build_results_by_id(full_results)

    total_input_ions = 0
    total_existing_ions = 0
    total_missing_ions = 0
    molecule_status: Dict[Any, Tuple[int, int, int]] = {}

    for record in selected_records:
        triplets = get_judge_triplets_from_record(record)
        cid = canonical_id(record.get("id"))
        matched, unmatched_existing = match_existing_results_for_record(
            record, triplets, results_by_id.get(cid, [])
        )
        existing_count = len(matched)
        missing_count = len(triplets) - existing_count
        total_input_ions += len(triplets)
        total_existing_ions += existing_count
        total_missing_ions += missing_count
        molecule_status[cid] = (len(triplets), existing_count, missing_count)
        if unmatched_existing:
            print(
                f"[WARN] id={record.get('id')} has {len(unmatched_existing)} existing "
                "judge_all entries that could not be aligned to current input ions; "
                "they will be preserved unchanged."
            )

    print("=" * 88)
    print(f"Selected molecules:               {len(selected_records)}")
    print(f"Input ions in selected range:     {total_input_ions}")
    print(f"Already judged ions:              {total_existing_ions}")
    print(f"Missing ions to judge:            {total_missing_ions}")
    print(f"Resume source:                    {args.output_file}")
    print("Resume key: molecule id + exact original triplet / triplet_index / target m/z")
    print("Mode: single-thread, insert one missing ion and save immediately")
    print("=" * 88)

    judged_count = 0

    for record_index, record in enumerate(selected_records, start=1):
        data_id = record.get("id")
        cid = canonical_id(data_id)
        name = record.get("name", "")
        smiles = record.get("smiles", "")
        formula = record.get("formula", "")
        mw = record.get("mw", "")
        spectrum = get_record_spectrum(record)
        triplets = get_judge_triplets_from_record(record)
        record["_judge_triplets_flat"] = triplets

        current_existing = build_results_by_id(full_results).get(cid, [])
        matched_by_index, unmatched_existing = match_existing_results_for_record(
            record, triplets, current_existing
        )
        missing_indices = [idx for idx in range(len(triplets)) if idx not in matched_by_index]

        print("=" * 88)
        print(
            f"[Molecule {record_index}/{len(selected_records)}] id={data_id}, name={name}, "
            f"ions={len(triplets)}, existing={len(matched_by_index)}, missing={len(missing_indices)}"
        )

        if not missing_indices:
            print(f"[Skip molecule] id={data_id}: every target ion already exists in judge_all.")
            continue

        if not spectrum:
            print(
                f"[Spectrum Warning] id={data_id}: no input_spectrum/spectrum found. "
                "Use the full generator result or provide --spectrum_source_file."
            )

        previous_verified_triplets: List[List[str]] = []
        previous_reliable_triplets: List[List[str]] = []

        for triplet_index, triplet in enumerate(triplets):
            # Existing ions are never regenerated. They are only restored as earlier context.
            existing = matched_by_index.get(triplet_index)
            if existing is not None:
                print(
                    f"[Skip existing ion] id={data_id}, triplet_index={triplet_index}, "
                    f"m/z={product_mz_from_result(existing)}"
                )
                restore_existing_context(
                    existing,
                    record,
                    previous_verified_triplets,
                    previous_reliable_triplets,
                )
                continue

            normalized_triplet = normalize_triplet(triplet)
            previous_verified_before = previous_verified_triplets[:]
            previous_reliable_before = previous_reliable_triplets[:]

            if normalized_triplet is None:
                fallback_triplet = fallback_corrected_triplet(record, triplet)
                result_item = {
                    "id": data_id,
                    "name": name,
                    "smiles": smiles,
                    "formula": formula,
                    "mw": mw,
                    "compound_class": record.get("compound_class", "alkene"),
                    "input_spectrum": {
                        mz: format_number(intensity) for mz, intensity in spectrum.items()
                    },
                    "triplet_index": triplet_index,
                    "original_triplet": triplet,
                    "judge_output": None,
                    "final_corrected_triplet": fallback_triplet,
                    "auto_fallback_used": True,
                    "semantic_retry_used": False,
                    "raw_response": "",
                    "parse_ok": False,
                    "error": "Invalid input triplet format.",
                    "warnings": ["Invalid input triplet format."],
                    "chemical_warnings": ["Invalid input triplet format."],
                    "context_safe": False,
                    "context_reliable_for_precursor": False,
                    "previous_verified_corrected_triplets_before_current": previous_verified_before,
                    "previous_reliable_precursor_triplets_before_current": previous_reliable_before,
                }
                full_results.append(result_item)
                full_results = order_full_results_without_changing_items(full_results, all_records)
                save_json_atomic(full_results, args.output_file)
                save_json_atomic(
                    rebuild_corrected_outputs_in_result_order(all_records, full_results),
                    args.corrected_output_file,
                )
                print(
                    f"[Inserted fallback] id={data_id}, triplet_index={triplet_index}; "
                    "final files updated."
                )
                continue

            judged_count += 1
            product_context = build_product_context(normalized_triplet[2], spectrum)
            print("-" * 88)
            print(
                f"[Judge missing {judged_count}/{total_missing_ions}] id={data_id}, "
                f"triplet_index={triplet_index}, target_mz={product_context.get('product_mz')}, "
                f"product_intensity={product_context.get('product_intensity')}"
            )

            user_prompt = build_user_prompt(
                record=record,
                triplet=normalized_triplet,
                triplet_index=triplet_index,
                previous_verified_triplets=previous_verified_before,
                previous_reliable_triplets=previous_reliable_before,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            parsed, raw_text, error = call_api_with_token_retry(
                client=client,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                base_max_tokens=args.max_tokens,
                use_json_mode=args.use_json_mode,
            )

            parse_ok = parsed is not None
            semantic_retry_used = False
            model_output_before_fallback: Optional[Dict[str, Any]] = None

            if parsed is not None:
                parsed = enforce_current_record_metadata(parsed, record, normalized_triplet)
                parsed, raw_text, warnings, semantic_retry_used = semantic_retry_if_needed(
                    client=client,
                    model=args.model,
                    messages=messages,
                    parsed=parsed,
                    raw_text=raw_text,
                    temperature=args.temperature,
                    base_max_tokens=args.max_tokens,
                    use_json_mode=args.use_json_mode,
                    max_retries=args.semantic_retry,
                    record=record,
                    original_triplet=normalized_triplet,
                )
                if parsed is not None:
                    parsed = enforce_current_record_metadata(parsed, record, normalized_triplet)
                    model_output_before_fallback = copy.deepcopy(parsed)
                    parsed = normalize_revise_without_change(parsed)
                    warnings = validate_judge_output(parsed)
                else:
                    warnings = ["Semantic retry returned no parsed output."]
            else:
                warnings = [error or "API/JSON parsing failed."]

            auto_fallback_used = False
            if parsed is None or has_hard_format_error(warnings):
                parsed = make_fallback_judge_output(
                    record=record,
                    original_triplet=normalized_triplet,
                    reason_prefix=(
                        "The API output was missing, invalid, or still violated the "
                        "strict no-delete schema after retry."
                    ),
                    model_output=model_output_before_fallback,
                )
                parsed = enforce_current_record_metadata(parsed, record, normalized_triplet)
                warnings = validate_judge_output(parsed)
                auto_fallback_used = True

            final_corrected = normalize_triplet(parsed.get("corrected_triplet"))
            if final_corrected is None:
                final_corrected = fallback_corrected_triplet(record, normalized_triplet)
                auto_fallback_used = True

            context_safe, chemical_warnings = is_context_safe_triplet(final_corrected, record)
            if auto_fallback_used:
                context_safe = False
                chemical_warnings = list(chemical_warnings) + [
                    "Automatic fallback results are excluded from verified downstream context."
                ]

            context_reliable = (
                context_safe
                and not auto_fallback_used
                and is_reliable_for_precursor_context(final_corrected, record)
            )

            if context_safe and not auto_fallback_used:
                previous_verified_triplets.append(final_corrected)
                if context_reliable:
                    previous_reliable_triplets.append(final_corrected)

            result_item = {
                "id": data_id,
                "name": name,
                "smiles": smiles,
                "formula": formula,
                "mw": mw,
                "compound_class": record.get("compound_class", "alkene"),
                "input_spectrum": {
                    mz: format_number(intensity) for mz, intensity in spectrum.items()
                },
                "spectrum_summary": build_spectrum_summary(spectrum),
                "current_product_context": product_context,
                "triplet_index": triplet_index,
                "original_triplet": normalized_triplet,
                "judge_output": parsed,
                "model_judge_output_before_fallback": model_output_before_fallback,
                "final_corrected_triplet": final_corrected,
                "auto_fallback_used": auto_fallback_used,
                "semantic_retry_used": semantic_retry_used,
                "raw_response": raw_text,
                "parse_ok": parse_ok,
                "error": None if parse_ok else error,
                "warnings": warnings,
                "chemical_warnings": chemical_warnings,
                "context_safe": context_safe,
                "context_reliable_for_precursor": context_reliable,
                "previous_verified_corrected_triplets_before_current": previous_verified_before,
                "previous_reliable_precursor_triplets_before_current": previous_reliable_before,
            }

            # Insert the new missing ion, reorder only the list, and immediately save.
            # Existing result dictionaries—including all later molecules—are untouched.
            full_results.append(result_item)
            full_results = order_full_results_without_changing_items(full_results, all_records)
            save_json_atomic(full_results, args.output_file)
            save_json_atomic(
                rebuild_corrected_outputs_in_result_order(all_records, full_results),
                args.corrected_output_file,
            )

            print(
                f"[Inserted and saved] id={data_id}, triplet_index={triplet_index}, "
                f"target_mz={product_context.get('product_mz')}, verdict={parsed.get('verdict')}, "
                f"final_result_count={len(full_results)}"
            )

            if args.sleep > 0:
                time.sleep(args.sleep)

    # Final normalization/save; no existing result content is rewritten by the program.
    full_results = order_full_results_without_changing_items(full_results, all_records)
    save_json_atomic(full_results, args.output_file)
    save_json_atomic(
        rebuild_corrected_outputs_in_result_order(all_records, full_results),
        args.corrected_output_file,
    )

    print("=" * 88)
    print("All missing ions have been processed.")
    print(f"Newly judged missing ions: {judged_count}")
    print(f"Final detailed results:    {args.output_file} ({len(full_results)} entries)")
    print(f"Final corrected results:   {args.corrected_output_file}")
    print("Existing later results were preserved; only missing-ion entries were inserted.")


if __name__ == "__main__":
    main()
