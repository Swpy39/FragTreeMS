import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except Exception as exc:
    raise ImportError(
        "RDKit is required for the Morgan-fingerprint global molecular branch."
    ) from exc
from torch.utils.data import DataLoader, Dataset


MZ_RE = re.compile(r"\(m/z\s*([0-9]+(?:\.[0-9]+)?)\)", flags=re.IGNORECASE)
PRECURSOR_MZ_RE = re.compile(r"^precursor_mz:\s*([0-9]+(?:\.[0-9]+)?)", flags=re.IGNORECASE)
GENERAL_TOKEN_RE = re.compile(
    r"[A-Za-z]+|\d+(?:\.\d+)?|[()+\-=/#:.\[\],|]+|[^\s]"
)
SMILES_TOKEN_RE = re.compile(
    r"\[[^\]]+\]|Br|Cl|Si|Na|Li|Mg|Al|Ca|Fe|Zn|Cu|Mn|Hg|Ag|Sn|"
    r"[A-Z][a-z]?|[bcnops]|%\d{2}|\d+|@@?|=|#|-|\+|\\|/|"
    r"\(|\)|\.|:|\*"
)
FORMULA_TOKEN_RE = re.compile(r"[A-Z][a-z]?|\d+(?:\.\d+)?|[+\-]")
ROOT_TEXT_RE = re.compile(
    r"^ROOT molecule name=(.*?) SMILES=(.*?) formula=(.*?) MW=(.*)$"
)
EDGE_TEXT_RE = re.compile(
    r"^EDGE precursor=(.*?) mechanism=(.*?) product=(.*)$"
)

MECHANISM_LABELS = [
    "Molecular ion",
    "Isotopic peak",
    "Alpha-cleavage",
    "Hydrogen transfer",
    "Sigma-bond cleavage",
    "Benzylic cleavage",
    "Allylic cleavage",
    "McLafferty rearrangement",
    "Radical-ion rearrangement",
    "Dehydrogenation / Sequential dehydrogenation",
    "Neutral loss",
    "Ring cleavage / Ring rearrangement",
    "Retro-Diels–Alder fragmentation",
]

MECH_TO_ID = {m: i + 1 for i, m in enumerate(MECHANISM_LABELS)}

TYPE_PAD = 0
TYPE_ROOT = 1
TYPE_STRUCTURE_EDGE = 2
TYPE_PRECURSOR_EDGE = 3
TYPE_ORPHAN_EDGE = 4

TYPE_TO_NAME = {
    TYPE_PAD: "PAD",
    TYPE_ROOT: "ROOT",
    TYPE_STRUCTURE_EDGE: "STRUCTURE_EDGE",
    TYPE_PRECURSOR_EDGE: "PRECURSOR_EDGE",
    TYPE_ORPHAN_EDGE: "ORPHAN_EDGE",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_cuda_inference() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    text = p.read_text(encoding="utf-8").strip()
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

    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Bad JSON at {path}:{line_no}: {e}\n{s[:500]}") from e
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def save_json_or_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return default


def normalize_mz_value(x: Any) -> Optional[str]:
    try:
        v = float(str(x).strip())
        return str(int(v)) if v.is_integer() else str(v)
    except Exception:
        return None


def format_mz_key(mz: float) -> str:
    return str(int(mz)) if float(mz).is_integer() else str(mz)


def parse_mz_from_product(product_ion: Any) -> Optional[str]:
    if not isinstance(product_ion, str):
        return None
    m = MZ_RE.search(product_ion)
    if not m:
        return None
    return normalize_mz_value(m.group(1))


def parse_precursor_mz(precursor: Any) -> Optional[str]:
    if not isinstance(precursor, str):
        return None
    m = PRECURSOR_MZ_RE.search(precursor.strip())
    if not m:
        return None
    return normalize_mz_value(m.group(1))


def normalize_triplet(x: Any) -> Optional[List[str]]:
    if isinstance(x, list) and len(x) >= 3:
        return [str(x[0]).strip(), str(x[1]).strip(), str(x[2]).strip()]
    return None


def extract_triplet_list(record: Dict[str, Any]) -> List[Any]:
    for key in (
        "triplets",
        "corrected_triplet",
        "corrected_triplets",
        "stage1_triplets",
        "predicted_triplets",
        "infer_result",
    ):
        value = record.get(key)
        if not isinstance(value, list):
            continue

        if all(isinstance(item, list) and len(item) >= 3 for item in value):
            return value


        for candidate in value:
            if (
                isinstance(candidate, list)
                and candidate
                and all(
                    isinstance(item, list) and len(item) >= 3
                    for item in candidate
                )
            ):
                return candidate
    return []


def extract_intensity_value(record: Dict[str, Any]) -> Any:
    for key in (
        "intensity",
        "intensities",
        "gold_intensity",
        "gold_spectrum",
        "input_spectrum",
        "spectrum",
    ):
        if key in record and record[key] is not None:
            return record[key]
    return {}


def normalize_intensity_mapping(value: Any) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if isinstance(value, dict):
        for mz, intensity in value.items():
            mz_key = normalize_mz_value(mz)
            if mz_key is not None:
                result[mz_key] = parse_float(intensity, 0.0)
        return result

    if not isinstance(value, list):
        return result

    for item in value:
        mz = None
        intensity = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            mz, intensity = item[0], item[1]
        elif isinstance(item, dict):
            for mz_key in ("mz", "m/z", "mass", "mass_to_charge"):
                if mz_key in item:
                    mz = item[mz_key]
                    break
            for int_key in ("intensity", "abundance", "relative_intensity"):
                if int_key in item:
                    intensity = item[int_key]
                    break
        normalized_mz = normalize_mz_value(mz)
        if normalized_mz is not None and intensity is not None:
            result[normalized_mz] = parse_float(intensity, 0.0)
    return result


def smiles_to_morgan_fingerprint(
    smiles: Any,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> List[float]:
    text = str(smiles or "").strip()
    if not text:
        return [0.0] * int(n_bits)
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return [0.0] * int(n_bits)
    bit_vector = AllChem.GetMorganFingerprintAsBitVect(
        mol, int(radius), nBits=int(n_bits), useChirality=True
    )
    array = np.zeros((int(n_bits),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array.tolist()


def normalize_prediction_by_base_peak(
    prediction: torch.Tensor,
    peak_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    masked = prediction * peak_mask.float()
    maximum = masked.max(dim=1, keepdim=True).values.clamp_min(eps)
    return (masked / maximum) * peak_mask.float()


def stable_hash_token(token: str, vocab_size: int) -> int:
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16) % (vocab_size - 1) + 1


def _prefixed_general_tokens(text: Any, namespace: str) -> List[str]:
    values = GENERAL_TOKEN_RE.findall(str(text))
    return [f"{namespace}::{token.lower()}" for token in values]


def _smiles_tokens(smiles: Any) -> List[str]:
    text = str(smiles or "").strip()
    values = SMILES_TOKEN_RE.findall(text)
    if not values and text:
        values = list(text)
    return [f"smiles::{token}" for token in values]


def _formula_tokens(formula: Any) -> List[str]:
    values = FORMULA_TOKEN_RE.findall(str(formula or ""))
    return [f"formula::{token}" for token in values]


def structured_text_tokens(text: Any) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return ["special::<empty>"]

    root_match = ROOT_TEXT_RE.match(raw)
    if root_match:
        name, smiles, formula, mw = root_match.groups()
        return (
            ["special::root", "field::name"]
            + _prefixed_general_tokens(name, "name")
            + ["field::smiles"]
            + _smiles_tokens(smiles)
            + ["field::formula"]
            + _formula_tokens(formula)
            + ["field::mw"]
            + _prefixed_general_tokens(mw, "mw")
        )

    edge_match = EDGE_TEXT_RE.match(raw)
    if edge_match:
        precursor, mechanism, product = edge_match.groups()
        precursor_text = precursor.strip()
        if precursor_text.lower().startswith("smiles_fragment:"):
            fragment = precursor_text.split(":", 1)[1].strip()
            precursor_tokens = (
                ["field::smiles_fragment"] + _smiles_tokens(fragment)
            )
        else:
            precursor_tokens = (
                ["field::precursor"]
                + _prefixed_general_tokens(precursor_text, "precursor")
            )

        formula_text = product.split("+", 1)[0].strip()
        return (
            ["special::edge"]
            + precursor_tokens
            + ["field::mechanism"]
            + _prefixed_general_tokens(mechanism, "mechanism")
            + ["field::product"]
            + _formula_tokens(formula_text)
            + _prefixed_general_tokens(product, "product")
        )

    return _prefixed_general_tokens(raw, "text") or ["special::<empty>"]


def text_to_ids(text: str, vocab_size: int, max_words: int) -> List[int]:
    tokens = structured_text_tokens(text)
    ids = [stable_hash_token(token, vocab_size) for token in tokens[:max_words]]
    if len(ids) < max_words:
        ids += [0] * (max_words - len(ids))
    return ids

def safe_get_intensity(intensity_dict: Any, mz: float, default: float = 0.0) -> float:
    if not isinstance(intensity_dict, dict):
        return default
    k1 = format_mz_key(mz)
    k2 = str(mz)
    value = intensity_dict.get(k1, intensity_dict.get(k2, default))
    return max(0.0, min(1.0, parse_float(value, default)))


THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
ALT_RE = re.compile(r"\s+alternatives?\s*:\s*\[.*?\]\s*$", flags=re.IGNORECASE)


def strip_think_for_prf(x: Any) -> str:
    return THINK_RE.sub("", str(x) if x is not None else "").strip()


def clean_source_field_for_prf(source: Any) -> str:
    s = strip_think_for_prf(source)
    if not s.startswith("smiles_fragment:"):
        return s
    s = s.split("|", 1)[0].strip()
    s = ALT_RE.sub("", s).strip()
    return s


def normalize_formula_for_prf(
    formula: str,
    *,
    ignore_radical_dot: bool = True,
    add_missing_charge: bool = True,
) -> str:
    s = str(formula).strip().replace(" ", "")
    if not s:
        return s
    if ignore_radical_dot:
        s = s.replace("·", "").replace(".", "")
    s = s.replace("++", "+").replace("--", "-")
    if add_missing_charge and s and not s.endswith("+") and not s.endswith("-"):
        s += "+"
    return s


def parse_product_ion_for_prf(
    product_ion: Any,
    *,
    ignore_radical_dot: bool = True,
    add_missing_charge: bool = True,
) -> Optional[Tuple[str, int]]:
    if not isinstance(product_ion, str):
        return None

    s = strip_think_for_prf(product_ion)
    m = MZ_RE.search(s)
    if not m:
        return None

    try:
        mz = float(m.group(1))
    except Exception:
        return None

    if not mz.is_integer():
        return None

    formula_part = s[:m.start()].strip()
    if not formula_part:
        return None

    formula = normalize_formula_for_prf(
        formula_part,
        ignore_radical_dot=ignore_radical_dot,
        add_missing_charge=add_missing_charge,
    )
    if not formula:
        return None

    return formula, int(mz)


def normalized_product_string_for_prf(product: Tuple[str, int]) -> str:
    return f"{product[0]} (m/z {product[1]})"


def normalize_triplet_for_prf(
    t: Any,
    *,
    ignore_radical_dot: bool = True,
    add_missing_charge: bool = True,
    validate_product: bool = True,
    validate_mechanism: bool = False,
) -> Optional[List[str]]:
    if not (isinstance(t, list) and len(t) == 3):
        return None

    precursor = clean_source_field_for_prf(t[0])
    mechanism = strip_think_for_prf(t[1])
    product_raw = strip_think_for_prf(t[2])

    if not precursor or not mechanism or not product_raw:
        return None

    if validate_mechanism and mechanism not in MECHANISM_LABELS:
        return None

    parsed = parse_product_ion_for_prf(
        product_raw,
        ignore_radical_dot=ignore_radical_dot,
        add_missing_charge=add_missing_charge,
    )

    if validate_product and parsed is None:
        return None

    product = normalized_product_string_for_prf(parsed) if parsed is not None else product_raw
    return [precursor, mechanism, product]


def normalize_triplets_for_prf(
    xs: Any,
    *,
    ignore_radical_dot: bool = True,
    add_missing_charge: bool = True,
    validate_product: bool = True,
    validate_mechanism: bool = False,
) -> List[List[str]]:
    if not isinstance(xs, list):
        return []

    if len(xs) == 3 and all(isinstance(x, str) for x in xs):
        nt = normalize_triplet_for_prf(
            xs,
            ignore_radical_dot=ignore_radical_dot,
            add_missing_charge=add_missing_charge,
            validate_product=validate_product,
            validate_mechanism=validate_mechanism,
        )
        return [nt] if nt is not None else []

    out: List[List[str]] = []
    for t in xs:
        nt = normalize_triplet_for_prf(
            t,
            ignore_radical_dot=ignore_radical_dot,
            add_missing_charge=add_missing_charge,
            validate_product=validate_product,
            validate_mechanism=validate_mechanism,
        )
        if nt is not None:
            out.append(nt)
    return out


def get_gold_triplets_for_prf(
    record: Dict[str, Any],
    *,
    ignore_radical_dot: bool = True,
    add_missing_charge: bool = True,
    validate_mechanism: bool = False,
) -> List[List[str]]:
    candidate_keys = [
        "corrected_triplet",
        "corrected_triplets",
        "gold_triplets",
        "label_triplets",
        "triplets",
    ]

    for key in candidate_keys:
        if isinstance(record.get(key), list):
            return normalize_triplets_for_prf(
                record[key],
                ignore_radical_dot=ignore_radical_dot,
                add_missing_charge=add_missing_charge,
                validate_product=True,
                validate_mechanism=validate_mechanism,
            )

    output = record.get("output")
    if isinstance(output, list):
        return normalize_triplets_for_prf(
            output,
            ignore_radical_dot=ignore_radical_dot,
            add_missing_charge=add_missing_charge,
            validate_product=True,
            validate_mechanism=validate_mechanism,
        )

    if isinstance(output, dict):
        if isinstance(output.get("triplets"), list):
            return normalize_triplets_for_prf(
                output["triplets"],
                ignore_radical_dot=ignore_radical_dot,
                add_missing_charge=add_missing_charge,
                validate_product=True,
                validate_mechanism=validate_mechanism,
            )

        merged = []
        for key in ["primary_triplets", "completion_triplets"]:
            if isinstance(output.get(key), list):
                merged.extend(output[key])
        if merged:
            return normalize_triplets_for_prf(
                merged,
                ignore_radical_dot=ignore_radical_dot,
                add_missing_charge=add_missing_charge,
                validate_product=True,
                validate_mechanism=validate_mechanism,
            )

    return []


def get_pred_triplets_for_prf(
    record: Dict[str, Any],
    *,
    ignore_radical_dot: bool = True,
    add_missing_charge: bool = True,
    validate_mechanism: bool = False,
) -> List[List[str]]:
    infer_obj = record.get("infer_object")
    if isinstance(infer_obj, dict):
        if isinstance(infer_obj.get("triplets"), list):
            return normalize_triplets_for_prf(
                infer_obj["triplets"],
                ignore_radical_dot=ignore_radical_dot,
                add_missing_charge=add_missing_charge,
                validate_product=True,
                validate_mechanism=validate_mechanism,
            )

        merged = []
        for key in ["primary_triplets", "completion_triplets"]:
            if isinstance(infer_obj.get(key), list):
                merged.extend(infer_obj[key])
        if merged:
            return normalize_triplets_for_prf(
                merged,
                ignore_radical_dot=ignore_radical_dot,
                add_missing_charge=add_missing_charge,
                validate_product=True,
                validate_mechanism=validate_mechanism,
            )

    for key in ["infer_result", "pred_triplets", "predicted_triplets", "generated_triplets"]:
        if isinstance(record.get(key), list):
            return normalize_triplets_for_prf(
                record[key],
                ignore_radical_dot=ignore_radical_dot,
                add_missing_charge=add_missing_charge,
                validate_product=True,
                validate_mechanism=validate_mechanism,
            )

    return []


ProductUnitForPRF = Tuple[str, int]


def extract_product_units_for_prf(triplets: List[List[str]], unique: bool = True) -> List[ProductUnitForPRF]:
    out: List[ProductUnitForPRF] = []
    seen: Set[ProductUnitForPRF] = set()

    for t in triplets:
        parsed = parse_product_ion_for_prf(t[2])
        if parsed is None:
            continue
        if unique and parsed in seen:
            continue
        seen.add(parsed)
        out.append(parsed)

    return out


def product_unit_to_str_for_prf(unit: ProductUnitForPRF) -> str:
    return f"{unit[0]} (m/z {unit[1]})"


def safe_div_for_prf(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def compute_prf_sets_for_prf(gold_set: Set[Any], pred_set: Set[Any]) -> Dict[str, Any]:
    hit_set = gold_set & pred_set
    missing_set = gold_set - pred_set
    extra_set = pred_set - gold_set

    recall = safe_div_for_prf(len(hit_set), len(gold_set))
    precision = safe_div_for_prf(len(hit_set), len(pred_set))
    f1 = safe_div_for_prf(2.0 * precision * recall, precision + recall)

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "n_gold": len(gold_set),
        "n_pred": len(pred_set),
        "n_hit": len(hit_set),
        "hit_set": hit_set,
        "missing_set": missing_set,
        "extra_set": extra_set,
    }


@dataclass
class PathToken:
    text: str
    token_type: int
    mechanism_id: int
    parent_mz: float
    product_mz: float


@dataclass
class EdgeInfo:
    parent: str
    child: str
    text: str
    token_type: int
    mechanism_id: int
    parent_mz: float
    product_mz: float
    order: int
    triplet: List[str]


@dataclass
class PeakPathExample:
    record_index: int
    record_id: Any
    name: str
    smiles: str
    formula: str
    mw: float
    triplets: List[List[str]]
    mz_values: List[float]
    product_texts: List[str]
    paths: List[List[PathToken]]
    gold_targets: List[float]
    has_gold: bool
    original_candidate_count: int
    truncated_candidate_count: int
    morgan_fp: List[float]


def molecule_root_token(record: Dict[str, Any]) -> PathToken:
    name = str(record.get("name", "")).strip()
    smiles = str(record.get("smiles", record.get("SMILES", ""))).strip()
    formula = str(record.get("formula", record.get("Formula", ""))).strip()
    mw = parse_float(record.get("mw", record.get("MW", 0.0)))
    text = f"ROOT molecule name={name} SMILES={smiles} formula={formula} MW={mw}"
    return PathToken(
        text=text,
        token_type=TYPE_ROOT,
        mechanism_id=0,
        parent_mz=0.0,
        product_mz=0.0,
    )


def edge_text_from_triplet(t: List[str]) -> str:
    return f"EDGE precursor={t[0]} mechanism={t[1]} product={t[2]}"


def build_edge_from_triplet(
    t: List[str],
    order: int,
    molecular_mw: float,
) -> Optional[EdgeInfo]:
    precursor, mechanism, product = t
    child = parse_mz_from_product(product)
    if child is None:
        return None

    child_float = parse_float(child)
    precursor_mz = parse_precursor_mz(precursor)

    if precursor.lower().startswith("smiles_fragment:"):
        parent = "ROOT"
        parent_mz_value = molecular_mw
        token_type = TYPE_STRUCTURE_EDGE
    elif precursor_mz is not None:
        parent = precursor_mz
        parent_mz_value = parse_float(precursor_mz)
        token_type = TYPE_PRECURSOR_EDGE
    else:
        parent = "ROOT"
        parent_mz_value = molecular_mw
        token_type = TYPE_ORPHAN_EDGE

    return EdgeInfo(
        parent=parent,
        child=child,
        text=edge_text_from_triplet(t),
        token_type=token_type,
        mechanism_id=MECH_TO_ID.get(mechanism, 0),
        parent_mz=parent_mz_value,
        product_mz=child_float,
        order=order,
        triplet=t,
    )


def build_tree_paths(
    record: Dict[str, Any],
    record_index: int,
    max_peaks: int,
    morgan_radius: int,
    morgan_fp_dim: int,
) -> Optional[PeakPathExample]:
    raw_triplets = extract_triplet_list(record)
    if not isinstance(raw_triplets, list):
        return None

    triplets: List[List[str]] = []
    seen_triplets = set()
    for item in raw_triplets:
        triplet = normalize_triplet(item)
        if triplet is None:
            continue
        key = json.dumps(triplet, ensure_ascii=False, sort_keys=True)
        if key in seen_triplets:
            continue
        seen_triplets.add(key)
        triplets.append(triplet)

    if not triplets:
        return None

    molecular_mw = parse_float(record.get("mw", record.get("MW", 0.0)))
    edges: List[EdgeInfo] = []
    product_text_by_mz: Dict[str, str] = {}
    first_edge_by_mz: Dict[str, EdgeInfo] = {}

    for order, triplet in enumerate(triplets):
        edge = build_edge_from_triplet(triplet, order, molecular_mw)
        if edge is None:
            continue
        edges.append(edge)
        product_text_by_mz.setdefault(edge.child, triplet[2])
        first_edge_by_mz.setdefault(edge.child, edge)

    if not edges or not product_text_by_mz:
        return None

    adjacency: Dict[str, List[EdgeInfo]] = {}
    for edge in edges:
        adjacency.setdefault(edge.parent, []).append(edge)
    for parent in adjacency:
        adjacency[parent].sort(
            key=lambda edge: (edge.order, edge.token_type, edge.child)
        )

    parent_map: Dict[str, Tuple[str, EdgeInfo]] = {}
    queue = ["ROOT"]
    visited = {"ROOT"}
    while queue:
        current = queue.pop(0)
        for edge in adjacency.get(current, []):
            if edge.child in visited:
                continue
            visited.add(edge.child)
            parent_map[edge.child] = (current, edge)
            queue.append(edge.child)

    root_token = molecule_root_token(record)

    def edge_to_path_token(
        edge: EdgeInfo,
        token_type: Optional[int] = None,
    ) -> PathToken:
        return PathToken(
            text=edge.text,
            token_type=edge.token_type if token_type is None else token_type,
            mechanism_id=edge.mechanism_id,
            parent_mz=edge.parent_mz,
            product_mz=edge.product_mz,
        )

    def recover_path(mz: str) -> List[PathToken]:
        if mz not in parent_map:
            edge = first_edge_by_mz[mz]
            return [
                root_token,
                edge_to_path_token(edge, TYPE_ORPHAN_EDGE),
            ]

        reversed_tokens: List[PathToken] = []
        node = mz
        seen_nodes = set()
        while node != "ROOT" and node in parent_map and node not in seen_nodes:
            seen_nodes.add(node)
            parent, edge = parent_map[node]
            reversed_tokens.append(edge_to_path_token(edge))
            node = parent

        return [root_token] + list(reversed(reversed_tokens))

    intensity = record.get("intensity", {})
    has_gold = isinstance(intensity, dict) and len(intensity) > 0
    if not isinstance(intensity, dict):
        intensity = {}

    mz_keys = sorted(
        product_text_by_mz.keys(),
        key=lambda value: float(value),
        reverse=True,
    )

    original_candidate_count = len(mz_keys)
    if len(mz_keys) > max_peaks:
        mz_keys = mz_keys[:max_peaks]
    truncated_candidate_count = original_candidate_count - len(mz_keys)

    mz_values: List[float] = []
    product_texts: List[str] = []
    paths: List[List[PathToken]] = []
    gold_targets: List[float] = []

    for mz in mz_keys:
        mz_float = float(mz)
        mz_values.append(mz_float)
        product_texts.append(product_text_by_mz[mz])
        paths.append(recover_path(mz))
        gold_targets.append(
            safe_get_intensity(intensity, mz_float, 0.0)
            if has_gold
            else 0.0
        )

    if not mz_values:
        return None

    morgan_fp = smiles_to_morgan_fingerprint(
        record.get("smiles", record.get("SMILES", "")),
        radius=morgan_radius,
        n_bits=morgan_fp_dim,
    )

    return PeakPathExample(
        record_index=record_index,
        record_id=record.get("id", None),
        name=str(record.get("name", "")),
        smiles=str(record.get("smiles", record.get("SMILES", ""))),
        formula=str(record.get("formula", record.get("Formula", ""))),
        mw=molecular_mw,
        triplets=triplets,
        mz_values=mz_values,
        product_texts=product_texts,
        paths=paths,
        gold_targets=gold_targets,
        has_gold=has_gold,
        original_candidate_count=original_candidate_count,
        truncated_candidate_count=truncated_candidate_count,
        morgan_fp=morgan_fp,
    )


class Stage2InferDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        *,
        vocab_size: int,
        max_peaks: int,
        max_path_len: int,
        max_words_per_token: int,
        morgan_fp_dim: int,
        morgan_radius: int,
    ) -> None:
        self.records = records
        self.vocab_size = vocab_size
        self.max_peaks = max_peaks
        self.max_path_len = max_path_len
        self.max_words_per_token = max_words_per_token
        self.morgan_fp_dim = int(morgan_fp_dim)
        self.morgan_radius = int(morgan_radius)

        self.examples: List[PeakPathExample] = []
        self.skipped_indices: List[int] = []

        for index, record in enumerate(records):
            example = build_tree_paths(
                record,
                index,
                max_peaks=max_peaks,
                morgan_radius=self.morgan_radius,
                morgan_fp_dim=self.morgan_fp_dim,
            )
            if example is None:
                self.skipped_indices.append(index)
                continue
            self.examples.append(example)

        original_counts = [
            example.original_candidate_count
            for example in self.examples
        ]
        truncated_molecules = sum(
            example.truncated_candidate_count > 0
            for example in self.examples
        )
        removed_candidates = sum(
            example.truncated_candidate_count
            for example in self.examples
        )

        print(
            f"[Dataset] input_records={len(records)} usable={len(self.examples)} "
            f"skipped={len(self.skipped_indices)}"
        )
        if original_counts:
            print(
                "[Candidate truncation] "
                f"max_original_candidates={max(original_counts)} "
                f"mean_original_candidates="
                f"{sum(original_counts) / len(original_counts):.2f} "
                f"max_peaks={max_peaks} "
                f"molecules_over_max_peaks={truncated_molecules} "
                f"total_candidates_truncated={removed_candidates}"
            )
            if truncated_molecules == 0:
                print(
                    "[Candidate truncation] none; every molecule has "
                    f"<= {max_peaks} Stage1 candidate peaks."
                )
            else:
                print(
                    "[Candidate truncation] high-m/z candidates were retained "
                    "to match training; gold intensity was not used."
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PeakPathExample:
        return self.examples[index]


class Stage2InferCollator:
    def __init__(
        self,
        *,
        vocab_size: int,
        max_peaks: int,
        max_path_len: int,
        max_words_per_token: int,
        morgan_fp_dim: int,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_peaks = max_peaks
        self.max_path_len = max_path_len
        self.max_words = max_words_per_token
        self.morgan_fp_dim = int(morgan_fp_dim)

    def __call__(self, batch: List[PeakPathExample]) -> Dict[str, Any]:
        batch_size = len(batch)
        max_peaks = min(
            self.max_peaks,
            max(len(item.mz_values) for item in batch),
        )
        max_path_len = min(
            self.max_path_len,
            max(len(path) for item in batch for path in item.paths),
        )
        max_words = self.max_words

        path_token_ids = torch.zeros(
            (batch_size, max_peaks, max_path_len, max_words),
            dtype=torch.long,
        )
        product_token_ids = torch.zeros(
            (batch_size, max_peaks, max_words),
            dtype=torch.long,
        )
        path_type_ids = torch.zeros(
            (batch_size, max_peaks, max_path_len),
            dtype=torch.long,
        )
        path_mech_ids = torch.zeros(
            (batch_size, max_peaks, max_path_len),
            dtype=torch.long,
        )
        path_parent_mz = torch.zeros(
            (batch_size, max_peaks, max_path_len),
            dtype=torch.float32,
        )
        path_product_mz = torch.zeros(
            (batch_size, max_peaks, max_path_len),
            dtype=torch.float32,
        )

        path_mask = torch.zeros(
            (batch_size, max_peaks, max_path_len),
            dtype=torch.bool,
        )
        peak_mask = torch.zeros(
            (batch_size, max_peaks),
            dtype=torch.bool,
        )

        mz = torch.zeros(
            (batch_size, max_peaks),
            dtype=torch.float32,
        )
        target = torch.zeros(
            (batch_size, max_peaks),
            dtype=torch.float32,
        )
        mw = torch.zeros((batch_size,), dtype=torch.float32)
        morgan_fp = torch.zeros(
            (batch_size, self.morgan_fp_dim), dtype=torch.float32
        )

        meta: List[Dict[str, Any]] = []

        for batch_index, example in enumerate(batch):
            mw[batch_index] = float(example.mw)
            fp_values = example.morgan_fp[: self.morgan_fp_dim]
            if fp_values:
                morgan_fp[batch_index, : len(fp_values)] = torch.tensor(
                    fp_values, dtype=torch.float32
                )
            peak_count = min(max_peaks, len(example.mz_values))

            meta.append(
                {
                    "record_index": example.record_index,
                    "id": example.record_id,
                    "name": example.name,
                    "smiles": example.smiles,
                    "formula": example.formula,
                    "mw": example.mw,
                    "triplets": example.triplets,
                    "mz_values": example.mz_values[:peak_count],
                    "product_texts": example.product_texts[:peak_count],
                    "gold_targets": example.gold_targets[:peak_count],
                    "has_gold": example.has_gold,
                }
            )

            for peak_index in range(peak_count):
                peak_mask[batch_index, peak_index] = True
                mz[batch_index, peak_index] = float(
                    example.mz_values[peak_index]
                )
                target[batch_index, peak_index] = float(
                    example.gold_targets[peak_index]
                )

                product_token_ids[batch_index, peak_index] = torch.tensor(
                    text_to_ids(
                        example.product_texts[peak_index],
                        self.vocab_size,
                        max_words,
                    ),
                    dtype=torch.long,
                )

                path = example.paths[peak_index]
                if len(path) > max_path_len:
                    path = [path[0]] + path[-(max_path_len - 1):]

                for path_index, token in enumerate(path[:max_path_len]):
                    path_mask[
                        batch_index,
                        peak_index,
                        path_index,
                    ] = True
                    path_token_ids[
                        batch_index,
                        peak_index,
                        path_index,
                    ] = torch.tensor(
                        text_to_ids(
                            token.text,
                            self.vocab_size,
                            max_words,
                        ),
                        dtype=torch.long,
                    )
                    path_type_ids[
                        batch_index,
                        peak_index,
                        path_index,
                    ] = int(token.token_type)
                    path_mech_ids[
                        batch_index,
                        peak_index,
                        path_index,
                    ] = int(token.mechanism_id)
                    path_parent_mz[
                        batch_index,
                        peak_index,
                        path_index,
                    ] = float(token.parent_mz)
                    path_product_mz[
                        batch_index,
                        peak_index,
                        path_index,
                    ] = float(token.product_mz)

        return {
            "path_token_ids": path_token_ids,
            "product_token_ids": product_token_ids,
            "path_type_ids": path_type_ids,
            "path_mech_ids": path_mech_ids,
            "path_parent_mz": path_parent_mz,
            "path_product_mz": path_product_mz,
            "path_mask": path_mask,
            "peak_mask": peak_mask,
            "mz": mz,
            "mw": mw,
            "morgan_fp": morgan_fp,
            "target": target,
            "meta": meta,
        }


class StructuredTreePathAttentionModel(nn.Module):
    PATH_NUMERIC_DIM = 9

    def __init__(
        self,
        *,
        vocab_size: int = 65536,
        d_model: int = 1152,
        n_heads: int = 18,
        interaction_layers: int = 10,
        dropout: float = 0.08,
        max_path_len: int = 128,
        n_token_types: int = 5,
        n_mechanisms: int = 14,
        morgan_fp_dim: int = 4096,
        use_global_context: bool = True,
        use_path_attention: bool = True,
        use_inter_peak_interaction: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        self.d_model = d_model
        self.max_path_len = max_path_len
        self.morgan_fp_dim = int(morgan_fp_dim)


        self.use_global_context = bool(use_global_context)
        self.use_path_attention = bool(use_path_attention)
        self.use_inter_peak_interaction = bool(use_inter_peak_interaction)

        self.word_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.path_pos_embedding = nn.Embedding(max_path_len, d_model)
        self.path_type_embedding = nn.Embedding(n_token_types, d_model, padding_idx=0)
        self.mechanism_embedding = nn.Embedding(n_mechanisms, d_model, padding_idx=0)

        self.token_mz_mlp = nn.Sequential(
            nn.Linear(self.PATH_NUMERIC_DIM, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.product_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.peak_mz_mlp = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.global_mol_encoder = nn.Sequential(
            nn.Linear(self.morgan_fp_dim, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.global_fusion_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.global_fusion_norm = nn.LayerNorm(d_model)
        self.global_token_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        self.path_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.path_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.path_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.tree_interaction = nn.TransformerEncoder(
                encoder_layer,
                num_layers=interaction_layers,
                norm=nn.LayerNorm(d_model),
                enable_nested_tensor=False,
            )
        except TypeError:
            self.tree_interaction = nn.TransformerEncoder(
                encoder_layer,
                num_layers=interaction_layers,
                norm=nn.LayerNorm(d_model),
            )

        self.context_fusion_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.context_fusion_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.word_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.path_pos_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.path_type_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.mechanism_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.word_embedding.weight[0].zero_()
            self.path_type_embedding.weight[0].zero_()
            self.mechanism_embedding.weight[0].zero_()

    def mean_word_embed(self, ids: torch.Tensor) -> torch.Tensor:
        embedding = self.word_embedding(ids)
        mask = (ids != 0).unsqueeze(-1)
        denominator = mask.sum(dim=-2).clamp(min=1)
        return (embedding * mask).sum(dim=-2) / denominator

    @staticmethod
    def peak_mz_features(mz: torch.Tensor, mw: torch.Tensor) -> torch.Tensor:
        mw_safe = mw.clamp(min=1.0)
        return torch.stack(
            [
                mz / 1000.0,
                mw_safe / 1000.0,
                torch.log1p(mz.clamp_min(0.0)) / 10.0,
                mz / mw_safe,
            ],
            dim=-1,
        )

    @staticmethod
    def path_numeric_features(
        parent_mz: torch.Tensor,
        product_mz: torch.Tensor,
        mw: torch.Tensor,
    ) -> torch.Tensor:
        mw_safe = mw.clamp(min=1.0)
        neutral_loss = parent_mz - product_mz
        return torch.stack(
            [
                product_mz / 1000.0,
                parent_mz / 1000.0,
                neutral_loss / 1000.0,
                mw_safe / 1000.0,
                torch.log1p(product_mz.clamp_min(0.0)) / 10.0,
                torch.log1p(parent_mz.clamp_min(0.0)) / 10.0,
                product_mz / mw_safe,
                parent_mz / mw_safe,
                neutral_loss / mw_safe,
            ],
            dim=-1,
        )

    def forward(
        self,
        *,
        path_token_ids: torch.Tensor,
        product_token_ids: torch.Tensor,
        path_type_ids: torch.Tensor,
        path_mech_ids: torch.Tensor,
        path_parent_mz: torch.Tensor,
        path_product_mz: torch.Tensor,
        path_mask: torch.Tensor,
        peak_mask: torch.Tensor,
        mz: torch.Tensor,
        mw: torch.Tensor,
        morgan_fp: torch.Tensor,
        return_logits: bool = False,
    ):
        device = path_token_ids.device
        bsz, max_p, max_l, _ = path_token_ids.shape

        path_embedding = self.mean_word_embed(path_token_ids)
        positions = torch.arange(max_l, device=device).view(1, 1, max_l)
        path_embedding = path_embedding + self.path_pos_embedding(positions)
        path_embedding = path_embedding + self.path_type_embedding(path_type_ids.clamp_min(0))
        path_embedding = path_embedding + self.mechanism_embedding(path_mech_ids.clamp_min(0))

        mw_for_path = mw.view(bsz, 1, 1).expand_as(path_product_mz)
        path_embedding = path_embedding + self.token_mz_mlp(
            self.path_numeric_features(path_parent_mz, path_product_mz, mw_for_path)
        )

        product_embedding = self.product_proj(self.mean_word_embed(product_token_ids))
        mw_for_peak = mw.view(bsz, 1).expand_as(mz)
        peak_query = product_embedding + self.peak_mz_mlp(
            self.peak_mz_features(mz, mw_for_peak)
        )


        if self.use_global_context:
            global_embedding = self.global_mol_encoder(morgan_fp.float())
            global_expand = global_embedding.unsqueeze(1).expand(-1, max_p, -1)
            local_global_gate = self.global_fusion_gate(
                torch.cat([peak_query, global_expand], dim=-1)
            )
            peak_query = self.global_fusion_norm(
                peak_query + local_global_gate * global_expand
            )
        else:
            global_embedding = None

        flat_query = peak_query.reshape(bsz * max_p, 1, self.d_model)
        flat_path = path_embedding.reshape(bsz * max_p, max_l, self.d_model)
        flat_path_mask = path_mask.reshape(bsz * max_p, max_l)
        flat_peak_mask = peak_mask.reshape(bsz * max_p)


        valid = flat_peak_mask & flat_path_mask.any(dim=-1)
        valid_f = valid.to(flat_path.dtype).view(-1, 1, 1)
        safe_flat_path = flat_path * valid_f
        safe_first_column = flat_path_mask[:, :1] | (~valid).unsqueeze(1)
        safe_flat_path_mask = torch.cat(
            [safe_first_column, flat_path_mask[:, 1:]], dim=1
        )


        if self.use_path_attention:
            attended, _ = self.path_attn(
                query=flat_query,
                key=safe_flat_path,
                value=safe_flat_path,
                key_padding_mask=~safe_flat_path_mask,
                need_weights=False,
            )
            attended = attended.squeeze(1)
        else:
            path_weight = flat_path_mask.to(flat_path.dtype).unsqueeze(-1)
            path_denominator = path_weight.sum(dim=1).clamp_min(1.0)
            attended = (flat_path * path_weight).sum(dim=1) / path_denominator
            attended = attended * valid.to(attended.dtype).unsqueeze(-1)

        attended = attended + self.path_ffn(attended)
        path_representation = attended * valid.to(attended.dtype).unsqueeze(-1)
        path_representation = self.path_norm(
            path_representation + peak_query.reshape(bsz * max_p, self.d_model)
        )
        peak_representation = path_representation.reshape(bsz, max_p, self.d_model)


        if self.use_inter_peak_interaction:
            if self.use_global_context:
                molecule_token = self.global_token_proj(global_embedding).unsqueeze(1)
                interaction_input = torch.cat(
                    [molecule_token, peak_representation], dim=1
                )
                interaction_mask = torch.cat(
                    [
                        torch.ones(
                            (bsz, 1),
                            dtype=torch.bool,
                            device=device,
                        ),
                        peak_mask,
                    ],
                    dim=1,
                )
                interacted_all = self.tree_interaction(
                    interaction_input,
                    src_key_padding_mask=~interaction_mask,
                )
                molecule_context = interacted_all[:, :1]
                interacted = interacted_all[:, 1:]
                context_expand = molecule_context.expand(-1, max_p, -1)
                context_gate = self.context_fusion_gate(
                    torch.cat([interacted, context_expand], dim=-1)
                )
                interacted = self.context_fusion_norm(
                    interacted + context_gate * context_expand
                )
            else:
                interacted = self.tree_interaction(
                    peak_representation,
                    src_key_padding_mask=~peak_mask,
                )
        else:
            interacted = peak_representation

        output_logits = self.output_head(interacted).squeeze(-1)
        prediction = F.softplus(output_logits, beta=1.0, threshold=20.0) * peak_mask.float()
        ranking_logits = output_logits.masked_fill(~peak_mask, -1.0e4)
        if return_logits:
            return prediction, ranking_logits
        return prediction


EXPECTED_MODEL_INPUT_VERSION = "path_parent_product_mz_morgan_globaltoken_softplus_fiveloss_v7"
EXPECTED_PATH_NUMERIC_DIM = 9


def default_model_config() -> Dict[str, Any]:
    return {
        "vocab_size": 65536,
        "d_model": 1152,
        "n_heads": 18,
        "interaction_layers": 10,
        "dropout": 0.08,
        "max_path_len": 128,
        "max_peaks": 320,
        "max_words_per_token": 64,
        "morgan_fp_dim": 4096,
        "morgan_radius": 2,
        "input_version": EXPECTED_MODEL_INPUT_VERSION,
        "path_numeric_dim": EXPECTED_PATH_NUMERIC_DIM,
        "smiles_tokenization": "field-aware-hashed",


        "use_global_context": True,
        "use_path_attention": True,
        "use_inter_peak_interaction": True,
        "ablate_intensity_loss": False,
        "ablate_spectral_loss": False,
        "ablate_ranking_loss": False,
        "ablate_base_peak_loss": False,
        "ablate_strong_peak_loss": False,
        "ablate_global_context": False,
        "ablate_path_attention": False,
        "ablate_inter_peak_interaction": False,
        "ablate_gold_guided_curriculum": False,
        "active_ablations": [],
        "ablation_metadata_found": False,
    }


def torch_load_compat(path: str, map_location: Any) -> Any:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def _validate_model_config(
    cfg: Dict[str, Any],
    *,
    source: str,
) -> None:
    numeric_dim = int(
        cfg.get("path_numeric_dim", EXPECTED_PATH_NUMERIC_DIM)
    )
    input_version = str(
        cfg.get("input_version", EXPECTED_MODEL_INPUT_VERSION)
    )
    tokenization = str(
        cfg.get("smiles_tokenization", "field-aware-hashed")
    )

    if numeric_dim != EXPECTED_PATH_NUMERIC_DIM:
        raise ValueError(
            f"{source} uses path_numeric_dim={numeric_dim}, but this "
            f"Stage2 inference requires {EXPECTED_PATH_NUMERIC_DIM}."
        )
    if input_version != EXPECTED_MODEL_INPUT_VERSION:
        raise ValueError(
            f"{source} uses input_version={input_version!r}, but this "
            f"inference code requires {EXPECTED_MODEL_INPUT_VERSION!r}."
        )
    if tokenization != "field-aware-hashed":
        raise ValueError(
            f"{source} uses smiles_tokenization={tokenization!r}, but this "
            "inference code requires 'field-aware-hashed'."
        )


def load_train_config(
    config_file: Optional[str],
    checkpoint_path: str,
) -> Dict[str, Any]:
    cfg = default_model_config()

    checkpoint_abs = os.path.abspath(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_abs)
    parent_dir = os.path.dirname(checkpoint_dir)

    candidates: List[str] = []
    if config_file:
        candidates.append(config_file)
    candidates.append(os.path.join(checkpoint_dir, "model_config.json"))
    candidates.append(os.path.join(parent_dir, "model_config.json"))

    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue

        with open(candidate, "r", encoding="utf-8") as handle:
            obj = json.load(handle)

        model_obj = obj.get("model", obj)
        data_obj = obj.get("data", {})
        ablation_obj = obj.get("ablation", {})
        cfg.update(
            {
                "vocab_size": int(
                    model_obj.get("vocab_size", cfg["vocab_size"])
                ),
                "d_model": int(
                    model_obj.get("d_model", cfg["d_model"])
                ),
                "n_heads": int(
                    model_obj.get("n_heads", cfg["n_heads"])
                ),
                "interaction_layers": int(
                    model_obj.get(
                        "interaction_layers",
                        cfg["interaction_layers"],
                    )
                ),
                "dropout": float(
                    model_obj.get("dropout", cfg["dropout"])
                ),
                "max_path_len": int(
                    model_obj.get(
                        "max_path_len",
                        cfg["max_path_len"],
                    )
                ),
                "max_peaks": int(
                    data_obj.get("max_peaks", cfg["max_peaks"])
                ),
                "max_words_per_token": int(
                    data_obj.get(
                        "max_words_per_token",
                        cfg["max_words_per_token"],
                    )
                ),
                "morgan_fp_dim": int(
                    model_obj.get(
                        "morgan_fp_dim",
                        data_obj.get("morgan_fp_dim", cfg["morgan_fp_dim"]),
                    )
                ),
                "morgan_radius": int(
                    model_obj.get(
                        "morgan_radius",
                        data_obj.get("morgan_radius", cfg["morgan_radius"]),
                    )
                ),
                "input_version": str(
                    model_obj.get(
                        "input_version",
                        cfg["input_version"],
                    )
                ),
                "path_numeric_dim": int(
                    model_obj.get(
                        "path_numeric_dim",
                        cfg["path_numeric_dim"],
                    )
                ),
                "smiles_tokenization": str(
                    model_obj.get(
                        "smiles_tokenization",
                        cfg["smiles_tokenization"],
                    )
                ),
                "use_global_context": bool(
                    model_obj.get(
                        "use_global_context",
                        not bool(ablation_obj.get("ablate_global_context", False)),
                    )
                ),
                "use_path_attention": bool(
                    model_obj.get(
                        "use_path_attention",
                        not bool(ablation_obj.get("ablate_path_attention", False)),
                    )
                ),
                "use_inter_peak_interaction": bool(
                    model_obj.get(
                        "use_inter_peak_interaction",
                        not bool(ablation_obj.get("ablate_inter_peak_interaction", False)),
                    )
                ),
            }
        )

        for ablation_key in (
            "ablate_intensity_loss",
            "ablate_spectral_loss",
            "ablate_ranking_loss",
            "ablate_base_peak_loss",
            "ablate_strong_peak_loss",
            "ablate_global_context",
            "ablate_path_attention",
            "ablate_inter_peak_interaction",
            "ablate_gold_guided_curriculum",
        ):
            if ablation_key in ablation_obj:
                cfg[ablation_key] = bool(ablation_obj[ablation_key])

        if isinstance(ablation_obj.get("active_ablations"), list):
            cfg["active_ablations"] = list(ablation_obj["active_ablations"])

        cfg["ablation_metadata_found"] = bool(
            ablation_obj
            or "use_global_context" in model_obj
            or "use_path_attention" in model_obj
            or "use_inter_peak_interaction" in model_obj
        )


        cfg["ablate_global_context"] = not bool(cfg["use_global_context"])
        cfg["ablate_path_attention"] = not bool(cfg["use_path_attention"])
        cfg["ablate_inter_peak_interaction"] = not bool(cfg["use_inter_peak_interaction"])

        _validate_model_config(cfg, source=candidate)
        print(f"[INFO] Loaded model config from: {candidate}")
        return cfg

    try:
        checkpoint = torch_load_compat(
            checkpoint_path,
            map_location="cpu",
        )
        saved_args = (
            checkpoint.get("args", {})
            if isinstance(checkpoint, dict)
            else {}
        )
        if isinstance(saved_args, dict):
            for key in (
                "vocab_size",
                "d_model",
                "n_heads",
                "interaction_layers",
                "dropout",
                "max_path_len",
                "max_peaks",
                "max_words_per_token",
                "morgan_fp_dim",
                "morgan_radius",
            ):
                if key in saved_args:
                    cfg[key] = saved_args[key]

            ablation_keys = (
                "ablate_intensity_loss",
                "ablate_spectral_loss",
                "ablate_ranking_loss",
                "ablate_base_peak_loss",
                "ablate_strong_peak_loss",
                "ablate_global_context",
                "ablate_path_attention",
                "ablate_inter_peak_interaction",
                "ablate_gold_guided_curriculum",
            )
            found_ablation_key = False
            for key in ablation_keys:
                if key in saved_args:
                    cfg[key] = bool(saved_args[key])
                    found_ablation_key = True

            if isinstance(saved_args.get("active_ablations"), list):
                cfg["active_ablations"] = list(saved_args["active_ablations"])
                found_ablation_key = True

            cfg["ablation_metadata_found"] = bool(found_ablation_key)
            cfg["use_global_context"] = not bool(cfg["ablate_global_context"])
            cfg["use_path_attention"] = not bool(cfg["ablate_path_attention"])
            cfg["use_inter_peak_interaction"] = not bool(cfg["ablate_inter_peak_interaction"])

        if isinstance(checkpoint, dict):
            cfg["input_version"] = checkpoint.get(
                "model_input_version",
                cfg["input_version"],
            )

            state = checkpoint.get("model", {})
            if isinstance(state, dict):
                state = strip_prefix_if_present(state, "_orig_mod.")
                state = strip_prefix_if_present(state, "module.")
                weight = state.get("token_mz_mlp.0.weight")
                if torch.is_tensor(weight):
                    cfg["path_numeric_dim"] = int(weight.shape[1])

        _validate_model_config(
            cfg,
            source="checkpoint metadata",
        )
        print("[INFO] Loaded model config from checkpoint.")
        return cfg
    except Exception as exc:
        raise RuntimeError(
            "Could not load a compatible Stage2 configuration. "
            "Provide the model_config.json generated by "
            "stage2_train_union_ablation.py."
        ) from exc


ABLATION_DISPLAY_NAMES = {
    "ablate_intensity_loss": "w/o Intensity loss",
    "ablate_spectral_loss": "w/o Spectral loss",
    "ablate_ranking_loss": "w/o Ranking loss",
    "ablate_base_peak_loss": "w/o Base-peak loss",
    "ablate_strong_peak_loss": "w/o Strong-peak loss",
    "ablate_global_context": "w/o Global molecular context",
    "ablate_path_attention": "w/o Path attention",
    "ablate_inter_peak_interaction": "w/o Inter-peak interaction",
    "ablate_gold_guided_curriculum": "w/o Gold-guided curriculum",
}

ARCHITECTURE_ABLATION_KEYS = {
    "ablate_global_context",
    "ablate_path_attention",
    "ablate_inter_peak_interaction",
}


def resolve_inference_ablation_configuration(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    resolved = dict(cfg)
    metadata_found = bool(resolved.get("ablation_metadata_found", False))

    for key in ABLATION_DISPLAY_NAMES:
        saved_value = bool(resolved.get(key, False))
        cli_value = getattr(args, key, None)

        if cli_value is None:
            effective = saved_value
        else:
            cli_value = bool(cli_value)
            if (
                metadata_found
                and cli_value != saved_value
                and not args.allow_ablation_override
            ):
                raise ValueError(
                    f"Inference ablation mismatch for {key}: checkpoint/config "
                    f"says {saved_value}, but CLI requests {cli_value}. Use the "
                    "checkpoint produced by the matching training ablation. "
                    "Only use --allow_ablation_override for debugging."
                )
            effective = cli_value

        resolved[key] = effective

    resolved["use_global_context"] = not bool(
        resolved["ablate_global_context"]
    )
    resolved["use_path_attention"] = not bool(
        resolved["ablate_path_attention"]
    )
    resolved["use_inter_peak_interaction"] = not bool(
        resolved["ablate_inter_peak_interaction"]
    )

    active = [
        display_name
        for key, display_name in ABLATION_DISPLAY_NAMES.items()
        if bool(resolved.get(key, False))
    ]
    resolved["active_ablations"] = active
    resolved["is_full_model"] = len(active) == 0
    return resolved


def print_inference_ablation_configuration(cfg: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("[INFERENCE ABLATION CONFIGURATION]")
    print("=" * 80)
    if cfg.get("is_full_model", False):
        print("[ABLATION] Full model: all ablation switches are OFF.")
    else:
        for name in cfg.get("active_ablations", []):
            print(f"[ABLATION] {name}")

    print(
        "[FORWARD SWITCHES] "
        f"global_context={cfg['use_global_context']}, "
        f"path_attention={cfg['use_path_attention']}, "
        f"inter_peak_interaction={cfg['use_inter_peak_interaction']}"
    )
    print(
        "[NOTE] Loss/curriculum ablation flags describe how the checkpoint "
        "was trained; only the three architecture switches above alter the "
        "inference forward path."
    )
    print("=" * 80 + "\n")


def strip_prefix_if_present(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> Dict[str, Any]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch_load_compat(
        checkpoint_path,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    state = strip_prefix_if_present(state, "_orig_mod.")
    state = strip_prefix_if_present(state, "module.")

    numeric_weight = state.get("token_mz_mlp.0.weight")
    if torch.is_tensor(numeric_weight):
        found_dim = int(numeric_weight.shape[1])
        if found_dim != EXPECTED_PATH_NUMERIC_DIM:
            raise ValueError(
                "The checkpoint belongs to the old 4-dimensional path "
                f"feature model (found {found_dim}) or another incompatible "
                f"model. Expected {EXPECTED_PATH_NUMERIC_DIM}."
            )

    missing, unexpected = model.load_state_dict(
        state,
        strict=strict,
    )
    if missing:
        print(
            f"[WARN] Missing keys when loading checkpoint: {len(missing)}"
        )
        print("       first missing:", missing[:5])
    if unexpected:
        print(
            f"[WARN] Unexpected keys when loading checkpoint: "
            f"{len(unexpected)}"
        )
        print("       first unexpected:", unexpected[:5])

    model.to(device)
    model.eval()

    metadata: Dict[str, Any] = {}
    if isinstance(checkpoint, dict):
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "best_cosine": checkpoint.get("best_cosine"),
            "best_val": checkpoint.get("best_val"),
            "phase_name": checkpoint.get("phase_name"),
            "selected_model": checkpoint.get("selected_model"),
            "model_input_version": checkpoint.get(
                "model_input_version"
            ),
            "val_logs": checkpoint.get("val_logs", {}),
            "checkpoint_args": checkpoint.get("args", {}),
        }
        saved_args = checkpoint.get("args", {})
        if isinstance(saved_args, dict):
            metadata["active_ablations"] = saved_args.get("active_ablations", [])
            metadata["ablation_flags"] = {
                key: bool(saved_args.get(key, False))
                for key in ABLATION_DISPLAY_NAMES
            }
    return metadata


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    records: List[Dict[str, Any]],
    device: torch.device,
    *,
    amp: bool,
    amp_dtype: str,
    keep_gold: bool,
    keep_triplets: bool,
    keep_input_record: bool,
    save_pred_as_intensity: bool,
    normalize_base_peak: bool,
    round_digits: int,
) -> List[Dict[str, Any]]:
    outputs_by_index: Dict[int, Dict[str, Any]] = {}
    use_amp = amp and device.type == "cuda"
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    model.eval()

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)

        with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            pred = model(
                path_token_ids=batch["path_token_ids"],
                product_token_ids=batch["product_token_ids"],
                path_type_ids=batch["path_type_ids"],
                path_mech_ids=batch["path_mech_ids"],
                path_parent_mz=batch["path_parent_mz"],
                path_product_mz=batch["path_product_mz"],
                path_mask=batch["path_mask"],
                peak_mask=batch["peak_mask"],
                mz=batch["mz"],
                mw=batch["mw"],
                morgan_fp=batch["morgan_fp"],
            )


        if normalize_base_peak:
            pred = normalize_prediction_by_base_peak(
                pred,
                batch["peak_mask"],
            )
        else:
            pred = pred * batch["peak_mask"].float()

        pred = pred.detach().float().cpu()
        peak_mask = batch["peak_mask"].detach().cpu()

        for b, meta in enumerate(batch["meta"]):
            valid_n = int(peak_mask[b].sum().item())
            mz_values = meta["mz_values"][:valid_n]
            product_texts = meta["product_texts"][:valid_n]


            pred_values = [
                max(0.0, min(1.0, float(pred[b, i].item())))
                for i in range(valid_n)
            ]

            predicted_intensity: Dict[str, float] = {}
            predicted_peaks: List[Dict[str, Any]] = []

            for mz, product, yhat in zip(mz_values, product_texts, pred_values):
                mz_key = format_mz_key(float(mz))
                yhat_round = round(float(yhat), round_digits)
                predicted_intensity[mz_key] = yhat_round
                predicted_peaks.append({
                    "mz": int(float(mz)) if float(mz).is_integer() else float(mz),
                    "product_ion": product,
                    "pred_intensity": yhat_round,
                })

            predicted_peaks.sort(key=lambda x: float(x["mz"]), reverse=True)

            out = {
                "id": meta["id"],
                "name": meta["name"],
                "smiles": meta["smiles"],
                "formula": meta["formula"],
                "mw": meta["mw"],
                "predicted_intensity": predicted_intensity,
                "predicted_peaks": predicted_peaks,
            }

            if save_pred_as_intensity:
                out["intensity"] = predicted_intensity

            if keep_triplets:
                out["stage1_triplets"] = meta["triplets"]

                out["triplets"] = meta["triplets"]

            if keep_input_record:
                out["stage1_input_record"] = records[int(meta["record_index"])]

            outputs_by_index[int(meta["record_index"])] = out

        if batch_idx % 10 == 0 or batch_idx == len(loader):
            print(f"[INFO] inference batch {batch_idx}/{len(loader)} done")

    final_outputs: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        if idx in outputs_by_index:
            final_outputs.append(outputs_by_index[idx])
        else:
            final_outputs.append({
                "id": record.get("id", None),
                "name": record.get("name", ""),
                "smiles": record.get("smiles", record.get("SMILES", "")),
                "formula": record.get("formula", ""),
                "mw": record.get("mw", ""),
                "predicted_intensity": {},
                "predicted_peaks": [],
                "parse_ok": False,
                "error": "No valid Stage1 triplets or no valid product m/z could be parsed.",
            })

    return final_outputs


def vector_cosine(pred: List[float], gold: List[float]) -> Optional[float]:
    if not pred or not gold or len(pred) != len(gold):
        return None
    p = torch.tensor(pred, dtype=torch.float32)
    y = torch.tensor(gold, dtype=torch.float32)
    if p.norm().item() <= 1e-12 or y.norm().item() <= 1e-12:
        return None
    return float(torch.dot(p, y) / (p.norm() * y.norm() + 1e-8))


def spectral_cosine(pred: List[float], gold: List[float]) -> Optional[float]:
    return vector_cosine(pred, gold)


def pearson_corr(pred: List[float], gold: List[float]) -> Optional[float]:
    if len(pred) < 2 or len(pred) != len(gold):
        return None
    p = torch.tensor(pred, dtype=torch.float32)
    y = torch.tensor(gold, dtype=torch.float32)
    p = p - p.mean()
    y = y - y.mean()
    denom = p.norm() * y.norm()
    if denom.item() <= 1e-12:
        return None
    return float(torch.dot(p, y) / (denom + 1e-8))


def rankdata_average(values: List[float]) -> List[float]:
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def spearman_corr(pred: List[float], gold: List[float]) -> Optional[float]:
    if len(pred) < 2 or len(pred) != len(gold):
        return None
    return pearson_corr(rankdata_average(pred), rankdata_average(gold))


def sort_mz_keys(keys: List[Any]) -> List[str]:
    uniq = []
    seen = set()
    for k in keys:
        nk = normalize_mz_value(k)
        if nk is None or nk in seen:
            continue
        seen.add(nk)
        uniq.append(nk)
    uniq.sort(key=lambda x: float(x))
    return uniq


def normalize_spectrum_for_plot(spec: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(spec, dict) or not spec:
        return {}
    out: Dict[str, float] = {}
    max_v = 0.0
    for k, v in spec.items():
        nk = normalize_mz_value(k)
        if nk is None:
            continue
        fv = max(0.0, parse_float(v, 0.0))
        out[nk] = fv
        max_v = max(max_v, fv)
    if max_v <= 1e-12:
        return {k: 0.0 for k in out}
    return {k: v / max_v for k, v in out.items()}


def spectrum_from_intensity(intensity: Any) -> Dict[str, float]:
    normalized = normalize_intensity_mapping(intensity)
    return {
        key: max(0.0, min(1.0, parse_float(value, 0.0)))
        for key, value in normalized.items()
    }


def normalize_product_formula(
    formula: Any,
    *,
    ignore_radical_dot: bool = True,
    add_missing_charge: bool = True,
) -> str:
    s = str(formula).strip().replace(" ", "")
    if not s:
        return s

    if ignore_radical_dot:
        s = s.replace("·", "").replace(".", "")

    s = s.replace("++", "+").replace("--", "-")

    if add_missing_charge and s and not s.endswith("+") and not s.endswith("-"):
        s += "+"

    return s


def parse_product_ion_unit(
    product_ion: Any,
    *,
    ignore_radical_dot: bool = True,
    add_missing_charge: bool = True,
) -> Optional[Tuple[str, str]]:
    if not isinstance(product_ion, str):
        return None

    m = MZ_RE.search(product_ion)
    if not m:
        return None

    mz = normalize_mz_value(m.group(1))
    if mz is None:
        return None

    formula_part = product_ion[:m.start()].strip()
    formula = normalize_product_formula(
        formula_part,
        ignore_radical_dot=ignore_radical_dot,
        add_missing_charge=add_missing_charge,
    )

    if not formula:
        return None

    return formula, mz


def product_unit_to_str(unit: Tuple[str, str]) -> str:
    return f"{unit[0]} (m/z {unit[1]})"


def product_unit_set_from_triplets(record: Dict[str, Any]) -> Set[Tuple[str, str]]:
    raw_triplets = record.get("triplets", record.get("infer_result", []))
    out: Set[Tuple[str, str]] = set()
    if not isinstance(raw_triplets, list):
        return out

    for item in raw_triplets:
        t = normalize_triplet(item)
        if t is None:
            continue
        unit = parse_product_ion_unit(t[2])
        if unit is not None:
            out.add(unit)

    return out


def mz_set_from_triplets(record: Dict[str, Any]) -> Set[str]:
    raw_triplets = record.get("triplets", record.get("infer_result", []))
    out: Set[str] = set()
    if not isinstance(raw_triplets, list):
        return out
    for item in raw_triplets:
        t = normalize_triplet(item)
        if t is None:
            continue
        mz = parse_mz_from_product(t[2])
        if mz is not None:
            out.add(mz)
    return out


def topk_keys_by_value(spec: Dict[str, float], k: int) -> List[str]:
    items = sorted(spec.items(), key=lambda kv: (float(kv[1]), float(kv[0])), reverse=True)
    return [x[0] for x in items[:max(1, k)]]


def metrics_on_keys(pred_spec: Dict[str, float], gold_spec: Dict[str, float], keys: List[str]) -> Dict[str, Any]:
    keys = sort_mz_keys(keys)
    if not keys:
        return {
            "num_peaks": 0,
            "cosine": None,
            "mae": None,
            "mse": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
            "weighted_mae_by_gold": None,
            "max_abs_error": None,
        }

    pred_vec = [float(pred_spec.get(k, 0.0)) for k in keys]
    gold_vec = [float(gold_spec.get(k, 0.0)) for k in keys]
    errors = [abs(p - g) for p, g in zip(pred_vec, gold_vec)]
    mse = sum((p - g) ** 2 for p, g in zip(pred_vec, gold_vec)) / len(keys)
    mae = sum(errors) / len(keys)

    gold_sum = sum(gold_vec)
    if gold_sum > 1e-12:
        weighted_mae = sum(abs(p - g) * g for p, g in zip(pred_vec, gold_vec)) / gold_sum
    else:
        weighted_mae = None

    return {
        "num_peaks": len(keys),
        "cosine": vector_cosine(pred_vec, gold_vec),
        "mae": mae,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "pearson": pearson_corr(pred_vec, gold_vec),
        "spearman": spearman_corr(pred_vec, gold_vec),
        "weighted_mae_by_gold": weighted_mae,
        "max_abs_error": max(errors) if errors else None,
    }


def round_nested(obj: Any, digits: int) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_nested(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_nested(v, digits) for v in obj]
    return obj


def compute_full_stage1_vs_gold_metrics(
    *,
    pred_spec: Dict[str, float],
    stage1_record: Dict[str, Any],
    gold_record: Dict[str, Any],
    round_digits: int,
) -> Dict[str, Any]:
    gold_spec = spectrum_from_intensity(gold_record.get("intensity", {}))
    pred_spec = spectrum_from_intensity(pred_spec)


    merged_for_stage1_prf = dict(stage1_record)

    for key_gold in [
        "corrected_triplet",
        "corrected_triplets",
        "gold_triplets",
        "label_triplets",
        "triplets",
        "output",
    ]:
        if key_gold in gold_record:
            merged_for_stage1_prf[key_gold] = gold_record[key_gold]
            break

    for k in ["id", "name", "smiles", "SMILES", "formula", "mw"]:
        if k not in merged_for_stage1_prf and k in gold_record:
            merged_for_stage1_prf[k] = gold_record[k]


    has_standard_pred_field = any(
        isinstance(stage1_record.get(k), list)
        for k in ["infer_result", "pred_triplets", "predicted_triplets", "generated_triplets"]
    ) or isinstance(stage1_record.get("infer_object"), dict)

    if not has_standard_pred_field and isinstance(stage1_record.get("triplets"), list):
        merged_for_stage1_prf["infer_result"] = stage1_record["triplets"]

    pred_triplets_for_prf = get_pred_triplets_for_prf(merged_for_stage1_prf)
    gold_triplets_for_prf = get_gold_triplets_for_prf(merged_for_stage1_prf)

    pred_product_units = set(extract_product_units_for_prf(pred_triplets_for_prf, unique=True))
    gold_product_units = set(extract_product_units_for_prf(gold_triplets_for_prf, unique=True))
    product_prf = compute_prf_sets_for_prf(gold_product_units, pred_product_units)

    product_matched = product_prf["hit_set"]
    product_missing = product_prf["missing_set"]
    product_extra = product_prf["extra_set"]


    pred_mz_from_product_units = set(str(x[1]) for x in pred_product_units)
    gold_mz_from_product_units = set(str(x[1]) for x in gold_product_units)
    mz_prf = compute_prf_sets_for_prf(gold_mz_from_product_units, pred_mz_from_product_units)

    matched_mz = mz_prf["hit_set"]
    missing_mz = mz_prf["missing_set"]
    extra_mz = mz_prf["extra_set"]


    pred_mz_keys_from_intensity = set(sort_mz_keys(list(pred_spec.keys())))
    gold_mz_keys_from_intensity = set(sort_mz_keys(list(gold_spec.keys())))

    weighted_matched_gold_mz = gold_mz_keys_from_intensity & pred_mz_from_product_units
    weighted_missing_gold_mz = gold_mz_keys_from_intensity - pred_mz_from_product_units

    gold_total_intensity = sum(gold_spec.get(k, 0.0) for k in gold_mz_keys_from_intensity)
    covered_gold_intensity = sum(gold_spec.get(k, 0.0) for k in weighted_matched_gold_mz)
    missing_gold_intensity = sum(gold_spec.get(k, 0.0) for k in weighted_missing_gold_mz)

    weighted_matched_pred_mz = pred_mz_keys_from_intensity & gold_mz_keys_from_intensity
    weighted_extra_pred_mz = pred_mz_keys_from_intensity - gold_mz_keys_from_intensity

    pred_total_intensity = sum(pred_spec.get(k, 0.0) for k in pred_mz_keys_from_intensity)
    matched_pred_intensity = sum(pred_spec.get(k, 0.0) for k in weighted_matched_pred_mz)
    extra_pred_intensity = sum(pred_spec.get(k, 0.0) for k in weighted_extra_pred_mz)

    weighted_mz_recall = covered_gold_intensity / gold_total_intensity if gold_total_intensity > 1e-12 else None
    weighted_mz_precision = matched_pred_intensity / pred_total_intensity if pred_total_intensity > 1e-12 else None
    if weighted_mz_precision is not None and weighted_mz_recall is not None and (weighted_mz_precision + weighted_mz_recall) > 0:
        weighted_mz_f1 = 2 * weighted_mz_precision * weighted_mz_recall / (weighted_mz_precision + weighted_mz_recall)
    else:
        weighted_mz_f1 = None

    missing_intensity_fraction = missing_gold_intensity / gold_total_intensity if gold_total_intensity > 1e-12 else None
    extra_predicted_intensity_fraction = extra_pred_intensity / pred_total_intensity if pred_total_intensity > 1e-12 else None


    pred_mz_keys = pred_mz_keys_from_intensity
    gold_mz_keys = gold_mz_keys_from_intensity
    union_mz_keys = pred_mz_keys | gold_mz_keys

    gold_base_mz = None
    if gold_spec:
        gold_base_mz = max(gold_spec.keys(), key=lambda k: (gold_spec.get(k, 0.0), float(k)))

    pred_base_mz = None
    if pred_spec:
        pred_base_mz = max(pred_spec.keys(), key=lambda k: (pred_spec.get(k, 0.0), float(k)))

    stage1_candidate_contains_gold_base = bool(gold_base_mz in pred_mz_from_product_units) if gold_base_mz is not None else None
    pred_base_mz_hit = bool(pred_base_mz == gold_base_mz) if pred_base_mz is not None and gold_base_mz is not None else None

    topk_eval: Dict[str, Any] = {}
    for k in [1, 3, 5, 7, 9, 11, 13, 15]:
        gold_top_list = topk_keys_by_value(gold_spec, k)
        pred_top_list = topk_keys_by_value(pred_spec, k)

        gold_top = set(gold_top_list)
        pred_top = set(pred_top_list)


        if gold_top:
            stage1_hit = len(gold_top & pred_mz_from_product_units)
            stage1_topk_recall = stage1_hit / len(gold_top)
        else:
            stage1_hit = 0
            stage1_topk_recall = None


        pred_hit = len(pred_top & gold_top)
        pred_precision = pred_hit / len(pred_top) if pred_top else None
        pred_recall = pred_hit / len(gold_top) if gold_top else None
        if pred_precision is not None and pred_recall is not None and (pred_precision + pred_recall) > 0:
            pred_f1 = 2 * pred_precision * pred_recall / (pred_precision + pred_recall)
        else:
            pred_f1 = None

        topk_eval[f"top{k}_stage1_gold_coverage_recall"] = stage1_topk_recall
        topk_eval[f"top{k}_stage1_gold_coverage_hit_count"] = stage1_hit
        topk_eval[f"top{k}_pred_precision"] = pred_precision
        topk_eval[f"top{k}_pred_recall"] = pred_recall
        topk_eval[f"top{k}_pred_f1"] = pred_f1
        topk_eval[f"top{k}_pred_overlap_count"] = pred_hit


        topk_eval[f"gold_top{k}_covered_by_stage1_mz_recall"] = stage1_topk_recall
        topk_eval[f"pred_top{k}_overlap_gold_top{k}"] = pred_recall

    triplet_mz_prf = compute_prf_sets_for_prf(gold_mz_from_product_units, pred_mz_from_product_units)

    metrics = {
        "stage1_product_ion_prf": {
            "num_stage1_product_ions": product_prf["n_pred"],
            "num_gold_product_ions": product_prf["n_gold"],
            "num_matched_product_ions": product_prf["n_hit"],
            "num_missing_gold_product_ions": len(product_missing),
            "num_extra_predicted_product_ions": len(product_extra),
            "precision": product_prf["precision"],
            "recall": product_prf["recall"],
            "f1": product_prf["f1"],
            "missing_gold_product_ions": sorted([product_unit_to_str_for_prf(x) for x in product_missing]),
            "extra_predicted_product_ions": sorted([product_unit_to_str_for_prf(x) for x in product_extra]),
        },
        "stage1_mz_prf": {
            "num_predicted_mz_peaks": mz_prf["n_pred"],
            "num_gold_mz_peaks": mz_prf["n_gold"],
            "num_matched_mz_peaks": mz_prf["n_hit"],
            "num_missing_gold_mz_peaks": len(missing_mz),
            "num_extra_predicted_mz_peaks": len(extra_mz),
            "precision": mz_prf["precision"],
            "recall": mz_prf["recall"],
            "f1": mz_prf["f1"],
            "missing_gold_mz": sorted(missing_mz, key=lambda x: float(x)),
            "extra_predicted_mz": sorted(extra_mz, key=lambda x: float(x)),
        },
        "stage1_weighted_mz_prf": {
            "precision": weighted_mz_precision,
            "recall": weighted_mz_recall,
            "f1": weighted_mz_f1,
            "covered_gold_intensity": covered_gold_intensity,
            "total_gold_intensity": gold_total_intensity,
            "matched_predicted_intensity": matched_pred_intensity,
            "total_predicted_intensity": pred_total_intensity,
            "missing_gold_intensity_fraction": missing_intensity_fraction,
            "extra_predicted_intensity_fraction": extra_predicted_intensity_fraction,
        },
        "stage1_peak_generation": {
            "num_stage1_predicted_label_peaks": product_prf["n_pred"],
            "num_gold_label_peaks": product_prf["n_gold"],
            "num_matched_label_peaks": product_prf["n_hit"],
            "num_missing_gold_label_peaks": len(product_missing),
            "num_extra_predicted_label_peaks": len(product_extra),
            "peak_precision": product_prf["precision"],
            "peak_recall": product_prf["recall"],
            "peak_f1": product_prf["f1"],
            "gold_base_mz": gold_base_mz,
            "predicted_base_mz": pred_base_mz,
            "stage1_candidate_contains_gold_base": stage1_candidate_contains_gold_base,
            "predicted_base_mz_hit": pred_base_mz_hit,
            **topk_eval,
        },
        "stage1_triplet_mz_coverage": {
            "num_stage1_triplet_mz": triplet_mz_prf["n_pred"],
            "num_gold_triplet_mz": triplet_mz_prf["n_gold"],
            "num_matched_triplet_mz": triplet_mz_prf["n_hit"],
            "triplet_mz_precision": triplet_mz_prf["precision"],
            "triplet_mz_recall": triplet_mz_prf["recall"],
            "triplet_mz_f1": triplet_mz_prf["f1"],
        },
        "intensity_on_predicted_label_peaks": metrics_on_keys(
            pred_spec,
            gold_spec,
            sorted(pred_mz_keys, key=lambda x: float(x)),
        ),
        "intensity_on_all_gold_label_peaks": metrics_on_keys(
            pred_spec,
            gold_spec,
            sorted(gold_mz_keys, key=lambda x: float(x)),
        ),
        "intensity_on_union_peaks": metrics_on_keys(
            pred_spec,
            gold_spec,
            sorted(union_mz_keys, key=lambda x: float(x)),
        ),
    }

    return round_nested(metrics, round_digits)


def compute_metrics_for_record(pred: List[float], gold: List[float]) -> Dict[str, Any]:
    if not pred or not gold or len(pred) != len(gold):
        return {}
    base = metrics_on_keys(
        {str(i): p for i, p in enumerate(pred)},
        {str(i): g for i, g in enumerate(gold)},
        [str(i) for i in range(len(pred))],
    )
    return {
        "mae": base.get("mae"),
        "mse": base.get("mse"),
        "rmse": base.get("rmse"),
        "spectral_cosine": base.get("cosine"),
        "pearson": base.get("pearson"),
        "spearman": base.get("spearman"),
    }


def make_safe_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r'[\/:*?"<>|]+', '_', str(text)).strip()
    text = re.sub(r'\s+', '_', text)
    return text[:max_len] if text else 'unnamed'


def make_compact_plot_filename(prefix_index: int, rid: Any, formula: Any, *, digits: int = 3) -> str:
    try:
        order_text = f"{int(prefix_index):0{digits}d}"
    except Exception:
        order_text = str(prefix_index)
    rid_text = make_safe_filename(f"id{rid}", max_len=24)
    formula_text = make_safe_filename(formula or 'NA', max_len=40)
    return f"{order_text}_{rid_text}_{formula_text}.png"


def enrich_outputs_with_gold_evaluation(
    outputs: List[Dict[str, Any]],
    stage1_records: List[Dict[str, Any]],
    gold_records: List[Dict[str, Any]],
    *,
    keep_gold_triplets: bool,
    round_digits: int,
) -> List[Dict[str, Any]]:
    for output, stage1_record, gold_record in zip(outputs, stage1_records, gold_records):
        pred_spec = spectrum_from_intensity(output.get("predicted_intensity", {}))
        gold_spec = spectrum_from_intensity(gold_record.get("intensity", {}))

        if not pred_spec or not gold_spec:
            continue

        full_eval = compute_full_stage1_vs_gold_metrics(
            pred_spec=pred_spec,
            stage1_record=stage1_record,
            gold_record=gold_record,
            round_digits=round_digits,
        )

        output["gold_intensity"] = {
            k: round(v, round_digits)
            for k, v in sorted(gold_spec.items(), key=lambda kv: float(kv[0]))
        }


        for peak in output.get("predicted_peaks", []):
            mz_key = normalize_mz_value(peak.get("mz"))
            if mz_key is None:
                continue
            gold_y = float(gold_spec.get(mz_key, 0.0))
            pred_y = float(peak.get("pred_intensity", 0.0))
            peak["gold_intensity"] = round(gold_y, round_digits)
            peak["abs_error"] = round(abs(pred_y - gold_y), round_digits)
            peak["is_gold_label_peak"] = mz_key in gold_spec

        output["evaluation"] = full_eval


        output["metrics"] = {

            "stage1_peak_precision": full_eval["stage1_product_ion_prf"]["precision"],
            "stage1_peak_recall": full_eval["stage1_product_ion_prf"]["recall"],
            "stage1_peak_f1": full_eval["stage1_product_ion_prf"]["f1"],

            "stage1_product_precision": full_eval["stage1_product_ion_prf"]["precision"],
            "stage1_product_recall": full_eval["stage1_product_ion_prf"]["recall"],
            "stage1_product_f1": full_eval["stage1_product_ion_prf"]["f1"],


            "stage1_mz_precision": full_eval["stage1_mz_prf"]["precision"],
            "stage1_mz_recall": full_eval["stage1_mz_prf"]["recall"],
            "stage1_mz_f1": full_eval["stage1_mz_prf"]["f1"],


            "stage1_weighted_mz_precision": full_eval["stage1_weighted_mz_prf"]["precision"],
            "stage1_weighted_mz_recall": full_eval["stage1_weighted_mz_prf"]["recall"],
            "stage1_weighted_mz_f1": full_eval["stage1_weighted_mz_prf"]["f1"],


            "weighted_peak_recall_by_gold_intensity": full_eval["stage1_weighted_mz_prf"]["recall"],

            "missing_gold_intensity_fraction": full_eval["stage1_weighted_mz_prf"]["missing_gold_intensity_fraction"],
            "extra_predicted_intensity_fraction": full_eval["stage1_weighted_mz_prf"]["extra_predicted_intensity_fraction"],
            "predicted_base_mz_hit": full_eval["stage1_peak_generation"]["predicted_base_mz_hit"],
            "cosine_predicted_label_peaks": full_eval["intensity_on_predicted_label_peaks"]["cosine"],
            "cosine_all_label_peaks": full_eval["intensity_on_all_gold_label_peaks"]["cosine"],
            "cosine_union_peaks": full_eval["intensity_on_union_peaks"]["cosine"],
            "mae_predicted_label_peaks": full_eval["intensity_on_predicted_label_peaks"]["mae"],
            "mae_all_gold_label_peaks": full_eval["intensity_on_all_gold_label_peaks"]["mae"],
            "rmse_all_gold_label_peaks": full_eval["intensity_on_all_gold_label_peaks"]["rmse"],
            "pearson_all_gold_label_peaks": full_eval["intensity_on_all_gold_label_peaks"]["pearson"],
            "spearman_all_gold_label_peaks": full_eval["intensity_on_all_gold_label_peaks"]["spearman"],
        }


        for k in [1, 3, 5, 7, 9, 11, 13, 15]:
            stage1_block = full_eval["stage1_peak_generation"]
            output["metrics"][f"top{k}_stage1_gold_coverage_recall"] = stage1_block.get(f"top{k}_stage1_gold_coverage_recall")
            output["metrics"][f"top{k}_pred_precision"] = stage1_block.get(f"top{k}_pred_precision")
            output["metrics"][f"top{k}_pred_recall"] = stage1_block.get(f"top{k}_pred_recall")
            output["metrics"][f"top{k}_pred_f1"] = stage1_block.get(f"top{k}_pred_f1")
            output["metrics"][f"top{k}_pred_overlap_count"] = stage1_block.get(f"top{k}_pred_overlap_count")

        output["cosine_validation"] = {
            "cosine_predicted_label_peaks": full_eval["intensity_on_predicted_label_peaks"]["cosine"],
            "cosine_all_label_peaks": full_eval["intensity_on_all_gold_label_peaks"]["cosine"],
            "cosine_union_peaks": full_eval["intensity_on_union_peaks"]["cosine"],
            "num_predicted_label_peaks": full_eval["stage1_peak_generation"]["num_stage1_predicted_label_peaks"],
            "num_all_label_peaks": full_eval["stage1_peak_generation"]["num_gold_label_peaks"],
            "num_union_peaks": full_eval["intensity_on_union_peaks"]["num_peaks"],
        }

        output["stage1_recall_validation"] = {
            "main_product_recall_formula_mz": full_eval["stage1_product_ion_prf"]["recall"],
            "mz_recall": full_eval["stage1_mz_prf"]["recall"],
            "weighted_mz_recall_by_gold_intensity": full_eval["stage1_weighted_mz_prf"]["recall"],
        }

        if keep_gold_triplets:
            output["gold_triplets"] = gold_record.get("triplets", gold_record.get("infer_result", []))

    return outputs


def _apply_nature_spectrum_style(
    ax,
    *,
    xlabel: str = 'm/z',
    ylabel: str = 'Intensity',
    mirrored: bool = False,
) -> None:
    ax.set_facecolor('white')
    ax.grid(False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)

    ax.tick_params(
        axis='both', which='major',
        labelsize=13, width=0.9, length=4,
        direction='out', pad=4,
    )
    ax.tick_params(axis='both', which='minor', width=0.7, length=2.5, direction='out')

    ax.set_xlabel(xlabel, fontsize=18, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=10)

    if mirrored:
        ax.set_ylim(-1.02, 1.02)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_yticklabels(['100', '50', '0', '50', '100'])
        ax.axhline(0.0, linewidth=0.9, color='black', zorder=1)
    else:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0', '25', '50', '75', '100'])
        ax.axhline(0.0, linewidth=0.9, color='black', zorder=1)


def _set_compact_xrange(ax, x_values: List[float]) -> None:
    if not x_values:
        ax.set_xlim(0, 10)
        return
    xmin = min(x_values)
    xmax = max(x_values)
    span = max(1.0, xmax - xmin)
    margin = max(1.0, span * 0.02)
    ax.set_xlim(max(0.0, xmin - margin), xmax + margin)


def plot_mirror_spectrum(
    *,
    output_record: Dict[str, Any],
    gold_record: Dict[str, Any],
    save_path: str,
    round_digits: int,
) -> Dict[str, Any]:
    pred_spec = output_record.get('predicted_intensity', {})
    gold_spec = gold_record.get('intensity', output_record.get('gold_intensity', {}))

    pred_keys = sort_mz_keys(list(pred_spec.keys()))
    gold_keys = sort_mz_keys(list(gold_spec.keys()))
    union_keys = sort_mz_keys(list(set(pred_keys) | set(gold_keys)))

    pred_plot = normalize_spectrum_for_plot({k: pred_spec.get(k, 0.0) for k in union_keys})
    gold_plot = normalize_spectrum_for_plot({k: gold_spec.get(k, 0.0) for k in union_keys})

    x = [float(k) for k in union_keys]
    y_pred = [pred_plot.get(k, 0.0) for k in union_keys]
    y_gold = [-gold_plot.get(k, 0.0) for k in union_keys]

    cosine_info = output_record.get('cosine_validation', {})
    cos_pred = cosine_info.get('cosine_predicted_label_peaks')
    cos_all = cosine_info.get('cosine_all_label_peaks')

    fig, ax = plt.subplots(figsize=(13.5, 5.8), dpi=220)

    if x:
        ax.vlines(
            x,
            [0.0] * len(x),
            y_pred,
            linewidth=1.0,
            colors='#C73E3A',
            label='Pred',
            zorder=3,
        )
        ax.vlines(
            x,
            [0.0] * len(x),
            y_gold,
            linewidth=1.0,
            colors='black',
            label='True',
            zorder=2,
        )

    rid = output_record.get('id', gold_record.get('id', 'NA'))
    name = output_record.get('name', gold_record.get('name', ''))

    _apply_nature_spectrum_style(ax, xlabel='m/z', ylabel='Relative intensity', mirrored=True)
    _set_compact_xrange(ax, x)
    ax.legend(loc='upper right', fontsize=12, frameon=False, handlelength=1.2, borderpad=0.2)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    return {
        'id': rid,
        'name': name,
        'plot_file': save_path,
        'cosine_predicted_label_peaks': cos_pred,
        'cosine_all_label_peaks': cos_all,
        'num_predicted_label_peaks': len(pred_keys),
        'num_all_label_peaks': len(gold_keys),
    }


def _collect_cosine_ranked_items(
    outputs: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    *,
    ranking_metric: str,
) -> List[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
    ranked_items: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []

    for output, record in zip(outputs, records):
        gold_spec = record.get('intensity', output.get('gold_intensity', {}))
        if not isinstance(gold_spec, dict) or not gold_spec:
            continue

        cos_info = output.get('cosine_validation', {})
        if ranking_metric == 'predicted_label_peaks':
            score = cos_info.get('cosine_predicted_label_peaks')
        else:
            score = cos_info.get('cosine_all_label_peaks')

        if score is None:
            continue

        try:
            score_f = float(score)
        except Exception:
            continue

        if math.isnan(score_f) or math.isinf(score_f):
            continue

        ranked_items.append((score_f, output, record))

    return ranked_items


def generate_extreme_cosine_plots(
    outputs: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    *,
    output_file: str,
    k: int,
    ranking_metric: str,
    round_digits: int,
    mode: str,
    plot_dir: Optional[str] = None,
) -> Optional[str]:
    if mode not in {"best", "worst"}:
        raise ValueError(f"mode must be 'best' or 'worst', got: {mode}")

    ranked_items = _collect_cosine_ranked_items(
        outputs,
        records,
        ranking_metric=ranking_metric,
    )

    if not ranked_items:
        print(f'[WARN] No records with gold intensity and valid cosine score; skip {mode} plot generation.')
        return None

    ranked_items.sort(key=lambda x: x[0], reverse=(mode == "best"))
    selected_items = ranked_items[:max(1, int(k))]

    if plot_dir is None:
        suffix = '_top10_cosine_plots' if mode == "best" else '_worst10_cosine_plots'
        plot_dir = os.path.splitext(output_file)[0] + suffix

    os.makedirs(plot_dir, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    for rank, (score, output, record) in enumerate(selected_items, start=1):
        rid = output.get('id', record.get('id', 'NA'))
        name = output.get('name', record.get('name', ''))
        fname = make_compact_plot_filename(rank, rid, output.get('formula', record.get('formula', 'NA')), digits=2)
        save_path = os.path.join(plot_dir, fname)

        info = plot_mirror_spectrum(
            output_record=output,
            gold_record=record,
            save_path=save_path,
            round_digits=round_digits,
        )
        info['rank'] = rank
        info['ranking_mode'] = mode
        info['ranking_metric'] = ranking_metric
        info['ranking_score'] = round(float(score), round_digits)
        summary_rows.append(info)

    prefix = 'topk' if mode == "best" else 'worstk'

    summary_json = os.path.join(plot_dir, f'{prefix}_cosine_summary.json')
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    summary_csv = os.path.join(plot_dir, f'{prefix}_cosine_summary.csv')
    fieldnames = [
        'rank', 'id', 'name', 'ranking_mode', 'ranking_metric', 'ranking_score',
        'cosine_predicted_label_peaks', 'cosine_all_label_peaks',
        'num_predicted_label_peaks', 'num_all_label_peaks', 'plot_file'
    ]
    with open(summary_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    label = 'best' if mode == "best" else 'worst'
    print(f'[DONE] Saved {label}-{len(summary_rows)} cosine plots to: {plot_dir}')
    print(f'[DONE] Saved {label} plot summary to: {summary_json}')
    return plot_dir


def generate_topk_cosine_plots(
    outputs: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    *,
    output_file: str,
    top_k: int,
    ranking_metric: str,
    round_digits: int,
    plot_dir: Optional[str] = None,
) -> Optional[str]:
    return generate_extreme_cosine_plots(
        outputs,
        records,
        output_file=output_file,
        k=top_k,
        ranking_metric=ranking_metric,
        round_digits=round_digits,
        mode="best",
        plot_dir=plot_dir,
    )


def generate_worstk_cosine_plots(
    outputs: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    *,
    output_file: str,
    worst_k: int,
    ranking_metric: str,
    round_digits: int,
    plot_dir: Optional[str] = None,
) -> Optional[str]:
    return generate_extreme_cosine_plots(
        outputs,
        records,
        output_file=output_file,
        k=worst_k,
        ranking_metric=ranking_metric,
        round_digits=round_digits,
        mode="worst",
        plot_dir=plot_dir,
    )

def generate_id_ordered_plots(
    outputs: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    *,
    output_file: str,
    round_digits: int,
    plot_dir: Optional[str] = None,
) -> Optional[str]:
    items: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []

    for output, record in zip(outputs, records):
        gold_spec = record.get('intensity', output.get('gold_intensity', {}))
        if not isinstance(gold_spec, dict) or not gold_spec:
            continue

        rid = output.get('id', record.get('id', 'NA'))
        try:
            rid_sort = float(rid)
        except Exception:
            rid_sort = float('inf')

        items.append((rid_sort, output, record))

    if not items:
        print('[WARN] No records with gold intensity; skip id-ordered plot generation.')
        return None

    items.sort(key=lambda x: (x[0], str(x[1].get('id', x[2].get('id', 'NA')))))

    if plot_dir is None:
        plot_dir = os.path.splitext(output_file)[0] + '_id_ordered_plots'

    os.makedirs(plot_dir, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    for order, (_, output, record) in enumerate(items, start=1):
        rid = output.get('id', record.get('id', 'NA'))
        name = output.get('name', record.get('name', ''))
        fname = make_compact_plot_filename(order, rid, output.get('formula', record.get('formula', 'NA')), digits=3)
        save_path = os.path.join(plot_dir, fname)

        info = plot_mirror_spectrum(
            output_record=output,
            gold_record=record,
            save_path=save_path,
            round_digits=round_digits,
        )
        info['order'] = order
        summary_rows.append(info)

    summary_json = os.path.join(plot_dir, 'id_ordered_plots_summary.json')
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    summary_csv = os.path.join(plot_dir, 'id_ordered_plots_summary.csv')
    fieldnames = [
        'order', 'id', 'name',
        'cosine_predicted_label_peaks', 'cosine_all_label_peaks',
        'num_predicted_label_peaks', 'num_all_label_peaks', 'plot_file'
    ]
    with open(summary_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f'[DONE] Saved id-ordered plots to: {plot_dir}')
    print(f'[DONE] Saved id-ordered plot summary to: {summary_json}')
    return plot_dir


def plot_pred_only_spectrum(
    *,
    output_record: Dict[str, Any],
    save_path: str,
) -> Dict[str, Any]:
    pred_spec = output_record.get('predicted_intensity', {})
    pred_keys = sort_mz_keys(list(pred_spec.keys()))
    pred_plot = normalize_spectrum_for_plot({k: pred_spec.get(k, 0.0) for k in pred_keys})

    x = [float(k) for k in pred_keys]
    y_pred = [pred_plot.get(k, 0.0) for k in pred_keys]

    fig, ax = plt.subplots(figsize=(13.5, 3.8), dpi=220)

    if x:
        ax.vlines(
            x,
            [0.0] * len(x),
            y_pred,
            linewidth=1.0,
            colors='#C73E3A',
            zorder=3,
        )

    _apply_nature_spectrum_style(ax, xlabel='m/z', ylabel='Intensity', mirrored=False)
    _set_compact_xrange(ax, x)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    return {
        'id': output_record.get('id', 'NA'),
        'name': output_record.get('name', ''),
        'plot_file': save_path,
        'num_predicted_label_peaks': len(pred_keys),
    }


def generate_pred_only_id_ordered_plots(
    outputs: List[Dict[str, Any]],
    *,
    output_file: str,
    plot_dir: Optional[str] = None,
) -> Optional[str]:
    items: List[Tuple[float, Dict[str, Any]]] = []

    for output in outputs:
        pred_spec = output.get('predicted_intensity', {})
        if not isinstance(pred_spec, dict) or not pred_spec:
            continue

        rid = output.get('id', 'NA')
        try:
            rid_sort = float(rid)
        except Exception:
            rid_sort = float('inf')

        items.append((rid_sort, output))

    if not items:
        print('[WARN] No records with predicted intensity; skip prediction-only plot generation.')
        return None

    items.sort(key=lambda x: (x[0], str(x[1].get('id', 'NA'))))

    if plot_dir is None:
        plot_dir = os.path.splitext(output_file)[0] + '_pred_only_plots'

    os.makedirs(plot_dir, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    for order, (_, output) in enumerate(items, start=1):
        rid = output.get('id', 'NA')
        name = output.get('name', '')
        fname = make_compact_plot_filename(order, rid, output.get('formula', 'NA'), digits=3)
        save_path = os.path.join(plot_dir, fname)

        info = plot_pred_only_spectrum(
            output_record=output,
            save_path=save_path,
        )
        info['order'] = order
        summary_rows.append(info)

    summary_json = os.path.join(plot_dir, 'pred_only_plots_summary.json')
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    summary_csv = os.path.join(plot_dir, 'pred_only_plots_summary.csv')
    fieldnames = ['order', 'id', 'name', 'num_predicted_label_peaks', 'plot_file']
    with open(summary_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f'[DONE] Saved prediction-only plots to: {plot_dir}')
    print(f'[DONE] Saved prediction-only plot summary to: {summary_json}')
    return plot_dir


def plot_gold_only_spectrum(
    *,
    gold_record: Dict[str, Any],
    save_path: str,
) -> Dict[str, Any]:
    gold_spec = gold_record.get('intensity', {})
    gold_keys = sort_mz_keys(list(gold_spec.keys()))
    gold_plot = normalize_spectrum_for_plot({k: gold_spec.get(k, 0.0) for k in gold_keys})

    x = [float(k) for k in gold_keys]
    y_gold = [gold_plot.get(k, 0.0) for k in gold_keys]

    fig, ax = plt.subplots(figsize=(13.5, 3.8), dpi=220)

    if x:
        ax.vlines(
            x,
            [0.0] * len(x),
            y_gold,
            linewidth=1.0,
            colors='black',
            zorder=3,
        )

    _apply_nature_spectrum_style(ax, xlabel='m/z', ylabel='Intensity', mirrored=False)
    _set_compact_xrange(ax, x)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    return {
        'id': gold_record.get('id', 'NA'),
        'name': gold_record.get('name', ''),
        'plot_file': save_path,
        'num_gold_peaks': len(gold_keys),
    }


def generate_gold_only_id_ordered_plots(
    gold_records: List[Dict[str, Any]],
    *,
    output_file: str,
    plot_dir: Optional[str] = None,
) -> Optional[str]:
    items: List[Tuple[float, Dict[str, Any]]] = []

    for record in gold_records:
        gold_spec = record.get('intensity', {})
        if not isinstance(gold_spec, dict) or not gold_spec:
            continue

        rid = record.get('id', 'NA')
        try:
            rid_sort = float(rid)
        except Exception:
            rid_sort = float('inf')

        items.append((rid_sort, record))

    if not items:
        print('[WARN] No records with gold intensity; skip gold-only plot generation.')
        return None

    items.sort(key=lambda x: (x[0], str(x[1].get('id', 'NA'))))

    if plot_dir is None:
        plot_dir = os.path.splitext(output_file)[0] + '_gold_only_plots'

    os.makedirs(plot_dir, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    for order, (_, record) in enumerate(items, start=1):
        rid = record.get('id', 'NA')
        name = record.get('name', '')
        fname = make_compact_plot_filename(order, rid, record.get('formula', 'NA'), digits=3)
        save_path = os.path.join(plot_dir, fname)

        info = plot_gold_only_spectrum(
            gold_record=record,
            save_path=save_path,
        )
        info['order'] = order
        summary_rows.append(info)

    summary_json = os.path.join(plot_dir, 'gold_only_plots_summary.json')
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    summary_csv = os.path.join(plot_dir, 'gold_only_plots_summary.csv')
    fieldnames = ['order', 'id', 'name', 'num_gold_peaks', 'plot_file']
    with open(summary_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f'[DONE] Saved gold-only plots to: {plot_dir}')
    print(f'[DONE] Saved gold-only plot summary to: {summary_json}')
    return plot_dir


def build_metric_descriptions() -> Dict[str, str]:
    return {
        "num_records": "参与评估的 Stage1 预测样本与 gold 样本成功配对后的分子数量。",
        "num_records_with_metrics": "成功计算完整评估指标的分子数量。",

        "num_predicted_peaks": "Stage1 生成的 product ion 总数，按 formula+m/z 去重统计；若 Stage1 文件只有 triplets 字段，则会将其作为预测三元组统计。",
        "num_gold_label_peaks": "gold 三元组中的真实 product ion 总数，按 formula+m/z 去重统计。",
        "num_matched_label_peaks": "Stage1 生成结果与 gold 中 formula+m/z 同时匹配成功的 product ion 数量。",
        "num_missing_gold_label_peaks": "gold 中存在但 Stage1 没有生成出来的 product ion 数量。",
        "num_extra_predicted_label_peaks": "Stage1 多生成的、gold 中不存在的 product ion 数量。",

        "num_stage1_product_ions": "Stage1 生成的 product ion 数量，匹配单位为 formula+m/z；兼容 stage2_stage1_predict_test.jsonl 中 triplets 作为预测结果的格式。",
        "num_gold_product_ions": "gold 标准答案中的 product ion 数量，匹配单位为 formula+m/z。",
        "num_matched_product_ions": "Stage1 与 gold 在 formula+m/z 层面匹配成功的 product ion 数量。",
        "num_missing_gold_product_ions": "formula+m/z 层面被 Stage1 漏掉的 gold product ion 数量。",
        "num_extra_predicted_product_ions": "formula+m/z 层面 Stage1 额外生成的非 gold product ion 数量。",

        "num_predicted_mz_peaks": "Stage1 生成并进入 Stage2 预测的 m/z 峰数量。",
        "num_gold_mz_peaks": "gold intensity 谱图中的真实 m/z 峰数量。",
        "num_matched_mz_peaks": "预测谱图与 gold 谱图在 m/z 层面匹配成功的峰数量。",
        "num_missing_gold_mz_peaks": "gold intensity 中存在但预测结果没有覆盖的 m/z 峰数量。",
        "num_extra_predicted_mz_peaks": "预测结果中存在但 gold intensity 中不存在的额外 m/z 峰数量。",

        "micro_stage1_peak_precision": "主精确率，与独立 Stage1 评估脚本保持一致，要求 formula+m/z 同时匹配。",
        "micro_stage1_peak_recall": "主召回率，与独立 Stage1 评估脚本保持一致，要求 formula+m/z 同时匹配。",
        "micro_stage1_peak_f1": "主 F1，与独立 Stage1 评估脚本保持一致，是 formula+m/z 层面 precision 和 recall 的调和平均。",

        "micro_stage1_product_precision": "product-level micro precision，要求 product formula 和 m/z 同时匹配。",
        "micro_stage1_product_recall": "product-level micro recall，要求 product formula 和 m/z 同时匹配。",
        "micro_stage1_product_f1": "product-level micro F1，是 product-level precision 和 recall 的调和平均。",

        "micro_stage1_mz_precision": "m/z-level micro precision，只要求 m/z 匹配，不要求 formula 一致。",
        "micro_stage1_mz_recall": "m/z-level micro recall，只要求 m/z 匹配，用来衡量真实谱峰位置是否被覆盖。",
        "micro_stage1_mz_f1": "m/z-level micro F1，是 m/z-level precision 和 recall 的调和平均。",

        "mean_stage1_weighted_mz_precision": "按预测强度加权的 m/z-level precision，高强度额外预测峰会带来更大惩罚。",
        "mean_stage1_weighted_mz_recall": "按 gold 强度加权的 m/z-level recall，漏掉高强度真实峰会带来更大惩罚。",
        "mean_stage1_weighted_mz_f1": "按强度加权后的 m/z-level F1，综合衡量强峰层面的 precision 和 recall。",

        "base_peak_hit_rate": "预测最强峰的 m/z 与 gold base peak 的 m/z 一致的样本比例。",

        "mean_stage1_peak_precision": "按分子平均的主 precision，与独立 Stage1 评估脚本一致，要求 formula+m/z 同时匹配。",
        "mean_stage1_peak_recall": "按分子平均的主 recall，与独立 Stage1 评估脚本一致，要求 formula+m/z 同时匹配。",
        "mean_stage1_peak_f1": "按分子平均的主 F1，与独立 Stage1 评估脚本一致。",

        "mean_stage1_product_precision": "按分子平均的 product-level precision，要求 formula+m/z 同时匹配。",
        "mean_stage1_product_recall": "按分子平均的 product-level recall，要求 formula+m/z 同时匹配。",
        "mean_stage1_product_f1": "按分子平均的 product-level F1，要求 formula+m/z 同时匹配。",

        "mean_stage1_mz_precision": "按分子平均的 m/z-level precision，只要求 m/z 匹配。",
        "mean_stage1_mz_recall": "按分子平均的 m/z-level recall，只要求 m/z 匹配。",
        "mean_stage1_mz_f1": "按分子平均的 m/z-level F1，只要求 m/z 匹配。",

        "mean_weighted_peak_recall_by_gold_intensity": "兼容旧字段名，等价于按 gold 强度加权的 m/z-level recall。",
        "mean_missing_gold_intensity_fraction": "被漏掉的 gold m/z 峰所占真实总谱图强度的平均比例。",
        "mean_extra_predicted_intensity_fraction": "额外预测的非 gold m/z 峰所占预测总谱图强度的平均比例。",

        "mean_cosine_predicted_label_peaks": "只在预测出来的 m/z 峰集合上计算的平均谱图 cosine。",
        "mean_cosine_all_label_peaks": "在全部 gold m/z 峰上计算的平均谱图 cosine，未预测到的 gold 峰按 0 处理。",
        "mean_cosine_union_peaks": "在预测 m/z 与 gold m/z 的并集上计算的平均谱图 cosine，同时惩罚漏峰和多生成峰。",
        "mean_mae_predicted_label_peaks": "只在预测出来的 m/z 峰集合上计算的平均绝对强度误差。",
        "mean_mae_all_gold_label_peaks": "在全部 gold m/z 峰上计算的平均绝对强度误差。",
        "mean_rmse_all_gold_label_peaks": "在全部 gold m/z 峰上计算的均方根强度误差。",
        "mean_pearson_all_gold_label_peaks": "在全部 gold m/z 峰上预测强度与真实强度的平均 Pearson 相关系数。",
        "mean_spearman_all_gold_label_peaks": "在全部 gold m/z 峰上预测强度排序与真实强度排序的平均 Spearman 相关系数。",

        "mean_stage1_product_recall_formula_mz": "显式 product-level recall 别名，表示 formula+m/z 同时匹配的召回率。",
    }


def summarize_outputs(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_keys = [

        "stage1_peak_precision",
        "stage1_peak_recall",
        "stage1_peak_f1",
        "stage1_product_precision",
        "stage1_product_recall",
        "stage1_product_f1",


        "stage1_mz_precision",
        "stage1_mz_recall",
        "stage1_mz_f1",


        "stage1_weighted_mz_precision",
        "stage1_weighted_mz_recall",
        "stage1_weighted_mz_f1",

        "weighted_peak_recall_by_gold_intensity",
        "missing_gold_intensity_fraction",
        "extra_predicted_intensity_fraction",
        "cosine_predicted_label_peaks",
        "cosine_all_label_peaks",
        "cosine_union_peaks",
        "mae_predicted_label_peaks",
        "mae_all_gold_label_peaks",
        "rmse_all_gold_label_peaks",
        "pearson_all_gold_label_peaks",
        "spearman_all_gold_label_peaks",
    ]

    for k in [1, 3, 5, 7, 9, 11, 13, 15]:
        metric_keys.extend([
            f"top{k}_stage1_gold_coverage_recall",
            f"top{k}_pred_precision",
            f"top{k}_pred_recall",
            f"top{k}_pred_f1",
            f"top{k}_pred_overlap_count",
        ])

    values: Dict[str, List[float]] = {k: [] for k in metric_keys}

    n_records = len(outputs)
    n_records_with_metrics = 0


    n_pred_product = 0
    n_gold_product = 0
    n_hit_product = 0
    n_missing_product = 0
    n_extra_product = 0


    n_pred_mz = 0
    n_gold_mz = 0
    n_hit_mz = 0
    n_missing_mz = 0
    n_extra_mz = 0

    n_base_hit = 0
    n_base_hit_valid = 0

    for item in outputs:
        metrics = item.get("metrics", {})
        evaluation = item.get("evaluation", {})
        product_eval = evaluation.get("stage1_product_ion_prf", {})
        mz_eval = evaluation.get("stage1_mz_prf", {})

        if metrics:
            n_records_with_metrics += 1

        n_pred_product += int(product_eval.get("num_stage1_product_ions", 0) or 0)
        n_gold_product += int(product_eval.get("num_gold_product_ions", 0) or 0)
        n_hit_product += int(product_eval.get("num_matched_product_ions", 0) or 0)
        n_missing_product += int(product_eval.get("num_missing_gold_product_ions", 0) or 0)
        n_extra_product += int(product_eval.get("num_extra_predicted_product_ions", 0) or 0)

        n_pred_mz += int(mz_eval.get("num_predicted_mz_peaks", 0) or 0)
        n_gold_mz += int(mz_eval.get("num_gold_mz_peaks", 0) or 0)
        n_hit_mz += int(mz_eval.get("num_matched_mz_peaks", 0) or 0)
        n_missing_mz += int(mz_eval.get("num_missing_gold_mz_peaks", 0) or 0)
        n_extra_mz += int(mz_eval.get("num_extra_predicted_mz_peaks", 0) or 0)

        base_hit = metrics.get("predicted_base_mz_hit")
        if isinstance(base_hit, bool):
            n_base_hit_valid += 1
            n_base_hit += int(base_hit)

        for k in metric_keys:
            v = metrics.get(k)
            if isinstance(v, bool):
                values[k].append(float(v))
            elif isinstance(v, (int, float)) and not math.isnan(float(v)):
                values[k].append(float(v))


    micro_product_precision = n_hit_product / n_pred_product if n_pred_product else None
    micro_product_recall = n_hit_product / n_gold_product if n_gold_product else None
    if micro_product_precision is not None and micro_product_recall is not None and (micro_product_precision + micro_product_recall) > 0:
        micro_product_f1 = 2 * micro_product_precision * micro_product_recall / (micro_product_precision + micro_product_recall)
    else:
        micro_product_f1 = None


    micro_mz_precision = n_hit_mz / n_pred_mz if n_pred_mz else None
    micro_mz_recall = n_hit_mz / n_gold_mz if n_gold_mz else None
    if micro_mz_precision is not None and micro_mz_recall is not None and (micro_mz_precision + micro_mz_recall) > 0:
        micro_mz_f1 = 2 * micro_mz_precision * micro_mz_recall / (micro_mz_precision + micro_mz_recall)
    else:
        micro_mz_f1 = None

    summary: Dict[str, Any] = {
        "num_records": n_records,
        "num_records_with_metrics": n_records_with_metrics,


        "num_predicted_peaks": n_pred_product,
        "num_gold_label_peaks": n_gold_product,
        "num_matched_label_peaks": n_hit_product,
        "num_missing_gold_label_peaks": n_missing_product,
        "num_extra_predicted_label_peaks": n_extra_product,


        "num_stage1_product_ions": n_pred_product,
        "num_gold_product_ions": n_gold_product,
        "num_matched_product_ions": n_hit_product,
        "num_missing_gold_product_ions": n_missing_product,
        "num_extra_predicted_product_ions": n_extra_product,


        "num_predicted_mz_peaks": n_pred_mz,
        "num_gold_mz_peaks": n_gold_mz,
        "num_matched_mz_peaks": n_hit_mz,
        "num_missing_gold_mz_peaks": n_missing_mz,
        "num_extra_predicted_mz_peaks": n_extra_mz,


        "micro_stage1_peak_precision": micro_product_precision,
        "micro_stage1_peak_recall": micro_product_recall,
        "micro_stage1_peak_f1": micro_product_f1,
        "micro_stage1_product_precision": micro_product_precision,
        "micro_stage1_product_recall": micro_product_recall,
        "micro_stage1_product_f1": micro_product_f1,


        "micro_stage1_mz_precision": micro_mz_precision,
        "micro_stage1_mz_recall": micro_mz_recall,
        "micro_stage1_mz_f1": micro_mz_f1,

        "base_peak_hit_rate": n_base_hit / n_base_hit_valid if n_base_hit_valid else None,
    }

    for k in metric_keys:
        summary[f"mean_{k}"] = sum(values[k]) / len(values[k]) if values[k] else None


    summary["mean_stage1_product_recall_formula_mz"] = summary.get("mean_stage1_product_recall")
    summary["mean_stage1_weighted_mz_precision"] = summary.get("mean_stage1_weighted_mz_precision")
    summary["mean_stage1_weighted_mz_recall"] = summary.get("mean_stage1_weighted_mz_recall")
    summary["mean_stage1_weighted_mz_f1"] = summary.get("mean_stage1_weighted_mz_f1")

    descriptions = build_metric_descriptions()
    for k in [1, 3, 5, 7, 9, 11, 13, 15]:
        descriptions[f"mean_top{k}_stage1_gold_coverage_recall"] = (
            f"按分子平均的 gold top{k} 强峰覆盖率，表示 gold 最强 {k} 个 m/z 中有多少被 Stage1 生成树覆盖。"
        )
        descriptions[f"mean_top{k}_pred_precision"] = (
            f"按分子平均的 top{k} precision，表示预测最强 {k} 个峰中有多少也属于 gold 最强 {k} 个峰。"
        )
        descriptions[f"mean_top{k}_pred_recall"] = (
            f"按分子平均的 top{k} recall，表示 gold 最强 {k} 个峰中有多少出现在预测最强 {k} 个峰中。"
        )
        descriptions[f"mean_top{k}_pred_f1"] = (
            f"按分子平均的 top{k} F1，综合衡量预测 top{k} 与 gold top{k} 的重叠质量。"
        )
        descriptions[f"mean_top{k}_pred_overlap_count"] = (
            f"按分子平均的 top{k} 重叠峰数量，即预测 top{k} 与 gold top{k} 共有多少个 m/z。"
        )
    summary["metric_descriptions"] = descriptions
    return summary


def build_record_index(records: List[Dict[str, Any]], key_field: str = "id") -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for r in records:
        if key_field == "smiles":
            key = str(r.get("smiles", r.get("SMILES", ""))).strip()
        else:
            key = str(r.get("id", r.get("ID", ""))).strip()
        if key and key not in index:
            index[key] = r
    return index


def pair_stage1_with_gold_records(
    stage1_records: List[Dict[str, Any]],
    gold_records: List[Dict[str, Any]],
    *,
    pair_key: str,
    require_gold: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    gold_index = build_record_index(gold_records, key_field=pair_key)

    paired_stage1: List[Dict[str, Any]] = []
    paired_gold: List[Dict[str, Any]] = []
    missing_gold: List[Dict[str, Any]] = []

    for r in stage1_records:
        if pair_key == "smiles":
            key = str(r.get("smiles", r.get("SMILES", ""))).strip()
        else:
            key = str(r.get("id", r.get("ID", ""))).strip()

        gold = gold_index.get(key)
        if gold is None:
            missing_gold.append(r)
            if not require_gold:
                paired_stage1.append(r)
                paired_gold.append({})
            continue


        stage1_input = dict(r)
        for intensity_key in (
            "intensity",
            "intensities",
            "gold_intensity",
            "gold_spectrum",
            "input_spectrum",
            "spectrum",
        ):
            stage1_input.pop(intensity_key, None)

        for metadata_key in (
            "name",
            "smiles",
            "SMILES",
            "formula",
            "Formula",
            "mw",
            "MW",
        ):
            if (
                stage1_input.get(metadata_key) in (None, "")
                and gold.get(metadata_key) not in (None, "")
            ):
                stage1_input[metadata_key] = gold[metadata_key]

        normalized_gold = dict(gold)
        normalized_gold["intensity"] = normalize_intensity_mapping(
            extract_intensity_value(gold)
        )

        paired_stage1.append(stage1_input)
        paired_gold.append(normalized_gold)

    return paired_stage1, paired_gold, missing_gold


def filter_records(
    records: List[Dict[str, Any]],
    *,
    id_min: Optional[int],
    id_max: Optional[int],
    start: int,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []

    for r in records:
        rid_raw = r.get("id", r.get("ID", None))
        try:
            rid = int(rid_raw)
        except Exception:
            rid = None

        if id_min is not None:
            if rid is None or rid < id_min:
                continue
        if id_max is not None:
            if rid is None or rid > id_max:
                continue

        selected.append(r)

    if start > 0:
        selected = selected[start:]

    if limit is not None:
        selected = selected[:limit]

    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Stage2 Tree-Path Attention inference")

    p.add_argument("--stage1_file", type=str, required=True, help="Stage1-generated triplet JSON/JSONL file. This is model input.")
    p.add_argument("--gold_file", type=str, required=True, help="Gold JSON/JSONL file with true gold triplets and gold intensity.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt.")
    p.add_argument("--output_file", type=str, required=True, help="Output prediction JSONL/JSON file.")
    p.add_argument("--config_file", type=str, default=None, help="Optional model_config.json. Default: checkpoint directory/model_config.json")

    p.add_argument("--id_min", type=int, default=None, help="Only infer records with id >= id_min.")
    p.add_argument("--id_max", type=int, default=None, help="Only infer records with id <= id_max.")
    p.add_argument("--start", type=int, default=0, help="Start offset after id filtering.")
    p.add_argument("--limit", type=int, default=None, help="Infer at most N records after id filtering and start.")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)


    p.add_argument("--max_peaks", type=int, default=None)
    p.add_argument("--max_path_len", type=int, default=None)
    p.add_argument("--max_words_per_token", type=int, default=None)

    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_false", dest="amp")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--compile", action="store_true", default=False, help="Optional torch.compile for inference. Disabled by default for stability.")
    p.add_argument("--pair_key", type=str, default="id", choices=["id", "smiles"], help="Key used to match Stage1 records with gold records.")
    p.add_argument("--require_gold", action="store_true", default=True, help="Skip Stage1 records that cannot be matched to gold records.")
    p.add_argument("--allow_missing_gold", action="store_false", dest="require_gold", help="Allow unmatched Stage1 records; metrics will be missing.")
    p.add_argument("--keep_gold", action="store_true", default=True, help="Kept for compatibility; gold metrics are computed from --gold_file.")
    p.add_argument("--keep_stage1_triplets", action="store_true", help="Save Stage1-generated triplets in output.")
    p.add_argument("--keep_gold_triplets", action="store_true", help="Save gold triplets in output.")
    p.add_argument("--keep_triplets", action="store_true", help="Alias of --keep_stage1_triplets.")
    p.add_argument("--keep_input_record", action="store_true", help="Save the full original input record under input_record.")
    p.add_argument("--save_pred_as_intensity", action="store_true", help="Also save predicted_intensity under field name intensity.")
    p.add_argument(
        "--normalize_base_peak",
        action="store_true",
        default=True,
        help=(
            "Normalize raw Softplus intensities by the strongest candidate "
            "before clipping/rounding. Enabled by default because this is the "
            "exact checkpoint-selection path used during training."
        ),
    )
    p.add_argument(
        "--no_normalize_base_peak",
        action="store_false",
        dest="normalize_base_peak",
        help="Disable base-peak normalization (will not reproduce training validation).",
    )
    p.add_argument("--non_strict_load", action="store_true", help="Allow non-strict checkpoint loading. Not recommended unless debugging.")
    p.add_argument("--round_digits", type=int, default=6)

    p.add_argument("--make_topk_cosine_plots", action="store_true", default=True,
                   help="Generate mirror-spectrum plots for the top-k highest cosine similarity records.")
    p.add_argument("--no_make_topk_cosine_plots", action="store_false", dest="make_topk_cosine_plots",
                   help="Disable top-k cosine plot generation.")
    p.add_argument("--topk_cosine_plots", type=int, default=10,
                   help="Number of highest-cosine records to visualize. Default: 10")
    p.add_argument("--cosine_ranking_metric", type=str, default="all_label_peaks",
                   choices=["predicted_label_peaks", "all_label_peaks"],
                   help="Which cosine metric to use for ranking the top-k plots.")
    p.add_argument("--cosine_plot_dir", type=str, default=None,
                   help="Optional output directory for top-k cosine plots.")

    p.add_argument("--make_worstk_cosine_plots", action="store_true", default=True,
                   help="Generate mirror-spectrum plots for the worst-k lowest cosine similarity records.")
    p.add_argument("--no_make_worstk_cosine_plots", action="store_false", dest="make_worstk_cosine_plots",
                   help="Disable worst-k cosine plot generation.")
    p.add_argument("--worstk_cosine_plots", type=int, default=10,
                   help="Number of lowest-cosine records to visualize. Default: 10")
    p.add_argument("--worst_cosine_plot_dir", type=str, default=None,
                   help="Optional output directory for worst-k cosine plots.")

    p.add_argument("--make_id_ordered_plots", action="store_true", default=True,
                   help="Generate mirror-spectrum plots for all records sorted by id.")
    p.add_argument("--no_make_id_ordered_plots", action="store_false", dest="make_id_ordered_plots",
                   help="Disable id-ordered plot generation.")
    p.add_argument("--id_ordered_plot_dir", type=str, default=None,
                   help="Optional output directory for id-ordered plots.")

    p.add_argument("--make_pred_only_plots", action="store_true", default=True,
                   help="Generate prediction-only spectrum plots for all records sorted by id.")
    p.add_argument("--no_make_pred_only_plots", action="store_false", dest="make_pred_only_plots",
                   help="Disable prediction-only plot generation.")
    p.add_argument("--pred_only_plot_dir", type=str, default=None,
                   help="Optional output directory for prediction-only plots.")

    p.add_argument("--make_gold_only_plots", action="store_true", default=True,
                   help="Generate gold-only spectrum plots for all records sorted by id.")
    p.add_argument("--no_make_gold_only_plots", action="store_false", dest="make_gold_only_plots",
                   help="Disable gold-only plot generation.")
    p.add_argument("--gold_only_plot_dir", type=str, default=None,
                   help="Optional output directory for gold-only plots.")


    p.add_argument("--ablate_intensity_loss", action="store_true", default=None)
    p.add_argument("--ablate_spectral_loss", action="store_true", default=None)
    p.add_argument("--ablate_ranking_loss", action="store_true", default=None)
    p.add_argument("--ablate_base_peak_loss", action="store_true", default=None)
    p.add_argument("--ablate_strong_peak_loss", action="store_true", default=None)
    p.add_argument(
        "--ablate_global_context", "--no_global_context",
        dest="ablate_global_context", action="store_true", default=None,
        help="Use the w/o Global molecular context forward path.",
    )
    p.add_argument(
        "--ablate_path_attention", "--no_path_attention",
        dest="ablate_path_attention", action="store_true", default=None,
        help="Use masked mean pooling instead of path attention.",
    )
    p.add_argument(
        "--ablate_inter_peak_interaction", "--no_inter_peak_interaction",
        dest="ablate_inter_peak_interaction", action="store_true", default=None,
        help="Bypass the cross-peak Transformer interaction.",
    )
    p.add_argument(
        "--ablate_gold_guided_curriculum", "--no_gold_guided_curriculum",
        dest="ablate_gold_guided_curriculum", action="store_true", default=None,
        help="Assert that the checkpoint was trained without Gold-guided curriculum.",
    )
    p.add_argument(
        "--allow_ablation_override", action="store_true", default=False,
        help=(
            "Allow CLI ablation switches to conflict with checkpoint/model_config "
            "metadata. Debugging only; not recommended for reported experiments."
        ),
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    setup_cuda_inference()

    device = torch.device(args.device)
    print(f"[INFO] device={device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU={torch.cuda.get_device_name(0)}")
    print(f"[INFO] checkpoint={args.checkpoint}")

    cfg = load_train_config(args.config_file, args.checkpoint)
    cfg = resolve_inference_ablation_configuration(args, cfg)
    print_inference_ablation_configuration(cfg)

    if args.max_peaks is not None:
        cfg["max_peaks"] = args.max_peaks
    if args.max_path_len is not None:
        cfg["max_path_len"] = args.max_path_len
    if args.max_words_per_token is not None:
        cfg["max_words_per_token"] = args.max_words_per_token

    print("[INFO] Inference config:")
    print(json.dumps(cfg, ensure_ascii=False, indent=2))

    all_stage1_records = load_json_or_jsonl(args.stage1_file)
    all_gold_records = load_json_or_jsonl(args.gold_file)
    if not all_stage1_records:
        raise ValueError(f"No valid Stage1 records loaded from: {args.stage1_file}")
    if not all_gold_records:
        raise ValueError(f"No valid gold records loaded from: {args.gold_file}")

    selected_stage1_records = filter_records(
        all_stage1_records,
        id_min=args.id_min,
        id_max=args.id_max,
        start=args.start,
        limit=args.limit,
    )
    if not selected_stage1_records:
        raise ValueError(
            f"No Stage1 records left after filtering. input={len(all_stage1_records)}, "
            f"id_min={args.id_min}, id_max={args.id_max}, start={args.start}, limit={args.limit}"
        )

    records, gold_records, missing_gold_records = pair_stage1_with_gold_records(
        selected_stage1_records,
        all_gold_records,
        pair_key=args.pair_key,
        require_gold=args.require_gold,
    )

    if not records:
        raise ValueError(
            "No paired records available. Check --pair_key, --stage1_file, and --gold_file."
        )

    print(f"[INFO] Loaded Stage1 records: {len(all_stage1_records)}")
    print(f"[INFO] Loaded gold records: {len(all_gold_records)}")
    print(f"[INFO] Selected Stage1 records after filtering: {len(selected_stage1_records)}")
    print(f"[INFO] Paired records for inference/evaluation: {len(records)}")
    print(f"[INFO] Missing gold records: {len(missing_gold_records)}")

    dataset = Stage2InferDataset(
        records,
        vocab_size=int(cfg["vocab_size"]),
        max_peaks=int(cfg["max_peaks"]),
        max_path_len=int(cfg["max_path_len"]),
        max_words_per_token=int(cfg["max_words_per_token"]),
        morgan_fp_dim=int(cfg["morgan_fp_dim"]),
        morgan_radius=int(cfg["morgan_radius"]),
    )

    collator = Stage2InferCollator(
        vocab_size=int(cfg["vocab_size"]),
        max_peaks=int(cfg["max_peaks"]),
        max_path_len=int(cfg["max_path_len"]),
        max_words_per_token=int(cfg["max_words_per_token"]),
        morgan_fp_dim=int(cfg["morgan_fp_dim"]),
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(**loader_kwargs)

    model = StructuredTreePathAttentionModel(
        vocab_size=int(cfg["vocab_size"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        interaction_layers=int(cfg["interaction_layers"]),
        dropout=float(cfg.get("dropout", 0.0)),
        max_path_len=int(cfg["max_path_len"]),
        n_token_types=5,
        n_mechanisms=len(MECHANISM_LABELS) + 1,
        morgan_fp_dim=int(cfg["morgan_fp_dim"]),
        use_global_context=bool(cfg["use_global_context"]),
        use_path_attention=bool(cfg["use_path_attention"]),
        use_inter_peak_interaction=bool(cfg["use_inter_peak_interaction"]),
    )

    ckpt_meta = load_checkpoint_into_model(
        model,
        args.checkpoint,
        device,
        strict=not args.non_strict_load,
    )
    print(f"[INFO] checkpoint meta: {ckpt_meta}")

    if args.compile and hasattr(torch, "compile"):
        try:
            print("[INFO] torch.compile enabled for inference")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[WARN] torch.compile failed, continue without compile: {e}")

    outputs = run_inference(
        model=model,
        loader=loader,
        records=records,
        device=device,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        keep_gold=args.keep_gold,
        keep_triplets=(args.keep_triplets or args.keep_stage1_triplets),
        keep_input_record=args.keep_input_record,
        save_pred_as_intensity=args.save_pred_as_intensity,
        normalize_base_peak=args.normalize_base_peak,
        round_digits=args.round_digits,
    )

    outputs = enrich_outputs_with_gold_evaluation(
        outputs,
        records,
        gold_records,
        keep_gold_triplets=args.keep_gold_triplets,
        round_digits=args.round_digits,
    )


    save_json_or_jsonl(outputs, args.output_file)
    print(f"[DONE] Saved final predictions to: {args.output_file}")

    summary = summarize_outputs(outputs)
    summary["ablation_configuration"] = {
        "is_full_model": bool(cfg.get("is_full_model", False)),
        "active_ablations": list(cfg.get("active_ablations", [])),
        "ablate_intensity_loss": bool(cfg.get("ablate_intensity_loss", False)),
        "ablate_spectral_loss": bool(cfg.get("ablate_spectral_loss", False)),
        "ablate_ranking_loss": bool(cfg.get("ablate_ranking_loss", False)),
        "ablate_base_peak_loss": bool(cfg.get("ablate_base_peak_loss", False)),
        "ablate_strong_peak_loss": bool(cfg.get("ablate_strong_peak_loss", False)),
        "ablate_global_context": bool(cfg.get("ablate_global_context", False)),
        "ablate_path_attention": bool(cfg.get("ablate_path_attention", False)),
        "ablate_inter_peak_interaction": bool(cfg.get("ablate_inter_peak_interaction", False)),
        "ablate_gold_guided_curriculum": bool(cfg.get("ablate_gold_guided_curriculum", False)),
        "use_global_context": bool(cfg.get("use_global_context", True)),
        "use_path_attention": bool(cfg.get("use_path_attention", True)),
        "use_inter_peak_interaction": bool(cfg.get("use_inter_peak_interaction", True)),
    }


    expected_best = ckpt_meta.get("best_cosine")
    observed_union = summary.get("mean_cosine_union_peaks")
    reproduction: Dict[str, Any] = {
        "checkpoint_best_mean_cosine_union_peaks": expected_best,
        "standalone_mean_cosine_union_peaks": observed_union,
        "normalize_base_peak": bool(args.normalize_base_peak),
        "amp": bool(args.amp),
        "amp_dtype": args.amp_dtype,
        "batch_size": int(args.batch_size),
        "round_digits": int(args.round_digits),
        "active_ablations": list(cfg.get("active_ablations", [])),
        "use_global_context": bool(cfg.get("use_global_context", True)),
        "use_path_attention": bool(cfg.get("use_path_attention", True)),
        "use_inter_peak_interaction": bool(cfg.get("use_inter_peak_interaction", True)),
    }
    if isinstance(expected_best, (int, float)) and isinstance(
        observed_union, (int, float)
    ):
        reproduction["absolute_difference"] = abs(
            float(observed_union) - float(expected_best)
        )
    summary["checkpoint_reproduction"] = reproduction

    summary_file = os.path.splitext(args.output_file)[0] + "_summary.json"
    write_json(summary, summary_file)
    print(f"[DONE] Saved final summary to: {summary_file}")

    print("[CHECKPOINT REPRODUCTION]")
    print(json.dumps(reproduction, ensure_ascii=False, indent=2))
    if (
        isinstance(reproduction.get("absolute_difference"), (int, float))
        and reproduction["absolute_difference"] > 1e-3
    ):
        print(
            "[WARN] Standalone inference still differs from checkpoint "
            "validation by more than 1e-3. Re-run with the same batch size, "
            "AMP dtype, files, and no --compile; then inspect per-record input "
            "hashes."
        )

    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


    saved_outputs = load_json_or_jsonl(args.output_file)
    if len(saved_outputs) != len(outputs):
        raise RuntimeError(
            f"Saved-output verification failed: in_memory={len(outputs)}, "
            f"reloaded={len(saved_outputs)}"
        )
    print(
        f"[INFO] Verified saved inference results: {len(saved_outputs)} records. "
        "Starting plot generation."
    )

    if args.make_topk_cosine_plots:
        try:
            generate_topk_cosine_plots(
                saved_outputs,
                gold_records,
                output_file=args.output_file,
                top_k=args.topk_cosine_plots,
                ranking_metric=args.cosine_ranking_metric,
                round_digits=args.round_digits,
                plot_dir=args.cosine_plot_dir,
            )
        except Exception as e:
            print(f"[WARN] Failed to generate top-k cosine plots: {e}")

    if args.make_worstk_cosine_plots:
        try:
            generate_worstk_cosine_plots(
                saved_outputs,
                gold_records,
                output_file=args.output_file,
                worst_k=args.worstk_cosine_plots,
                ranking_metric=args.cosine_ranking_metric,
                round_digits=args.round_digits,
                plot_dir=args.worst_cosine_plot_dir,
            )
        except Exception as e:
            print(f"[WARN] Failed to generate worst-k cosine plots: {e}")

    if args.make_id_ordered_plots:
        try:
            generate_id_ordered_plots(
                saved_outputs,
                gold_records,
                output_file=args.output_file,
                round_digits=args.round_digits,
                plot_dir=args.id_ordered_plot_dir,
            )
        except Exception as e:
            print(f"[WARN] Failed to generate id-ordered plots: {e}")

    if args.make_pred_only_plots:
        try:
            generate_pred_only_id_ordered_plots(
                saved_outputs,
                output_file=args.output_file,
                plot_dir=args.pred_only_plot_dir,
            )
        except Exception as e:
            print(f"[WARN] Failed to generate prediction-only plots: {e}")

    if args.make_gold_only_plots:
        try:
            generate_gold_only_id_ordered_plots(
                gold_records,
                output_file=args.output_file,
                plot_dir=args.gold_only_plot_dir,
            )
        except Exception as e:
            print(f"[WARN] Failed to generate gold-only plots: {e}")


if __name__ == "__main__":
    main()
