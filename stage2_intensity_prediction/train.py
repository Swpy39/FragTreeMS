#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import bisect
import copy
import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except Exception as exc:
    raise ImportError(
        "RDKit is required for the Morgan-fingerprint global molecular branch."
    ) from exc
import torch.nn.functional as F
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Sampler,
)


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

EVAL_ROUND_DIGITS = 6

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_cuda_high_throughput() -> None:
    """Enable stable high-throughput settings for NVIDIA Ampere GPUs."""
    torch.autograd.set_detect_anomaly(False)
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    try:
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = 64
        dynamo.config.accumulated_cache_size_limit = 256
    except Exception:
        pass


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []

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
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError as e:
            print(f"[WARN] Bad JSON at {path}:{line_no}: {e}")
    return rows


def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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
    """Match the m/z key formatting used by stage2_infer_New.py."""
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


def get_record_id(record: Dict[str, Any]) -> str:
    rid = record.get("id", record.get("ID", ""))
    return str(rid).strip()


def extract_triplet_list(record: Dict[str, Any]) -> List[Any]:
    """
    Read Stage1 triplets from common infer-file fields.

    The preferred field is ``triplets``. The fallbacks keep this trainer
    compatible with repaired/unified inference JSONL files that may store the
    same Stage1 result under another name.
    """
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

        # Direct list of triplets: [[source, mechanism, product], ...]
        if all(isinstance(item, list) and len(item) >= 3 for item in value):
            return value

        # Some inference outputs wrap candidates one level deeper. Use the
        # first non-empty candidate list rather than mixing candidate rounds.
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
    """Return an intensity/spectrum field without changing its scale."""
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
    """
    Convert common spectrum representations to ``{mz: intensity}``.

    Supported inputs:
      * a mapping from m/z to intensity;
      * a list of ``[mz, intensity]`` pairs;
      * a list of dictionaries containing m/z and intensity keys.

    Values are not renormalized here, preserving the original trainer's target
    semantics. ``build_tree_paths`` still clips targets to [0, 1].
    """
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


def merge_infer_validation_records(
    infer_records: List[Dict[str, Any]],
    infer_gold_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build validation records from the infer file only.

    When ``--infer_val_gold_file`` is provided, only its target spectrum and
    missing molecule metadata are merged by id. No record from either training
    file participates in validation.
    """
    if not infer_gold_records:
        merged: List[Dict[str, Any]] = []
        for record in infer_records:
            row = dict(record)
            row["triplets"] = extract_triplet_list(row)
            row["intensity"] = normalize_intensity_mapping(
                extract_intensity_value(row)
            )
            merged.append(row)
        return merged

    gold_by_id = {
        get_record_id(record): record
        for record in infer_gold_records
        if get_record_id(record)
    }

    merged = []
    unmatched = 0
    for record in infer_records:
        row = dict(record)
        rid = get_record_id(row)
        gold = gold_by_id.get(rid)
        if gold is None:
            unmatched += 1
        else:
            row["intensity"] = normalize_intensity_mapping(
                extract_intensity_value(gold)
            )
            for key in ("name", "smiles", "SMILES", "formula", "Formula", "mw", "MW"):
                if row.get(key) in (None, "") and gold.get(key) not in (None, ""):
                    row[key] = gold[key]

        row["triplets"] = extract_triplet_list(row)
        if "intensity" not in row:
            row["intensity"] = normalize_intensity_mapping(
                extract_intensity_value(row)
            )
        merged.append(row)

    if unmatched:
        print(
            f"[WARN] infer validation records without matching infer gold id: "
            f"{unmatched}/{len(infer_records)}"
        )
    return merged


def smiles_to_morgan_fingerprint(
    smiles: Any,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> List[float]:
    """Return a fixed-size Morgan bit fingerprint for global structure context."""
    text = str(smiles or "").strip()
    if not text:
        return [0.0] * int(n_bits)

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return [0.0] * int(n_bits)

    bit_vector = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        int(radius),
        nBits=int(n_bits),
        useChirality=True,
    )
    array = np.zeros((int(n_bits),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array.tolist()


def normalize_prediction_by_base_peak(
    prediction: torch.Tensor,
    peak_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize each predicted spectrum so its strongest valid peak is 1."""
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
    """
    Tokenize known Stage2 fields with a SMILES-aware path while preserving the
    original fixed-size hash vocabulary. No explicit vocabulary is built.
    """
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


# ============================================================
# 2. Fragmentation tree representation
# ============================================================

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
    record_id: Any
    name: str
    smiles: str
    formula: str
    mw: float
    mz_values: List[float]
    product_texts: List[str]
    paths: List[List[PathToken]]
    targets: List[float]
    full_gold_intensity: Dict[str, float]
    true_base_mz: Optional[str]
    morgan_fp: List[float]
    source_type: str
    # Built exactly once during Dataset construction. The DataLoader no longer
    # performs regex tokenization, hashing, or Python text processing per epoch.
    tensor_cache: Optional[Dict[str, torch.Tensor]] = field(
        default=None,
        repr=False,
    )
    bucket_shape: Tuple[int, int] = (1, 1)


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
    *,
    morgan_radius: int,
    morgan_fp_dim: int,
) -> Optional[PeakPathExample]:
    """
    Build one unique ROOT-to-peak path per Stage1 candidate peak.

    The path now carries explicit parent m/z and product m/z values. Neutral
    loss is derived by the model as parent_mz - product_mz.
    """
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

    def edge_to_path_token(edge: EdgeInfo, token_type: Optional[int] = None) -> PathToken:
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

    intensity = normalize_intensity_mapping(
        extract_intensity_value(record)
    )
    # stage2_infer_New.py clips every gold intensity to [0, 1] before
    # computing mean_cosine_union_peaks. Keep the exact same semantics here.
    full_gold_intensity = {
        str(mz_key): max(0.0, min(1.0, parse_float(value, 0.0)))
        for mz_key, value in intensity.items()
    }

    true_base_mz: Optional[str] = None
    if full_gold_intensity:
        true_base_mz = max(
            full_gold_intensity.keys(),
            key=lambda key: (
                full_gold_intensity.get(key, 0.0),
                parse_float(key, 0.0),
            ),
        )

    smiles_text = str(
        record.get("smiles", record.get("SMILES", ""))
    ).strip()
    morgan_fp = smiles_to_morgan_fingerprint(
        smiles_text,
        radius=morgan_radius,
        n_bits=morgan_fp_dim,
    )

    mz_keys = sorted(
        product_text_by_mz.keys(),
        key=lambda value: float(value),
        reverse=True,
    )

    mz_values: List[float] = []
    product_texts: List[str] = []
    paths: List[List[PathToken]] = []
    targets: List[float] = []

    for mz in mz_keys:
        target = parse_float(
            intensity.get(mz, intensity.get(str(int(float(mz))), 0.0)),
            default=0.0,
        )
        mz_values.append(float(mz))
        product_texts.append(product_text_by_mz[mz])
        paths.append(recover_path(mz))
        targets.append(max(0.0, min(1.0, target)))

    if not mz_values:
        return None

    return PeakPathExample(
        record_id=record.get("id", None),
        name=str(record.get("name", "")),
        smiles=str(record.get("smiles", record.get("SMILES", ""))),
        formula=str(record.get("formula", record.get("Formula", ""))),
        mw=molecular_mw,
        mz_values=mz_values,
        product_texts=product_texts,
        paths=paths,
        targets=targets,
        full_gold_intensity=full_gold_intensity,
        true_base_mz=true_base_mz,
        morgan_fp=morgan_fp,
        source_type=str(record.get("_source_type", "")),
    )


# ============================================================
# 3. Dataset and collator
# ============================================================

def _round_up(value: int, multiple: int, maximum: int) -> int:
    value = max(1, min(int(value), int(maximum)))
    multiple = max(1, int(multiple))
    return min(int(maximum), ((value + multiple - 1) // multiple) * multiple)


def tensorize_example_once(
    example: PeakPathExample,
    args: argparse.Namespace,
) -> None:
    """Convert one molecule to reusable CPU tensors exactly once."""
    n_peaks = min(args.max_peaks, len(example.mz_values))
    max_path = min(
        args.max_path_len,
        max((len(path) for path in example.paths[:n_peaks]), default=1),
    )
    max_words = args.max_words_per_token

    path_token_ids = torch.zeros(
        (n_peaks, max_path, max_words), dtype=torch.int32
    )
    product_token_ids = torch.zeros(
        (n_peaks, max_words), dtype=torch.int32
    )
    path_type_ids = torch.zeros(
        (n_peaks, max_path), dtype=torch.int16
    )
    path_mech_ids = torch.zeros(
        (n_peaks, max_path), dtype=torch.int16
    )
    path_parent_mz = torch.zeros(
        (n_peaks, max_path), dtype=torch.float32
    )
    path_product_mz = torch.zeros(
        (n_peaks, max_path), dtype=torch.float32
    )
    path_mask = torch.zeros((n_peaks, max_path), dtype=torch.bool)

    for peak_index in range(n_peaks):
        product_token_ids[peak_index] = torch.tensor(
            text_to_ids(
                example.product_texts[peak_index],
                args.vocab_size,
                max_words,
            ),
            dtype=torch.int32,
        )
        path = example.paths[peak_index]
        if len(path) > max_path:
            path = [path[0]] + path[-(max_path - 1):]
        for path_index, token in enumerate(path[:max_path]):
            path_mask[peak_index, path_index] = True
            path_token_ids[peak_index, path_index] = torch.tensor(
                text_to_ids(token.text, args.vocab_size, max_words),
                dtype=torch.int32,
            )
            path_type_ids[peak_index, path_index] = int(token.token_type)
            path_mech_ids[peak_index, path_index] = int(token.mechanism_id)
            path_parent_mz[peak_index, path_index] = float(token.parent_mz)
            path_product_mz[peak_index, path_index] = float(token.product_mz)

    normalized_true_base = (
        normalize_mz_value(example.true_base_mz)
        if example.true_base_mz is not None
        else None
    )
    base_candidate_index = -1
    if normalized_true_base is not None:
        for index, candidate_mz in enumerate(example.mz_values[:n_peaks]):
            if format_mz_key(float(candidate_mz)) == normalized_true_base:
                base_candidate_index = index
                break

    example.tensor_cache = {
        "path_token_ids": path_token_ids,
        "product_token_ids": product_token_ids,
        "path_type_ids": path_type_ids,
        "path_mech_ids": path_mech_ids,
        "path_parent_mz": path_parent_mz,
        "path_product_mz": path_product_mz,
        "path_mask": path_mask,
        "mz": torch.tensor(example.mz_values[:n_peaks], dtype=torch.float32),
        "target": torch.tensor(example.targets[:n_peaks], dtype=torch.float32),
        "morgan_fp": torch.tensor(
            example.morgan_fp[: args.morgan_fp_dim], dtype=torch.float32
        ),
        "full_gold_l2_norm": torch.tensor(
            math.sqrt(
                sum(float(value) ** 2 for value in example.full_gold_intensity.values())
            ),
            dtype=torch.float32,
        ),
        "base_index": torch.tensor(base_candidate_index, dtype=torch.long),
        "base_present": torch.tensor(base_candidate_index >= 0, dtype=torch.bool),
    }
    example.bucket_shape = (
        _round_up(n_peaks, args.peak_bucket_multiple, args.max_peaks),
        _round_up(max_path, args.path_bucket_multiple, args.max_path_len),
    )


class Stage2TreeDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        *,
        source_type: str,
        args: argparse.Namespace,
    ) -> None:
        self.examples: List[PeakPathExample] = []
        skipped = 0
        original_peak_counts: List[int] = []
        truncated_molecules = 0
        truncated_peaks = 0
        total_candidate_peaks = 0
        preprocessing_start = time.time()

        for record in records:
            row = dict(record)
            row["_source_type"] = source_type
            example = build_tree_paths(
                row,
                morgan_radius=args.morgan_radius,
                morgan_fp_dim=args.morgan_fp_dim,
            )
            if example is None:
                skipped += 1
                continue

            original_count = len(example.mz_values)
            original_peak_counts.append(original_count)
            total_candidate_peaks += original_count

            if original_count > args.max_peaks:
                truncated_molecules += 1
                truncated_peaks += original_count - args.max_peaks
                indices = list(range(args.max_peaks))
                example = PeakPathExample(
                    record_id=example.record_id,
                    name=example.name,
                    smiles=example.smiles,
                    formula=example.formula,
                    mw=example.mw,
                    mz_values=[example.mz_values[i] for i in indices],
                    product_texts=[example.product_texts[i] for i in indices],
                    paths=[example.paths[i] for i in indices],
                    targets=[example.targets[i] for i in indices],
                    full_gold_intensity=dict(example.full_gold_intensity),
                    true_base_mz=example.true_base_mz,
                    morgan_fp=list(example.morgan_fp),
                    source_type=example.source_type,
                )

            tensorize_example_once(example, args)
            self.examples.append(example)

        bucket_counts: Dict[str, int] = defaultdict(int)
        for example in self.examples:
            bucket_counts[f"{example.bucket_shape[0]}x{example.bucket_shape[1]}"] += 1

        self.truncation_stats = {
            "source_type": source_type,
            "records": len(records),
            "usable": len(self.examples),
            "skipped": skipped,
            "max_original_candidates": max(original_peak_counts, default=0),
            "mean_original_candidates": (
                sum(original_peak_counts) / max(1, len(original_peak_counts))
            ),
            "molecules_over_max_peaks": truncated_molecules,
            "total_candidates_truncated": truncated_peaks,
            "max_peaks": args.max_peaks,
            "total_candidate_peaks_before_truncation": total_candidate_peaks,
            "pretokenized_in_memory": True,
            "preprocessing_seconds": round(time.time() - preprocessing_start, 3),
            "bucket_counts": dict(sorted(bucket_counts.items())),
        }

        print("[Dataset] " + json.dumps(self.truncation_stats, ensure_ascii=False))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> PeakPathExample:
        return self.examples[idx]


def resolve_dataset_example(dataset: Dataset, index: int) -> PeakPathExample:
    if isinstance(dataset, Stage2TreeDataset):
        return dataset.examples[index]
    if isinstance(dataset, ConcatDataset):
        dataset_index = bisect.bisect_right(dataset.cumulative_sizes, index)
        previous = 0 if dataset_index == 0 else dataset.cumulative_sizes[dataset_index - 1]
        return resolve_dataset_example(dataset.datasets[dataset_index], index - previous)
    item = dataset[index]
    if not isinstance(item, PeakPathExample):
        raise TypeError(f"Unsupported bucket dataset item: {type(item)!r}")
    return item


class BucketBatchSampler(Sampler[List[int]]):
    """Shape-aware sampler producing a small, stable set of compiled graphs."""
    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        sample_weights: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.sample_weights = sample_weights
        self.num_samples = int(num_samples or len(dataset))
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def __len__(self) -> int:
        # Exact for ordinary loaders. For weighted replay, estimate expected
        # samples per bucket from the sampling probabilities so the LR
        # scheduler remains close to the actual number of optimizer steps.
        bucket_mass: Dict[Tuple[int, int], float] = defaultdict(float)
        if self.sample_weights is None:
            for index in range(len(self.dataset)):
                bucket_mass[resolve_dataset_example(self.dataset, index).bucket_shape] += 1.0
        else:
            total_weight = float(self.sample_weights.sum().item())
            if total_weight <= 0.0:
                return math.ceil(self.num_samples / self.batch_size)
            for index, weight in enumerate(self.sample_weights.tolist()):
                key = resolve_dataset_example(self.dataset, index).bucket_shape
                bucket_mass[key] += self.num_samples * float(weight) / total_weight
        if self.drop_last:
            return sum(int(count) // self.batch_size for count in bucket_mass.values())
        return sum(math.ceil(max(0.0, count) / self.batch_size) for count in bucket_mass.values())

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1

        if self.sample_weights is not None:
            indices = torch.multinomial(
                self.sample_weights,
                self.num_samples,
                replacement=True,
                generator=generator,
            ).tolist()
        else:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in order]

        buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for index in indices:
            example = resolve_dataset_example(self.dataset, index)
            buckets[example.bucket_shape].append(index)

        batches: List[List[int]] = []
        remainders: List[int] = []
        for key in sorted(buckets):
            bucket_indices = buckets[key]
            if self.shuffle and len(bucket_indices) > 1:
                order = torch.randperm(
                    len(bucket_indices), generator=generator
                ).tolist()
                bucket_indices = [bucket_indices[i] for i in order]
            full_count = (len(bucket_indices) // self.batch_size) * self.batch_size
            for begin in range(0, full_count, self.batch_size):
                batches.append(bucket_indices[begin: begin + self.batch_size])
            remainders.extend(bucket_indices[full_count:])

        # Pack the small tail of neighbouring buckets together. This avoids
        # dozens of under-filled GPU steps when the dataset has many shapes;
        # the collator simply pads each mixed tail batch to its largest bucket.
        if not self.drop_last and remainders:
            remainders.sort(
                key=lambda index: resolve_dataset_example(
                    self.dataset, index
                ).bucket_shape
            )
            for begin in range(0, len(remainders), self.batch_size):
                batches.append(remainders[begin: begin + self.batch_size])

        if self.shuffle and len(batches) > 1:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]
        yield from batches


class Stage2Collator:
    """Fast collator: tensor copy/padding only; no tokenization or hashing."""
    def __init__(self, args: argparse.Namespace) -> None:
        self.max_peaks = args.max_peaks
        self.max_path_len = args.max_path_len
        self.max_words = args.max_words_per_token
        self.morgan_fp_dim = args.morgan_fp_dim

    def __call__(self, batch: List[PeakPathExample]) -> Dict[str, Any]:
        bsz = len(batch)
        max_p = min(self.max_peaks, max(item.bucket_shape[0] for item in batch))
        max_l = min(self.max_path_len, max(item.bucket_shape[1] for item in batch))
        max_w = self.max_words

        path_token_ids = torch.zeros((bsz, max_p, max_l, max_w), dtype=torch.long)
        product_token_ids = torch.zeros((bsz, max_p, max_w), dtype=torch.long)
        path_type_ids = torch.zeros((bsz, max_p, max_l), dtype=torch.long)
        path_mech_ids = torch.zeros((bsz, max_p, max_l), dtype=torch.long)
        path_parent_mz = torch.zeros((bsz, max_p, max_l), dtype=torch.float32)
        path_product_mz = torch.zeros((bsz, max_p, max_l), dtype=torch.float32)
        path_mask = torch.zeros((bsz, max_p, max_l), dtype=torch.bool)
        peak_mask = torch.zeros((bsz, max_p), dtype=torch.bool)
        mz = torch.zeros((bsz, max_p), dtype=torch.float32)
        target = torch.zeros((bsz, max_p), dtype=torch.float32)
        mw = torch.zeros((bsz,), dtype=torch.float32)
        morgan_fp = torch.zeros((bsz, self.morgan_fp_dim), dtype=torch.float32)
        base_index = torch.full((bsz,), -1, dtype=torch.long)
        base_present = torch.zeros((bsz,), dtype=torch.bool)
        full_gold_l2_norm = torch.zeros((bsz,), dtype=torch.float32)
        meta: List[Dict[str, Any]] = []

        for batch_index, example in enumerate(batch):
            cache = example.tensor_cache
            if cache is None:
                raise RuntimeError("Example tensor cache was not initialized")
            n_peaks = min(max_p, int(cache["mz"].numel()))
            n_path = min(max_l, int(cache["path_token_ids"].size(1)))

            path_token_ids[batch_index, :n_peaks, :n_path].copy_(
                cache["path_token_ids"][:n_peaks, :n_path]
            )
            product_token_ids[batch_index, :n_peaks].copy_(
                cache["product_token_ids"][:n_peaks]
            )
            path_type_ids[batch_index, :n_peaks, :n_path].copy_(
                cache["path_type_ids"][:n_peaks, :n_path]
            )
            path_mech_ids[batch_index, :n_peaks, :n_path].copy_(
                cache["path_mech_ids"][:n_peaks, :n_path]
            )
            path_parent_mz[batch_index, :n_peaks, :n_path].copy_(
                cache["path_parent_mz"][:n_peaks, :n_path]
            )
            path_product_mz[batch_index, :n_peaks, :n_path].copy_(
                cache["path_product_mz"][:n_peaks, :n_path]
            )
            path_mask[batch_index, :n_peaks, :n_path].copy_(
                cache["path_mask"][:n_peaks, :n_path]
            )
            peak_mask[batch_index, :n_peaks] = True
            mz[batch_index, :n_peaks].copy_(cache["mz"][:n_peaks])
            target[batch_index, :n_peaks].copy_(cache["target"][:n_peaks])
            mw[batch_index] = float(example.mw)
            morgan_fp[batch_index].copy_(cache["morgan_fp"])
            base_index[batch_index].copy_(cache["base_index"])
            base_present[batch_index].copy_(cache["base_present"])
            full_gold_l2_norm[batch_index].copy_(cache["full_gold_l2_norm"])

            meta.append({
                "id": example.record_id,
                "name": example.name,
                "smiles": example.smiles,
                "formula": example.formula,
                "mw": example.mw,
                "source_type": example.source_type,
                "mz_values": example.mz_values[:n_peaks],
                "full_gold_intensity": dict(example.full_gold_intensity),
                "true_base_mz": example.true_base_mz,
                "base_present": bool(cache["base_present"].item()),
            })

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
            "base_index": base_index,
            "base_present": base_present,
            "full_gold_l2_norm": full_gold_l2_norm,
            "target": target,
            "meta": meta,
        }


# ============================================================
# 4. Model
# ============================================================

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

        # Ablation switches.
        #
        # IMPORTANT:
        # All defaults are True, so running this script without any ablation
        # flags follows the exact same model path as the original full model.
        # The switches only change the forward path when explicitly enabled
        # through the corresponding CLI ablation flags.
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

        # ------------------------------------------------------------
        # Global molecular context.
        #
        # Full model (default): unchanged from the original implementation.
        # Ablation: skip Morgan-fingerprint encoding and the local-global
        # gated fusion completely.
        # ------------------------------------------------------------
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

        # Dense, compile-friendly path attention. Invalid rows receive one
        # temporary zero key and are masked back to zero after attention.
        valid = flat_peak_mask & flat_path_mask.any(dim=-1)
        valid_f = valid.to(flat_path.dtype).view(-1, 1, 1)
        safe_flat_path = flat_path * valid_f
        safe_first_column = flat_path_mask[:, :1] | (~valid).unsqueeze(1)
        safe_flat_path_mask = torch.cat(
            [safe_first_column, flat_path_mask[:, 1:]], dim=1
        )
        # ------------------------------------------------------------
        # ROOT-to-peak path aggregation.
        #
        # Full model (default): original multi-head path attention.
        # Ablation: masked mean pooling over the same path tokens, so the
        # input information and downstream network remain unchanged while
        # only the learnable attention operation is removed.
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Inter-peak interaction.
        #
        # Full model (default): exactly the original global molecule token +
        # Transformer interaction + context fusion.
        #
        # w/o Global molecular context:
        #   the Transformer still models interactions among candidate peaks,
        #   but no Morgan-derived molecule token or molecule-context fusion
        #   is used.
        #
        # w/o Inter-peak interaction:
        #   every candidate peak is sent directly to the output head after
        #   its local path representation is constructed.
        # ------------------------------------------------------------
        if self.use_inter_peak_interaction:
            if self.use_global_context:
                # Original full-model path.
                molecule_token = self.global_token_proj(global_embedding).unsqueeze(1)
                interaction_input = torch.cat(
                    [molecule_token, peak_representation],
                    dim=1,
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
                # Keep peak-to-peak Transformer interaction while removing
                # all global molecular context.
                interacted = self.tree_interaction(
                    peak_representation,
                    src_key_padding_mask=~peak_mask,
                )
        else:
            # Independent candidate-peak prediction after local path encoding.
            interacted = peak_representation

        output_logits = self.output_head(interacted).squeeze(-1)
        prediction = F.softplus(output_logits, beta=1.0, threshold=20.0) * peak_mask.float()
        ranking_logits = output_logits.masked_fill(~peak_mask, -1.0e4)
        if return_logits:
            return prediction, ranking_logits
        return prediction


# ============================================================
# 5. Losses
# ============================================================

def build_true_base_mask(
    base_index: torch.Tensor,
    base_present: torch.Tensor,
    peak_mask: torch.Tensor,
) -> torch.Tensor:
    safe_index = base_index.clamp(min=0, max=max(0, peak_mask.size(1) - 1))
    result = torch.zeros_like(peak_mask, dtype=torch.bool)
    result.scatter_(1, safe_index.unsqueeze(1), base_present.unsqueeze(1))
    return result & peak_mask


def masked_weighted_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    base_index: torch.Tensor,
    base_present: torch.Tensor,
    *,
    beta: float = 0.04,
    peak_weight_alpha: float = 2.0,
    base_peak_extra: float = 3.0,
    strong_peak_extra: float = 1.5,
    strong_peak_threshold: float = 0.30,
    strong_peak_gamma: float = 1.4,
) -> torch.Tensor:
    """Low-weight continuous intensity regression over every candidate."""
    mask_f = mask.float()
    true_base_mask = build_true_base_mask(base_index, base_present, mask)
    strong_mask = (target >= strong_peak_threshold) & mask
    weights = (
        1.0
        + peak_weight_alpha * target.clamp_min(0.0).pow(strong_peak_gamma)
        + strong_peak_extra * strong_mask.float()
        + base_peak_extra * true_base_mask.float()
    )
    token_loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    effective = weights * mask_f
    per_molecule = (token_loss * effective).sum(dim=1) / effective.sum(dim=1).clamp_min(1.0)
    valid = mask.any(dim=1).float()
    return (per_molecule * valid).sum() / valid.sum().clamp_min(1.0)


def strong_peak_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float = 0.30,
    top_k: int = 5,
    beta: float = 0.04,
) -> torch.Tensor:
    """Vectorized strong/top-k regression without Python-GPU synchronization."""
    peak_count = target.size(1)
    k = max(1, min(int(top_k), peak_count))
    masked_target = target.masked_fill(~mask, -1.0e4)
    top_indices = torch.topk(masked_target, k=k, dim=1).indices
    top_mask = torch.zeros_like(mask, dtype=torch.bool)
    top_mask.scatter_(1, top_indices, True)
    selected = mask & (target > 0.0) & ((target >= threshold) | top_mask)

    token_loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    weights = (1.0 + 2.0 * target.clamp_min(0.0)) * selected.float()
    per_molecule = (token_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    valid = selected.any(dim=1).float()
    return (per_molecule * valid).sum() / valid.sum().clamp_min(1.0)


def union_space_cosine_values(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    full_gold_l2_norm: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_f = mask.float()
    prediction = pred * mask_f
    candidate_gold = target * mask_f
    numerator = (prediction * candidate_gold).sum(dim=1)
    prediction_norm = torch.sqrt((prediction * prediction).sum(dim=1) + eps)
    complete_gold_norm = full_gold_l2_norm.float()
    valid = mask.any(dim=1) & (complete_gold_norm > 1e-12)
    cosine = numerator / (prediction_norm * complete_gold_norm + eps)
    return cosine, valid


def union_space_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    full_gold_l2_norm: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    cosine, valid = union_space_cosine_values(
        pred, target, mask, full_gold_l2_norm, eps
    )
    valid_f = valid.float()
    return ((1.0 - cosine) * valid_f).sum() / valid_f.sum().clamp_min(1.0)


def tree_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_label_gap: float = 0.02,
    base_margin: float = 0.03,
    gap_margin_scale: float = 0.20,
) -> torch.Tensor:
    """Fully vectorized pairwise ranking over all candidate intensities."""
    y_diff = target.unsqueeze(2) - target.unsqueeze(1)
    p_diff = pred.unsqueeze(2) - pred.unsqueeze(1)
    pair_valid = mask.unsqueeze(2) & mask.unsqueeze(1)
    pair_mask = pair_valid & (y_diff > min_label_gap)
    margin = base_margin + gap_margin_scale * y_diff.clamp(0.0, 1.0)
    pair_loss = F.relu(margin - p_diff) * y_diff.clamp_min(0.0)
    pair_mask_f = pair_mask.float()
    per_molecule = (pair_loss * pair_mask_f).sum(dim=(1, 2)) / pair_mask_f.sum(dim=(1, 2)).clamp_min(1.0)
    valid = pair_mask.any(dim=(1, 2)).float()
    return (per_molecule * valid).sum() / valid.sum().clamp_min(1.0)



def base_peak_classification_loss(
    logits: torch.Tensor,
    base_index: torch.Tensor,
    base_present: torch.Tensor,
    peak_mask: torch.Tensor,
    label_smoothing: float = 0.02,
) -> torch.Tensor:
    valid = base_present & (base_index >= 0) & peak_mask.any(dim=1)
    selected_logits = logits.masked_fill(~peak_mask, -1.0e4)
    safe_target = base_index.clamp(min=0)
    log_probability = F.log_softmax(selected_logits, dim=1)
    nll = -log_probability.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    candidate_count = peak_mask.sum(dim=1).float().clamp_min(1.0)
    # Smooth only across valid candidates. PyTorch's built-in label_smoothing
    # would also allocate probability mass to masked padding classes whose
    # logits are -1e4, producing an artificially enormous loss.
    smooth = -(log_probability * peak_mask.float()).sum(dim=1) / candidate_count
    smoothing = float(max(0.0, min(1.0, label_smoothing)))
    per_sample = (1.0 - smoothing) * nll + smoothing * smooth
    normalized = per_sample / torch.log(candidate_count.clamp_min(2.0)).clamp_min(math.log(2.0))
    valid_f = valid.float()
    return (normalized * valid_f).sum() / valid_f.sum().clamp_min(1.0)


def compute_loss(
    pred_raw: torch.Tensor,
    ranking_logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    base_index: torch.Tensor,
    base_present: torch.Tensor,
    full_gold_l2_norm: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pred = normalize_prediction_by_base_peak(pred_raw, mask)
    l_int = masked_weighted_smooth_l1(
        pred,
        target,
        mask,
        base_index,
        base_present,
        beta=args.smooth_l1_beta,
        peak_weight_alpha=args.peak_weight_alpha,
        base_peak_extra=args.base_peak_extra,
        strong_peak_extra=args.strong_peak_extra,
        strong_peak_threshold=args.strong_peak_threshold,
        strong_peak_gamma=args.strong_peak_gamma,
    )
    l_union = union_space_cosine_loss(pred, target, mask, full_gold_l2_norm)
    l_rank = tree_ranking_loss(
        pred,
        target,
        mask,
        min_label_gap=args.rank_min_gap,
        base_margin=args.rank_base_margin,
        gap_margin_scale=args.rank_gap_margin_scale,
    )
    l_base = base_peak_classification_loss(
        ranking_logits,
        base_index,
        base_present,
        mask,
        label_smoothing=args.base_label_smoothing,
    )
    l_strong = strong_peak_regression_loss(
        pred,
        target,
        mask,
        threshold=args.strong_peak_threshold,
        top_k=args.strong_peak_topk,
        beta=args.smooth_l1_beta,
    )
    total = (
        args.lambda_intensity * l_int
        + args.lambda_spectral * l_union
        + args.lambda_rank * l_rank
        + args.lambda_base * l_base
        + args.lambda_strong * l_strong
    )
    return total, {
        "loss_total": total.detach(),
        "loss_intensity": l_int.detach(),
        "loss_union_cosine": l_union.detach(),
        "loss_spectral": l_union.detach(),
        "loss_rank": l_rank.detach(),
        "loss_base": l_base.detach(),
        "loss_strong": l_strong.detach(),
    }


# ============================================================
# 6. Data loaders
# ============================================================

def make_dataset(
    records: List[Dict[str, Any]],
    *,
    source_type: str,
    args: argparse.Namespace,
) -> Optional[Stage2TreeDataset]:
    if not records:
        return None
    dataset = Stage2TreeDataset(records, source_type=source_type, args=args)
    return dataset if len(dataset) > 0 else None


def _loader_worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader_from_dataset(
    dataset: Optional[Dataset],
    *,
    args: argparse.Namespace,
    shuffle: bool,
) -> Optional[DataLoader]:
    if dataset is None or len(dataset) == 0:
        return None
    batch_sampler = BucketBatchSampler(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        seed=args.seed + (17 if shuffle else 0),
    )
    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": args.num_workers,
        "collate_fn": Stage2Collator(args),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": _loader_worker_init,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**loader_kwargs)


def make_stage1_replay_loader(
    stage1_dataset: Optional[Stage2TreeDataset],
    gold_dataset: Optional[Stage2TreeDataset],
    *,
    args: argparse.Namespace,
) -> Optional[DataLoader]:
    if stage1_dataset is None:
        return None
    if gold_dataset is None or args.phase_b_gold_ratio <= 0.0 or len(gold_dataset) == 0:
        return make_loader_from_dataset(stage1_dataset, args=args, shuffle=True)

    ratio = min(max(float(args.phase_b_gold_ratio), 0.0), 0.5)
    combined = ConcatDataset([stage1_dataset, gold_dataset])
    stage1_weight = (1.0 - ratio) / max(1, len(stage1_dataset))
    gold_weight = ratio / max(1, len(gold_dataset))
    sample_weights = torch.tensor(
        [stage1_weight] * len(stage1_dataset) + [gold_weight] * len(gold_dataset),
        dtype=torch.double,
    )
    batch_sampler = BucketBatchSampler(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed + 1701,
        sample_weights=sample_weights,
        num_samples=len(stage1_dataset),
    )
    loader_kwargs: Dict[str, Any] = {
        "dataset": combined,
        "batch_sampler": batch_sampler,
        "num_workers": args.num_workers,
        "collate_fn": Stage2Collator(args),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": _loader_worker_init,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    print(
        f"[Phase B replay] Stage1={1.0-ratio:.2%}, Gold={ratio:.2%}, "
        f"samples_per_epoch={len(stage1_dataset)}, bucketed=True"
    )
    return DataLoader(**loader_kwargs)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


# ============================================================
# 7. EMA
# ============================================================

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for name, ema_value in ema_state.items():
            source_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(
                    source_value.to(dtype=ema_value.dtype),
                    alpha=1.0 - self.decay,
                )
            else:
                ema_value.copy_(source_value)

    def reset_from(self, model: nn.Module) -> None:
        self.module.load_state_dict(model.state_dict(), strict=True)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.module.state_dict()


# ============================================================
# 8. Train / eval
# ============================================================

@torch.no_grad()
def spectral_cosine_values(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Candidate-peak cosine retained only as a diagnostic metric."""
    mask_f = mask.float()
    prediction = pred * mask_f
    gold = target * mask_f
    numerator = (prediction * gold).sum(dim=1)
    pred_norm = torch.sqrt((prediction * prediction).sum(dim=1) + eps)
    gold_norm = torch.sqrt((gold * gold).sum(dim=1) + eps)
    valid = (mask_f.sum(dim=1) > 0) & (gold_norm > math.sqrt(eps))
    cosine = numerator / (pred_norm * gold_norm + eps)
    return cosine, valid


def vector_cosine_like_inference(
    pred_values: List[float],
    gold_values: List[float],
) -> Optional[float]:
    """
    Exact scalar cosine convention used by stage2_infer_New.py.

    The vectors are float32, a zero-norm vector produces None, and the
    denominator contains 1e-8. This function is intentionally separate from
    the differentiable training loss because it is used only for checkpoint
    selection.
    """
    if not pred_values or not gold_values or len(pred_values) != len(gold_values):
        return None

    p = torch.tensor(pred_values, dtype=torch.float32)
    y = torch.tensor(gold_values, dtype=torch.float32)
    if p.norm().item() <= 1e-12 or y.norm().item() <= 1e-12:
        return None

    return float(torch.dot(p, y) / (p.norm() * y.norm() + 1e-8))


def union_peak_cosine_for_one_molecule(
    *,
    predicted_mz_values: List[float],
    predicted_values: List[float],
    full_gold_intensity: Dict[str, float],
    round_digits: int = EVAL_ROUND_DIGITS,
) -> Optional[float]:
    """
    Reproduce inference ``cosine_union_peaks`` for one molecule.

    - predicted spectrum keys: all retained Stage1 candidate m/z values;
    - gold spectrum keys: every m/z in the paired full Gold intensity mapping;
    - evaluation keys: predicted keys UNION gold keys;
    - a missing value on either side is filled with zero;
    - predictions are clipped and rounded before cosine, exactly as inference.
    """
    pred_spec: Dict[str, float] = {}
    for mz, value in zip(predicted_mz_values, predicted_values):
        key = format_mz_key(float(mz))
        clipped = max(0.0, min(1.0, float(value)))
        pred_spec[key] = round(clipped, round_digits)

    gold_spec: Dict[str, float] = {}
    for mz_key, value in (full_gold_intensity or {}).items():
        normalized_key = normalize_mz_value(mz_key)
        if normalized_key is None:
            continue
        gold_spec[normalized_key] = max(
            0.0,
            min(1.0, parse_float(value, 0.0)),
        )

    union_keys = sorted(
        set(pred_spec.keys()) | set(gold_spec.keys()),
        key=lambda key: float(key),
    )
    if not union_keys:
        return None

    pred_vector = [float(pred_spec.get(key, 0.0)) for key in union_keys]
    gold_vector = [float(gold_spec.get(key, 0.0)) for key in union_keys]
    return vector_cosine_like_inference(pred_vector, gold_vector)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """
    Validate using the exact final inference metric used for model selection.

    ``mean_cosine_union_peaks`` is computed per molecule on the union of:
      1. Stage1 candidate/predicted m/z peaks, and
      2. all m/z peaks in the paired Gold spectrum.

    Gold peaks missed by Stage1 receive predicted intensity 0. Extra Stage1
    peaks receive Gold intensity 0. The per-molecule cosine values are then
    macro-averaged, matching ``summarize_outputs`` in stage2_infer_New.py.
    """
    if loader is None or len(loader.dataset) == 0:
        return {
            "loss_total": 0.0,
            "loss_intensity": 0.0,
            "loss_spectral": 0.0,
            "loss_rank": 0.0,
            "loss_base": 0.0,
            "loss_strong": 0.0,
            "candidate_peak_cosine": 0.0,
            "mean_cosine_union_peaks": 0.0,
            "union_cosine_count": 0.0,
            "base_peak_candidate_recall": 0.0,
            "conditional_base_peak_accuracy": 0.0,
            "base_peak_accuracy": 0.0,
            "top1_gold_coverage_recall": 0.0,
            "top3_gold_coverage_recall": 0.0,
            "top5_gold_coverage_recall": 0.0,
        }

    model.eval()
    totals = {
        "loss_total": 0.0,
        "loss_intensity": 0.0,
        "loss_spectral": 0.0,
        "loss_rank": 0.0,
        "loss_base": 0.0,
        "loss_strong": 0.0,
    }
    molecule_count = 0

    candidate_cosine_sum = 0.0
    candidate_cosine_count = 0
    union_cosine_sum = 0.0
    union_cosine_count = 0

    base_candidate_total = 0
    base_candidate_hit = 0
    conditional_base_correct = 0
    overall_base_correct = 0
    topk_coverage_sums = {1: 0.0, 3: 0.0, 5: 0.0}
    topk_coverage_counts = {1: 0, 3: 0, 5: 0}

    # Match stage2_infer_New.py default numerical path.
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    )

    for batch in loader:
        batch = move_batch(batch, device)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            pred_raw, ranking_logits = model(
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
                return_logits=True,
            )

        pred = normalize_prediction_by_base_peak(
            pred_raw,
            batch["peak_mask"],
        )
        _, logs = compute_loss(
            pred_raw,
            ranking_logits,
            batch["target"],
            batch["peak_mask"],
            batch["base_index"],
            batch["base_present"],
            batch["full_gold_l2_norm"],
            args,
        )
        batch_size = int(batch["target"].size(0))
        molecule_count += batch_size
        for key in totals:
            totals[key] += float(logs[key].detach().float().cpu()) * batch_size

        # Diagnostic: old candidate-only cosine.
        candidate_cosine, candidate_valid = spectral_cosine_values(
            pred,
            batch["target"],
            batch["peak_mask"],
        )
        if candidate_valid.any():
            candidate_cosine_sum += float(
                candidate_cosine[candidate_valid].sum().float().cpu()
            )
            candidate_cosine_count += int(candidate_valid.sum().cpu())

        # Selection metric: exact inference union-peak cosine.
        pred_cpu = pred.detach().float().cpu()
        peak_mask_cpu = batch["peak_mask"].detach().cpu()

        for batch_index, meta in enumerate(batch["meta"]):
            valid_n = int(peak_mask_cpu[batch_index].sum().item())
            mz_values = [
                float(value)
                for value in meta.get("mz_values", [])[:valid_n]
            ]
            pred_values = [
                float(pred_cpu[batch_index, peak_index].item())
                for peak_index in range(valid_n)
            ]

            union_cosine = union_peak_cosine_for_one_molecule(
                predicted_mz_values=mz_values,
                predicted_values=pred_values,
                full_gold_intensity=meta.get("full_gold_intensity", {}),
                round_digits=EVAL_ROUND_DIGITS,
            )
            if union_cosine is not None and math.isfinite(union_cosine):
                union_cosine_sum += float(union_cosine)
                union_cosine_count += 1

            gold_spec = {
                normalize_mz_value(key): max(
                    0.0,
                    min(1.0, parse_float(value, 0.0)),
                )
                for key, value in meta.get(
                    "full_gold_intensity", {}
                ).items()
                if normalize_mz_value(key) is not None
            }
            candidate_set = {
                format_mz_key(float(value)) for value in mz_values
            }
            true_base_mz = normalize_mz_value(
                meta.get("true_base_mz")
            )
            if true_base_mz is not None:
                base_candidate_total += 1
                candidate_has_base = true_base_mz in candidate_set
                base_candidate_hit += int(candidate_has_base)

                if pred_values:
                    pred_base_index = max(
                        range(len(pred_values)),
                        key=lambda index: (
                            pred_values[index],
                            mz_values[index],
                        ),
                    )
                    predicted_base_mz = format_mz_key(
                        float(mz_values[pred_base_index])
                    )
                    base_correct = predicted_base_mz == true_base_mz
                    overall_base_correct += int(base_correct)
                    if candidate_has_base:
                        conditional_base_correct += int(base_correct)

            for top_k in (1, 3, 5):
                if not gold_spec:
                    continue
                gold_top = {
                    key
                    for key, _ in sorted(
                        gold_spec.items(),
                        key=lambda item: (item[1], parse_float(item[0])),
                        reverse=True,
                    )[:top_k]
                }
                if gold_top:
                    topk_coverage_sums[top_k] += (
                        len(gold_top & candidate_set) / len(gold_top)
                    )
                    topk_coverage_counts[top_k] += 1

    result = {
        key: value / max(1, molecule_count)
        for key, value in totals.items()
    }
    result["candidate_peak_cosine"] = (
        candidate_cosine_sum / candidate_cosine_count
        if candidate_cosine_count > 0
        else 0.0
    )
    result["mean_cosine_union_peaks"] = (
        union_cosine_sum / union_cosine_count
        if union_cosine_count > 0
        else 0.0
    )
    result["union_cosine_count"] = float(union_cosine_count)
    result["base_peak_candidate_recall"] = (
        base_candidate_hit / base_candidate_total
        if base_candidate_total > 0
        else 0.0
    )
    result["conditional_base_peak_accuracy"] = (
        conditional_base_correct / base_candidate_hit
        if base_candidate_hit > 0
        else 0.0
    )
    result["base_peak_accuracy"] = (
        overall_base_correct / base_candidate_total
        if base_candidate_total > 0
        else 0.0
    )
    for top_k in (1, 3, 5):
        result[f"top{top_k}_gold_coverage_recall"] = (
            topk_coverage_sums[top_k] / topk_coverage_counts[top_k]
            if topk_coverage_counts[top_k] > 0
            else 0.0
        )
    return result


def save_checkpoint(
    path: str,
    *,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_cosine: float,
    phase_name: str,
    args: argparse.Namespace,
    val_logs: Optional[Dict[str, float]] = None,
    selected_model: str = "raw",
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    selected_state = (
        ema.state_dict()
        if selected_model == "ema"
        else model.state_dict()
    )
    torch.save(
        {
            # Existing inference code can continue loading checkpoint["model"].
            "model": selected_state,
            "raw_model": model.state_dict(),
            "ema_model": ema.state_dict(),
            "selected_model": selected_model,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_cosine": best_cosine,
            "best_mean_cosine_union_peaks": best_cosine,
            "selection_metric": "mean_cosine_union_peaks",
            "best_val": -best_cosine,
            "phase_name": phase_name,
            "args": vars(args),
            "val_logs": val_logs or {},
            "model_input_version": "path_parent_product_mz_morgan_globaltoken_softplus_fiveloss_v7",
        },
        path,
    )


def load_selected_checkpoint_into_model(
    checkpoint_path: str,
    model: nn.Module,
    ema: ModelEMA,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    ema.reset_from(model)
    return checkpoint


def train_one_phase(
    *,
    model: nn.Module,
    forward_model: nn.Module,
    ema: ModelEMA,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    phase_name: str,
    epochs: int,
    global_step: int,
    global_best_cosine: float,
    phase_best_path: str,
) -> Dict[str, Any]:
    phase_best_cosine = float("-inf")
    no_improve = 0
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    log_keys = (
        "loss_total",
        "loss_intensity",
        "loss_spectral",
        "loss_rank",
        "loss_base",
        "loss_strong",
    )

    for epoch in range(1, epochs + 1):
        model.train()
        forward_model.train()
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        running = {
            key: torch.zeros((), device=device, dtype=torch.float32)
            for key in log_keys
        }
        seen_molecules = 0
        optimizer.zero_grad(set_to_none=True)
        total_micro_steps = len(train_loader)
        last_print_time = time.perf_counter()
        last_print_seen = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            group_start = ((step - 1) // args.grad_accum) * args.grad_accum + 1
            group_end = min(group_start + args.grad_accum - 1, total_micro_steps)
            accumulation_divisor = group_end - group_start + 1

            with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                pred_raw, ranking_logits = forward_model(
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
                    return_logits=True,
                )
                loss, logs = compute_loss(
                    pred_raw,
                    ranking_logits,
                    batch["target"],
                    batch["peak_mask"],
                    batch["base_index"],
                    batch["base_present"],
                    batch["full_gold_l2_norm"],
                    args,
                )
                backward_loss = loss / accumulation_divisor

            scaler.scale(backward_loss).backward()
            should_update = step % args.grad_accum == 0 or step == total_micro_steps
            if should_update:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
                global_step += 1

            batch_size = int(batch["target"].size(0))
            seen_molecules += batch_size
            for key in log_keys:
                running[key].add_(logs[key].float() * batch_size)

            if step % args.print_every == 0 or step == total_micro_steps:
                # One synchronization for all scalar logs, rather than one
                # .cpu() call per loss per micro-step.
                stacked = torch.stack([running[key] for key in log_keys]) / max(1, seen_molecules)
                values = stacked.detach().float().cpu().tolist()
                averages = dict(zip(log_keys, values))
                now = time.perf_counter()
                recent_rate = (seen_molecules - last_print_seen) / max(1e-6, now - last_print_time)
                last_print_time = now
                last_print_seen = seen_molecules
                print(
                    f"[{phase_name}] epoch={epoch}/{epochs} step={step}/{total_micro_steps} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} mol/s={recent_rate:.2f} "
                    f"total={averages['loss_total']:.5f} "
                    f"union={averages['loss_spectral']:.5f} "
                    f"rank={averages['loss_rank']:.5f} "
                    f"base={averages['loss_base']:.5f} "
                    f"strong={averages['loss_strong']:.5f} "
                    f"int={averages['loss_intensity']:.5f}"
                )

        stacked = torch.stack([running[key] for key in log_keys]) / max(1, seen_molecules)
        train_average = dict(zip(log_keys, stacked.detach().float().cpu().tolist()))
        val_logs = evaluate(ema.module, val_loader, device, args) if val_loader is not None else {
            **train_average,
            "candidate_peak_cosine": 1.0 - train_average["loss_spectral"],
            "mean_cosine_union_peaks": 1.0 - train_average["loss_spectral"],
            "union_cosine_count": 0.0,
            "base_peak_candidate_recall": 0.0,
            "conditional_base_peak_accuracy": 0.0,
            "base_peak_accuracy": 0.0,
            "top1_gold_coverage_recall": 0.0,
            "top3_gold_coverage_recall": 0.0,
            "top5_gold_coverage_recall": 0.0,
        }
        val_cosine = float(val_logs["mean_cosine_union_peaks"])
        epoch_seconds = time.perf_counter() - epoch_start
        peak_allocated_gb = 0.0
        peak_reserved_gb = 0.0
        if device.type == "cuda":
            peak_allocated_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024 ** 3

        log_line = {
            "phase": phase_name,
            "epoch": epoch,
            "global_step": global_step,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(epoch_seconds, 3),
            "molecules_per_second": seen_molecules / max(epoch_seconds, 1e-6),
            "cuda_peak_allocated_gb": round(peak_allocated_gb, 3),
            "cuda_peak_reserved_gb": round(peak_reserved_gb, 3),
            "train": train_average,
            "val_ema": val_logs,
            "selection_metric": "mean_cosine_union_peaks",
            "validation_source": args.infer_val_file,
        }
        append_jsonl(log_line, os.path.join(args.output_dir, "train_log.jsonl"))

        print(
            f"[{phase_name}] epoch={epoch} done | seconds={epoch_seconds:.2f} "
            f"mol/s={seen_molecules/max(epoch_seconds,1e-6):.2f} "
            f"cuda_peak={peak_allocated_gb:.2f}/{peak_reserved_gb:.2f}GB "
            f"val_union_cosine={val_cosine:.6f} "
            f"val_candidate_cosine={val_logs['candidate_peak_cosine']:.6f} "
            f"base_acc={val_logs['base_peak_accuracy']:.4f}"
        )

        save_checkpoint(
            os.path.join(args.output_dir, "last.pt"),
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_cosine=max(global_best_cosine, val_cosine),
            phase_name=phase_name,
            args=args,
            val_logs=val_logs,
            selected_model="raw",
        )

        if val_cosine > phase_best_cosine + args.min_delta:
            phase_best_cosine = val_cosine
            no_improve = 0
            save_checkpoint(
                phase_best_path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_cosine=phase_best_cosine,
                phase_name=phase_name,
                args=args,
                val_logs=val_logs,
                selected_model="ema",
            )
            print(
                f"[SAVE] {Path(phase_best_path).name} updated: "
                f"EMA mean_cosine_union_peaks={phase_best_cosine:.6f}"
            )
        else:
            no_improve += 1

        if val_cosine > global_best_cosine + args.min_delta:
            global_best_cosine = val_cosine
            save_checkpoint(
                os.path.join(args.output_dir, "best.pt"),
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_cosine=global_best_cosine,
                phase_name=phase_name,
                args=args,
                val_logs=val_logs,
                selected_model="ema",
            )
            print(
                f"[SAVE] best.pt updated: EMA mean_cosine_union_peaks={global_best_cosine:.6f}"
            )

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(
                f"[EARLY STOP] phase={phase_name}, no cosine improvement "
                f"for {no_improve} epochs."
            )
            break

    return {
        "global_step": global_step,
        "global_best_cosine": global_best_cosine,
        "phase_best_cosine": phase_best_cosine,
    }


# ============================================================
# 9. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Optimized Stage2 Structured Tree-Path Attention trainer"
    )

    parser.add_argument(
        "--gold_train_file",
        type=str,
        default="./dataset/train/stage2_gold_train.jsonl",
    )
    parser.add_argument(
        "--stage1_train_file",
        type=str,
        default="./dataset/train/stage2_stage1_predict_train.jsonl",
        help=(
            "Stage1-generated training file. Prefer the same processed/filtered "
            "distribution used by final inference; all candidates are trained "
            "with one continuous intensity objective."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/stage2_treepath_attention_optimized",
    )
    parser.add_argument(
        "--infer_val_file",
        type=str,
        required=True,
        help=(
            "Independent infer JSON/JSONL used exclusively for validation "
            "cosine and best-checkpoint selection."
        ),
    )
    parser.add_argument(
        "--infer_val_gold_file",
        type=str,
        default="",
        help=(
            "Optional held-out gold JSON/JSONL containing target intensities. "
            "It is merged with --infer_val_file by molecule id. This is not "
            "the training gold file."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--prefetch_factor", type=int, default=4)

    # Single-stage Stage1-tree adaptation schedule.
    # Gold data are retained only for the optional Phase-B replay mixture;
    # there is no separate Gold-tree warmup phase.
    parser.add_argument("--stage1_epochs", type=int, default=350)
    parser.add_argument("--stage1_lr", type=float, default=5.0e-5)
    parser.add_argument("--phase_b_gold_ratio", type=float, default=0.08)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # Smaller default capacity; still large enough for structured tree inputs.
    parser.add_argument("--vocab_size", type=int, default=65536)
    parser.add_argument("--d_model", type=int, default=1152)
    parser.add_argument("--n_heads", type=int, default=18)
    parser.add_argument("--interaction_layers", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.08)

    parser.add_argument("--max_peaks", type=int, default=320)
    parser.add_argument("--max_path_len", type=int, default=128)
    parser.add_argument("--max_words_per_token", type=int, default=48)
    parser.add_argument("--morgan_fp_dim", type=int, default=4096)
    parser.add_argument("--morgan_radius", type=int, default=2)

    # Similarity-first five-loss objective. Presence/extra losses are fully removed.
    # Raw loss magnitudes differ, so these are optimization coefficients rather
    # than percentages. Intensity regression is deliberately weak; exact
    # union-space cosine and continuous ranking dominate.
    parser.add_argument("--lambda_intensity", type=float, default=0.05)
    parser.add_argument("--lambda_spectral", type=float, default=3.00)
    parser.add_argument("--lambda_rank", type=float, default=0.00)
    parser.add_argument("--lambda_base", type=float, default=0.00)
    parser.add_argument("--lambda_strong", type=float, default=0.00)

    # ============================================================
    # Ablation switches
    #
    # All switches are OFF by default. Therefore, the command used for the
    # original full model does not need to change and all original weights,
    # schedules, dimensions, and optimization settings remain unchanged.
    #
    # Loss-level ablations:
    #   each switch only sets the corresponding lambda to 0.0.
    #
    # Architecture/training ablations:
    #   --ablate_global_context
    #   --ablate_path_attention
    #   --ablate_inter_peak_interaction
    #   --ablate_gold_guided_curriculum
    # ============================================================
    parser.add_argument(
        "--ablate_intensity_loss",
        action="store_true",
        default=False,
        help="Ablation: set lambda_intensity=0 while leaving all other settings unchanged.",
    )
    parser.add_argument(
        "--ablate_spectral_loss",
        action="store_true",
        default=False,
        help="Ablation: set lambda_spectral=0 while leaving all other settings unchanged.",
    )
    parser.add_argument(
        "--ablate_ranking_loss",
        action="store_true",
        default=False,
        help="Ablation: set lambda_rank=0 while leaving all other settings unchanged.",
    )
    parser.add_argument(
        "--ablate_base_peak_loss",
        action="store_true",
        default=False,
        help="Ablation: set lambda_base=0 while leaving all other settings unchanged.",
    )
    parser.add_argument(
        "--ablate_strong_peak_loss",
        action="store_true",
        default=False,
        help="Ablation: set lambda_strong=0 while leaving all other settings unchanged.",
    )
    parser.add_argument(
        "--ablate_global_context",
        "--no_global_context",
        dest="ablate_global_context",
        action="store_true",
        default=False,
        help=(
            "Ablation: remove the Morgan-fingerprint global molecular branch, "
            "local-global gated fusion, and global molecule token/context fusion."
        ),
    )
    parser.add_argument(
        "--ablate_path_attention",
        "--no_path_attention",
        dest="ablate_path_attention",
        action="store_true",
        default=False,
        help=(
            "Ablation: replace learnable ROOT-to-peak path attention with "
            "masked mean pooling over the same path tokens."
        ),
    )
    parser.add_argument(
        "--ablate_inter_peak_interaction",
        "--no_inter_peak_interaction",
        dest="ablate_inter_peak_interaction",
        action="store_true",
        default=False,
        help=(
            "Ablation: bypass the cross-peak Transformer interaction and "
            "predict each candidate intensity from its local path representation."
        ),
    )
    parser.add_argument(
        "--ablate_gold_guided_curriculum",
        "--no_gold_guided_curriculum",
        dest="ablate_gold_guided_curriculum",
        action="store_true",
        default=False,
        help=(
            "Ablation: remove Phase-B Gold replay while leaving the Stage1-tree "
            "training schedule unchanged. The Gold-tree warmup phase is not "
            "present in this trainer."
        ),
    )

    parser.add_argument("--smooth_l1_beta", type=float, default=0.04)
    parser.add_argument("--peak_weight_alpha", type=float, default=2.0)
    parser.add_argument("--base_peak_extra", type=float, default=3.0)
    parser.add_argument("--strong_peak_extra", type=float, default=1.5)
    parser.add_argument("--strong_peak_threshold", type=float, default=0.30)
    parser.add_argument("--strong_peak_gamma", type=float, default=1.4)
    parser.add_argument("--strong_peak_topk", type=int, default=5)
    parser.add_argument("--rank_min_gap", type=float, default=0.015)
    parser.add_argument("--rank_base_margin", type=float, default=0.03)
    parser.add_argument("--rank_gap_margin_scale", type=float, default=0.22)

    parser.add_argument("--base_label_smoothing", type=float, default=0.01)
    parser.add_argument("--peak_bucket_multiple", type=int, default=32)
    parser.add_argument("--path_bucket_multiple", type=int, default=8)
    parser.add_argument(
        "--auto_batch_size",
        action="store_true",
        default=True,
        help="Probe the largest safe batch on the most expensive training shape.",
    )
    parser.add_argument(
        "--no_auto_batch_size",
        action="store_false",
        dest="auto_batch_size",
    )
    parser.add_argument("--auto_batch_memory_fraction", type=float, default=0.86)
    parser.add_argument("--auto_batch_max", type=int, default=64)
    parser.add_argument("--auto_batch_step", type=int, default=4)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
    )
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument(
        "--compile_model",
        action="store_true",
        default=True,
        help="Use torch.compile for the training forward/backward path.",
    )
    parser.add_argument(
        "--no_compile_model",
        action="store_false",
        dest="compile_model",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
    )

    return parser.parse_args()


def apply_ablation_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """
    Apply optional ablation switches after CLI parsing.

    When no ablation switch is supplied, this function changes nothing.
    Therefore the original full-model hyperparameters and training schedule
    remain exactly the same.
    """
    active: List[str] = []

    # ------------------------------------------------------------
    # Five loss-level ablations.
    # Only the selected coefficient is set to zero.
    # ------------------------------------------------------------
    if args.ablate_intensity_loss:
        args.lambda_intensity = 0.0
        active.append("w/o Intensity loss")

    if args.ablate_spectral_loss:
        args.lambda_spectral = 0.0
        active.append("w/o Spectral loss")

    if args.ablate_ranking_loss:
        args.lambda_rank = 0.0
        active.append("w/o Ranking loss")

    if args.ablate_base_peak_loss:
        args.lambda_base = 0.0
        active.append("w/o Base-peak loss")

    if args.ablate_strong_peak_loss:
        args.lambda_strong = 0.0
        active.append("w/o Strong-peak loss")

    # ------------------------------------------------------------
    # Architecture ablations.
    # These do not alter dimensions, parameter initialization order,
    # optimizer defaults, loss weights, or data preprocessing.
    # ------------------------------------------------------------
    if args.ablate_global_context:
        active.append("w/o Global molecular context")

    if args.ablate_path_attention:
        active.append("w/o Path attention")

    if args.ablate_inter_peak_interaction:
        active.append("w/o Inter-peak interaction")

    # ------------------------------------------------------------
    # Gold-guided curriculum ablation.
    #
    # This trainer has no separate Gold-tree warmup. The existing ablation
    # switch is kept fully backward-compatible and now removes only the
    # Phase-B Gold replay mixture. Stage1 epochs and all other settings stay
    # unchanged.
    # ------------------------------------------------------------
    if args.ablate_gold_guided_curriculum:
        args.phase_b_gold_ratio = 0.0
        active.append("w/o Gold-guided curriculum")

    # Do not allow an accidentally empty objective.
    lambda_values = [
        float(args.lambda_intensity),
        float(args.lambda_spectral),
        float(args.lambda_rank),
        float(args.lambda_base),
        float(args.lambda_strong),
    ]
    if all(abs(value) <= 0.0 for value in lambda_values):
        raise ValueError(
            "All five Stage2 loss coefficients are zero. "
            "At least one loss must remain active."
        )

    args.active_ablations = active
    args.is_full_model = len(active) == 0

    return args


def print_ablation_configuration(args: argparse.Namespace) -> None:
    print("\n" + "=" * 80)
    print("[ABLATION CONFIGURATION]")
    print("=" * 80)

    if args.is_full_model:
        print("[ABLATION] Full model: no ablation switch is active.")
    else:
        for name in args.active_ablations:
            print(f"[ABLATION] {name}")

    print(
        "[LOSS WEIGHTS] "
        f"intensity={args.lambda_intensity}, "
        f"spectral={args.lambda_spectral}, "
        f"rank={args.lambda_rank}, "
        f"base={args.lambda_base}, "
        f"strong={args.lambda_strong}"
    )
    print(
        "[MODEL SWITCHES] "
        f"global_context={not args.ablate_global_context}, "
        f"path_attention={not args.ablate_path_attention}, "
        f"inter_peak_interaction={not args.ablate_inter_peak_interaction}"
    )
    print(
        "[CURRICULUM] "
        "gold_warmup=False, "
        f"stage1_epochs={args.stage1_epochs}, "
        f"phase_b_gold_ratio={args.phase_b_gold_ratio}"
    )
    print("=" * 80 + "\n")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(min_lr_ratio, float(step + 1) / float(max(1, warmup_steps)))

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_optimizer_and_scheduler(
    model: nn.Module,
    *,
    learning_rate: float,
    epochs: int,
    loader: DataLoader,
    args: argparse.Namespace,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    decay_parameters: List[nn.Parameter] = []
    no_decay_parameters: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if (
            parameter.ndim < 2
            or lowered.endswith(".bias")
            or "norm" in lowered
            or "embedding" in lowered
        ):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    parameter_groups = [
        {"params": decay_parameters, "weight_decay": args.weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    optimizer_kwargs: Dict[str, Any] = {
        "lr": learning_rate,
        "betas": (0.9, 0.95),
    }
    if next(model.parameters()).is_cuda:
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_kwargs)

    total_steps = max(
        1,
        math.ceil(len(loader) / max(1, args.grad_accum)) * epochs,
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=0.05,
    )
    return optimizer, scheduler


def select_probe_dataset(
    stage1_dataset: Optional[Stage2TreeDataset],
    gold_dataset: Optional[Stage2TreeDataset],
) -> Optional[Stage2TreeDataset]:
    candidates = [
        dataset for dataset in (stage1_dataset, gold_dataset)
        if dataset is not None and len(dataset) > 0
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda dataset: max(
            (
                example.bucket_shape[0]
                * example.bucket_shape[1]
                for example in dataset.examples
            ),
            default=0,
        ),
    )


def auto_tune_batch_size(
    model: nn.Module,
    dataset: Optional[Stage2TreeDataset],
    device: torch.device,
    args: argparse.Namespace,
) -> int:
    """Probe the largest safe batch on the most expensive observed shape."""
    if (
        not args.auto_batch_size
        or device.type != "cuda"
        or dataset is None
        or len(dataset) == 0
    ):
        return int(args.batch_size)

    largest = max(
        dataset.examples,
        key=lambda example: (
            example.bucket_shape[0] * example.bucket_shape[1],
            example.bucket_shape[0],
            example.bucket_shape[1],
        ),
    )
    start = max(1, int(args.batch_size))
    maximum = max(start, int(args.auto_batch_max))
    step = max(1, int(args.auto_batch_step))
    target_fraction = min(max(float(args.auto_batch_memory_fraction), 0.50), 0.95)
    candidates = list(range(start, maximum + 1, step))
    if candidates[-1] != maximum:
        candidates.append(maximum)

    total_memory = torch.cuda.get_device_properties(device).total_memory
    best = 0
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    collator = Stage2Collator(args)
    original_training = model.training
    model.train()

    print(
        f"[AutoBatch] probing shape={largest.bucket_shape}, "
        f"candidates={candidates}, target_memory={target_fraction:.0%}"
    )
    for candidate in candidates:
        batch = None
        pred_raw = None
        ranking_logits = None
        loss = None
        probe_logs = None
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            batch = collator([largest] * candidate)
            batch = move_batch(batch, device)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                pred_raw, ranking_logits = model(
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
                    return_logits=True,
                )
                loss, probe_logs = compute_loss(
                    pred_raw,
                    ranking_logits,
                    batch["target"],
                    batch["peak_mask"],
                    batch["base_index"],
                    batch["base_present"],
                    batch["full_gold_l2_norm"],
                    args,
                )
            loss.backward()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            fraction = peak / max(1, total_memory)
            print(
                f"[AutoBatch] batch={candidate}: "
                f"peak={peak/1024**3:.2f}GB ({fraction:.1%})"
            )
            if fraction <= target_fraction:
                best = candidate
            else:
                break
        except torch.cuda.OutOfMemoryError:
            print(f"[AutoBatch] batch={candidate}: OOM")
            break
        finally:
            model.zero_grad(set_to_none=True)
            del batch, pred_raw, ranking_logits, loss, probe_logs
            torch.cuda.empty_cache()

    if not original_training:
        model.eval()
    if best <= 0:
        best = max(1, min(start, int(args.batch_size)))
        print(
            f"[AutoBatch] no candidate met the memory target; "
            f"falling back to batch={best}."
        )
    else:
        print(f"[AutoBatch] selected batch_size={best}")
    return best


def main() -> None:
    args = parse_args()
    args = apply_ablation_overrides(args)
    set_seed(args.seed)
    setup_cuda_high_throughput()

    os.makedirs(args.output_dir, exist_ok=True)
    write_json(vars(args), os.path.join(args.output_dir, "train_args.json"))
    print_ablation_configuration(args)

    device = torch.device(args.device)
    print(f"[INFO] device={device}")
    print(
        f"[INFO] amp={args.amp and device.type == 'cuda'}, "
        f"amp_dtype={args.amp_dtype}"
    )
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        )
        print(f"[INFO] GPU={gpu_name}, total_memory={total_gb:.2f} GB")

    gold_records = load_json_or_jsonl(args.gold_train_file)
    stage1_records = load_json_or_jsonl(args.stage1_train_file)
    infer_records_raw = load_json_or_jsonl(args.infer_val_file)
    infer_gold_records = (
        load_json_or_jsonl(args.infer_val_gold_file)
        if args.infer_val_gold_file
        else []
    )

    if not gold_records and not stage1_records:
        raise FileNotFoundError(
            "No training records loaded. Check --gold_train_file and "
            "--stage1_train_file."
        )
    if not infer_records_raw:
        raise FileNotFoundError(
            "No independent validation records loaded from --infer_val_file. "
            "Training/gold-training data are intentionally not used as "
            "validation fallback."
        )

    infer_val_records = merge_infer_validation_records(
        infer_records_raw,
        infer_gold_records,
    )

    print(
        f"[INFO] gold_train_records={len(gold_records)} "
        f"from {args.gold_train_file}"
    )
    print(
        f"[INFO] stage1_train_records={len(stage1_records)} "
        f"from {args.stage1_train_file}"
    )
    print(
        f"[INFO] infer_val_records={len(infer_val_records)} "
        f"from {args.infer_val_file}"
    )
    if args.infer_val_gold_file:
        print(
            f"[INFO] infer_val_gold_records={len(infer_gold_records)} "
            f"from {args.infer_val_gold_file}"
        )

    gold_train_ids = {get_record_id(r) for r in gold_records if get_record_id(r)}
    stage1_train_ids = {get_record_id(r) for r in stage1_records if get_record_id(r)}
    infer_val_ids = {get_record_id(r) for r in infer_val_records if get_record_id(r)}
    train_union_ids = gold_train_ids | stage1_train_ids
    overlap_ids = sorted(train_union_ids & infer_val_ids)
    if overlap_ids:
        print(
            f"[WARN] train/infer-validation id overlap: {len(overlap_ids)}. "
            "These records will make checkpoint selection optimistic."
        )

    write_json(
        {
            "seed": args.seed,
            "validation_source": "infer_file_only",
            "gold_train_file": args.gold_train_file,
            "stage1_train_file": args.stage1_train_file,
            "infer_val_file": args.infer_val_file,
            "infer_val_gold_file": args.infer_val_gold_file,
            "gold_train_ids": sorted(gold_train_ids),
            "stage1_train_ids": sorted(stage1_train_ids),
            "infer_val_ids": sorted(infer_val_ids),
            "train_infer_overlap_ids": overlap_ids,
        },
        os.path.join(args.output_dir, "data_manifest.json"),
    )

    # Every training record is used for training. Validation is constructed
    # exclusively from --infer_val_file (plus optional held-out infer gold).
    gold_train_dataset = make_dataset(
        gold_records,
        source_type="gold_train",
        args=args,
    )
    stage1_train_dataset = make_dataset(
        stage1_records,
        source_type="stage1_train",
        args=args,
    )
    infer_val_dataset = make_dataset(
        infer_val_records,
        source_type="infer_val",
        args=args,
    )

    if infer_val_dataset is None or len(infer_val_dataset) == 0:
        raise RuntimeError(
            "All --infer_val_file records were skipped. Confirm that each "
            "validation record contains Stage1 triplets and molecule metadata."
        )

    positive_target_molecules = sum(
        1
        for example in infer_val_dataset.examples
        if any(value > 0 for value in example.full_gold_intensity.values())
    )
    if positive_target_molecules == 0:
        raise RuntimeError(
            "The infer validation set contains no positive target intensity. "
            "Put target intensities in --infer_val_file or provide the matching "
            "held-out --infer_val_gold_file. Training gold data will not be "
            "used as a fallback."
        )
    print(
        f"[INFO] infer validation molecules with positive targets: "
        f"{positive_target_molecules}/{len(infer_val_dataset)}"
    )

    truncation_report = {
        name: dataset.truncation_stats
        for name, dataset in {
            "gold_train": gold_train_dataset,
            "stage1_train": stage1_train_dataset,
            "infer_val": infer_val_dataset,
        }.items()
        if dataset is not None
    }
    write_json(
        truncation_report,
        os.path.join(args.output_dir, "candidate_truncation_report.json"),
    )

    model = StructuredTreePathAttentionModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        interaction_layers=args.interaction_layers,
        dropout=args.dropout,
        max_path_len=args.max_path_len,
        n_token_types=5,
        n_mechanisms=len(MECHANISM_LABELS) + 1,
        morgan_fp_dim=args.morgan_fp_dim,
        use_global_context=not args.ablate_global_context,
        use_path_attention=not args.ablate_path_attention,
        use_inter_peak_interaction=not args.ablate_inter_peak_interaction,
    ).to(device)

    probe_dataset = select_probe_dataset(stage1_train_dataset, gold_train_dataset)
    tuned_batch_size = auto_tune_batch_size(model, probe_dataset, device, args)
    if tuned_batch_size != args.batch_size:
        print(f"[INFO] batch_size adjusted: {args.batch_size} -> {tuned_batch_size}")
        args.batch_size = tuned_batch_size
        write_json(vars(args), os.path.join(args.output_dir, "train_args.json"))

    stage1_train_loader = make_stage1_replay_loader(
        stage1_train_dataset, gold_train_dataset, args=args
    )
    infer_val_loader = make_loader_from_dataset(
        infer_val_dataset, args=args, shuffle=False
    )
    main_val_loader = infer_val_loader
    if stage1_train_loader is None:
        raise RuntimeError(
            "All Stage1 training records were skipped during tree construction."
        )

    ema = ModelEMA(model, decay=args.ema_decay)

    forward_model: nn.Module = model
    if args.compile_model and hasattr(torch, "compile"):
        try:
            forward_model = torch.compile(
                model,
                mode=args.compile_mode,
                dynamic=False,
            )
            print(
                f"[INFO] torch.compile enabled: mode={args.compile_mode}, "
                "dynamic=False"
            )
        except Exception as exc:
            forward_model = model
            print(f"[WARN] torch.compile unavailable; using eager mode: {exc}")

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"[INFO] trainable_parameters={trainable_parameters:,}")
    print(
        "[INFO] checkpoint selection: highest infer-validation EMA "
        "mean_cosine_union_peaks"
    )
    print(
        f"[INFO] validation source is strictly --infer_val_file: "
        f"{args.infer_val_file}"
    )

    # BF16 does not require dynamic loss scaling; FP16 keeps GradScaler.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            args.amp
            and device.type == "cuda"
            and args.amp_dtype == "fp16"
        ),
    )

    model_config = {
        "model": {
            "name": "StructuredTreePathAttentionModelV7FiveLoss",
            "input_version": "path_parent_product_mz_morgan_globaltoken_softplus_fiveloss_v7",
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "interaction_layers": args.interaction_layers,
            "dropout": args.dropout,
            "max_path_len": args.max_path_len,
            "path_numeric_dim": StructuredTreePathAttentionModel.PATH_NUMERIC_DIM,
            "mechanism_labels": MECHANISM_LABELS,
            "token_types": TYPE_TO_NAME,
            "smiles_tokenization": "field-aware-hashed",
            "global_structure_branch": "morgan_fingerprint_gated_fusion_plus_global_token",
            "morgan_fp_dim": args.morgan_fp_dim,
            "morgan_radius": args.morgan_radius,
            "use_global_context": not args.ablate_global_context,
            "use_path_attention": not args.ablate_path_attention,
            "use_inter_peak_interaction": not args.ablate_inter_peak_interaction,
        },
        "ablation": {
            "is_full_model": args.is_full_model,
            "active_ablations": list(args.active_ablations),
            "ablate_intensity_loss": args.ablate_intensity_loss,
            "ablate_spectral_loss": args.ablate_spectral_loss,
            "ablate_ranking_loss": args.ablate_ranking_loss,
            "ablate_base_peak_loss": args.ablate_base_peak_loss,
            "ablate_strong_peak_loss": args.ablate_strong_peak_loss,
            "ablate_global_context": args.ablate_global_context,
            "ablate_path_attention": args.ablate_path_attention,
            "ablate_inter_peak_interaction": args.ablate_inter_peak_interaction,
            "ablate_gold_guided_curriculum": args.ablate_gold_guided_curriculum,
        },
        "loss": {
            "types": [
                "weighted_continuous_intensity_smooth_l1",
                "union_space_cosine",
                "pairwise_ranking",
                "true_base_peak_classification",
                "strong_peak_regression",
            ],
            "lambda_intensity": args.lambda_intensity,
            "lambda_spectral": args.lambda_spectral,
            "lambda_rank": args.lambda_rank,
            "lambda_base": args.lambda_base,
            "lambda_strong": args.lambda_strong,
            "smooth_l1_beta": args.smooth_l1_beta,
            "peak_weight_alpha": args.peak_weight_alpha,
            "base_peak_extra": args.base_peak_extra,
            "strong_peak_extra": args.strong_peak_extra,
            "strong_peak_threshold": args.strong_peak_threshold,
            "strong_peak_gamma": args.strong_peak_gamma,
            "strong_peak_topk": args.strong_peak_topk,
            "spectral_training_space": "stage1_candidates_union_complete_gold",
            "rank_min_gap": args.rank_min_gap,
            "rank_base_margin": args.rank_base_margin,
            "rank_gap_margin_scale": args.rank_gap_margin_scale,
            "base_label_smoothing": args.base_label_smoothing,
            "intensity_reduction": "per_molecule_then_batch_mean",
            "prediction_normalization": "per_spectrum_base_peak",
            "output_parameterization": "single_softplus_continuous_intensity",
            "base_peak_definition": "argmax_complete_gold_spectrum",
        },
        "schedule": {
            "gold_warmup": False,
            "stage1_epochs": args.stage1_epochs,
            "stage1_lr": args.stage1_lr,
            "phase_b_gold_ratio": args.phase_b_gold_ratio,
            "warmup_ratio": args.warmup_ratio,
            "ema_decay": args.ema_decay,
            "compile_model": args.compile_model,
            "compile_mode": args.compile_mode,
            "batch_size": args.batch_size,
            "auto_batch_size": args.auto_batch_size,
            "auto_batch_memory_fraction": args.auto_batch_memory_fraction,
            "selection_metric": "mean_cosine_union_peaks",
            "selection_peak_space": "stage1_predicted_mz_union_full_gold_mz",
            "selection_round_digits": EVAL_ROUND_DIGITS,
            "selection_missing_peak_value": 0.0,
            "selection_reduction": "per_molecule_macro_mean",
        },
        "data": {
            "gold_train_file": args.gold_train_file,
            "stage1_train_file": args.stage1_train_file,
            "infer_val_file": args.infer_val_file,
            "infer_val_gold_file": args.infer_val_gold_file,
            "validation_source": "infer_file_only",
            "max_peaks": args.max_peaks,
            "max_path_len": args.max_path_len,
            "max_words_per_token": args.max_words_per_token,
            "peak_bucket_multiple": args.peak_bucket_multiple,
            "path_bucket_multiple": args.path_bucket_multiple,
            "morgan_fp_dim": args.morgan_fp_dim,
            "morgan_radius": args.morgan_radius,
            "gold_train_records": len(gold_records),
            "stage1_train_records": len(stage1_records),
            "infer_val_records": len(infer_val_records),
            "train_infer_overlap_ids": len(overlap_ids),
        },
    }
    write_json(
        model_config,
        os.path.join(args.output_dir, "model_config.json"),
    )

    global_step = 0
    global_best_cosine = float("-inf")

    if stage1_train_loader is not None and args.stage1_epochs > 0:
        print("\n" + "=" * 80)
        print("[TRAIN] Stage1-generated-tree adaptation with Gold replay")
        print("=" * 80)

        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            learning_rate=args.stage1_lr,
            epochs=args.stage1_epochs,
            loader=stage1_train_loader,
            args=args,
        )
        phase_b = train_one_phase(
            model=model,
            forward_model=forward_model,
            ema=ema,
            train_loader=stage1_train_loader,
            val_loader=main_val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            args=args,
            phase_name="stage1_tree_adaptation",
            epochs=args.stage1_epochs,
            global_step=global_step,
            global_best_cosine=global_best_cosine,
            phase_best_path=os.path.join(args.output_dir, "best_stage1.pt"),
        )
        global_step = int(phase_b["global_step"])
        global_best_cosine = float(phase_b["global_best_cosine"])

    print("\n[DONE]")
    print(
        f"Best infer-validation EMA mean_cosine_union_peaks: "
        f"{global_best_cosine:.6f}"
    )
    print(f"Best checkpoint: {os.path.join(args.output_dir, 'best.pt')}")
    print(
        f"Best Stage1 checkpoint: "
        f"{os.path.join(args.output_dir, 'best_stage1.pt')}"
    )
    print(f"Last checkpoint: {os.path.join(args.output_dir, 'last.pt')}")
    print(f"Training log: {os.path.join(args.output_dir, 'train_log.jsonl')}")
    print(
        "Candidate truncation report: "
        f"{os.path.join(args.output_dir, 'candidate_truncation_report.json')}"
    )


if __name__ == "__main__":
    main()
