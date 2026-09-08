#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


MECHANISMS: Tuple[str, ...] = (
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
)

SMILES_SOURCE_RE = re.compile(
    r"^\s*smiles_fragment\s*:\s*(.*?)\s*$",
    re.IGNORECASE,
)
PRECURSOR_SOURCE_RE = re.compile(
    r"^\s*precursor_mz\s*:\s*(-?\d+)\s*$",
    re.IGNORECASE,
)
MZ_RE = re.compile(r"m/z\s*[:=]?\s*(-?\d+)", re.IGNORECASE)
PRODUCT_FORMULA_RE = re.compile(
    r"^\s*(.*?)\s*\+\s*\(\s*m/z\s*[:=]?\s*(-?\d+)\s*\)\s*$",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are IonRecall-FITGraph, an EI mass-spectrometry mechanism-triplet graph generator.
Given only the molecular name, SMILES, formula, molecular weight, and compound class,
first recover a high-recall inventory of product ions, then explain those ions with
label fragments and precursor-ion transition triplets.
Omission of a plausible product ion is more serious than adding one extra plausible ion.
Use only the allowed mechanism names.
Do not output reasoning, spectrum intensity, markdown, or commentary.
Output only the required FITGraph text format and end with [GRAPH_END]."""

FORMAT_INSTRUCTION = """Required ion-first format:
[COUNTS]	fragments=<int>	ions=<int>	fragment_edges=<int>	ion_edges=<int>
[IONS]
I0	<mz>	<formula>	<molecular_ion|fragment_ion|referenced_precursor>
[/IONS]
[FRAGMENTS]
F0	"<raw fragment text>"
[/FRAGMENTS]
[FRAGMENT_TO_ION]
F0	I0	<mechanism>
[/FRAGMENT_TO_ION]
[ION_TO_ION]
I0	I1	<mechanism>
[/ION_TO_ION]
[GRAPH_END]

Allowed mechanisms:
- Molecular ion
- Isotopic peak
- Alpha-cleavage
- Sigma-bond cleavage
- Benzylic cleavage
- Allylic cleavage
- McLafferty rearrangement
- Neutral loss
- Retro-Diels–Alder fragmentation
- Hydrogen transfer
- Radical-ion rearrangement
- Dehydrogenation / Sequential dehydrogenation
- Ring cleavage / Ring rearrangement

Rules:
- Generate the most complete plausible product-ion inventory before fragment and edge sections.
- Fragment text is a JSON-quoted string.
- IDs must be unique and all edge references must exist.
- Preserve learned fragment labels even when they are not valid SMILES.
- Use one ion node per m/z.
- Do not invent spectrum intensity.
- Output [GRAPH_END] exactly once at the end."""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} top-level value must be an object")
            records.append(obj)
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_fragment(value: Any) -> str:
    return normalize_text(value)


def parse_product(product: Any) -> Tuple[str, Optional[int]]:
    text = str(product or "").strip()
    match = PRODUCT_FORMULA_RE.match(text)
    if match:
        return match.group(1).strip(), int(match.group(2))
    mz_match = MZ_RE.search(text)
    if not mz_match:
        return "", None
    mz = int(mz_match.group(1))
    prefix = text[: mz_match.start()].strip()
    prefix = re.sub(r"\+\s*\(?\s*$", "", prefix).strip()
    return prefix, mz


def format_product(formula: str, mz: int) -> str:
    safe_formula = normalize_text(formula) or "Unknown"
    return f"{safe_formula}+ (m/z {int(mz)})"


def build_user_prompt(record: Dict[str, Any]) -> str:
    return (
        f"Name: {record.get('name', '')}\n"
        f"SMILES: {record.get('smiles', '')}\n"
        f"Formula: {record.get('formula', '')}\n"
        f"MW: {record.get('mw', '')}\n"
        f"Compound class: {record.get('compound_class', '')}\n\n"
        f"{FORMAT_INSTRUCTION}"
    )


def build_chat_prompt(tokenizer: Any, record: Dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(record)},
    ]
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_representation_prompt(tokenizer: Any, record: Dict[str, Any]) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Encode the molecule using only the supplied name, SMILES, formula, "
                "molecular weight, and compound class."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Name: {record.get('name', '')}\n"
                f"SMILES: {record.get('smiles', '')}\n"
                f"Formula: {record.get('formula', '')}\n"
                f"MW: {record.get('mw', '')}\n"
                f"Compound class: {record.get('compound_class', '')}"
            ),
        },
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=False)
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_fitgraph(record: Dict[str, Any]) -> Dict[str, Any]:
    triplets = record.get("corrected_triplet", [])
    if not isinstance(triplets, list):
        triplets = []

    parsed: List[Dict[str, Any]] = []
    seen_triplets: Set[Tuple[str, str, str, int]] = set()

    for raw in triplets:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            continue
        source, mechanism, product = map(str, raw)
        source = source.strip()
        mechanism = mechanism.strip()
        formula, mz = parse_product(product)
        if mz is None or mz <= 0:
            continue
        if mechanism not in MECHANISMS:
            # Preserve data while keeping a fixed label space.
            mechanism = "Sigma-bond cleavage"

        fragment_match = SMILES_SOURCE_RE.match(source)
        precursor_match = PRECURSOR_SOURCE_RE.match(source)

        if fragment_match:
            source_type = "fragment"
            source_value: Any = normalize_fragment(fragment_match.group(1))
        elif precursor_match:
            source_type = "precursor"
            source_value = int(precursor_match.group(1))
        else:
            # Non-standard GPT source is still retained as a fragment label.
            source_type = "fragment"
            source_value = normalize_fragment(source)

        key = (source_type, str(source_value), mechanism, int(mz))
        if key in seen_triplets:
            continue
        seen_triplets.add(key)
        parsed.append(
            {
                "source_type": source_type,
                "source_value": source_value,
                "mechanism": mechanism,
                "formula": normalize_text(formula),
                "mz": int(mz),
            }
        )

    fragments_in_order: List[str] = []
    fragment_seen: Set[str] = set()
    for item in parsed:
        if item["source_type"] != "fragment":
            continue
        fragment = str(item["source_value"])
        if fragment not in fragment_seen:
            fragment_seen.add(fragment)
            fragments_in_order.append(fragment)

    fragment_to_id = {
        fragment: f"F{index}"
        for index, fragment in enumerate(fragments_in_order)
    }

    ion_formula_by_mz: Dict[int, str] = {}
    ion_types_by_mz: Dict[int, str] = {}

    # Product ions.
    for item in parsed:
        mz = int(item["mz"])
        if mz not in ion_formula_by_mz or (
            not ion_formula_by_mz[mz] and item["formula"]
        ):
            ion_formula_by_mz[mz] = str(item["formula"])
        if item["mechanism"] == "Molecular ion":
            ion_types_by_mz[mz] = "molecular_ion"
        else:
            ion_types_by_mz.setdefault(mz, "fragment_ion")

    # Precursor references can point to a node not otherwise present.
    for item in parsed:
        if item["source_type"] == "precursor":
            parent_mz = int(item["source_value"])
            ion_formula_by_mz.setdefault(parent_mz, "")
            ion_types_by_mz.setdefault(parent_mz, "referenced_precursor")

    # Prefer declared molecular weight as molecular ion only if it already exists.
    try:
        declared_mw = int(round(float(record.get("mw", 0))))
    except Exception:
        declared_mw = 0
    if declared_mw in ion_formula_by_mz:
        ion_types_by_mz[declared_mw] = "molecular_ion"

    # Molecular ion first, then descending m/z.
    sorted_mz = sorted(
        ion_formula_by_mz,
        key=lambda mz: (
            0 if ion_types_by_mz.get(mz) == "molecular_ion" else 1,
            -int(mz),
        ),
    )
    mz_to_ion_id = {
        int(mz): f"I{index}"
        for index, mz in enumerate(sorted_mz)
    }

    ions = [
        {
            "ion_id": mz_to_ion_id[int(mz)],
            "mz": int(mz),
            "formula": ion_formula_by_mz[int(mz)],
            "ion_type": ion_types_by_mz.get(int(mz), "fragment_ion"),
        }
        for mz in sorted_mz
    ]

    fragment_to_ion: List[Dict[str, Any]] = []
    ion_to_ion: List[Dict[str, Any]] = []
    seen_fi: Set[Tuple[str, str, str]] = set()
    seen_ii: Set[Tuple[str, str, str]] = set()

    for item in parsed:
        child = mz_to_ion_id[int(item["mz"])]
        if item["source_type"] == "fragment":
            fragment_id = fragment_to_id[str(item["source_value"])]
            key = (fragment_id, child, str(item["mechanism"]))
            if key not in seen_fi:
                seen_fi.add(key)
                fragment_to_ion.append(
                    {
                        "fragment": fragment_id,
                        "ion": child,
                        "mechanism": str(item["mechanism"]),
                    }
                )
        else:
            parent_mz = int(item["source_value"])
            parent = mz_to_ion_id[parent_mz]
            key = (parent, child, str(item["mechanism"]))
            if parent != child and key not in seen_ii:
                seen_ii.add(key)
                ion_to_ion.append(
                    {
                        "parent": parent,
                        "child": child,
                        "mechanism": str(item["mechanism"]),
                    }
                )

    graph = {
        "fragments": [
            {
                "fragment_id": fragment_to_id[fragment],
                "raw_fragment": fragment,
            }
            for fragment in fragments_in_order
        ],
        "ions": ions,
        "fragment_to_ion": fragment_to_ion,
        "ion_to_ion": ion_to_ion,
    }
    graph["counts"] = {
        "fragments": len(graph["fragments"]),
        "ions": len(graph["ions"]),
        "fragment_edges": len(fragment_to_ion),
        "ion_edges": len(ion_to_ion),
    }
    return graph


def graph_to_dsl(graph: Dict[str, Any]) -> str:
    """Serialize FITGraph with the ion inventory before fragments and edges."""
    counts = graph.get("counts", {})
    lines: List[str] = [
        (
            "[COUNTS]\t"
            f"fragments={int(counts.get('fragments', 0))}\t"
            f"ions={int(counts.get('ions', 0))}\t"
            f"fragment_edges={int(counts.get('fragment_edges', 0))}\t"
            f"ion_edges={int(counts.get('ion_edges', 0))}"
        ),
        "[IONS]",
    ]

    for ion in graph.get("ions", []):
        lines.append(
            f"{ion.get('ion_id')}\t{int(ion.get('mz', 0))}\t"
            f"{str(ion.get('formula', ''))}\t"
            f"{str(ion.get('ion_type', 'fragment_ion'))}"
        )
    lines.append("[/IONS]")
    lines.append("[FRAGMENTS]")

    for fragment in graph.get("fragments", []):
        quoted = json.dumps(
            str(fragment.get("raw_fragment", "")),
            ensure_ascii=False,
        )
        lines.append(f"{fragment.get('fragment_id')}\t{quoted}")
    lines.append("[/FRAGMENTS]")
    lines.append("[FRAGMENT_TO_ION]")

    for edge in graph.get("fragment_to_ion", []):
        lines.append(
            f"{edge.get('fragment')}\t{edge.get('ion')}\t"
            f"{edge.get('mechanism')}"
        )
    lines.append("[/FRAGMENT_TO_ION]")
    lines.append("[ION_TO_ION]")

    for edge in graph.get("ion_to_ion", []):
        lines.append(
            f"{edge.get('parent')}\t{edge.get('child')}\t"
            f"{edge.get('mechanism')}"
        )
    lines.append("[/ION_TO_ION]")
    lines.append("[GRAPH_END]")
    return "\n".join(lines)


def _extract_graph_text(text: str) -> str:
    raw = str(text or "")
    start = raw.find("[COUNTS]")
    if start < 0:
        return raw.strip()
    end = raw.find("[GRAPH_END]", start)
    if end < 0:
        return raw[start:].strip()
    return raw[start : end + len("[GRAPH_END]")].strip()


def parse_fitgraph_dsl(text: str) -> Tuple[Dict[str, Any], bool, List[str]]:
    graph_text = _extract_graph_text(text)
    errors: List[str] = []
    if not graph_text:
        return {
            "counts": {},
            "fragments": [],
            "ions": [],
            "fragment_to_ion": [],
            "ion_to_ion": [],
        }, False, ["empty_output"]

    lines = [line.rstrip() for line in graph_text.splitlines() if line.strip()]
    section: Optional[str] = None
    counts: Dict[str, int] = {}
    fragments: List[Dict[str, Any]] = []
    ions: List[Dict[str, Any]] = []
    fragment_edges: List[Dict[str, Any]] = []
    ion_edges: List[Dict[str, Any]] = []

    fragment_ids: Set[str] = set()
    ion_ids: Set[str] = set()

    for line in lines:
        if line.startswith("[COUNTS]"):
            for key, value in re.findall(
                r"(fragments|ions|fragment_edges|ion_edges)=(-?\d+)",
                line,
            ):
                counts[key] = max(0, int(value))
            continue
        if line == "[FRAGMENTS]":
            section = "fragments"
            continue
        if line == "[/FRAGMENTS]":
            section = None
            continue
        if line == "[IONS]":
            section = "ions"
            continue
        if line == "[/IONS]":
            section = None
            continue
        if line == "[FRAGMENT_TO_ION]":
            section = "fragment_edges"
            continue
        if line == "[/FRAGMENT_TO_ION]":
            section = None
            continue
        if line == "[ION_TO_ION]":
            section = "ion_edges"
            continue
        if line == "[/ION_TO_ION]":
            section = None
            continue
        if line == "[GRAPH_END]":
            section = None
            continue

        parts = line.split("\t")
        if section == "fragments":
            if len(parts) < 2:
                errors.append("fragment_line_shape")
                continue
            fragment_id = parts[0].strip()
            raw_text = "\t".join(parts[1:]).strip()
            try:
                raw_fragment = json.loads(raw_text)
            except Exception:
                raw_fragment = raw_text.strip('"')
                errors.append(f"fragment_not_json_quoted:{fragment_id}")
            if fragment_id in fragment_ids:
                errors.append(f"duplicate_fragment_id:{fragment_id}")
                continue
            fragment_ids.add(fragment_id)
            fragments.append(
                {
                    "fragment_id": fragment_id,
                    "raw_fragment": str(raw_fragment),
                }
            )
        elif section == "ions":
            if len(parts) < 4:
                errors.append("ion_line_shape")
                continue
            ion_id = parts[0].strip()
            try:
                mz = int(parts[1].strip())
            except Exception:
                errors.append(f"invalid_mz:{ion_id}")
                continue
            formula = parts[2].strip()
            ion_type = parts[3].strip()
            if ion_id in ion_ids:
                errors.append(f"duplicate_ion_id:{ion_id}")
                continue
            if mz <= 0:
                errors.append(f"nonpositive_mz:{ion_id}")
                continue
            ion_ids.add(ion_id)
            ions.append(
                {
                    "ion_id": ion_id,
                    "mz": mz,
                    "formula": formula,
                    "ion_type": ion_type,
                }
            )
        elif section == "fragment_edges":
            if len(parts) < 3:
                errors.append("fragment_edge_line_shape")
                continue
            fragment_edges.append(
                {
                    "fragment": parts[0].strip(),
                    "ion": parts[1].strip(),
                    "mechanism": "\t".join(parts[2:]).strip(),
                }
            )
        elif section == "ion_edges":
            if len(parts) < 3:
                errors.append("ion_edge_line_shape")
                continue
            ion_edges.append(
                {
                    "parent": parts[0].strip(),
                    "child": parts[1].strip(),
                    "mechanism": "\t".join(parts[2:]).strip(),
                }
            )

    valid_fragment_edges: List[Dict[str, Any]] = []
    seen_fi: Set[Tuple[str, str, str]] = set()
    for edge in fragment_edges:
        key = (
            edge["fragment"],
            edge["ion"],
            edge["mechanism"],
        )
        if key in seen_fi:
            continue
        seen_fi.add(key)
        if edge["fragment"] not in fragment_ids:
            errors.append(f"unknown_fragment_ref:{edge['fragment']}")
            continue
        if edge["ion"] not in ion_ids:
            errors.append(f"unknown_ion_ref:{edge['ion']}")
            continue
        if edge["mechanism"] not in MECHANISMS:
            errors.append(f"invalid_mechanism:{edge['mechanism']}")
            continue
        valid_fragment_edges.append(edge)

    ion_by_id = {str(item["ion_id"]): item for item in ions}
    valid_ion_edges: List[Dict[str, Any]] = []
    seen_ii: Set[Tuple[str, str, str]] = set()
    for edge in ion_edges:
        key = (edge["parent"], edge["child"], edge["mechanism"])
        if key in seen_ii:
            continue
        seen_ii.add(key)
        if edge["parent"] not in ion_ids or edge["child"] not in ion_ids:
            errors.append("unknown_ion_transition_ref")
            continue
        if edge["parent"] == edge["child"]:
            errors.append("ion_self_loop")
            continue
        if edge["mechanism"] not in MECHANISMS:
            errors.append(f"invalid_mechanism:{edge['mechanism']}")
            continue
        parent_mz = int(ion_by_id[edge["parent"]]["mz"])
        child_mz = int(ion_by_id[edge["child"]]["mz"])
        if edge["mechanism"] != "Isotopic peak" and parent_mz <= child_mz:
            errors.append(
                f"nondecreasing_transition:{parent_mz}->{child_mz}"
            )
            continue
        valid_ion_edges.append(edge)

    actual_counts = {
        "fragments": len(fragments),
        "ions": len(ions),
        "fragment_edges": len(valid_fragment_edges),
        "ion_edges": len(valid_ion_edges),
    }
    for key, actual in actual_counts.items():
        declared = counts.get(key)
        if declared is not None and declared != actual:
            errors.append(f"count_mismatch:{key}:{declared}:{actual}")

    graph = {
        "counts": actual_counts,
        "fragments": fragments,
        "ions": ions,
        "fragment_to_ion": valid_fragment_edges,
        "ion_to_ion": valid_ion_edges,
    }
    parse_ok = "[GRAPH_END]" in graph_text and len(ions) > 0
    return graph, parse_ok, errors


def graph_to_triplets(graph: Dict[str, Any]) -> List[List[str]]:
    fragment_map = {
        str(item.get("fragment_id")): str(item.get("raw_fragment", ""))
        for item in graph.get("fragments", [])
    }
    ion_map = {
        str(item.get("ion_id")): item
        for item in graph.get("ions", [])
    }
    triplets: List[List[str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for edge in graph.get("fragment_to_ion", []):
        fragment = fragment_map.get(str(edge.get("fragment")), "")
        ion = ion_map.get(str(edge.get("ion")))
        if not ion:
            continue
        source = f"smiles_fragment: {fragment}"
        mechanism = str(edge.get("mechanism", ""))
        product = format_product(
            str(ion.get("formula", "")),
            int(ion.get("mz", 0)),
        )
        key = (source, mechanism, product)
        if key not in seen:
            seen.add(key)
            triplets.append(list(key))

    for edge in graph.get("ion_to_ion", []):
        parent = ion_map.get(str(edge.get("parent")))
        child = ion_map.get(str(edge.get("child")))
        if not parent or not child:
            continue
        source = f"precursor_mz: {int(parent.get('mz', 0))}"
        mechanism = str(edge.get("mechanism", ""))
        product = format_product(
            str(child.get("formula", "")),
            int(child.get("mz", 0)),
        )
        key = (source, mechanism, product)
        if key not in seen:
            seen.add(key)
            triplets.append(list(key))

    triplets.sort(key=lambda item: -(parse_product(item[2])[1] or -1))
    return triplets


def _jaccard(first: Set[Any], second: Set[Any]) -> float:
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def graph_signatures(graph: Dict[str, Any]) -> Dict[str, Set[Any]]:
    fragment_map = {
        str(item.get("fragment_id")): normalize_fragment(
            item.get("raw_fragment", "")
        ).lower()
        for item in graph.get("fragments", [])
    }
    ion_map = {
        str(item.get("ion_id")): item
        for item in graph.get("ions", [])
    }

    fragments = set(fragment_map.values())
    ions = {
        (int(item.get("mz", 0)), normalize_text(item.get("formula", "")))
        for item in graph.get("ions", [])
    }
    fragment_edges: Set[Any] = set()
    for edge in graph.get("fragment_to_ion", []):
        ion = ion_map.get(str(edge.get("ion")))
        if not ion:
            continue
        fragment_edges.add(
            (
                fragment_map.get(str(edge.get("fragment")), ""),
                int(ion.get("mz", 0)),
                str(edge.get("mechanism", "")),
            )
        )

    transitions: Set[Any] = set()
    for edge in graph.get("ion_to_ion", []):
        parent = ion_map.get(str(edge.get("parent")))
        child = ion_map.get(str(edge.get("child")))
        if not parent or not child:
            continue
        transitions.add(
            (
                str(edge.get("mechanism", "")),
                int(parent.get("mz", 0)) - int(child.get("mz", 0)),
                normalize_text(parent.get("formula", "")),
                normalize_text(child.get("formula", "")),
            )
        )
    return {
        "fragments": fragments,
        "ions": ions,
        "fragment_edges": fragment_edges,
        "transitions": transitions,
    }


def output_similarity(
    first_graph: Dict[str, Any],
    second_graph: Dict[str, Any],
) -> float:
    first = graph_signatures(first_graph)
    second = graph_signatures(second_graph)
    return (
        0.25 * _jaccard(first["fragments"], second["fragments"])
        + 0.25 * _jaccard(first["ions"], second["ions"])
        + 0.25 * _jaccard(
            first["fragment_edges"],
            second["fragment_edges"],
        )
        + 0.25 * _jaccard(
            first["transitions"],
            second["transitions"],
        )
    )


def build_similarity_neighbors(
    graphs: Sequence[Dict[str, Any]],
    top_k: int,
    min_similarity: float,
) -> List[List[Tuple[int, float]]]:
    signatures = [graph_signatures(graph) for graph in graphs]
    neighbors: List[List[Tuple[int, float]]] = [[] for _ in graphs]

    for i in range(len(graphs)):
        scored: List[Tuple[float, int]] = []
        first = signatures[i]
        for j in range(len(graphs)):
            if i == j:
                continue
            second = signatures[j]
            score = (
                0.25 * _jaccard(first["fragments"], second["fragments"])
                + 0.25 * _jaccard(first["ions"], second["ions"])
                + 0.25 * _jaccard(
                    first["fragment_edges"],
                    second["fragment_edges"],
                )
                + 0.25 * _jaccard(
                    first["transitions"],
                    second["transitions"],
                )
            )
            if score >= min_similarity:
                scored.append((score, j))
        scored.sort(key=lambda item: (-item[0], item[1]))
        neighbors[i] = [
            (index, float(score))
            for score, index in scored[:top_k]
        ]
    return neighbors


def _safe_prf(matched: int, predicted: int, gold: int) -> Dict[str, float]:
    precision = matched / predicted if predicted else 0.0
    recall = matched / gold if gold else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def graph_metric_sets(graph: Dict[str, Any]) -> Dict[str, Set[Any]]:
    fragments = {
        normalize_fragment(item.get("raw_fragment", "")).lower()
        for item in graph.get("fragments", [])
    }
    ion_map = {
        str(item.get("ion_id")): item
        for item in graph.get("ions", [])
    }
    ions = {
        int(item.get("mz", 0))
        for item in graph.get("ions", [])
        if int(item.get("mz", 0)) > 0
    }
    ion_formula = {
        (
            int(item.get("mz", 0)),
            normalize_text(item.get("formula", "")),
        )
        for item in graph.get("ions", [])
        if int(item.get("mz", 0)) > 0
    }

    fragment_edges: Set[Any] = set()
    for edge in graph.get("fragment_to_ion", []):
        ion = ion_map.get(str(edge.get("ion")))
        fragment_item = next(
            (
                item
                for item in graph.get("fragments", [])
                if str(item.get("fragment_id")) == str(edge.get("fragment"))
            ),
            None,
        )
        if not ion or not fragment_item:
            continue
        fragment_edges.add(
            (
                normalize_fragment(
                    fragment_item.get("raw_fragment", "")
                ).lower(),
                str(edge.get("mechanism", "")),
                int(ion.get("mz", 0)),
                normalize_text(ion.get("formula", "")),
            )
        )

    ion_edges: Set[Any] = set()
    for edge in graph.get("ion_to_ion", []):
        parent = ion_map.get(str(edge.get("parent")))
        child = ion_map.get(str(edge.get("child")))
        if not parent or not child:
            continue
        ion_edges.add(
            (
                int(parent.get("mz", 0)),
                str(edge.get("mechanism", "")),
                int(child.get("mz", 0)),
                normalize_text(child.get("formula", "")),
            )
        )

    triplets = {
        tuple(item)
        for item in graph_to_triplets(graph)
    }
    return {
        "fragments": fragments,
        "ions": ions,
        "ion_formula": ion_formula,
        "fragment_edges": fragment_edges,
        "ion_edges": ion_edges,
        "triplets": triplets,
    }


def strong_ion_set(
    record: Dict[str, Any],
    threshold: float,
) -> Set[int]:
    spectrum = record.get("input_spectrum", {})
    if not isinstance(spectrum, dict):
        return set()
    result: Set[int] = set()
    for mz, intensity in spectrum.items():
        try:
            mz_value = int(round(float(mz)))
            intensity_value = float(intensity)
        except Exception:
            continue
        if intensity_value >= threshold:
            result.add(mz_value)
    return result


def evaluate_predictions(
    rows: Sequence[Dict[str, Any]],
    strong_threshold: float,
) -> Dict[str, Any]:
    categories = (
        "fragments",
        "ions",
        "ion_formula",
        "fragment_edges",
        "ion_edges",
        "triplets",
    )
    micro = {
        category: {"matched": 0, "predicted": 0, "gold": 0}
        for category in categories
    }
    macro_values: Dict[str, List[Dict[str, float]]] = {
        category: [] for category in categories
    }

    parse_ok_count = 0
    valid_reference_count = 0
    predicted_ion_total = 0
    gold_ion_total = 0
    strong_matched = 0
    strong_gold = 0

    for row in rows:
        gold_graph = row["gold_graph"]
        pred_graph = row["predicted_graph"]
        gold_sets = graph_metric_sets(gold_graph)
        pred_sets = graph_metric_sets(pred_graph)

        if row.get("parse_ok"):
            parse_ok_count += 1
        if not row.get("validation_errors"):
            valid_reference_count += 1

        for category in categories:
            gold_set = gold_sets[category]
            pred_set = pred_sets[category]
            matched = len(gold_set & pred_set)
            micro[category]["matched"] += matched
            micro[category]["predicted"] += len(pred_set)
            micro[category]["gold"] += len(gold_set)
            macro_values[category].append(
                _safe_prf(matched, len(pred_set), len(gold_set))
            )

        predicted_ion_total += len(pred_sets["ions"])
        gold_ion_total += len(gold_sets["ions"])
        strong = strong_ion_set(row["record"], strong_threshold)
        strong_gold += len(strong)
        strong_matched += len(strong & pred_sets["ions"])

    summary: Dict[str, Any] = {
        "evaluated_molecules": len(rows),
        "parse_rate": parse_ok_count / max(1, len(rows)),
        "valid_reference_rate": valid_reference_count / max(1, len(rows)),
        "predicted_ion_total": predicted_ion_total,
        "gold_ion_total": gold_ion_total,
        "predicted_gold_ion_count_ratio": (
            predicted_ion_total / max(1, gold_ion_total)
        ),
        "strong_ion_recall": strong_matched / max(1, strong_gold),
        "strong_ion_matched": strong_matched,
        "strong_ion_gold": strong_gold,
    }

    for category in categories:
        values = micro[category]
        prf = _safe_prf(
            values["matched"],
            values["predicted"],
            values["gold"],
        )
        summary[f"micro_{category}_precision"] = prf["precision"]
        summary[f"micro_{category}_recall"] = prf["recall"]
        summary[f"micro_{category}_f1"] = prf["f1"]
        summary[f"micro_{category}_matched"] = values["matched"]
        summary[f"micro_{category}_predicted"] = values["predicted"]
        summary[f"micro_{category}_gold"] = values["gold"]

        for metric in ("precision", "recall", "f1"):
            summary[f"macro_{category}_{metric}"] = (
                sum(item[metric] for item in macro_values[category])
                / max(1, len(macro_values[category]))
            )
    return summary


def serializable_record_key(record: Dict[str, Any], index: int) -> str:
    value = record.get("id")
    return str(value) if value is not None else f"__index_{index}"


def product_ion_mzs_from_record(record: Dict[str, Any]) -> Set[int]:
    result: Set[int] = set()
    triplets = record.get("corrected_triplet", [])
    if not isinstance(triplets, list):
        return result
    for triplet in triplets:
        if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
            continue
        _, mz = parse_product(triplet[2])
        if mz is not None and int(mz) > 0:
            result.add(int(mz))
    return result


def product_ion_mzs_from_graph(graph: Dict[str, Any]) -> Set[int]:
    ion_map = {
        str(ion.get("ion_id")): int(ion.get("mz", 0))
        for ion in graph.get("ions", [])
        if int(ion.get("mz", 0)) > 0
    }
    result: Set[int] = set()
    for edge in graph.get("fragment_to_ion", []):
        mz = ion_map.get(str(edge.get("ion")))
        if mz is not None:
            result.add(mz)
    for edge in graph.get("ion_to_ion", []):
        mz = ion_map.get(str(edge.get("child")))
        if mz is not None:
            result.add(mz)
    return result


def triplet_similarity_signatures_from_graph(
    graph: Dict[str, Any],
) -> Set[Tuple[str, str, int]]:
    signatures: Set[Tuple[str, str, int]] = set()
    for triplet in graph_to_triplets(graph):
        if len(triplet) != 3:
            continue
        source, mechanism, product = map(str, triplet)
        _, mz = parse_product(product)
        if mz is None:
            continue
        source_type = (
            "precursor"
            if PRECURSOR_SOURCE_RE.match(source)
            else "fragment"
        )
        signatures.add((source_type, mechanism.strip(), int(mz)))
    return signatures


def jaccard_nonempty(
    first: Set[Any],
    second: Set[Any],
) -> float:
    if not first and not second:
        return 0.0
    union = first | second
    if not union:
        return 0.0
    return len(first & second) / len(union)


def output_graph_similarity(
    first_graph: Dict[str, Any],
    second_graph: Dict[str, Any],
) -> float:
    """
    Recall-oriented graph similarity:
      70% product-ion Jaccard + 30% mechanism-triplet Jaccard.
    """
    ion_score = jaccard_nonempty(
        product_ion_mzs_from_graph(first_graph),
        product_ion_mzs_from_graph(second_graph),
    )
    triplet_score = jaccard_nonempty(
        triplet_similarity_signatures_from_graph(first_graph),
        triplet_similarity_signatures_from_graph(second_graph),
    )
    return 0.70 * ion_score + 0.30 * triplet_score


def ion_prf_beta(
    predicted: Set[int],
    gold: Set[int],
    beta: float = 2.0,
) -> Dict[str, float]:
    matched = len(predicted & gold)
    precision = matched / max(1, len(predicted))
    recall = matched / max(1, len(gold))
    beta2 = float(beta) ** 2
    denominator = beta2 * precision + recall
    fbeta = (
        (1.0 + beta2) * precision * recall / denominator
        if denominator > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "fbeta": fbeta,
        "matched": matched,
        "predicted": len(predicted),
        "gold": len(gold),
    }


def macro_ion_fbeta_similarity(
    rows: Sequence[Dict[str, Any]],
    beta: float = 2.0,
) -> Dict[str, float]:
    f_values: List[float] = []
    graph_values: List[float] = []
    recalls: List[float] = []
    precisions: List[float] = []

    for row in rows:
        predicted = product_ion_mzs_from_graph(row["predicted_graph"])
        gold = product_ion_mzs_from_graph(row["gold_graph"])
        score = ion_prf_beta(predicted, gold, beta=beta)
        f_values.append(score["fbeta"])
        recalls.append(score["recall"])
        precisions.append(score["precision"])
        graph_values.append(
            output_graph_similarity(
                row["predicted_graph"],
                row["gold_graph"],
            )
        )

    denominator = max(1, len(rows))
    return {
        "macro_product_ion_precision": sum(precisions) / denominator,
        "macro_product_ion_recall": sum(recalls) / denominator,
        f"macro_product_ion_f{beta:g}": sum(f_values) / denominator,
        "macro_output_graph_similarity": (
            sum(graph_values) / denominator
        ),
    }


def build_repair_chat_prompt(
    tokenizer: Any,
    record: Dict[str, Any],
    draft_graph: Dict[str, Any],
    candidate_mzs: Sequence[int],
) -> str:
    candidate_text = ", ".join(
        str(int(value))
        for value in sorted(
            {int(value) for value in candidate_mzs if int(value) > 0},
            reverse=True,
        )
    )
    user_content = (
        f"Name: {record.get('name', '')}\n"
        f"SMILES: {record.get('smiles', '')}\n"
        f"Formula: {record.get('formula', '')}\n"
        f"MW: {record.get('mw', '')}\n"
        f"Compound class: {record.get('compound_class', '')}\n\n"
        "The following first-pass FITGraph may be incomplete.\n"
        f"High-recall candidate product m/z values: {candidate_text or 'none'}\n"
        "Keep all chemically supported existing content. Recover omitted "
        "product ions and add their most plausible mechanism triplets. "
        "Return one complete corrected ion-first FITGraph.\n\n"
        "Draft FITGraph:\n"
        f"{graph_to_dsl(draft_graph)}\n\n"
        f"{FORMAT_INSTRUCTION}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                SYSTEM_PROMPT
                + "\nThis is a no-delete recall-repair pass. Do not remove "
                "an existing ion or triplet unless it is structurally invalid."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def corrupt_graph_for_repair(
    graph: Dict[str, Any],
    rng: Any,
    drop_ratio_min: float,
    drop_ratio_max: float,
) -> Tuple[Dict[str, Any], List[int]]:
    product_mzs = sorted(product_ion_mzs_from_graph(graph))
    if len(product_mzs) < 2:
        return graph, []

    ratio = rng.uniform(float(drop_ratio_min), float(drop_ratio_max))
    drop_count = max(1, int(round(len(product_mzs) * ratio)))
    drop_count = min(drop_count, len(product_mzs) - 1)
    dropped_mzs = set(rng.sample(product_mzs, drop_count))

    ion_by_id = {
        str(ion.get("ion_id")): ion
        for ion in graph.get("ions", [])
    }
    dropped_ids = {
        ion_id
        for ion_id, ion in ion_by_id.items()
        if int(ion.get("mz", 0)) in dropped_mzs
    }

    draft = {
        "fragments": [
            dict(item) for item in graph.get("fragments", [])
        ],
        "ions": [
            dict(item)
            for item in graph.get("ions", [])
            if str(item.get("ion_id")) not in dropped_ids
        ],
        "fragment_to_ion": [
            dict(edge)
            for edge in graph.get("fragment_to_ion", [])
            if str(edge.get("ion")) not in dropped_ids
        ],
        "ion_to_ion": [
            dict(edge)
            for edge in graph.get("ion_to_ion", [])
            if (
                str(edge.get("parent")) not in dropped_ids
                and str(edge.get("child")) not in dropped_ids
            )
        ],
    }

    referenced_fragments = {
        str(edge.get("fragment"))
        for edge in draft["fragment_to_ion"]
    }
    draft["fragments"] = [
        item
        for item in draft["fragments"]
        if str(item.get("fragment_id")) in referenced_fragments
    ]
    draft["counts"] = {
        "fragments": len(draft["fragments"]),
        "ions": len(draft["ions"]),
        "fragment_edges": len(draft["fragment_to_ion"]),
        "ion_edges": len(draft["ion_to_ion"]),
    }
    return draft, sorted(dropped_mzs, reverse=True)


def merge_graphs_no_delete(
    record: Dict[str, Any],
    first_graph: Dict[str, Any],
    repaired_graph: Dict[str, Any],
) -> Dict[str, Any]:
    combined: List[List[str]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for graph in (first_graph, repaired_graph):
        for triplet in graph_to_triplets(graph):
            key = tuple(map(str, triplet))
            if key in seen:
                continue
            seen.add(key)
            combined.append(list(key))

    rebuilt_record = dict(record)
    rebuilt_record["corrected_triplet"] = combined
    return build_fitgraph(rebuilt_record)


def strict_inference_row(
    record: Dict[str, Any],
    graph: Dict[str, Any],
    parse_ok: bool,
) -> Dict[str, Any]:
    try:
        mw_value = float(record.get("mw"))
        nested_mw: Any = (
            int(mw_value)
            if mw_value.is_integer()
            else mw_value
        )
    except Exception:
        nested_mw = record.get("mw")

    corrected = record.get("corrected_triplet", [])
    if not isinstance(corrected, list):
        corrected = []

    return {
        "id": record.get("id"),
        "name": record.get("name"),
        "smiles": record.get("smiles"),
        "formula": record.get("formula"),
        "mw": record.get("mw"),
        "input": {
            "SMILES": record.get("smiles"),
            "formula": record.get("formula"),
            "mw": nested_mw,
        },
        "infer_result": graph_to_triplets(graph),
        "parse_ok": bool(parse_ok),
        "corrected_triplet": corrected,
    }
