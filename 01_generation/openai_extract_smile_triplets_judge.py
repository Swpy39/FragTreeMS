# -*- coding: utf-8 -*-
"""
python openai_extract_smile_triplets_judge.py --input_file ./outputs/openai_gpt41_test_results_judge_all.json --output_file ./outputs/openai_gpt41_test_results_judge_extract.json

"""

import argparse
import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="Full judge result JSON file, e.g., openai_gpt41_test_results_judge_all_1-74.json")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Extracted corrected triplet output JSON file.")
    parser.add_argument("--sort_by_triplet_index", action="store_true",
                        help="Sort triplets inside each molecule by triplet_index. Recommended if the input order may be shuffled.")
    parser.add_argument("--drop_invalid", action="store_true",
                        help="Drop entries whose corrected triplet cannot be found or is not a valid 3-element list.")
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_triplet(x: Any) -> Optional[List[str]]:
    if isinstance(x, list) and len(x) == 3:
        return [str(x[0]), str(x[1]), str(x[2])]
    return None


def get_corrected_triplet(item: Dict[str, Any]) -> Optional[List[str]]:
    """
    Priority:
    1. final_corrected_triplet
    2. judge_output.corrected_triplet
    3. model_judge_output_before_fallback.corrected_triplet
    """
    triplet = normalize_triplet(item.get("final_corrected_triplet"))
    if triplet is not None:
        return triplet

    judge_output = item.get("judge_output")
    if isinstance(judge_output, dict):
        triplet = normalize_triplet(judge_output.get("corrected_triplet"))
        if triplet is not None:
            return triplet

    model_output = item.get("model_judge_output_before_fallback")
    if isinstance(model_output, dict):
        triplet = normalize_triplet(model_output.get("corrected_triplet"))
        if triplet is not None:
            return triplet

    return None


def extract_corrected_outputs(
    full_results: List[Dict[str, Any]],
    sort_by_triplet_index: bool = False,
    drop_invalid: bool = False,
) -> List[Dict[str, Any]]:
    grouped: "OrderedDict[Any, Dict[str, Any]]" = OrderedDict()

    # Temporarily keep triplet_index for optional sorting.
    temp_triplets: Dict[Any, List[Tuple[int, List[str]]]] = {}

    for item in full_results:
        if not isinstance(item, dict):
            continue

        data_id = item.get("id", None)

        if data_id not in grouped:
            grouped[data_id] = {
                "id": data_id,
                "name": item.get("name", ""),
                "smiles": item.get("smiles", ""),
                "formula": item.get("formula", ""),
                "mw": item.get("mw", ""),
                "corrected_triplet": [],
            }
            temp_triplets[data_id] = []

        triplet = get_corrected_triplet(item)

        if triplet is None:
            if drop_invalid:
                continue
            # Keep a visible placeholder only when the user does not request dropping invalid entries.
            # Usually this should not happen if judge_all was produced successfully.
            continue

        triplet_index = item.get("triplet_index", len(temp_triplets[data_id]))
        try:
            triplet_index = int(triplet_index)
        except Exception:
            triplet_index = len(temp_triplets[data_id])

        temp_triplets[data_id].append((triplet_index, triplet))

    for data_id, output_item in grouped.items():
        pairs = temp_triplets.get(data_id, [])
        if sort_by_triplet_index:
            pairs = sorted(pairs, key=lambda x: x[0])
        output_item["corrected_triplet"] = [triplet for _, triplet in pairs]

    return list(grouped.values())


def main() -> None:
    args = parse_args()

    full_results = load_json(args.input_file)

    if isinstance(full_results, dict):
        full_results = [full_results]

    if not isinstance(full_results, list):
        raise ValueError("Input file must be a JSON list or a single JSON object.")

    extracted = extract_corrected_outputs(
        full_results=full_results,
        sort_by_triplet_index=args.sort_by_triplet_index,
        drop_invalid=args.drop_invalid,
    )

    save_json(extracted, args.output_file)

    total_triplets = sum(len(x.get("corrected_triplet", [])) for x in extracted)

    print("=" * 80)
    print("Extraction finished.")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Molecules: {len(extracted)}")
    print(f"Corrected triplets: {total_triplets}")
    print("=" * 80)


if __name__ == "__main__":
    main()
