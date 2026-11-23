import argparse
import json
import os
from typing import Any, Dict, List

from infer_unidirectional_github import (
    enrich_outputs_with_gold_evaluation,
    load_json_or_jsonl,
    save_json_or_jsonl,
    summarize_outputs,
)

def record_key(record: Dict[str, Any], key_name: str) -> str:
    if key_name == "id":
        value = record.get("id", record.get("ID"))
    else:
        value = record.get("smiles", record.get("SMILES"))
    if value in (None, ""):
        return ""
    return str(value).strip()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", required=True)
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--stage1_file", default="")
    parser.add_argument("--pair_key", choices=("id", "smiles"), default="id")
    parser.add_argument("--round_digits", type=int, default=6)
    parser.add_argument("--keep_gold_triplets", action="store_true")
    args = parser.parse_args()

    predictions = load_json_or_jsonl(args.prediction_file)
    gold_records = load_json_or_jsonl(args.gold_file)
    if not predictions:
        raise ValueError("No prediction records were loaded.")
    if not gold_records:
        raise ValueError("No gold records were loaded.")

    gold_index = {
        record_key(record, args.pair_key): record
        for record in gold_records
        if record_key(record, args.pair_key)
    }
    stage1_index = {}
    if args.stage1_file:
        stage1_records = load_json_or_jsonl(args.stage1_file)
        stage1_index = {
            record_key(record, args.pair_key): record
            for record in stage1_records
            if record_key(record, args.pair_key)
        }

    paired_predictions: List[Dict[str, Any]] = []
    paired_stage1: List[Dict[str, Any]] = []
    paired_gold: List[Dict[str, Any]] = []
    missing_gold = []
    missing_stage1_triplets = []

    for prediction in predictions:
        key = record_key(prediction, args.pair_key)
        gold = gold_index.get(key)
        if not key or gold is None:
            missing_gold.append(key or "<missing key>")
            continue

        if stage1_index:
            stage1 = stage1_index.get(key)
            if stage1 is None:
                missing_stage1_triplets.append(key)
                continue
            stage1 = dict(stage1)
        else:
            triplets = prediction.get("stage1_triplets", prediction.get("triplets"))
            if not isinstance(triplets, list):
                missing_stage1_triplets.append(key)
                continue
            stage1 = dict(prediction)
            stage1["triplets"] = triplets

        paired_predictions.append(dict(prediction))
        paired_stage1.append(stage1)
        paired_gold.append(dict(gold))

    if missing_gold:
        raise ValueError(
            "Gold records are missing for " + str(len(missing_gold))
            + " predictions; examples: " + ", ".join(missing_gold[:10])
        )
    if missing_stage1_triplets:
        raise ValueError(
            "Stage1 records/triplets are missing for "
            + str(len(missing_stage1_triplets))
            + " predictions. Include stage1_triplets in predictions or pass --stage1_file."
        )
    if not paired_predictions:
        raise ValueError("No prediction records could be matched to gold.")

    evaluated = enrich_outputs_with_gold_evaluation(
        paired_predictions,
        paired_stage1,
        paired_gold,
        keep_gold_triplets=args.keep_gold_triplets,
        round_digits=args.round_digits,
    )
    metrics = summarize_outputs(evaluated)
    metrics["num_matched_records"] = len(evaluated)
    metrics["prediction_file"] = os.path.abspath(args.prediction_file)
    metrics["gold_file"] = os.path.abspath(args.gold_file)
    if args.stage1_file:
        metrics["stage1_file"] = os.path.abspath(args.stage1_file)

    save_json_or_jsonl(evaluated, args.output_file)
    metrics_file = args.output_file + ".metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[DONE] evaluated_records={len(evaluated)}")
    print(f"[DONE] per_sample={args.output_file}")
    print(f"[DONE] metrics={metrics_file}")

if __name__ == "__main__":
    main()
