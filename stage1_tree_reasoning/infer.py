from __future__ import annotations

CHECKPOINT_DIR = "./outputs/aromatic_hydrocarbon/Qwen25-14B_44_fitgraph/final_checkpoint"
TEST_DATA_FILE = "./data/aromatic_hydrocarbon/infer.jsonl"
OUTPUT_DIR = "./outputs/aromatic_hydrocarbon/Qwen25-14B_44_fitgraph/inference"
OUTPUT_FILENAME = "infer_result.jsonl"

MODEL_NAME_OR_PATH = "/hpc2hdd/home/fye374/models/Qwen2.5-14B-Instruct"
RANDOM_SEED = 44

BATCH_SIZE_PER_GPU = 4
MAX_INPUT_LENGTH = 4096
MAX_NEW_TOKENS = 8192

ATTN_IMPLEMENTATION = "sdpa"
USE_BF16 = True
MERGE_ADAPTER_FOR_INFERENCE = False
RESUME = False
REQUIRE_GOLD_LABELS = True

ION_PROBABILITY_THRESHOLD = 0.20
ION_COUNT_EXPANSION = 1.15
ION_COUNT_MARGIN = 1
MAX_AUXILIARY_ION_CANDIDATES = 64
MAX_REPAIR_ROUNDS = 1

STRONG_PEAK_THRESHOLD = 100.0
EVALUATION_F_BETA = 2.0

import json
import math
import os
import random
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from fittree import (
    build_chat_prompt,
    build_fitgraph,
    build_repair_chat_prompt,
    evaluate_predictions,
    macro_ion_fbeta_similarity,
    merge_graphs_no_delete,
    output_graph_similarity,
    parse_fitgraph_dsl,
    product_ion_mzs_from_graph,
    read_jsonl,
    strict_inference_row,
    write_jsonl,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (SCRIPT_DIR / path).resolve()


class IonRecallGraphModel(nn.Module):
    def __init__(
        self,
        language_model: nn.Module,
        hidden_size: int,
        max_mz: int,
        similarity_dim: int,
    ) -> None:
        super().__init__()
        self.lm = language_model
        self.hidden_size = int(hidden_size)
        self.max_mz = int(max_mz)
        self.ion_head = nn.Linear(self.hidden_size, self.max_mz + 1)
        self.count_head = nn.Linear(self.hidden_size, 1)
        self.similarity_projection = nn.Sequential(
            nn.Linear(self.hidden_size, int(similarity_dim)),
            nn.GELU(),
            nn.Linear(int(similarity_dim), int(similarity_dim)),
        )

    def encode_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).float()
        return self.ion_head(pooled), F.softplus(self.count_head(pooled).squeeze(-1))


def distributed_setup() -> Tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=12),
        )
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    return rank, world_size, local_rank, device


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def safe_destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as exc:
            print(
                f"[warning] destroy_process_group failed: {exc}",
                flush=True,
            )


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def strict_key(record: Dict[str, Any]) -> str:
    return json.dumps(
        [
            record.get("id"),
            record.get("name"),
            record.get("smiles"),
            record.get("formula"),
            record.get("mw"),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_input(
    records: Sequence[Dict[str, Any]],
    path: Path,
) -> None:
    required = (
        "id",
        "name",
        "smiles",
        "formula",
        "mw",
        "compound_class",
    )
    errors: List[str] = []
    for index, record in enumerate(records):
        missing = [key for key in required if key not in record]
        if missing:
            errors.append(f"index={index} missing={missing}")
        if (
            REQUIRE_GOLD_LABELS
            and not isinstance(record.get("corrected_triplet"), list)
        ):
            errors.append(
                f"index={index} corrected_triplet is not a list"
            )

    if errors:
        raise ValueError(
            f"{path} has invalid records:\n"
            + "\n".join(errors[:20])
        )


def load_config(checkpoint_dir: Path) -> Dict[str, Any]:
    config_path = checkpoint_dir / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("architecture") != "IonRecall-FITGraph-V2":
        raise RuntimeError(
            "The checkpoint is not an IonRecall-FITGraph-V2 checkpoint."
        )
    return config


def load_model(
    checkpoint_dir: Path,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[IonRecallGraphModel, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir / "adapter"),
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = (
        torch.bfloat16
        if USE_BF16 and torch.cuda.is_available()
        else torch.float32
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    language_model = PeftModel.from_pretrained(
        base,
        str(checkpoint_dir / "adapter"),
        is_trainable=False,
    )
    if MERGE_ADAPTER_FOR_INFERENCE:
        language_model = language_model.merge_and_unload()

    language_model.config.use_cache = True
    language_model.config.pad_token_id = tokenizer.pad_token_id

    model = IonRecallGraphModel(
        language_model,
        hidden_size=int(config["hidden_size"]),
        max_mz=int(config["ion_head_max_mz"]),
        similarity_dim=int(config["similarity_projection_dim"]),
    )
    auxiliary = safe_torch_load(checkpoint_dir / "auxiliary_heads.pt")
    model.ion_head.load_state_dict(auxiliary["ion_head"])
    model.count_head.load_state_dict(auxiliary["count_head"])
    model.similarity_projection.load_state_dict(
        auxiliary["similarity_projection"]
    )

    model = model.to(device)
    model.eval()
    return model, tokenizer


def clean_generation_config(language_model: Any) -> None:
    config = language_model.generation_config
    config.do_sample = False
    for name in ("temperature", "top_p", "top_k"):
        if hasattr(config, name):
            setattr(config, name, None)


def generate_texts(
    language_model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> List[str]:
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    encoded = {
        key: value.to(device, non_blocking=True)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        generated = language_model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_width = encoded["input_ids"].shape[1]
    return tokenizer.batch_decode(
        generated[:, prompt_width:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def generate_with_oom_fallback(
    language_model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> List[str]:
    try:
        return generate_texts(
            language_model,
            tokenizer,
            prompts,
            device,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) <= 1:
            raise
        middle = len(prompts) // 2
        return (
            generate_with_oom_fallback(
                language_model,
                tokenizer,
                prompts[:middle],
                device,
            )
            + generate_with_oom_fallback(
                language_model,
                tokenizer,
                prompts[middle:],
                device,
            )
        )


def predict_auxiliary_batch(
    model: IonRecallGraphModel,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    encoded = {
        key: value.to(device, non_blocking=True)
        for key, value in encoded.items()
    }
    with torch.inference_mode():
        logits, counts = model.encode_prompt(
            encoded["input_ids"],
            encoded["attention_mask"],
        )
    return torch.sigmoid(logits.float()).cpu(), counts.float().cpu()


def select_candidates(
    probabilities: torch.Tensor,
    count_prediction: float,
    record: Dict[str, Any],
    config: Dict[str, Any],
) -> List[int]:
    threshold = float(
        config.get(
            "ion_probability_threshold",
            ION_PROBABILITY_THRESHOLD,
        )
    )
    expansion = float(
        config.get("ion_count_expansion", ION_COUNT_EXPANSION)
    )
    margin = int(
        config.get("ion_count_margin", ION_COUNT_MARGIN)
    )
    maximum_candidates = int(
        config.get(
            "max_auxiliary_ion_candidates",
            MAX_AUXILIARY_ION_CANDIDATES,
        )
    )
    max_mz = int(config["ion_head_max_mz"])
    isotope_margin = int(config["ion_isotope_margin"])

    try:
        molecular_mw = int(round(float(record.get("mw", 0))))
    except Exception:
        molecular_mw = 0

    valid_max = min(
        max_mz,
        max(1, molecular_mw + isotope_margin),
    )

    scores = probabilities.clone()
    scores[0] = -1.0
    if valid_max < max_mz:
        scores[valid_max + 1 :] = -1.0

    threshold_indices = {
        int(index)
        for index in torch.nonzero(
            scores >= threshold,
            as_tuple=False,
        ).flatten().tolist()
        if int(index) > 0
    }

    top_k = int(math.ceil(max(1.0, count_prediction) * expansion)) + margin
    top_k = min(max(1, top_k), valid_max, maximum_candidates)
    top_indices = {
        int(index)
        for index in torch.topk(scores, k=top_k).indices.tolist()
        if int(index) > 0 and float(scores[index]) >= 0.0
    }

    candidates = threshold_indices | top_indices
    if len(candidates) > maximum_candidates:
        candidates = set(
            sorted(
                candidates,
                key=lambda value: float(scores[value]),
                reverse=True,
            )[:maximum_candidates]
        )
    return sorted(candidates, reverse=True)


def predict_batch(
    model: IonRecallGraphModel,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    prompts = [
        build_chat_prompt(tokenizer, record)
        for record in records
    ]
    probabilities, counts = predict_auxiliary_batch(
        model,
        tokenizer,
        prompts,
        device,
    )
    first_texts = generate_with_oom_fallback(
        model.lm,
        tokenizer,
        prompts,
        device,
    )

    intermediate: List[Dict[str, Any]] = []
    repair_prompts: List[str] = []
    repair_positions: List[int] = []

    for position, (record, first_text) in enumerate(
        zip(records, first_texts)
    ):
        first_graph, first_ok, first_errors = parse_fitgraph_dsl(
            first_text
        )
        candidates = select_candidates(
            probabilities[position],
            float(counts[position]),
            record,
            config,
        )
        first_mzs = product_ion_mzs_from_graph(first_graph)
        missing = sorted(
            set(candidates) - first_mzs,
            reverse=True,
        )

        intermediate.append(
            {
                "record": record,
                "first_graph": first_graph,
                "first_parse_ok": first_ok,
                "first_errors": first_errors,
                "first_text": first_text,
                "candidates": candidates,
                "missing": missing,
                "count_prediction": float(counts[position]),
            }
        )

        if missing and MAX_REPAIR_ROUNDS > 0:
            repair_positions.append(position)
            repair_prompts.append(
                build_repair_chat_prompt(
                    tokenizer,
                    record,
                    first_graph,
                    missing,
                )
            )

    repair_texts = (
        generate_with_oom_fallback(
            model.lm,
            tokenizer,
            repair_prompts,
            device,
        )
        if repair_prompts
        else []
    )
    repair_by_position = {
        position: text
        for position, text in zip(repair_positions, repair_texts)
    }

    outputs: List[Dict[str, Any]] = []
    for position, item in enumerate(intermediate):
        final_graph = item["first_graph"]
        final_ok = bool(item["first_parse_ok"])
        repair_text = repair_by_position.get(position, "")
        repair_ok = False
        repair_errors: List[str] = []

        if repair_text:
            repaired_graph, repair_ok, repair_errors = (
                parse_fitgraph_dsl(repair_text)
            )
            if repair_ok:
                final_graph = merge_graphs_no_delete(
                    item["record"],
                    item["first_graph"],
                    repaired_graph,
                )
                final_ok = final_ok or repair_ok

        final_mzs = product_ion_mzs_from_graph(final_graph)
        stage2_candidate_mzs = sorted(
            final_mzs | set(item["candidates"]),
            reverse=True,
        )
        unresolved = sorted(
            set(item["candidates"]) - final_mzs,
            reverse=True,
        )

        outputs.append(
            {
                "record": item["record"],
                "first_graph": item["first_graph"],
                "final_graph": final_graph,
                "parse_ok": final_ok,
                "first_parse_ok": item["first_parse_ok"],
                "repair_parse_ok": repair_ok,
                "validation_errors": (
                    list(item["first_errors"]) + repair_errors
                ),
                "first_generation": item["first_text"],
                "repair_generation": repair_text,
                "auxiliary_candidate_mzs": item["candidates"],
                "stage2_candidate_mzs": stage2_candidate_mzs,
                "missing_before_repair": item["missing"],
                "unresolved_after_repair": unresolved,
                "count_prediction": item["count_prediction"],
            }
        )

    return outputs


def add_recall_metrics(
    rows: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> None:
    head_recall = 0.0
    first_recall = 0.0
    final_recall = 0.0
    union_recall = 0.0
    first_similarity = 0.0
    final_similarity = 0.0

    for row in rows:
        gold = product_ion_mzs_from_graph(row["gold_graph"])
        first = product_ion_mzs_from_graph(row["first_graph"])
        final = product_ion_mzs_from_graph(row["predicted_graph"])
        head = set(row["auxiliary_candidate_mzs"])
        union = set(row["stage2_candidate_mzs"])

        denominator = max(1, len(gold))
        head_recall += len(head & gold) / denominator
        first_recall += len(first & gold) / denominator
        final_recall += len(final & gold) / denominator
        union_recall += len(union & gold) / denominator
        first_similarity += output_graph_similarity(
            row["first_graph"],
            row["gold_graph"],
        )
        final_similarity += output_graph_similarity(
            row["predicted_graph"],
            row["gold_graph"],
        )

    count = max(1, len(rows))
    metrics.update(
        {
            "macro_auxiliary_head_product_ion_recall": (
                head_recall / count
            ),
            "macro_first_pass_product_ion_recall": (
                first_recall / count
            ),
            "macro_final_triplet_grounded_product_ion_recall": (
                final_recall / count
            ),
            "macro_stage2_candidate_union_product_ion_recall": (
                union_recall / count
            ),
            "macro_repair_recall_gain": (
                final_recall - first_recall
            ) / count,
            "macro_first_pass_graph_similarity": (
                first_similarity / count
            ),
            "macro_final_graph_similarity": (
                final_similarity / count
            ),
        }
    )

def set_all_seeds(seed: int, rank: int) -> None:
    actual_seed = int(seed) + int(rank)
    random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)


def main() -> None:
    rank, world_size, _, device = distributed_setup()
    set_all_seeds(RANDOM_SEED, rank)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    checkpoint_dir = resolve_path(CHECKPOINT_DIR)
    input_file = resolve_path(TEST_DATA_FILE)
    output_file = resolve_path(str(Path(OUTPUT_DIR) / OUTPUT_FILENAME))
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(
            "[config] "
            + json.dumps(
                {
                    "checkpoint_dir": str(checkpoint_dir),
                    "output_directory": str(output_file.parent),
                    "output_file": str(output_file),
                    "test_data_file": str(input_file),
                    "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
                    "max_input_length": MAX_INPUT_LENGTH,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "attn_implementation": ATTN_IMPLEMENTATION,
                    "use_bf16": USE_BF16,
                    "merge_adapter": MERGE_ADAPTER_FOR_INFERENCE,
                    "resume": RESUME,
                    "world_size": world_size,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)

    required_checkpoint_files = (
        checkpoint_dir / "adapter",
        checkpoint_dir / "auxiliary_heads.pt",
        checkpoint_dir / "model_config.json",
    )
    missing_checkpoint_items = [
        str(path) for path in required_checkpoint_files if not path.exists()
    ]
    if missing_checkpoint_items:
        raise FileNotFoundError(
            "CHECKPOINT_DIR is not a complete IonRecall-FITGraph-V2 checkpoint. "
            "Missing: " + ", ".join(missing_checkpoint_items)
        )

    if not input_file.is_file():
        raise FileNotFoundError(input_file)

    config = load_config(checkpoint_dir)
    records = read_jsonl(input_file)
    validate_input(records, input_file)

    state_file = output_file.with_suffix(".run_state.json")
    candidate_file = output_file.with_suffix(
        ".ion_candidates.jsonl"
    )
    evaluation_file = output_file.with_suffix(
        ".evaluation_state.jsonl"
    )
    metrics_file = output_file.with_suffix(".metrics.json")
    summary_file = output_file.with_suffix(".summary.json")

    checkpoint_identity = str(checkpoint_dir.resolve())
    if RESUME and state_file.exists():
        existing_state = json.loads(
            state_file.read_text(encoding="utf-8")
        )
        if existing_state.get("checkpoint_dir") != checkpoint_identity:
            raise RuntimeError(
                "Existing output was generated with a different checkpoint. "
                "Delete the old output files or change OUTPUT_DIR."
            )

    completed: Dict[str, Dict[str, Any]] = {}
    existing_candidates: Dict[str, Dict[str, Any]] = {}
    existing_evaluation: Dict[str, Dict[str, Any]] = {}

    if RESUME and output_file.exists():
        for row in read_jsonl(output_file):
            completed[strict_key(row)] = row
    if RESUME and candidate_file.exists():
        for row in read_jsonl(candidate_file):
            existing_candidates[str(row["_key"])] = row
    if RESUME and evaluation_file.exists():
        for row in read_jsonl(evaluation_file):
            existing_evaluation[str(row["_key"])] = row

    model, tokenizer = load_model(
        checkpoint_dir,
        config,
        device,
    )
    clean_generation_config(model.lm)

    local_indices = [
        index
        for index in range(rank, len(records), world_size)
        if strict_key(records[index]) not in completed
    ]

    local_output: List[Dict[str, Any]] = []
    for start in range(0, len(local_indices), BATCH_SIZE_PER_GPU):
        index_batch = local_indices[start : start + BATCH_SIZE_PER_GPU]
        record_batch = [records[index] for index in index_batch]
        predicted_batch = predict_batch(
            model,
            tokenizer,
            record_batch,
            config,
            device,
        )

        for index, predicted in zip(index_batch, predicted_batch):
            record = records[index]
            key = strict_key(record)
            local_output.append(
                {
                    "_index": index,
                    "_key": key,
                    "strict_row": strict_inference_row(
                        record,
                        predicted["final_graph"],
                        predicted["parse_ok"],
                    ),
                    "candidate_row": {
                        "_index": index,
                        "_key": key,
                        "id": record.get("id"),
                        "auxiliary_candidate_mzs": (
                            predicted["auxiliary_candidate_mzs"]
                        ),
                        "stage2_candidate_mzs": (
                            predicted["stage2_candidate_mzs"]
                        ),
                        "missing_before_repair": (
                            predicted["missing_before_repair"]
                        ),
                        "unresolved_after_repair": (
                            predicted["unresolved_after_repair"]
                        ),
                        "predicted_product_ion_count": len(
                            product_ion_mzs_from_graph(
                                predicted["final_graph"]
                            )
                        ),
                        "auxiliary_count_prediction": (
                            predicted["count_prediction"]
                        ),
                    },
                    "evaluation_row": {
                        "_index": index,
                        "_key": key,
                        "record": record,
                        "gold_graph": (
                            build_fitgraph(record)
                            if isinstance(
                                record.get("corrected_triplet"),
                                list,
                            )
                            else {}
                        ),
                        "first_graph": predicted["first_graph"],
                        "predicted_graph": predicted["final_graph"],
                        "parse_ok": predicted["parse_ok"],
                        "validation_errors": (
                            predicted["validation_errors"]
                        ),
                        "auxiliary_candidate_mzs": (
                            predicted["auxiliary_candidate_mzs"]
                        ),
                        "stage2_candidate_mzs": (
                            predicted["stage2_candidate_mzs"]
                        ),
                    },
                }
            )

        print(
            f"[rank {rank}] processed="
            f"{min(start + len(index_batch), len(local_indices))}/"
            f"{len(local_indices)}",
            flush=True,
        )

    shard_dir = output_file.parent / (
        output_file.stem + "_distributed_shards"
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        shard_dir / f"rank_{rank:03d}.jsonl",
        local_output,
    )
    barrier()

    if rank == 0:
        generated: Dict[str, Dict[str, Any]] = {}
        for shard_rank in range(world_size):
            shard = shard_dir / f"rank_{shard_rank:03d}.jsonl"
            if not shard.exists():
                continue
            for row in read_jsonl(shard):
                generated[str(row["_key"])] = row

        final_rows: List[Dict[str, Any]] = []
        candidate_rows: List[Dict[str, Any]] = []
        evaluation_rows: List[Dict[str, Any]] = []

        for index, record in enumerate(records):
            key = strict_key(record)
            if key in generated:
                item = generated[key]
                strict_row = item["strict_row"]
                candidate_row = item["candidate_row"]
                evaluation_row = item["evaluation_row"]
            elif key in completed:
                strict_row = completed[key]
                candidate_row = existing_candidates.get(key)
                evaluation_row = existing_evaluation.get(key)
                if candidate_row is None or evaluation_row is None:
                    temporary = dict(record)
                    temporary["corrected_triplet"] = strict_row.get(
                        "infer_result",
                        [],
                    )
                    reconstructed = build_fitgraph(temporary)
                    candidate_row = {
                        "_index": index,
                        "_key": key,
                        "id": record.get("id"),
                        "auxiliary_candidate_mzs": [],
                        "stage2_candidate_mzs": sorted(
                            product_ion_mzs_from_graph(reconstructed),
                            reverse=True,
                        ),
                        "missing_before_repair": [],
                        "unresolved_after_repair": [],
                        "predicted_product_ion_count": len(
                            product_ion_mzs_from_graph(reconstructed)
                        ),
                        "auxiliary_count_prediction": None,
                    }
                    evaluation_row = {
                        "_index": index,
                        "_key": key,
                        "record": record,
                        "gold_graph": build_fitgraph(record),
                        "first_graph": reconstructed,
                        "predicted_graph": reconstructed,
                        "parse_ok": bool(strict_row.get("parse_ok")),
                        "validation_errors": [],
                        "auxiliary_candidate_mzs": [],
                        "stage2_candidate_mzs": (
                            candidate_row["stage2_candidate_mzs"]
                        ),
                    }
            else:
                continue

            final_rows.append(strict_row)
            candidate_rows.append(candidate_row)
            evaluation_rows.append(evaluation_row)

        write_jsonl(output_file, final_rows)
        write_jsonl(candidate_file, candidate_rows)
        write_jsonl(evaluation_file, evaluation_rows)

        if REQUIRE_GOLD_LABELS:
            metrics = evaluate_predictions(
                evaluation_rows,
                STRONG_PEAK_THRESHOLD,
            )
            metrics.update(
                macro_ion_fbeta_similarity(
                    evaluation_rows,
                    beta=EVALUATION_F_BETA,
                )
            )
            add_recall_metrics(evaluation_rows, metrics)
        else:
            metrics = {
                "evaluated_molecules": 0,
                "message": "Gold labels were not required.",
            }

        metrics.update(
            {
                "checkpoint_dir": checkpoint_identity,
                "input_file": str(input_file),
                "output_file": str(output_file),
            }
        )
        metrics_file.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary = {
            "input_records": len(records),
            "output_records": len(final_rows),
            "world_size": world_size,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "checkpoint_dir": checkpoint_identity,
            "output_file": str(output_file),
            "ion_candidates_file": str(candidate_file),
            "metrics_file": str(metrics_file),
            "main_output_schema": [
                "id",
                "name",
                "smiles",
                "formula",
                "mw",
                "input",
                "infer_result",
                "parse_ok",
                "corrected_triplet",
            ],
        }
        summary_file.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        state_file.write_text(
            json.dumps(
                {
                    "checkpoint_dir": checkpoint_identity,
                    "completed_records": len(final_rows),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "[metrics]\n"
            + json.dumps(metrics, ensure_ascii=False, indent=2),
            flush=True,
        )
        print(
            "[done]\n"
            + json.dumps(summary, ensure_ascii=False, indent=2),
            flush=True,
        )

        shutil.rmtree(shard_dir, ignore_errors=True)

    barrier()


if __name__ == "__main__":
    try:
        main()
    finally:
        safe_destroy_process_group()
