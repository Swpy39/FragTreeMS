#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

MODEL_NAME_OR_PATH = "xx"

TRAIN_DATA_FILE = "xx"

SELECTION_DATA_FILE = "xx"

OUTPUT_DIR = "xx"

print(OUTPUT_DIR)

RANDOM_SEED = 44
MAX_SEQ_LENGTH = 8192
MAX_REPRESENTATION_LENGTH = 512

NUM_TRAIN_EPOCHS = 30
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 3.0e-5
AUXILIARY_LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0

LORA_RANK = 96
LORA_ALPHA = 192
LORA_DROPOUT = 0.05

ATTN_IMPLEMENTATION = "sdpa"
USE_BF16 = True
GRADIENT_CHECKPOINTING = True

DATALOADER_NUM_WORKERS = 4
DATALOADER_PREFETCH_FACTOR = 4
LOG_EVERY_UPDATES = 5

BASE_TOKEN_WEIGHT = 1.0
ION_RELATED_TOKEN_WEIGHT = 1.5

ION_HEAD_MAX_MZ = 1024
ION_ISOTOPE_MARGIN = 4

ION_POS_WEIGHT = 2.0
ION_POS_GAMMA = 0.0
ION_NEG_GAMMA = 4.0
ION_NEGATIVE_CLIP = 0.05

ION_COUNT_LOSS_WEIGHT = 0.20
ION_COUNT_UNDER_WEIGHT = 2.0
ION_COUNT_OVER_WEIGHT = 0.5

LAMBDA_ION_MAX = 1.0
LAMBDA_SIMILARITY_MAX = 0.03
SIMILARITY_START_EPOCH = 2
SIMILARITY_PROJECTION_DIM = 256

USE_RECALL_REPAIR_TRAINING = True
REPAIR_SAMPLE_RATIO = 0.25
REPAIR_DROP_MIN = 0.10
REPAIR_DROP_MAX = 0.30

SELECTION_START_EPOCH = 30
SELECTION_EVERY_N_EPOCHS = 30
SELECT_FINAL_EPOCH = True

SELECTION_BATCH_SIZE_PER_GPU = 4
SELECTION_MAX_INPUT_LENGTH = 4096
SELECTION_MAX_NEW_TOKENS = 8192
STRONG_PEAK_THRESHOLD = 100.0

ION_PROBABILITY_THRESHOLD = 0.20
ION_COUNT_EXPANSION = 1.15
ION_COUNT_MARGIN = 1
MAX_AUXILIARY_ION_CANDIDATES = 64
MAX_REPAIR_ROUNDS = 1

SELECTION_F_BETA = 2.0


RESUME_CHECKPOINT_DIR = None


import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from fitgraph_ion_recall_common_v2 import (
    build_chat_prompt,
    build_fitgraph,
    build_repair_chat_prompt,
    corrupt_graph_for_repair,
    evaluate_predictions,
    graph_to_dsl,
    graph_to_triplets,
    macro_ion_fbeta_similarity,
    merge_graphs_no_delete,
    output_graph_similarity,
    parse_fitgraph_dsl,
    product_ion_mzs_from_graph,
    product_ion_mzs_from_record,
    read_jsonl,
    serializable_record_key,
    strict_inference_row,
    triplet_similarity_signatures_from_graph,
    write_jsonl,
)



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


def is_rank0(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def set_all_seeds(seed: int, rank: int) -> None:
    actual_seed = int(seed) + int(rank)
    random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def configure_torch() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_updates: int,
    warmup_updates: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_updates:
            return max(1e-8, step / max(1, warmup_updates))
        progress = (
            (step - warmup_updates)
            / max(1, total_updates - warmup_updates)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )



def validate_record_schema(
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
        "corrected_triplet",
    )
    errors: List[str] = []

    for index, record in enumerate(records):
        missing = [key for key in required if key not in record]
        if missing:
            errors.append(
                f"line/index {index}: missing={missing}"
            )
            continue
        if not isinstance(record.get("corrected_triplet"), list):
            errors.append(
                f"line/index {index}: corrected_triplet is not a list"
            )

    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(
            f"{path} contains invalid records:\n{preview}"
        )


def warn_train_selection_overlap(
    train_records: Sequence[Dict[str, Any]],
    selection_records: Sequence[Dict[str, Any]],
    rank: int,
) -> None:
    if not is_rank0(rank):
        return

    def key(record: Dict[str, Any]) -> Tuple[str, str, str]:
        return (
            str(record.get("smiles", "")).strip(),
            str(record.get("formula", "")).strip(),
            str(record.get("mw", "")).strip(),
        )

    train_keys = {key(record) for record in train_records}
    selection_keys = {key(record) for record in selection_records}
    overlap = train_keys & selection_keys
    print(
        "[split-check] "
        + json.dumps(
            {
                "train_records": len(train_records),
                "selection_records": len(selection_records),
                "exact_structure_overlap": len(overlap),
                "selection_set_warning": (
                    "This file is used for checkpoint selection and is not "
                    "an unbiased final test set."
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if overlap:
        print(
            "[warning] train/selection overlap examples: "
            + json.dumps(list(overlap)[:10], ensure_ascii=False),
            flush=True,
        )


def build_target_character_weights(target: str) -> List[float]:
    """
    Use only two token weights:
      - BASE_TOKEN_WEIGHT for ordinary graph text;
      - ION_RELATED_TOKEN_WEIGHT for ions=<count>, ion m/z and ion formula.

    IDs, formatting, ion type, fragments and edges remain at the base weight.
    """
    weights = [BASE_TOKEN_WEIGHT] * len(target)
    section: Optional[str] = None
    cursor = 0

    for line in target.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        line_start = cursor
        line_end = cursor + len(line)

        if stripped.startswith("[COUNTS]"):
            match = __import__("re").search(
                r"\bions=(\d+)",
                stripped,
            )
            if match:
                start = line_start + match.start(1)
                end = line_start + match.end(1)
                weights[start:end] = [
                    ION_RELATED_TOKEN_WEIGHT
                ] * (end - start)

        if stripped == "[IONS]":
            section = "ions"
        elif stripped == "[/IONS]":
            section = None
        elif (
            stripped.startswith("[")
            and stripped.endswith("]")
            and stripped not in ("[IONS]", "[/IONS]")
        ):
            section = None
        elif section == "ions" and stripped:
            # I0<TAB>55<TAB>C4H7<TAB>fragment_ion
            positions: List[Tuple[int, int]] = []
            part_start = 0
            for part in stripped.split("\t"):
                part_end = part_start + len(part)
                positions.append((part_start, part_end))
                part_start = part_end + 1

            # m/z column and formula column only.
            for column in (1, 2):
                if column >= len(positions):
                    continue
                local_start, local_end = positions[column]
                start = line_start + local_start
                end = line_start + local_end
                weights[start:end] = [
                    ION_RELATED_TOKEN_WEIGHT
                ] * max(0, end - start)

        cursor = line_end

    return weights


def target_token_weights_from_offsets(
    target: str,
    offsets: Sequence[Tuple[int, int]],
) -> List[float]:
    character_weights = build_target_character_weights(target)
    token_weights: List[float] = []

    for start, end in offsets:
        start = int(start)
        end = int(end)
        if end <= start or not character_weights:
            token_weights.append(BASE_TOKEN_WEIGHT)
            continue

        start = max(0, min(start, len(character_weights)))
        end = max(start, min(end, len(character_weights)))
        if end <= start:
            token_weights.append(BASE_TOKEN_WEIGHT)
            continue

        span = character_weights[start:end]
        token_weights.append(sum(span) / len(span))

    return token_weights


def tokenize_training_example(
    tokenizer: Any,
    record: Dict[str, Any],
    graph: Dict[str, Any],
    *,
    repair_draft: Optional[Dict[str, Any]] = None,
    repair_candidates: Optional[Sequence[int]] = None,
) -> Optional[Dict[str, Any]]:
    if repair_draft is None:
        prompt = build_chat_prompt(tokenizer, record)
        example_type = "direct"
    else:
        prompt = build_repair_chat_prompt(
            tokenizer,
            record,
            repair_draft,
            repair_candidates or [],
        )
        example_type = "repair"

    target = graph_to_dsl(graph)

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    target_encoded = tokenizer(
        target,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    target_ids = list(target_encoded["input_ids"])
    offsets = target_encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError(
            "A fast tokenizer with offset_mapping support is required."
        )

    target_weights = target_token_weights_from_offsets(
        target,
        offsets,
    )

    if tokenizer.eos_token_id is not None:
        target_ids.append(int(tokenizer.eos_token_id))
        target_weights.append(BASE_TOKEN_WEIGHT)

    if len(prompt_ids) + len(target_ids) > MAX_SEQ_LENGTH:
        return None

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    loss_weights = [0.0] * len(prompt_ids) + target_weights

    try:
        molecular_mw = int(round(float(record.get("mw", 0))))
    except Exception:
        molecular_mw = 0

    product_mzs = sorted(product_ion_mzs_from_record(record))
    if product_mzs and max(product_mzs) > ION_HEAD_MAX_MZ:
        raise ValueError(
            f"Record id={record.get('id')} has product m/z "
            f"{max(product_mzs)} > ION_HEAD_MAX_MZ={ION_HEAD_MAX_MZ}."
        )

    valid_mz_max = min(
        ION_HEAD_MAX_MZ,
        max(
            molecular_mw + ION_ISOTOPE_MARGIN,
            max(product_mzs, default=1),
        ),
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_weights": loss_weights,
        "prompt_length": len(prompt_ids),
        "product_mzs": product_mzs,
        "product_ion_count": len(product_mzs),
        "valid_mz_max": int(valid_mz_max),
        "ion_signature": set(product_ion_mzs_from_graph(graph)),
        "triplet_signature": set(
            triplet_similarity_signatures_from_graph(graph)
        ),
        "example_type": example_type,
        "record_id": record.get("id"),
    }


def prepare_training_examples(
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    rank: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    examples: List[Dict[str, Any]] = []
    graphs: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    rng = random.Random(RANDOM_SEED)

    for index, record in enumerate(records):
        graph = build_fitgraph(record)
        direct = tokenize_training_example(
            tokenizer,
            record,
            graph,
        )
        if direct is None:
            dropped.append(
                {
                    "id": record.get("id"),
                    "type": "direct",
                    "reason": "sequence_too_long",
                }
            )
            continue

        examples.append(direct)
        graphs.append(graph)

        if (
            USE_RECALL_REPAIR_TRAINING
            and rng.random() < REPAIR_SAMPLE_RATIO
        ):
            repair_graph, dropped_mzs = corrupt_graph_for_repair(
                graph,
                rng,
                REPAIR_DROP_MIN,
                REPAIR_DROP_MAX,
            )
            if dropped_mzs:
                repair = tokenize_training_example(
                    tokenizer,
                    record,
                    graph,
                    repair_draft=repair_graph,
                    repair_candidates=dropped_mzs,
                )
                if repair is not None:
                    examples.append(repair)
                    graphs.append(graph)
                else:
                    dropped.append(
                        {
                            "id": record.get("id"),
                            "type": "repair",
                            "reason": "sequence_too_long",
                        }
                    )

    if is_rank0(rank):
        type_counts: Dict[str, int] = {}
        for example in examples:
            key = str(example["example_type"])
            type_counts[key] = type_counts.get(key, 0) + 1

        print(
            "[prepared] "
            + json.dumps(
                {
                    "source_records": len(records),
                    "training_examples": len(examples),
                    "example_types": type_counts,
                    "dropped": len(dropped),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    return examples, graphs, dropped


def compute_positive_frequency_weights(
    records: Sequence[Dict[str, Any]],
) -> torch.Tensor:
    frequency = torch.zeros(
        ION_HEAD_MAX_MZ + 1,
        dtype=torch.float32,
    )
    for record in records:
        for mz in product_ion_mzs_from_record(record):
            if 0 < mz <= ION_HEAD_MAX_MZ:
                frequency[mz] += 1.0

    nonzero = frequency[frequency > 0]
    median_frequency = (
        float(nonzero.median().item()) if nonzero.numel() else 1.0
    )

    weights = torch.ones_like(frequency)
    observed = frequency > 0
    weights[observed] = torch.sqrt(
        torch.tensor(median_frequency) / frequency[observed]
    ).clamp(min=1.0, max=3.0)
    weights[0] = 0.0
    return weights


class RecallGraphDataset(Dataset):
    def __init__(self, examples: Sequence[Dict[str, Any]]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.examples[index]


def set_jaccard_nonempty(
    first: Set[Any],
    second: Set[Any],
) -> float:
    if not first and not second:
        return 0.0
    union = first | second
    return len(first & second) / max(1, len(union))


class RecallGraphCollator:
    def __init__(
        self,
        pad_token_id: int,
        ion_head_max_mz: int,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.ion_head_max_mz = int(ion_head_max_mz)

    @staticmethod
    def _padded_width(
        values: Sequence[Sequence[Any]],
        multiple: int = 8,
    ) -> int:
        maximum = max(len(value) for value in values)
        return int(math.ceil(maximum / multiple) * multiple)

    def __call__(
        self,
        examples: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        input_values = [example["input_ids"] for example in examples]
        labels_values = [example["labels"] for example in examples]
        weight_values = [example["loss_weights"] for example in examples]

        width = self._padded_width(input_values)
        batch_size = len(examples)

        input_ids = torch.full(
            (batch_size, width),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, width),
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, width),
            -100,
            dtype=torch.long,
        )
        loss_weights = torch.zeros(
            (batch_size, width),
            dtype=torch.float32,
        )

        ion_targets = torch.zeros(
            (batch_size, self.ion_head_max_mz + 1),
            dtype=torch.float32,
        )
        ion_valid_mask = torch.zeros_like(ion_targets)
        ion_count_targets = torch.zeros(
            batch_size,
            dtype=torch.float32,
        )

        for row, example in enumerate(examples):
            length = len(example["input_ids"])
            input_ids[row, :length] = torch.tensor(
                example["input_ids"],
                dtype=torch.long,
            )
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(
                example["labels"],
                dtype=torch.long,
            )
            loss_weights[row, :length] = torch.tensor(
                example["loss_weights"],
                dtype=torch.float32,
            )

            valid_max = int(example["valid_mz_max"])
            ion_valid_mask[row, 1 : valid_max + 1] = 1.0
            for mz in example["product_mzs"]:
                ion_targets[row, int(mz)] = 1.0
            ion_count_targets[row] = float(
                example["product_ion_count"]
            )

        pair_similarity = torch.zeros(
            (batch_size, batch_size),
            dtype=torch.float32,
        )
        for left in range(batch_size):
            pair_similarity[left, left] = 1.0
            for right in range(left + 1, batch_size):
                ion_score = set_jaccard_nonempty(
                    examples[left]["ion_signature"],
                    examples[right]["ion_signature"],
                )
                triplet_score = set_jaccard_nonempty(
                    examples[left]["triplet_signature"],
                    examples[right]["triplet_signature"],
                )
                value = 0.70 * ion_score + 0.30 * triplet_score
                pair_similarity[left, right] = value
                pair_similarity[right, left] = value

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": loss_weights,
            "ion_targets": ion_targets,
            "ion_valid_mask": ion_valid_mask,
            "ion_count_targets": ion_count_targets,
            "pair_similarity": pair_similarity,
        }


# ===========================================================================
# Model
# ===========================================================================

def masked_prompt_pool(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    prompt_mask = (
        labels.eq(-100) & attention_mask.bool()
    ).unsqueeze(-1)
    mask = prompt_mask.to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return summed / denominator


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

        self.ion_head = nn.Linear(
            self.hidden_size,
            self.max_mz + 1,
        )
        self.count_head = nn.Linear(self.hidden_size, 1)
        self.similarity_projection = nn.Sequential(
            nn.Linear(self.hidden_size, int(similarity_dim)),
            nn.GELU(),
            nn.Linear(int(similarity_dim), int(similarity_dim)),
        )
        self.reset_auxiliary_parameters()

    def reset_auxiliary_parameters(self) -> None:
        nn.init.xavier_uniform_(self.ion_head.weight)
        nn.init.zeros_(self.ion_head.bias)
        nn.init.xavier_uniform_(self.count_head.weight)
        nn.init.zeros_(self.count_head.bias)
        for module in self.similarity_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        molecule_repr = masked_prompt_pool(
            hidden,
            labels,
            attention_mask,
        ).float()

        ion_logits = self.ion_head(molecule_repr)
        count_prediction = F.softplus(
            self.count_head(molecule_repr).squeeze(-1)
        )
        similarity_repr = F.normalize(
            self.similarity_projection(molecule_repr),
            dim=-1,
        )

        return {
            "logits": outputs.logits,
            "ion_logits": ion_logits,
            "count_prediction": count_prediction,
            "similarity_repr": similarity_repr,
        }

    def encode_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prompt-only inference for auxiliary ion proposals.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1.0)
        ).float()
        return (
            self.ion_head(pooled),
            F.softplus(self.count_head(pooled).squeeze(-1)),
        )


def load_model_and_tokenizer(
    device: torch.device,
) -> Tuple[IonRecallGraphModel, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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

    resume_dir = (
        Path(RESUME_CHECKPOINT_DIR)
        if RESUME_CHECKPOINT_DIR
        else None
    )
    if resume_dir is not None:
        adapter_path = resume_dir / "adapter"
        language_model = PeftModel.from_pretrained(
            base,
            str(adapter_path),
            is_trainable=True,
        )
    else:
        lora = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
        )
        language_model = get_peft_model(base, lora)

    if GRADIENT_CHECKPOINTING:
        try:
            language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False
                }
            )
        except TypeError:
            language_model.gradient_checkpointing_enable()
        if hasattr(language_model, "enable_input_require_grads"):
            language_model.enable_input_require_grads()

    language_model.config.use_cache = False
    language_model.config.pad_token_id = tokenizer.pad_token_id

    hidden_size = int(language_model.config.hidden_size)
    model = IonRecallGraphModel(
        language_model,
        hidden_size=hidden_size,
        max_mz=ION_HEAD_MAX_MZ,
        similarity_dim=SIMILARITY_PROJECTION_DIM,
    )

    if resume_dir is not None:
        state = safe_torch_load(resume_dir / "auxiliary_heads.pt")
        model.ion_head.load_state_dict(state["ion_head"])
        model.count_head.load_state_dict(state["count_head"])
        model.similarity_projection.load_state_dict(
            state["similarity_projection"]
        )

    model = model.to(device)
    return model, tokenizer


# ===========================================================================
# Losses
# ===========================================================================

def graph_weighted_loss_per_molecule(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize each molecule independently, then average molecules.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = loss_weights[:, 1:].contiguous().float()

    valid = shift_labels.ne(-100)
    token_loss = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)

    valid_float = valid.to(token_loss.dtype)
    effective_weights = shift_weights * valid_float

    weighted_per_molecule = (
        (token_loss * effective_weights).sum(dim=1)
        / effective_weights.sum(dim=1).clamp_min(1e-8)
    )
    unweighted_per_molecule = (
        (token_loss * valid_float).sum(dim=1)
        / valid_float.sum(dim=1).clamp_min(1.0)
    )

    valid_molecule = valid.any(dim=1)
    return (
        weighted_per_molecule[valid_molecule].mean(),
        unweighted_per_molecule[valid_molecule].mean(),
    )


def asymmetric_ion_inventory_loss(
    ion_logits: torch.Tensor,
    ion_targets: torch.Tensor,
    ion_valid_mask: torch.Tensor,
    count_prediction: torch.Tensor,
    count_targets: torch.Tensor,
    positive_frequency_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = torch.sigmoid(ion_logits.float())
    targets = ion_targets.float()
    valid = ion_valid_mask.float()

    positive_mask = targets * valid
    negative_mask = (1.0 - targets) * valid

    positive_term = -(
        (1.0 - probabilities).pow(ION_POS_GAMMA)
        * torch.log(probabilities.clamp_min(1e-8))
        * positive_mask
        * positive_frequency_weights.unsqueeze(0)
        * ION_POS_WEIGHT
    )

    negative_probability = probabilities
    if ION_NEGATIVE_CLIP > 0:
        negative_log_argument = (
            1.0 - probabilities + ION_NEGATIVE_CLIP
        ).clamp(max=1.0)
    else:
        negative_log_argument = (
            1.0 - probabilities
        ).clamp_min(1e-8)

    negative_term = -(
        negative_probability.pow(ION_NEG_GAMMA)
        * torch.log(negative_log_argument.clamp_min(1e-8))
        * negative_mask
    )

    positive_per_molecule = (
        positive_term.sum(dim=1)
        / positive_mask.sum(dim=1).clamp_min(1.0)
    )
    negative_per_molecule = (
        negative_term.sum(dim=1)
        / negative_mask.sum(dim=1).clamp_min(1.0)
    )
    presence_loss = (
        positive_per_molecule + negative_per_molecule
    ).mean()

    denominator = count_targets.clamp_min(1.0)
    under = F.relu(count_targets - count_prediction) / denominator
    over = F.relu(count_prediction - count_targets) / denominator
    count_loss = (
        ION_COUNT_UNDER_WEIGHT * under.square()
        + ION_COUNT_OVER_WEIGHT * over.square()
    ).mean()

    total = presence_loss + ION_COUNT_LOSS_WEIGHT * count_loss
    return total, presence_loss, count_loss


def graph_similarity_regression_loss(
    similarity_repr: torch.Tensor,
    target_similarity: torch.Tensor,
) -> torch.Tensor:
    batch_size = similarity_repr.shape[0]
    if batch_size < 2:
        return similarity_repr.sum() * 0.0

    cosine = similarity_repr @ similarity_repr.transpose(0, 1)
    predicted = (cosine + 1.0) * 0.5

    pair_mask = torch.triu(
        torch.ones_like(predicted, dtype=torch.bool),
        diagonal=1,
    )
    return F.smooth_l1_loss(
        predicted[pair_mask],
        target_similarity.float()[pair_mask],
        beta=0.05,
        reduction="mean",
    )


def ion_lambda(epoch: int) -> float:
    return 0.5 * LAMBDA_ION_MAX if epoch == 1 else LAMBDA_ION_MAX


def similarity_lambda(epoch: int) -> float:
    if epoch < SIMILARITY_START_EPOCH:
        return 0.0
    if epoch == SIMILARITY_START_EPOCH:
        return 0.5 * LAMBDA_SIMILARITY_MAX
    return LAMBDA_SIMILARITY_MAX


# ===========================================================================
# Selection inference
# ===========================================================================

def clean_generation_config(language_model: nn.Module) -> None:
    config = language_model.generation_config
    config.do_sample = False
    for name in ("temperature", "top_p", "top_k"):
        if hasattr(config, name):
            setattr(config, name, None)


def generate_texts(
    language_model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> List[str]:
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=SELECTION_MAX_INPUT_LENGTH,
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
            max_new_tokens=SELECTION_MAX_NEW_TOKENS,
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
    language_model: nn.Module,
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
    raw_model: IonRecallGraphModel,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=SELECTION_MAX_INPUT_LENGTH,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    encoded = {
        key: value.to(device, non_blocking=True)
        for key, value in encoded.items()
    }
    with torch.inference_mode():
        logits, count = raw_model.encode_prompt(
            encoded["input_ids"],
            encoded["attention_mask"],
        )
    return torch.sigmoid(logits.float()).cpu(), count.float().cpu()


def select_auxiliary_candidates(
    probabilities: torch.Tensor,
    count_prediction: float,
    record: Dict[str, Any],
) -> List[int]:
    try:
        molecular_mw = int(round(float(record.get("mw", 0))))
    except Exception:
        molecular_mw = 0

    valid_max = min(
        ION_HEAD_MAX_MZ,
        max(1, molecular_mw + ION_ISOTOPE_MARGIN),
    )

    scores = probabilities.clone()
    scores[0] = -1.0
    if valid_max < ION_HEAD_MAX_MZ:
        scores[valid_max + 1 :] = -1.0

    threshold_indices = {
        int(index)
        for index in torch.nonzero(
            scores >= ION_PROBABILITY_THRESHOLD,
            as_tuple=False,
        ).flatten().tolist()
        if int(index) > 0
    }

    top_k = (
        int(math.ceil(max(1.0, count_prediction) * ION_COUNT_EXPANSION))
        + ION_COUNT_MARGIN
    )
    top_k = min(
        max(1, top_k),
        valid_max,
        MAX_AUXILIARY_ION_CANDIDATES,
    )
    top_indices = {
        int(index)
        for index in torch.topk(scores, k=top_k).indices.tolist()
        if int(index) > 0 and float(scores[index]) >= 0.0
    }

    candidates = threshold_indices | top_indices
    if len(candidates) > MAX_AUXILIARY_ION_CANDIDATES:
        candidates = set(
            sorted(
                candidates,
                key=lambda value: float(scores[value]),
                reverse=True,
            )[:MAX_AUXILIARY_ION_CANDIDATES]
        )
    return sorted(candidates, reverse=True)


def predict_records_batch(
    raw_model: IonRecallGraphModel,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    prompts = [
        build_chat_prompt(tokenizer, record)
        for record in records
    ]

    probabilities, count_predictions = predict_auxiliary_batch(
        raw_model,
        tokenizer,
        prompts,
        device,
    )
    first_texts = generate_with_oom_fallback(
        raw_model.lm,
        tokenizer,
        prompts,
        device,
    )

    intermediate: List[Dict[str, Any]] = []
    repair_prompts: List[str] = []
    repair_positions: List[int] = []

    for position, (record, text) in enumerate(
        zip(records, first_texts)
    ):
        first_graph, first_ok, first_errors = parse_fitgraph_dsl(text)
        auxiliary_candidates = select_auxiliary_candidates(
            probabilities[position],
            float(count_predictions[position]),
            record,
        )
        first_product_mzs = product_ion_mzs_from_graph(first_graph)
        missing = sorted(
            set(auxiliary_candidates) - first_product_mzs,
            reverse=True,
        )

        intermediate.append(
            {
                "record": record,
                "first_graph": first_graph,
                "first_parse_ok": first_ok,
                "first_errors": first_errors,
                "first_text": text,
                "auxiliary_candidates": auxiliary_candidates,
                "missing_before_repair": missing,
                "count_prediction": float(count_predictions[position]),
                "head_probabilities": probabilities[position],
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

    repair_texts: List[str] = []
    if repair_prompts:
        repair_texts = generate_with_oom_fallback(
            raw_model.lm,
            tokenizer,
            repair_prompts,
            device,
        )

    repair_by_position = {
        position: text
        for position, text in zip(repair_positions, repair_texts)
    }

    rows: List[Dict[str, Any]] = []
    for position, item in enumerate(intermediate):
        first_graph = item["first_graph"]
        final_graph = first_graph
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
                    first_graph,
                    repaired_graph,
                )
                final_ok = final_ok or repair_ok

        final_product_mzs = product_ion_mzs_from_graph(final_graph)
        unresolved = sorted(
            set(item["auxiliary_candidates"]) - final_product_mzs,
            reverse=True,
        )

        rows.append(
            {
                "record": item["record"],
                "first_graph": first_graph,
                "predicted_graph": final_graph,
                "parse_ok": final_ok,
                "validation_errors": (
                    list(item["first_errors"]) + list(repair_errors)
                ),
                "raw_generation": item["first_text"],
                "repair_generation": repair_text,
                "repair_parse_ok": repair_ok,
                "auxiliary_candidates": item["auxiliary_candidates"],
                "missing_before_repair": item["missing_before_repair"],
                "unresolved_after_repair": unresolved,
                "count_prediction": item["count_prediction"],
            }
        )

    return rows


def add_auxiliary_selection_metrics(
    rows: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> None:
    head_recalls: List[float] = []
    first_recalls: List[float] = []
    final_recalls: List[float] = []
    first_similarity: List[float] = []
    final_similarity: List[float] = []

    for row in rows:
        gold_graph = row["gold_graph"]
        gold = product_ion_mzs_from_graph(gold_graph)
        first = product_ion_mzs_from_graph(row["first_graph"])
        final = product_ion_mzs_from_graph(row["predicted_graph"])
        head = set(row.get("auxiliary_candidates", []))

        head_recalls.append(len(head & gold) / max(1, len(gold)))
        first_recalls.append(len(first & gold) / max(1, len(gold)))
        final_recalls.append(len(final & gold) / max(1, len(gold)))
        first_similarity.append(
            output_graph_similarity(row["first_graph"], gold_graph)
        )
        final_similarity.append(
            output_graph_similarity(row["predicted_graph"], gold_graph)
        )

    count = max(1, len(rows))
    metrics["macro_auxiliary_head_product_ion_recall"] = (
        sum(head_recalls) / count
    )
    metrics["macro_first_pass_product_ion_recall"] = (
        sum(first_recalls) / count
    )
    metrics["macro_final_product_ion_recall"] = (
        sum(final_recalls) / count
    )
    metrics["macro_repair_recall_gain"] = (
        metrics["macro_final_product_ion_recall"]
        - metrics["macro_first_pass_product_ion_recall"]
    )
    metrics["macro_first_pass_graph_similarity"] = (
        sum(first_similarity) / count
    )
    metrics["macro_final_graph_similarity"] = (
        sum(final_similarity) / count
    )


def save_recall_checkpoint(
    raw_model: IonRecallGraphModel,
    tokenizer: Any,
    directory: Path,
    metadata: Dict[str, Any],
) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    adapter_dir = directory / "adapter"
    raw_model.lm.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    torch.save(
        {
            "ion_head": raw_model.ion_head.state_dict(),
            "count_head": raw_model.count_head.state_dict(),
            "similarity_projection": (
                raw_model.similarity_projection.state_dict()
            ),
            "metadata": metadata,
        },
        directory / "auxiliary_heads.pt",
    )

    config = {
        "architecture": "IonRecall-FITGraph-V2",
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "hidden_size": raw_model.hidden_size,
        "ion_head_max_mz": ION_HEAD_MAX_MZ,
        "ion_isotope_margin": ION_ISOTOPE_MARGIN,
        "similarity_projection_dim": SIMILARITY_PROJECTION_DIM,
        "ion_probability_threshold": ION_PROBABILITY_THRESHOLD,
        "ion_count_expansion": ION_COUNT_EXPANSION,
        "ion_count_margin": ION_COUNT_MARGIN,
        "max_auxiliary_ion_candidates": MAX_AUXILIARY_ION_CANDIDATES,
        "graph_format": "ion_first_fitgraph_v2",
        "selection_f_beta": SELECTION_F_BETA,
        "metadata": metadata,
    }
    (directory / "model_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_selection(
    model: nn.Module,
    tokenizer: Any,
    selection_records: Sequence[Dict[str, Any]],
    epoch: int,
    output_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    raw_model = unwrap_model(model)
    raw_model.eval()
    raw_model.lm.config.use_cache = True
    tokenizer.padding_side = "left"
    clean_generation_config(raw_model.lm)

    local_indices = list(
        range(rank, len(selection_records), world_size)
    )
    local_rows: List[Dict[str, Any]] = []

    for start in range(
        0,
        len(local_indices),
        SELECTION_BATCH_SIZE_PER_GPU,
    ):
        index_batch = local_indices[
            start : start + SELECTION_BATCH_SIZE_PER_GPU
        ]
        record_batch = [
            selection_records[index] for index in index_batch
        ]
        predicted = predict_records_batch(
            raw_model,
            tokenizer,
            record_batch,
            device,
        )

        for index, item in zip(index_batch, predicted):
            record = selection_records[index]
            local_rows.append(
                {
                    "index": index,
                    "id": record.get("id"),
                    "record": record,
                    "gold_graph": build_fitgraph(record),
                    **item,
                    "infer_result": graph_to_triplets(
                        item["predicted_graph"]
                    ),
                }
            )

        print(
            f"[selection rank={rank}] "
            f"processed={min(start + len(index_batch), len(local_indices))}/"
            f"{len(local_indices)}",
            flush=True,
        )

    epoch_dir = output_dir / "selection" / f"epoch_{epoch:03d}"
    shard_dir = epoch_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        shard_dir / f"rank_{rank:03d}.jsonl",
        local_rows,
    )
    barrier()

    metrics: Optional[Dict[str, Any]] = None
    if is_rank0(rank):
        merged: List[Dict[str, Any]] = []
        for shard_rank in range(world_size):
            merged.extend(
                read_jsonl(
                    shard_dir / f"rank_{shard_rank:03d}.jsonl"
                )
            )
        merged.sort(key=lambda row: int(row["index"]))
        write_jsonl(epoch_dir / "predictions.jsonl", merged)

        metrics = evaluate_predictions(
            merged,
            STRONG_PEAK_THRESHOLD,
        )
        metrics.update(
            macro_ion_fbeta_similarity(
                merged,
                beta=SELECTION_F_BETA,
            )
        )
        add_auxiliary_selection_metrics(merged, metrics)
        metrics["epoch"] = epoch
        metrics["selection_data_file"] = str(
            Path(SELECTION_DATA_FILE).resolve()
        )
        metrics["selection_data_is_not_unbiased_test"] = True

        metric_name = (
            f"macro_product_ion_f{SELECTION_F_BETA:g}"
        )
        metrics["checkpoint_selection_metric"] = metric_name
        metrics["checkpoint_selection_value"] = metrics[metric_name]

        (epoch_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        save_recall_checkpoint(
            raw_model,
            tokenizer,
            epoch_dir / "checkpoint",
            metadata={
                "epoch": epoch,
                "selection_metrics": metrics,
            },
        )

        history_path = output_dir / "selection_history.jsonl"
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(metrics, ensure_ascii=False) + "\n"
            )

        print(
            "[selection] "
            + json.dumps(metrics, ensure_ascii=False, indent=2),
            flush=True,
        )

        shutil.rmtree(shard_dir, ignore_errors=True)

    barrier()
    raw_model.lm.config.use_cache = False
    tokenizer.padding_side = "right"
    raw_model.train()
    return metrics


# ===========================================================================
# Main training
# ===========================================================================

def main() -> None:
    rank, world_size, local_rank, device = distributed_setup()
    configure_torch()
    set_all_seeds(RANDOM_SEED, rank)

    train_path = Path(TRAIN_DATA_FILE).resolve()
    selection_path = Path(SELECTION_DATA_FILE).resolve()
    output_dir = Path(OUTPUT_DIR).resolve()

    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_jsonl(train_path)
    selection_records = read_jsonl(selection_path)
    validate_record_schema(train_records, train_path)
    validate_record_schema(selection_records, selection_path)
    warn_train_selection_overlap(
        train_records,
        selection_records,
        rank,
    )

    model, tokenizer = load_model_and_tokenizer(device)

    examples, graphs, dropped = prepare_training_examples(
        tokenizer,
        train_records,
        rank,
    )
    if not examples:
        raise RuntimeError("No usable training examples")

    positive_frequency_weights = (
        compute_positive_frequency_weights(train_records).to(device)
    )

    if is_rank0(rank):
        prepared_dir = output_dir / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(prepared_dir / "dropped.jsonl", dropped)
        write_jsonl(
            prepared_dir / "train_graphs.jsonl",
            [
                {
                    "record": record,
                    "fitgraph": build_fitgraph(record),
                    "target_dsl": graph_to_dsl(
                        build_fitgraph(record)
                    ),
                }
                for record in train_records
            ],
        )
        torch.save(
            positive_frequency_weights.cpu(),
            prepared_dir / "ion_positive_frequency_weights.pt",
        )

        run_config = {
            key: value
            for key, value in globals().items()
            if key.isupper()
            and isinstance(
                value,
                (str, int, float, bool, type(None)),
            )
        }
        (output_dir / "training_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    barrier()

    dataset = RecallGraphDataset(examples)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=RANDOM_SEED,
            drop_last=False,
        )
        if world_size > 1
        else None
    )

    collator = RecallGraphCollator(
        tokenizer.pad_token_id,
        ION_HEAD_MAX_MZ,
    )

    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "sampler": sampler,
        "shuffle": sampler is None,
        "collate_fn": collator,
        "num_workers": DATALOADER_NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if DATALOADER_NUM_WORKERS > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": DATALOADER_PREFETCH_FACTOR,
            }
        )
    loader = DataLoader(**loader_kwargs)

    if is_rank0(rank) and hasattr(model.lm, "print_trainable_parameters"):
        model.lm.print_trainable_parameters()

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    raw_model = unwrap_model(model)
    lm_parameters = [
        parameter
        for parameter in raw_model.lm.parameters()
        if parameter.requires_grad
    ]
    auxiliary_parameters = [
        *raw_model.ion_head.parameters(),
        *raw_model.count_head.parameters(),
        *raw_model.similarity_projection.parameters(),
    ]

    optimizer_groups = [
        {
            "params": lm_parameters,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": auxiliary_parameters,
            "lr": AUXILIARY_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
    ]
    try:
        optimizer = AdamW(
            optimizer_groups,
            betas=(0.9, 0.95),
            fused=torch.cuda.is_available(),
        )
    except TypeError:
        optimizer = AdamW(
            optimizer_groups,
            betas=(0.9, 0.95),
        )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    updates_per_epoch = math.ceil(
        len(loader) / GRADIENT_ACCUMULATION_STEPS
    )
    total_updates = updates_per_epoch * NUM_TRAIN_EPOCHS
    warmup_updates = int(round(total_updates * WARMUP_RATIO))
    scheduler = cosine_schedule(
        optimizer,
        total_updates,
        warmup_updates,
    )

    best_value = -1.0
    best_recall = -1.0
    global_update = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, NUM_TRAIN_EPOCHS + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_start = time.time()
        running = {
            "graph": 0.0,
            "graph_unweighted": 0.0,
            "ion": 0.0,
            "ion_presence": 0.0,
            "ion_count": 0.0,
            "similarity": 0.0,
            "total": 0.0,
            "micro": 0,
        }

        lambda_ion = ion_lambda(epoch)
        lambda_similarity = similarity_lambda(epoch)

        for micro_step, batch in enumerate(loader, start=1):
            batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }

            should_sync = (
                micro_step % GRADIENT_ACCUMULATION_STEPS == 0
                or micro_step == len(loader)
            )
            group_start = (
                ((micro_step - 1) // GRADIENT_ACCUMULATION_STEPS)
                * GRADIENT_ACCUMULATION_STEPS
                + 1
            )
            group_end = min(
                group_start + GRADIENT_ACCUMULATION_STEPS - 1,
                len(loader),
            )
            accumulation_divisor = group_end - group_start + 1

            sync_context = (
                nullcontext()
                if should_sync or not isinstance(model, DDP)
                else model.no_sync()
            )

            with sync_context:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                graph_loss, graph_unweighted = (
                    graph_weighted_loss_per_molecule(
                        outputs["logits"],
                        batch["labels"],
                        batch["loss_weights"],
                    )
                )

                ion_loss, ion_presence, ion_count = (
                    asymmetric_ion_inventory_loss(
                        outputs["ion_logits"],
                        batch["ion_targets"],
                        batch["ion_valid_mask"],
                        outputs["count_prediction"],
                        batch["ion_count_targets"],
                        positive_frequency_weights,
                    )
                )

                similarity_loss = (
                    graph_similarity_regression_loss(
                        outputs["similarity_repr"],
                        batch["pair_similarity"],
                    )
                )

                total_loss = (
                    graph_loss
                    + lambda_ion * ion_loss
                    + lambda_similarity * similarity_loss
                )
                (
                    total_loss / accumulation_divisor
                ).backward()

            running["graph"] += float(graph_loss.detach().cpu())
            running["graph_unweighted"] += float(
                graph_unweighted.detach().cpu()
            )
            running["ion"] += float(ion_loss.detach().cpu())
            running["ion_presence"] += float(
                ion_presence.detach().cpu()
            )
            running["ion_count"] += float(ion_count.detach().cpu())
            running["similarity"] += float(
                similarity_loss.detach().cpu()
            )
            running["total"] += float(total_loss.detach().cpu())
            running["micro"] += 1

            if should_sync:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    MAX_GRAD_NORM,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

                if (
                    is_rank0(rank)
                    and global_update % LOG_EVERY_UPDATES == 0
                ):
                    denominator = max(1, running["micro"])
                    print(
                        f"[train] epoch={epoch} "
                        f"update={global_update}/{total_updates} "
                        f"total={running['total']/denominator:.4f} "
                        f"graph={running['graph']/denominator:.4f} "
                        f"graph_unweighted="
                        f"{running['graph_unweighted']/denominator:.4f} "
                        f"ion={running['ion']/denominator:.4f} "
                        f"ion_presence="
                        f"{running['ion_presence']/denominator:.4f} "
                        f"ion_count="
                        f"{running['ion_count']/denominator:.4f} "
                        f"similarity="
                        f"{running['similarity']/denominator:.4f} "
                        f"lambda_ion={lambda_ion:.3f} "
                        f"lambda_similarity={lambda_similarity:.3f} "
                        f"lr_lm={scheduler.get_last_lr()[0]:.3e} "
                        f"lr_aux={scheduler.get_last_lr()[1]:.3e}",
                        flush=True,
                    )
                    for key in running:
                        running[key] = 0 if key == "micro" else 0.0

        if is_rank0(rank):
            print(
                f"[epoch] epoch={epoch} "
                f"training_seconds={time.time()-epoch_start:.2f}",
                flush=True,
            )

        scheduled = (
            epoch >= SELECTION_START_EPOCH
            and (
                (epoch - SELECTION_START_EPOCH)
                % SELECTION_EVERY_N_EPOCHS
                == 0
            )
        )
        final = SELECT_FINAL_EPOCH and epoch == NUM_TRAIN_EPOCHS

        if scheduled or final:
            metrics = run_selection(
                model,
                tokenizer,
                selection_records,
                epoch,
                output_dir,
                rank,
                world_size,
                device,
            )

            if is_rank0(rank) and metrics is not None:
                metric_name = (
                    f"macro_product_ion_f{SELECTION_F_BETA:g}"
                )
                value = float(metrics[metric_name])
                recall = float(
                    metrics["macro_product_ion_recall"]
                )

                improved = (
                    value > best_value + 1e-12
                    or (
                        abs(value - best_value) <= 1e-12
                        and recall > best_recall
                    )
                )
                if improved:
                    best_value = value
                    best_recall = recall
                    save_recall_checkpoint(
                        unwrap_model(model),
                        tokenizer,
                        output_dir / "best_checkpoint",
                        metadata={
                            "epoch": epoch,
                            "selection_metric": metric_name,
                            "selection_value": value,
                            "selection_recall": recall,
                            "selection_metrics": metrics,
                            "warning": (
                                "SELECTION_DATA_FILE was used for checkpoint "
                                "selection and is not an unbiased final test."
                            ),
                        },
                    )
                    print(
                        f"[best] epoch={epoch} "
                        f"{metric_name}={value:.6f} "
                        f"product_ion_recall={recall:.6f}",
                        flush=True,
                    )
            barrier()

    barrier()
    if is_rank0(rank):
        save_recall_checkpoint(
            unwrap_model(model),
            tokenizer,
            output_dir / "final_checkpoint",
            metadata={
                "epoch": NUM_TRAIN_EPOCHS,
                "global_updates": global_update,
            },
        )
        (output_dir / "training_complete.json").write_text(
            json.dumps(
                {
                    "epochs": NUM_TRAIN_EPOCHS,
                    "global_updates": global_update,
                    "best_selection_fbeta": best_value,
                    "best_selection_recall": best_recall,
                    "best_checkpoint": str(
                        output_dir / "best_checkpoint"
                    ),
                    "final_checkpoint": str(
                        output_dir / "final_checkpoint"
                    ),
                    "selection_data_warning": (
                        "The selection file is not an unbiased final test."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[done] best checkpoint: "
            f"{output_dir / 'best_checkpoint'}",
            flush=True,
        )

    barrier()


if __name__ == "__main__":
    try:
        main()
    finally:
        safe_destroy_process_group()
