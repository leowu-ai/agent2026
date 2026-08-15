#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def normalize_question(text: str) -> str:
    return " ".join(str(text).strip().split())


def checkpoint_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def encode_text_batch(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return CONCH v1 projected and pre-projection pooled text features."""
    text_tower = model.text
    projection = text_tower.text_projection
    if projection is None:
        raise RuntimeError("CONCH text tower has no text_projection parameter")

    # Official CoCa.encode_text removes the final padded token to make room for CLS.
    input_ids = token_ids[:, :-1]
    text_tower.text_projection = None
    try:
        tower_output = text_tower(input_ids)
    finally:
        text_tower.text_projection = projection
    preprojection = (
        tower_output[0] if isinstance(tower_output, tuple) else tower_output
    )
    projected = preprojection @ projection
    return (
        F.normalize(projected.float(), dim=-1),
        F.normalize(preprojection.float(), dim=-1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CONCH v1 text features for WSI VQA questions"
    )
    parser.add_argument(
        "--vqa_json",
        default="/home/wl/agent_2026/dataset/WsiVQA_test.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="/data_nas2/zd/paper1/models/conch/pytorch_model.bin",
    )
    parser.add_argument(
        "--output",
        default=(
            "/home/wl/agent_2026/dataset/"
            "WsiVQA_conch_v1_question_features.npz"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mc_only", action="store_true")
    args = parser.parse_args()

    from conch.open_clip_custom import (
        create_model_from_pretrained,
        get_tokenizer,
        tokenize,
    )

    source = Path(args.vqa_json)
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(f"VQA dataset not found: {source}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CONCH checkpoint not found: {checkpoint}")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("VQA JSON must contain a list")
    if args.mc_only:
        rows = [row for row in rows if row.get("Choice", row.get("choices"))]

    questions: List[str] = []
    case_ids: List[str] = []
    is_multiple_choice: List[bool] = []
    for index, row in enumerate(rows):
        question = normalize_question(row.get("Question", row.get("question", "")))
        if not question:
            raise ValueError(f"VQA row {index} has an empty question")
        questions.append(question)
        case_ids.append(str(row.get("Id", row.get("case_id", "")))[:12])
        is_multiple_choice.append(bool(row.get("Choice", row.get("choices"))))

    unique_questions = list(dict.fromkeys(questions))
    unique_lookup = {question: index for index, question in enumerate(unique_questions)}
    question_to_unique = np.asarray(
        [unique_lookup[question] for question in questions], dtype=np.int64
    )

    device = torch.device(args.device)
    model, _ = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path=str(checkpoint),
        device=device,
    )
    model.eval()
    tokenizer = get_tokenizer()

    projected_batches = []
    preprojection_batches = []
    with torch.inference_mode():
        for start in range(0, len(unique_questions), args.batch_size):
            texts = unique_questions[start:start + args.batch_size]
            token_ids = tokenize(tokenizer, texts).to(device)
            projected, preprojection = encode_text_batch(model, token_ids)
            projected_batches.append(projected.cpu().numpy())
            preprojection_batches.append(preprojection.cpu().numpy())
            print(
                f"encoded {min(start + len(texts), len(unique_questions))}/"
                f"{len(unique_questions)} unique questions",
                flush=True,
            )

    unique_projected = np.concatenate(projected_batches).astype(np.float32)
    unique_preprojection = np.concatenate(preprojection_batches).astype(np.float32)
    if unique_projected.shape != (len(unique_questions), 512):
        raise RuntimeError(f"Unexpected projected shape: {unique_projected.shape}")
    if unique_preprojection.shape != (len(unique_questions), 768):
        raise RuntimeError(
            f"Unexpected pre-projection shape: {unique_preprojection.shape}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        unique_questions=np.asarray(unique_questions),
        question_to_unique=question_to_unique,
        case_ids=np.asarray(case_ids),
        questions=np.asarray(questions),
        is_multiple_choice=np.asarray(is_multiple_choice, dtype=np.bool_),
        projected_512=unique_projected,
        preprojection_768=unique_preprojection,
    )

    index_path = output.with_name(f"{output.stem}_index.csv")
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "row_index", "case_id", "question", "unique_feature_index",
                "is_multiple_choice",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for index, (case_id, question, unique_index, is_mc) in enumerate(zip(
            case_ids, questions, question_to_unique, is_multiple_choice
        )):
            writer.writerow({
                "row_index": index,
                "case_id": case_id,
                "question": question,
                "unique_feature_index": int(unique_index),
                "is_multiple_choice": bool(is_mc),
            })

    metadata = {
        "source_vqa_json": str(source.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "model": "CONCH v1 conch_ViT-B-16",
        "row_count": len(rows),
        "unique_question_count": len(unique_questions),
        "mc_only": bool(args.mc_only),
        "features": {
            "projected_512": {
                "shape": list(unique_projected.shape),
                "l2_normalized": True,
                "meaning": "Official CONCH v1 image-text contrastive space.",
            },
            "preprojection_768": {
                "shape": list(unique_preprojection.shape),
                "l2_normalized": True,
                "meaning": (
                    "CONCH v1 pooled text state before text_projection. It is not "
                    "trained or validated against CONCH v1.5 patch features."
                ),
            },
        },
        "mapping": (
            "question_to_unique maps each VQA row to a row in projected_512 and "
            "preprojection_768."
        ),
    }
    metadata_path = output.with_name(f"{output.stem}_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved features: {output}")
    print(f"saved index: {index_path}")
    print(f"saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
