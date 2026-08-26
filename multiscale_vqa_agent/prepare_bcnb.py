#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


SCALES = (1024, 2048, 4096)
CHOICE_IDS = ("A", "B", "C", "D")


def normalize_slide_id(value: str) -> str:
    value = str(value).strip()
    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]
    if not value:
        raise ValueError("Empty Slide value")
    return value


def convert_vqa_csv(source: Path, destination: Path):
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"ID", "Slide", "Task", "Question", "Answer", *CHOICE_IDS}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"BCNB CSV is missing columns: {sorted(missing)}")

    converted = []
    for row_index, row in enumerate(rows):
        choices = [str(row[key]).strip() for key in CHOICE_IDS if str(row[key]).strip()]
        answer_id = str(row["Answer"]).strip().upper()
        if answer_id not in CHOICE_IDS:
            raise ValueError(f"Row {row_index} has invalid Answer {answer_id!r}")
        answer_text = str(row[answer_id]).strip()
        if not answer_text or answer_text not in choices:
            raise ValueError(f"Row {row_index} answer is not a supplied choice")
        converted.append({
            "Id": normalize_slide_id(row["Slide"]),
            "QuestionId": str(row["ID"]).strip(),
            "Task": str(row["Task"]).strip(),
            "Question": str(row["Question"]).strip(),
            "Choice": choices,
            "Answer": answer_text,
            "AnswerId": answer_id,
        })

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return converted


def write_manifests(rows, feature_root: Path, output_dir: Path):
    slide_ids = sorted(
        {normalize_slide_id(row["Id"]) for row in rows},
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for scale in SCALES:
        feature_dir = feature_root / str(scale)
        records = []
        for slide_id in slide_ids:
            path = feature_dir / f"{slide_id}_0_{scale}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing scale-{scale} feature: {path}")
            payload = np.load(path, allow_pickle=True).item()
            features = np.asarray(payload.get("feature"))
            coordinates = payload.get("index")
            if features.ndim != 2 or features.shape[1] != 768:
                raise ValueError(f"Unexpected feature shape in {path}: {features.shape}")
            if coordinates is None or len(coordinates) != len(features):
                raise ValueError(f"Feature/coordinate count mismatch in {path}")
            records.append({
                "case_id": slide_id,
                "slide_id": path.stem,
                "feature_path": str(path.resolve()),
                "split": "external",
            })

        manifest_path = output_dir / f"aligned_manifest_{scale}.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        manifests[str(scale)] = str(manifest_path.resolve())
    return manifests


def write_config(
    base_config: Path,
    destination: Path,
    vqa_json: Path,
    wsi_root: Path,
    manifests,
    question_features: Path,
    output_dir: Path,
):
    config = json.loads(base_config.read_text(encoding="utf-8"))
    config["vqa_json"] = str(vqa_json.resolve())
    config["wsi_root"] = str(wsi_root.resolve())
    config["output_dir"] = str(output_dir.resolve())
    config["scale_manifests"] = manifests
    config["retrieval"]["question_feature_path"] = str(question_features.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root.parent / "dataset"
    default_data_dir = data_root / "SlideBench-VQA-BCNB-adapted"
    parser = argparse.ArgumentParser(description="Prepare BCNB for the multiscale VQA agent")
    parser.add_argument("--csv", default=str(data_root / "SlideBench-VQA-BCNB.csv"))
    parser.add_argument(
        "--feature_root",
        default="/data_nas3/wl/BCNB_handle/features/conch_v1_5_new",
    )
    parser.add_argument(
        "--wsi_root",
        default="/data_local/public_files/public_datasets/zd/BCNB/WSIs",
    )
    parser.add_argument("--output_dir", default=str(default_data_dir))
    parser.add_argument(
        "--base_config",
        default=str(project_root / "multiscale_vqa_agent/config.servers.json"),
    )
    parser.add_argument(
        "--config_output",
        default=str(project_root / "multiscale_vqa_agent/config.bcnb.servers.json"),
    )
    parser.add_argument(
        "--question_features",
        default=str(default_data_dir / "SlideBench-VQA-BCNB_conch_v1_question_features.npz"),
    )
    parser.add_argument(
        "--run_output",
        default="/data_nas3/wl/BCNB_VQA_outputs/bcnb_live_metrics_v1",
    )
    args = parser.parse_args()

    source = Path(args.csv)
    feature_root = Path(args.feature_root)
    wsi_root = Path(args.wsi_root)
    output_dir = Path(args.output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not wsi_root.is_dir():
        raise FileNotFoundError(wsi_root)

    vqa_json = output_dir / "SlideBench-VQA-BCNB.json"
    rows = convert_vqa_csv(source, vqa_json)
    manifests = write_manifests(rows, feature_root, output_dir)
    write_config(
        Path(args.base_config),
        Path(args.config_output),
        vqa_json,
        wsi_root,
        manifests,
        Path(args.question_features),
        Path(args.run_output),
    )
    print(f"prepared questions={len(rows)} slides={len({row['Id'] for row in rows})}")
    print(f"VQA JSON: {vqa_json}")
    print(f"manifests: {json.dumps(manifests, ensure_ascii=False)}")
    print(f"config: {args.config_output}")


if __name__ == "__main__":
    main()
