#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiscale_vqa_agent.mc_pipeline import MultipleChoiceVQAPipeline


def main():
    agent_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run multiple-choice WSI VQA with live accuracy")
    parser.add_argument("--config", default=str(agent_dir / "config.servers.json"))
    parser.add_argument("--vqa_json", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--answerability_only", action="store_true")
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument(
        "--answerability_labels",
        default=None,
        help="Optional Gold labels used only after inference for evaluation.",
    )
    args = parser.parse_args()

    pipeline = MultipleChoiceVQAPipeline(
        args.config, answerability_only=args.answerability_only
    )
    output = pipeline.run_multiple_choice(
        vqa_path=args.vqa_json,
        output_path=args.output,
        metrics_path=args.metrics,
        limit=args.limit,
        crop_patches=not args.no_crop,
        resume=not args.no_resume,
        answerability_labels=args.answerability_labels,
    )
    print(f"Saved answers: {output}")


if __name__ == "__main__":
    main()
