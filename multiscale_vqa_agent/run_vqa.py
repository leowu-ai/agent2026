#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiscale_vqa_agent.pipeline import MultiScaleVQAPipeline


def main():
    parser = argparse.ArgumentParser(description="Run relation-guided multi-scale WSI VQA")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--vqa_json", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--planner_only", action="store_true")
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()
    pipeline = MultiScaleVQAPipeline(args.config, planner_only=args.planner_only)
    output = pipeline.run(
        vqa_path=args.vqa_json,
        output_path=args.output,
        limit=args.limit,
        crop_patches=not args.no_crop,
        resume=not args.no_resume,
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
