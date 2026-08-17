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
    parser.add_argument("--answerability_only", action="store_true")
    parser.add_argument(
        "--precomputed_answerability",
        default=None,
        help=(
            "Optional frozen answerability JSONL. When provided, online "
            "AnswerabilityAgent calls are disabled and missing keys are fatal."
        ),
    )
    parser.add_argument("--mc_only", action="store_true")
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument(
        "--agent_mode",
        choices=("legacy", "hierarchical_rag"),
        default="legacy",
        help="Evidence acquisition mode. Defaults to the unchanged legacy pipeline.",
    )
    parser.add_argument(
        "--knowledge_base",
        default=None,
        help="Knowledge-base ZIP required by --agent_mode hierarchical_rag.",
    )
    parser.add_argument(
        "--morphology_retrieval_mode",
        choices=("broad", "question_similarity"),
        default=None,
        help="Optional override for morphology-only patch retrieval.",
    )
    parser.add_argument(
        "--partial_retrieval_mode",
        choices=(
            "selected_phenotype", "question_similarity",
            "hybrid_question_prototype",
        ),
        default=None,
        help="Optional override for partial-route visual patch retrieval.",
    )
    parser.add_argument(
        "--direct_retrieval_mode",
        choices=(
            "selected_phenotype", "question_similarity",
            "hybrid_question_prototype",
        ),
        default=None,
        help="Optional override for direct-route visual patch retrieval.",
    )
    parser.add_argument(
        "--answerability_labels",
        default=None,
        help="Optional Gold labels used only after inference for evaluation.",
    )
    args = parser.parse_args()
    pipeline = MultiScaleVQAPipeline(
        args.config,
        planner_only=args.planner_only,
        answerability_only=args.answerability_only,
        precomputed_answerability=args.precomputed_answerability,
        morphology_retrieval_mode=args.morphology_retrieval_mode,
        partial_retrieval_mode=args.partial_retrieval_mode,
        direct_retrieval_mode=args.direct_retrieval_mode,
        agent_mode=args.agent_mode,
        knowledge_base=args.knowledge_base,
    )
    output = pipeline.run(
        vqa_path=args.vqa_json,
        output_path=args.output,
        limit=args.limit,
        crop_patches=not args.no_crop,
        resume=not args.no_resume,
        multiple_choice_only=args.mc_only,
        answerability_labels=args.answerability_labels,
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
