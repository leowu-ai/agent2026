import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .fusion_evidence import indexed_choices
from .live_metrics import LiveAccuracyTracker
from .pipeline import MultiScaleVQAPipeline


class MultipleChoiceVQAPipeline(MultiScaleVQAPipeline):
    """Multiple-choice runner with resumable, per-question live accuracy."""

    def run_multiple_choice(
        self,
        vqa_path: Optional[str] = None,
        output_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        limit: Optional[int] = None,
        crop_patches: bool = True,
        resume: bool = True,
        answerability_labels: Optional[str] = None,
        comparison_answers: Optional[str] = None,
    ) -> Path:
        if self.answerability_only:
            return super().run(
                vqa_path=vqa_path,
                output_path=output_path,
                limit=limit,
                resume=resume,
                multiple_choice_only=True,
                answerability_labels=answerability_labels,
            )
        source = Path(vqa_path or self.config["vqa_json"])
        with source.open(encoding="utf-8") as handle:
            all_items = json.load(handle)
        items = [item for item in all_items if item.get("Choice", item.get("choices"))]
        if limit is not None:
            items = items[:limit]

        destination = Path(
            output_path or (Path(self.config["output_dir"]) / "mc_answers.jsonl")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path = Path(metrics_path) if metrics_path else destination.with_name(
            f"{destination.stem}_metrics.json"
        )
        history_path = snapshot_path.with_name(f"{snapshot_path.stem}_history.csv")
        completed = self._completed_keys(destination) if resume else set()
        mode = "a" if resume and destination.exists() else "w"
        tracker = LiveAccuracyTracker(
            snapshot_path,
            history_path,
            selected_total=len(items),
            existing_answers=destination if resume else None,
        )

        grouped: Dict[str, List[Any]] = defaultdict(list)
        for item in items:
            case_id = str(item.get("Id", item.get("case_id", "")))[:12]
            question = str(item.get("Question", item.get("question", "")))
            if (case_id, question) not in completed:
                grouped[case_id].append(item)

        print(
            f"multiple_choice selected={len(items)} completed={len(completed)} "
            f"remaining={sum(len(rows) for rows in grouped.values())}",
            flush=True,
        )
        with destination.open(mode, encoding="utf-8") as handle:
            for case_number, (case_id, case_items) in enumerate(grouped.items(), 1):
                print(
                    f"[{case_number}/{len(grouped)}] plan {case_id} "
                    f"({len(case_items)} MC questions)",
                    flush=True,
                )
                planned = [
                    (item, self.planner.plan(self._planner_item(item)))
                    for item in case_items
                ]

                print(
                    f"infer {case_id} ({len(planned)} MC questions)",
                    flush=True,
                )
                try:
                    scale_results = self.g2p.infer_case(case_id)
                except Exception as error:
                    for item, plan in planned:
                        self._save_mc_result(
                            handle, self._error_result(item, plan, error), tracker
                        )
                    continue

                evidence_cache = {}
                for item, plan in planned:
                    try:
                        result = self._run_question(
                            item, plan, scale_results, evidence_cache, crop_patches
                        )
                    except Exception as error:
                        result = self._error_result(item, plan, error)
                    self._save_mc_result(handle, result, tracker)

                del scale_results, evidence_cache
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        tracker.save_snapshot()
        self._evaluate_if_requested(destination, answerability_labels)
        if comparison_answers:
            if not answerability_labels:
                raise ValueError(
                    "--comparison_answers requires --answerability_labels for "
                    "post-inference transition analysis"
                )
            from .gate_run_analysis import write_gate_run_analysis

            summary = write_gate_run_analysis(
                destination, Path(comparison_answers), Path(answerability_labels)
            )
            print(
                "Gate transition analysis: "
                + json.dumps(summary, ensure_ascii=False),
                flush=True,
            )
            from .full_run_analysis import write_full_run_analysis

            full_summary = write_full_run_analysis(
                destination, Path(comparison_answers), Path(answerability_labels)
            )
            print(
                "Full run comparison analysis: "
                + json.dumps(full_summary, ensure_ascii=False),
                flush=True,
            )
        return destination

    @staticmethod
    def _error_result(item: Dict[str, Any], plan: Any, error: Exception) -> Dict[str, Any]:
        choices = list(item.get("Choice", item.get("choices", [])) or [])
        return {
            "case_id": plan.case_id,
            "question": plan.question,
            "choices": choices,
            "choice_options": indexed_choices(choices),
            "reference_answer": item.get("Answer", item.get("answer")),
            "input": item,
            "plan": plan.to_dict(),
            "error": f"{type(error).__name__}: {error}",
        }

    @staticmethod
    def _save_mc_result(handle: Any, result: Dict[str, Any], tracker: LiveAccuracyTracker):
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()
        evaluation = tracker.update(result)
        snapshot = tracker.snapshot()
        result_accuracy = snapshot["accuracy"]
        supported_accuracy = snapshot["supported_accuracy"]
        accuracy_text = "nan" if result_accuracy is None else f"{result_accuracy:.4f}"
        supported_text = "nan" if supported_accuracy is None else f"{supported_accuracy:.4f}"
        print(
            f"live_mc processed={snapshot['processed']}/{snapshot['selected_total']} "
            f"correct={snapshot['correct']} errors={snapshot['errors']} "
            f"acc={accuracy_text} supported_acc={supported_text} "
            f"last_correct={evaluation['correct']}",
            flush=True,
        )
