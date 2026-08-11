#!/usr/bin/env python3
"""Direct Qwen-VLM baseline using only whole-slide overview thumbnails."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiscale_vqa_agent.clients import OpenAICompatibleClient, parse_json_response
from multiscale_vqa_agent.live_metrics import LiveAccuracyTracker


ANSWERABILITY_SYSTEM_PROMPT = """You are the answerability gate for a direct breast pathology WSI baseline. Decide whether the supplied whole-slide overview thumbnail(s) contain enough information to give a reasonably grounded answer to the question. Use only the question, choices, and images.

Return JSON only: {"can_answer":true,"confidence":0.0,"reason":"one short sentence"}. can_answer must be a JSON boolean, never a string.

Set can_answer=true for targets reasonably supported by the available WSI morphology, including diagnosis, histologic type/subtype/grade, visible morphology, necrosis, in-situ disease, microcalcification, lymphovascular invasion, and morphology-linked categorical phenotypes when the images provide reasonable predictive evidence.

Set can_answer=false for targets requiring exact measurements, age, treatment, procedure, clinical history, follow-up, specimen metadata or orientation, report-only information, exact assay measurements, or information clearly unavailable from the supplied thumbnails.

Judge the information available to this direct thumbnail baseline, not whether the question is generally easy. Do not answer the multiple-choice question in this stage."""


ANSWER_SYSTEM_PROMPT = """You are answering a breast pathology multiple-choice question using only whole-slide overview thumbnails from one patient.
The images are low-resolution H&E overviews, so do not claim findings that are not visibly supported. You must still choose the most defensible supplied option.
Return JSON only: {"answer_id":"A","confidence":0.0,"explanation":"one short sentence","limitations":"one short sentence"}.
answer_id must be exactly one supplied option ID. Do not return the option text as answer_id, refuse, or return null."""


def indexed_choices(choices: Sequence[Any]) -> List[Dict[str, str]]:
    if len(choices) > 26:
        raise ValueError("At most 26 choices are supported")
    return [
        {"id": chr(ord("A") + index), "text": str(choice)}
        for index, choice in enumerate(choices)
    ]


def parse_answer(raw: Optional[str], choices: Sequence[Any]) -> Optional[Dict[str, Any]]:
    """Parse only an explicit supplied option ID; never guess a default choice."""
    options = {row["id"]: row["text"] for row in indexed_choices(choices)}
    parsed = parse_json_response(raw)
    answer_id = parsed.get("answer_id") if isinstance(parsed, dict) else None
    if not isinstance(answer_id, str):
        match = re.search(r'["\']answer_id["\']\s*:\s*["\']([A-Z])["\']', raw or "", re.I)
        answer_id = match.group(1) if match else None
    answer_id = answer_id.strip().upper() if isinstance(answer_id, str) else None
    if answer_id not in options:
        return None
    return {
        "answer_id": answer_id,
        "answer": options[answer_id],
        "confidence": parsed.get("confidence") if isinstance(parsed, dict) else None,
        "explanation": parsed.get("explanation", "") if isinstance(parsed, dict) else "",
        "limitations": parsed.get("limitations", "") if isinstance(parsed, dict) else "",
    }


def parse_answerability(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    parsed = parse_json_response(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("can_answer"), bool):
        return None
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "can_answer": parsed["can_answer"],
        "confidence": max(0.0, min(confidence, 1.0)),
        "reason": str(parsed.get("reason", "")).strip(),
        "fallback_used": False,
    }


class WSIThumbnailIndex:
    def __init__(self, wsi_root: Path, cache_root: Path, size: int, max_slides: int):
        self.wsi_root = Path(wsi_root)
        self.cache_root = Path(cache_root)
        self.size = int(size)
        self.max_slides = int(max_slides)
        self._by_case: Optional[Dict[str, List[Path]]] = None

    def thumbnails(self, case_id: str) -> Tuple[List[str], List[str]]:
        slides = self._slides(case_id)[: self.max_slides]
        targets = [self._make_thumbnail(case_id, slide) for slide in slides]
        return [str(path) for path in targets], [str(path) for path in slides]

    def _slides(self, case_id: str) -> List[Path]:
        if self._by_case is None:
            self._by_case = {}
            for path in self.wsi_root.rglob("*.svs"):
                self._by_case.setdefault(path.name[:12], []).append(path)
        candidates = self._by_case.get(case_id, [])
        return sorted(candidates, key=self._slide_priority)

    @staticmethod
    def _slide_priority(path: Path) -> Tuple[int, str]:
        name = path.name.upper()
        # Diagnostic slides are the fairest first choice for a pathology VLM baseline.
        priority = 0 if re.search(r"-DX\d", name) else 1 if "-TS" in name else 2
        return priority, name

    def _make_thumbnail(self, case_id: str, slide_path: Path) -> Path:
        target_dir = self.cache_root / case_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{slide_path.stem}.overview_{self.size}.jpg"
        if target.exists() and target.stat().st_size > 0:
            return target
        try:
            import openslide
        except ImportError as error:
            raise RuntimeError("openslide-python is required to create WSI thumbnails") from error
        with openslide.OpenSlide(str(slide_path)) as slide:
            image = slide.get_thumbnail((self.size, self.size)).convert("RGB")
        image.save(target, "JPEG", quality=90)
        return target


def completed_keys(path: Path) -> set:
    keys = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((str(row.get("case_id", "")), str(row.get("question", ""))))
    return keys


def evidence_packet(question: str, choices: Sequence[Any], image_count: int) -> Dict[str, Any]:
    return {
        "question": question,
        "choices": indexed_choices(choices),
        "image_context": f"{image_count} whole-slide overview thumbnail(s) from the same patient",
    }


def format_answerability_prompt(
    question: str, choices: Sequence[Any], image_count: int
) -> str:
    packet = evidence_packet(question, choices, image_count)
    packet["output_schema"] = {
        "can_answer": True,
        "confidence": 0.0,
        "reason": "one short sentence",
    }
    return "Decide answerability only.\n" + json.dumps(packet, ensure_ascii=False)


def format_user_prompt(question: str, choices: Sequence[Any], image_count: int) -> str:
    packet = evidence_packet(question, choices, image_count)
    return "Select one option ID.\n" + json.dumps(packet, ensure_ascii=False)


def save_result(handle: Any, row: Dict[str, Any], tracker: LiveAccuracyTracker):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    evaluation = tracker.update(row)
    snapshot = tracker.snapshot()
    accuracy = snapshot["accuracy"]
    print(
        f"direct_qwen_wsi processed={snapshot['processed']}/{snapshot['selected_total']} "
        f"correct={snapshot['correct']} errors={snapshot['errors']} "
        f"acc={'nan' if accuracy is None else f'{accuracy:.4f}'} "
        f"last_correct={evaluation['correct']}",
        flush=True,
    )


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = Path(args.vqa_json or config["vqa_json"])
    items = [row for row in json.loads(source.read_text(encoding="utf-8")) if row.get("Choice", row.get("choices"))]
    if args.limit is not None:
        items = items[: args.limit]

    destination = Path(
        args.output
        or Path(config["output_dir"]) / "qwen_wsi_selective" / "mc_answers.jsonl"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics) if args.metrics else destination.with_name("mc_answers_metrics.json")
    history_path = metrics_path.with_name(f"{metrics_path.stem}_history.csv")
    cache_root = Path(args.thumbnail_dir or destination.parent / "thumbnails")
    completed = completed_keys(destination) if args.resume else set()
    mode = "a" if args.resume and destination.exists() else "w"
    tracker = LiveAccuracyTracker(
        metrics_path,
        history_path,
        selected_total=len(items),
        existing_answers=destination if args.resume else None,
    )
    client = OpenAICompatibleClient(config["qwen"])
    thumbnails = WSIThumbnailIndex(Path(config["wsi_root"]), cache_root, args.thumbnail_size, args.max_slides)

    pending = [
        item for item in items
        if (str(item.get("Id", item.get("case_id", "")))[:12], str(item.get("Question", item.get("question", "")))) not in completed
    ]
    print(f"direct_qwen_wsi selected={len(items)} completed={len(completed)} remaining={len(pending)}", flush=True)
    with destination.open(mode, encoding="utf-8") as handle:
        for item in pending:
            case_id = str(item.get("Id", item.get("case_id", "")))[:12]
            question = str(item.get("Question", item.get("question", "")))
            choices = list(item.get("Choice", item.get("choices", [])))
            row: Dict[str, Any] = {
                "case_id": case_id,
                "question": question,
                "choices": choices,
                "choice_options": indexed_choices(choices),
                "reference_answer": item.get("Answer", item.get("answer")),
                "plan": {"supported": False, "task_match": "direct_qwen_wsi", "target_phenotypes": []},
                "task_match": "direct_qwen_wsi",
            }
            try:
                image_paths, slide_paths = thumbnails.thumbnails(case_id)
                if not image_paths:
                    raise FileNotFoundError(f"No WSI found for {case_id}")
                gate_raw = client.chat(
                    system=ANSWERABILITY_SYSTEM_PROMPT,
                    user=format_answerability_prompt(
                        question, choices, len(image_paths)
                    ),
                    images=image_paths,
                    temperature=0.0,
                    max_tokens=160,
                    response_format={"type": "json_object"},
                    retries=2,
                )
                assessment = parse_answerability(gate_raw)
                if assessment is None:
                    assessment = {
                        "can_answer": True,
                        "confidence": 0.0,
                        "reason": "Answerability response was invalid; continuing without abstention.",
                        "fallback_used": True,
                    }
                row.update({
                    "thumbnail_paths": image_paths,
                    "wsi_paths": slide_paths,
                    "predicted_can_answer": assessment["can_answer"],
                    "predicted_answerability": (
                        "answerable" if assessment["can_answer"] else "unanswerable"
                    ),
                    "answerability_confidence": assessment["confidence"],
                    "answerability_reason": assessment["reason"],
                    "answerability_fallback_used": assessment["fallback_used"],
                    "answerability_raw_response": gate_raw,
                    "abstained": not assessment["can_answer"],
                })
                if not assessment["can_answer"]:
                    row.update({
                        "plan": {},
                        "task_match": "answerability_abstain",
                        "agent_answer": None,
                        "json_parse_success": True,
                    })
                    save_result(handle, row, tracker)
                    continue

                raw = client.chat(
                    system=ANSWER_SYSTEM_PROMPT,
                    user=format_user_prompt(question, choices, len(image_paths)),
                    images=image_paths,
                    temperature=0.0,
                    max_tokens=256,
                    response_format={"type": "json_object"},
                    retries=2,
                )
                answer = parse_answer(raw, choices)
                if answer is None:
                    raise ValueError("Qwen response did not contain a valid supplied answer_id")
                row.update({
                    "raw_response": raw,
                    "agent_answer": {**answer, "json_parse_success": True},
                    "json_parse_success": True,
                })
            except Exception as error:
                row.setdefault("thumbnail_paths", [])
                row.setdefault("wsi_paths", [])
                row.setdefault("predicted_can_answer", True)
                row.setdefault("predicted_answerability", "answerable")
                row.setdefault("answerability_confidence", 0.0)
                row.setdefault(
                    "answerability_reason",
                    "Baseline processing failed; retaining a non-rejecting fallback.",
                )
                row.setdefault("answerability_fallback_used", True)
                row.setdefault("abstained", False)
                row.update({
                    "error": f"{type(error).__name__}: {error}",
                    "json_parse_success": False,
                })
            save_result(handle, row, tracker)
    tracker.save_snapshot()
    if args.answerability_labels:
        from multiscale_vqa_agent.answerability_evaluation import evaluate_answerability

        summary = evaluate_answerability(
            destination, Path(args.answerability_labels)
        )
        print(
            f"Qwen selective evaluation: {json.dumps(summary, ensure_ascii=False)}",
            flush=True,
        )
    return destination


def build_parser() -> argparse.ArgumentParser:
    agent_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Selective direct Qwen-VLM baseline on whole-slide thumbnails"
    )
    parser.add_argument("--config", default=str(agent_dir / "config.servers.json"))
    parser.add_argument("--vqa_json", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument(
        "--answerability_labels",
        default=None,
        help="Optional binary Gold labels read only after inference.",
    )
    parser.add_argument("--thumbnail_dir", default=None)
    parser.add_argument("--thumbnail_size", type=int, default=1536)
    parser.add_argument("--max_slides", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser


def main():
    output = run(build_parser().parse_args())
    print(f"Saved direct Qwen WSI answers: {output}")


if __name__ == "__main__":
    main()
