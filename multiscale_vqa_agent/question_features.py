import re
from pathlib import Path

import numpy as np


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


class QuestionFeatureStore:
    """Strict lookup for frozen CONCH v1 question features."""

    FEATURE_NAME = "CONCH_v1_preprojection_768"

    def __init__(self, path: str):
        self.path = Path(path)
        with np.load(self.path) as payload:
            if "unique_questions" not in payload or "preprojection_768" not in payload:
                raise ValueError(
                    "Question feature NPZ must contain unique_questions and "
                    "preprojection_768"
                )
            questions = [normalize_question(value) for value in payload["unique_questions"]]
            features = np.asarray(payload["preprojection_768"], dtype=np.float32)

        if features.ndim != 2 or features.shape[1] != 768:
            raise ValueError(
                f"Expected question features with shape [N, 768], got {features.shape}"
            )
        if len(questions) != features.shape[0]:
            raise ValueError(
                "Question count does not match preprojection_768 rows: "
                f"{len(questions)} != {features.shape[0]}"
            )
        if not np.isfinite(features).all():
            raise ValueError("Question features contain NaN or Inf")
        if any(not question for question in questions):
            raise ValueError("Question feature store contains an empty question")
        if len(set(questions)) != len(questions):
            raise ValueError("Question feature store contains duplicate normalized questions")

        norms = np.linalg.norm(features, axis=1)
        if np.any(norms <= 1e-8):
            raise ValueError("Question feature store contains a zero-norm feature")
        self.features = features / norms[:, None]
        self.question_to_index = {
            question: index for index, question in enumerate(questions)
        }
        print(
            f"question_features count={len(questions)} dim={features.shape[1]} "
            f"source={self.FEATURE_NAME}",
            flush=True,
        )

    def lookup(self, question: str) -> np.ndarray:
        normalized = normalize_question(question)
        try:
            return self.features[self.question_to_index[normalized]].copy()
        except KeyError as error:
            raise ValueError(
                f"Question not found in frozen feature store: {question!r}"
            ) from error

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def __len__(self) -> int:
        return int(self.features.shape[0])
