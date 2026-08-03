from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .scoring import TopTokenArtifact
from .taxonomy import Taxonomy


@dataclass(frozen=True)
class RecognitionResult:
    predictions: list[str | None]
    exact_confusion: np.ndarray
    broad_confusion: np.ndarray
    summary: dict[str, float | int]


def predict_from_top_tokens(
    decoded_tokens: np.ndarray,
    emotion_words: set[str],
    rank_limit: int = 100,
) -> list[str | None]:
    """Reproduce the original decode-concatenate-regex-first-match rule."""
    if rank_limit <= 0 or rank_limit > decoded_tokens.shape[1]:
        raise ValueError(f"rank_limit must be in [1, {decoded_tokens.shape[1]}]")

    predictions: list[str | None] = []
    for row in decoded_tokens[:, :rank_limit]:
        word_string = "".join(
            str(token).encode("unicode_escape").decode() for token in row
        )
        words = re.findall(r"'\s*|\w+", word_string)
        predictions.append(next((word for word in words if word in emotion_words), None))
    return predictions


def confusion_matrix(
    ground_truth: list[str], predictions: list[str | None], labels: list[str]
) -> np.ndarray:
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground truth and predictions must have equal length")
    index = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, predicted in zip(ground_truth, predictions):
        if actual not in index:
            raise ValueError(f"Unknown ground-truth label: {actual}")
        if predicted is not None:
            matrix[index[actual], index[predicted]] += 1
    return matrix


def evaluate_recognition(
    artifact: TopTokenArtifact,
    taxonomy: Taxonomy,
    rank_limit: int = 100,
) -> RecognitionResult:
    if artifact.ground_truth is None:
        raise ValueError("Recognition artifact has no ground-truth labels")
    predictions = predict_from_top_tokens(
        artifact.decoded_tokens, set(taxonomy.labels), rank_limit
    )
    exact_confusion = confusion_matrix(
        artifact.ground_truth, predictions, list(taxonomy.labels)
    )
    broad_truth = [taxonomy.primary_by_label[label] for label in artifact.ground_truth]
    broad_predictions = [
        taxonomy.primary_by_label[prediction] if prediction is not None else None
        for prediction in predictions
    ]
    broad_confusion = confusion_matrix(
        broad_truth, broad_predictions, list(taxonomy.primary_labels)
    )

    total = len(artifact.ground_truth)
    covered = sum(prediction is not None for prediction in predictions)
    exact_correct = sum(
        actual == predicted
        for actual, predicted in zip(artifact.ground_truth, predictions)
    )
    broad_correct = sum(
        actual == predicted
        for actual, predicted in zip(broad_truth, broad_predictions)
    )
    exact_evaluated = int(exact_confusion.sum())
    broad_evaluated = int(broad_confusion.sum())
    return RecognitionResult(
        predictions=predictions,
        exact_confusion=exact_confusion,
        broad_confusion=broad_confusion,
        summary={
            "examples": total,
            "covered": covered,
            "coverage": covered / total if total else 0.0,
            # The paper plotting code divides by the confusion-matrix sum, so
            # unmatched rows are excluded from these two primary accuracies.
            "exact_accuracy": (
                exact_correct / exact_evaluated if exact_evaluated else 0.0
            ),
            "broad_accuracy": (
                broad_correct / broad_evaluated if broad_evaluated else 0.0
            ),
            "exact_accuracy_all_examples": (
                exact_correct / total if total else 0.0
            ),
            "broad_accuracy_all_examples": (
                broad_correct / total if total else 0.0
            ),
            "rank_limit": rank_limit,
            "prediction_semantics": "legacy-paper-code",
        },
    )


def _write_matrix(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def write_recognition_outputs(
    results: dict[str, RecognitionResult],
    taxonomy: Taxonomy,
    ground_truth_by_persona: dict[str, list[str]],
    metadata_by_persona: dict[str, dict],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "persona",
        "examples",
        "covered",
        "coverage",
        "exact_accuracy",
        "broad_accuracy",
        "exact_accuracy_all_examples",
        "broad_accuracy_all_examples",
        "rank_limit",
        "prediction_semantics",
    ]
    with (output_dir / "accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for persona, result in results.items():
            writer.writerow({"persona": persona, **result.summary})

    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata_by_persona, handle, indent=2)

    for persona, result in results.items():
        _write_matrix(
            output_dir / f"confusion_135_{persona}.csv",
            result.exact_confusion,
            list(taxonomy.labels),
        )
        _write_matrix(
            output_dir / f"confusion_6_{persona}.csv",
            result.broad_confusion,
            list(taxonomy.primary_labels),
        )
        with (output_dir / f"predictions_{persona}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["ground_truth", "prediction"])
            writer.writerows(
                zip(ground_truth_by_persona[persona], result.predictions)
            )

    personas = list(results)
    exact = [results[name].summary["exact_accuracy"] for name in personas]
    broad = [results[name].summary["broad_accuracy"] for name in personas]
    x = np.arange(len(personas))
    figure, axis = plt.subplots(figsize=(max(6, len(personas) * 0.8), 4))
    axis.bar(x - 0.18, exact, 0.36, label="135-way")
    axis.bar(x + 0.18, broad, 0.36, label="6-way")
    axis.set_xticks(x, personas, rotation=35, ha="right")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "accuracy.pdf", bbox_inches="tight")
    plt.close(figure)
