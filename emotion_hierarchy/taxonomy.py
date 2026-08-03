from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Taxonomy:
    labels: tuple[str, ...]
    primary_by_label: dict[str, str]
    primary_labels: tuple[str, ...]


def load_taxonomy(path: str | Path) -> Taxonomy:
    """Load the ordered Shaver et al. taxonomy used by the experiments."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)

    primary_labels = tuple(raw["primary_emotions"])
    children = raw["children"]
    labels = list(primary_labels)
    primary_by_label = {label: label for label in primary_labels}

    for primary in primary_labels:
        group = children[primary]
        labels.extend(group)
        primary_by_label.update({label: primary for label in group})

    if len(labels) != 135:
        raise ValueError(f"Expected 135 emotion labels, found {len(labels)}")
    if len(set(labels)) != len(labels):
        raise ValueError("Emotion labels must be unique")

    return Taxonomy(tuple(labels), primary_by_label, primary_labels)
