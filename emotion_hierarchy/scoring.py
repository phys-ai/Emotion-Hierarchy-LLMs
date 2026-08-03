from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


HIERARCHY_SUFFIX = " The emotion in this sentence is"
RECOGNITION_SUFFIX = "I think the emotion involved in this situation is"
ARTIFACT_FORMAT = "emotion-hierarchy-top-tokens-v1"


@dataclass(frozen=True)
class TopTokenArtifact:
    """The compact equivalent of the full-vocabulary logits used originally."""

    probabilities: np.ndarray
    token_ids: np.ndarray
    decoded_tokens: np.ndarray
    metadata: dict
    ground_truth: list[str] | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_lines(path: str | Path, limit: int | None = None) -> list[str]:
    with Path(path).open(encoding="utf-8") as handle:
        # The original input has three empty rows and the original script scored
        # them. Preserve physical records rather than silently filtering them.
        lines = [line.rstrip("\r\n") for line in handle]
    return lines if limit is None else lines[:limit]


def read_scenarios(
    path: str | Path, limit: int | None = None
) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            texts.append(record["text"].strip())
            labels.append(record["label"])
            if limit is not None and len(texts) >= limit:
                break
    return texts, labels


def format_hierarchy_prompts(sentences: Iterable[str]) -> list[str]:
    return [sentence.strip() + HIERARCHY_SUFFIX for sentence in sentences]


def format_recognition_prompts(
    scenarios: Iterable[str], persona_prefix: str
) -> list[str]:
    suffix = f"{persona_prefix}{RECOGNITION_SUFFIX}"
    return [scenario.rstrip() + "\n" + suffix for scenario in scenarios]


def _decode_top_tokens(tokenizer, token_ids: np.ndarray) -> np.ndarray:
    rows = [
        [
            tokenizer.decode(int(token_id))
            for token_id in row
        ]
        for row in token_ids
    ]
    return np.asarray(rows, dtype=np.str_)


def score_with_transformers(
    prompts: list[str],
    model_name: str,
    *,
    batch_size: int = 8,
    top_k: int = 100,
    max_length: int | None = None,
    device_map: str = "auto",
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> TopTokenArtifact:
    """Save the top next-token IDs and full-softmax probabilities.

    This mirrors the original inference pipeline while avoiding storage of an
    N x vocabulary-size tensor. Emotion filtering intentionally happens later.
    """
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not prompts:
        raise ValueError("At least one prompt is required")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        device_map=device_map,
        torch_dtype="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    effective_max_length = max_length or tokenizer.model_max_length
    if effective_max_length > 1_000_000:
        effective_max_length = 8192

    probability_batches: list[np.ndarray] = []
    token_id_batches: list[np.ndarray] = []
    probability_dtype: str | None = None
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=effective_max_length,
        )
        input_device = next(model.parameters()).device
        encoded = {key: value.to(input_device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits[:, -1, :]
            probabilities = torch.softmax(logits, dim=-1)
            top_probabilities, top_token_ids = probabilities.topk(top_k, dim=-1)

        probability_dtype = probability_dtype or str(top_probabilities.dtype)

        probability_batches.append(
            top_probabilities.detach().float().cpu().numpy().astype(np.float32)
        )
        token_id_batches.append(
            top_token_ids.detach().cpu().numpy().astype(np.int64)
        )
        print(f"scored {min(start + len(batch), len(prompts))}/{len(prompts)}")

    probabilities = np.concatenate(probability_batches, axis=0)
    token_ids = np.concatenate(token_id_batches, axis=0)
    decoded_tokens = _decode_top_tokens(tokenizer, token_ids)
    metadata = {
        "artifact_format": ARTIFACT_FORMAT,
        "backend": "transformers",
        "model": model_name,
        "requested_revision": revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "top_k": top_k,
        "probability_dtype_before_storage": probability_dtype,
        "max_length": effective_max_length,
    }
    return TopTokenArtifact(probabilities, token_ids, decoded_tokens, metadata)


def score_with_nnsight(
    prompts: list[str],
    model_name: str,
    *,
    batch_size: int = 32,
    top_k: int = 100,
    device_map: str = "auto",
    revision: str | None = None,
    remote: bool = False,
) -> TopTokenArtifact:
    """Run the original NNsight tracing path used for the Llama experiments."""
    import nnsight
    import torch
    from nnsight import LanguageModel

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not prompts:
        raise ValueError("At least one prompt is required")

    model_options = {"device_map": device_map}
    if revision is not None:
        model_options["revision"] = revision
    model = LanguageModel(model_name, **model_options)

    probability_batches: list[np.ndarray] = []
    token_id_batches: list[np.ndarray] = []
    probability_dtype: str | None = None
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        with model.trace(batch, remote=remote):
            logits = model.lm_head.output[:, -1, :].save()
        probabilities = torch.softmax(logits, dim=-1)
        top_probabilities, top_token_ids = probabilities.topk(top_k, dim=-1)
        probability_dtype = probability_dtype or str(top_probabilities.dtype)
        probability_batches.append(
            top_probabilities.detach().float().cpu().numpy().astype(np.float32)
        )
        token_id_batches.append(
            top_token_ids.detach().cpu().numpy().astype(np.int64)
        )
        print(f"scored {min(start + len(batch), len(prompts))}/{len(prompts)}")

    probabilities = np.concatenate(probability_batches, axis=0)
    token_ids = np.concatenate(token_id_batches, axis=0)
    decoded_tokens = _decode_top_tokens(model.tokenizer, token_ids)
    metadata = {
        "artifact_format": ARTIFACT_FORMAT,
        "backend": "nnsight",
        "remote": remote,
        "model": model_name,
        "requested_revision": revision,
        "resolved_revision": None,
        "model_class": type(model).__name__,
        "tokenizer_class": type(model.tokenizer).__name__,
        "nnsight_version": getattr(nnsight, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "top_k": top_k,
        "probability_dtype_before_storage": probability_dtype,
        "max_length": None,
    }
    return TopTokenArtifact(probabilities, token_ids, decoded_tokens, metadata)


def save_top_tokens(path: str | Path, artifact: TopTokenArtifact) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probabilities": np.asarray(artifact.probabilities, dtype=np.float32),
        "token_ids": np.asarray(artifact.token_ids, dtype=np.int64),
        "decoded_tokens": np.asarray(artifact.decoded_tokens, dtype=np.str_),
        "metadata": np.asarray(json.dumps(artifact.metadata, sort_keys=True)),
    }
    if artifact.ground_truth is not None:
        payload["ground_truth"] = np.asarray(artifact.ground_truth, dtype=np.str_)
    np.savez_compressed(output, **payload)


def load_top_tokens(path: str | Path) -> TopTokenArtifact:
    with np.load(path, allow_pickle=False) as archive:
        artifact = TopTokenArtifact(
            probabilities=archive["probabilities"],
            token_ids=archive["token_ids"],
            decoded_tokens=archive["decoded_tokens"],
            metadata=json.loads(str(archive["metadata"])),
            ground_truth=(
                archive["ground_truth"].tolist()
                if "ground_truth" in archive.files
                else None
            ),
        )
    if artifact.metadata.get("artifact_format") != ARTIFACT_FORMAT:
        raise ValueError("Unsupported top-token artifact format")
    if not (
        artifact.probabilities.shape
        == artifact.token_ids.shape
        == artifact.decoded_tokens.shape
    ):
        raise ValueError("Top-token arrays must have identical shapes")
    if artifact.ground_truth is not None and len(artifact.ground_truth) != len(
        artifact.probabilities
    ):
        raise ValueError("Ground-truth length does not match artifact rows")
    return artifact
