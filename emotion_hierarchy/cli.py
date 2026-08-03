from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .hierarchy import extract_hierarchy, plot_hierarchy, write_hierarchy_outputs
from .recognition import evaluate_recognition, write_recognition_outputs
from .scoring import (
    format_hierarchy_prompts,
    format_recognition_prompts,
    load_top_tokens,
    read_lines,
    read_scenarios,
    save_top_tokens,
    score_with_nnsight,
    score_with_transformers,
    sha256_file,
)
from .taxonomy import load_taxonomy


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Hugging Face model id or path")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--taxonomy", type=Path, default=Path("data/emotion_taxonomy.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of ranked next-token IDs to retain (paper default: 100)",
    )
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--revision", help="Model repository commit or tag")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--backend", choices=["transformers", "nnsight"], default="transformers"
    )
    parser.add_argument(
        "--remote", action="store_true", help="Use the NNsight remote backend"
    )


def _score_prompts(args: argparse.Namespace, prompts: list[str]):
    if args.backend == "nnsight":
        if args.max_length is not None:
            raise ValueError("--max-length is only supported by the transformers backend")
        if args.trust_remote_code:
            raise ValueError(
                "--trust-remote-code is only supported by the transformers backend"
            )
        return score_with_nnsight(
            prompts,
            args.model,
            batch_size=args.batch_size,
            top_k=args.top_k,
            device_map=args.device_map,
            revision=args.revision,
            remote=args.remote,
        )
    if args.remote:
        raise ValueError("--remote requires --backend nnsight")
    return score_with_transformers(
        prompts,
        args.model,
        batch_size=args.batch_size,
        top_k=args.top_k,
        max_length=args.max_length,
        device_map=args.device_map,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )


def score_hierarchy(args: argparse.Namespace) -> None:
    load_taxonomy(args.taxonomy)  # Validate before starting expensive inference.
    sentences = read_lines(args.prompts, args.limit)
    prompts = format_hierarchy_prompts(sentences)
    artifact = _score_prompts(args, prompts)
    metadata = {
        **artifact.metadata,
        "task": "hierarchy",
        "input": str(args.prompts),
        "input_sha256": sha256_file(args.prompts),
        "taxonomy_sha256": sha256_file(args.taxonomy),
        "examples": len(prompts),
    }
    save_top_tokens(args.output, replace(artifact, metadata=metadata))


def score_recognition(args: argparse.Namespace) -> None:
    load_taxonomy(args.taxonomy)
    with args.personas.open(encoding="utf-8") as handle:
        personas = json.load(handle)
    if args.persona not in personas:
        raise ValueError(f"Unknown persona: {args.persona}")
    scenarios, ground_truth = read_scenarios(args.scenarios, args.limit)
    persona_prefix = personas[args.persona]
    prompts = format_recognition_prompts(scenarios, persona_prefix)
    artifact = _score_prompts(args, prompts)
    metadata = {
        **artifact.metadata,
        "task": "recognition",
        "persona": args.persona,
        "persona_prefix": persona_prefix,
        "input": str(args.scenarios),
        "input_sha256": sha256_file(args.scenarios),
        "personas_sha256": sha256_file(args.personas),
        "taxonomy_sha256": sha256_file(args.taxonomy),
        "examples": len(prompts),
    }
    save_top_tokens(
        args.output,
        replace(artifact, metadata=metadata, ground_truth=ground_truth),
    )


def analyze_hierarchy(args: argparse.Namespace) -> None:
    taxonomy = load_taxonomy(args.taxonomy)
    artifact = load_top_tokens(args.tokens)
    result = extract_hierarchy(
        artifact,
        taxonomy,
        threshold=args.threshold,
        rank_limit=args.rank_limit,
    )
    write_hierarchy_outputs(result, args.output_dir)
    plot_hierarchy(result.graph, taxonomy, args.output_dir / "hierarchy.pdf")


def analyze_recognition(args: argparse.Namespace) -> None:
    taxonomy = load_taxonomy(args.taxonomy)
    results = {}
    ground_truth_by_persona = {}
    metadata_by_persona = {}
    for specification in args.tokens:
        persona, separator, path = specification.partition("=")
        if not separator:
            raise ValueError("Each --tokens value must be PERSONA=PATH")
        artifact = load_top_tokens(path)
        if artifact.ground_truth is None:
            raise ValueError(f"Recognition artifact lacks ground truth: {path}")
        results[persona] = evaluate_recognition(
            artifact, taxonomy, rank_limit=args.rank_limit
        )
        ground_truth_by_persona[persona] = artifact.ground_truth
        metadata_by_persona[persona] = artifact.metadata
    write_recognition_outputs(
        results,
        taxonomy,
        ground_truth_by_persona,
        metadata_by_persona,
        args.output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="emotion-hierarchy")
    commands = parser.add_subparsers(dest="command", required=True)

    score_tree = commands.add_parser("score-hierarchy")
    _add_inference_arguments(score_tree)
    score_tree.add_argument("--prompts", required=True, type=Path)
    score_tree.set_defaults(function=score_hierarchy)

    score_bias = commands.add_parser("score-recognition")
    _add_inference_arguments(score_bias)
    score_bias.add_argument("--scenarios", required=True, type=Path)
    score_bias.add_argument("--personas", required=True, type=Path)
    score_bias.add_argument("--persona", required=True)
    score_bias.set_defaults(function=score_recognition)

    tree = commands.add_parser("hierarchy")
    tree.add_argument("--tokens", required=True, type=Path)
    tree.add_argument("--output-dir", required=True, type=Path)
    tree.add_argument("--taxonomy", type=Path, default=Path("data/emotion_taxonomy.json"))
    tree.add_argument("--threshold", type=float, default=0.3)
    tree.add_argument("--rank-limit", type=int, default=100)
    tree.set_defaults(function=analyze_hierarchy)

    recognition = commands.add_parser("recognition")
    recognition.add_argument(
        "--tokens", action="append", required=True, help="PERSONA=top_tokens.npz"
    )
    recognition.add_argument("--output-dir", required=True, type=Path)
    recognition.add_argument("--taxonomy", type=Path, default=Path("data/emotion_taxonomy.json"))
    recognition.add_argument("--rank-limit", type=int, default=100)
    recognition.set_defaults(function=analyze_recognition)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
