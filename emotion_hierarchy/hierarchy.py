from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .scoring import TopTokenArtifact
from .taxonomy import Taxonomy


@dataclass(frozen=True)
class TokenPair:
    """A specific emotion token and the broader token it implies."""

    child_token_id: int
    parent_token_id: int
    child: str
    parent: str
    probability: float


@dataclass(frozen=True)
class HierarchyResult:
    graph: nx.DiGraph
    cooccurrence: dict[int, dict[int, np.float32]]
    token_frequencies: dict[int, np.float32]
    candidates: list[TokenPair]
    rank_limit: int
    threshold: float
    artifact_metadata: dict


def _escaped(text: str) -> str:
    return text.encode("unicode_escape").decode()


def compute_cooccurrence(
    artifact: TopTokenArtifact,
    emotion_words: set[str],
    rank_limit: int = 100,
) -> tuple[
    dict[int, dict[int, np.float32]],
    dict[int, np.float32],
    dict[int, str],
]:
    """Reproduce the original token-ID-level weighted co-occurrence loop."""
    if rank_limit <= 0 or rank_limit > artifact.token_ids.shape[1]:
        raise ValueError(
            f"rank_limit must be in [1, {artifact.token_ids.shape[1]}]"
        )

    cooccurrence: dict[int, dict[int, np.float32]] = {}
    decoded_by_id: dict[int, str] = {}

    for row_index in range(artifact.token_ids.shape[0]):
        token_ids = artifact.token_ids[row_index, :rank_limit]
        probabilities = artifact.probabilities[row_index, :rank_limit]
        decoded = artifact.decoded_tokens[row_index, :rank_limit]
        normalized = [str(token).strip() for token in decoded]

        for token_id, raw in zip(token_ids, decoded):
            decoded_by_id.setdefault(int(token_id), str(raw))

        for first in range(rank_limit):
            if normalized[first] not in emotion_words:
                continue
            token_a = int(token_ids[first])
            probability_a = np.float32(probabilities[first])
            for second in range(first + 1, rank_limit):
                if normalized[second] not in emotion_words:
                    continue
                token_b = int(token_ids[second])
                probability_b = np.float32(probabilities[second])
                weight = np.float32(probability_a * probability_b)

                neighbors_a = cooccurrence.setdefault(token_a, {})
                neighbors_a[token_b] = np.float32(
                    neighbors_a.get(token_b, np.float32(0.0)) + weight
                )
                neighbors_b = cooccurrence.setdefault(token_b, {})
                neighbors_b[token_a] = np.float32(
                    neighbors_b.get(token_a, np.float32(0.0)) + weight
                )

    token_frequencies: dict[int, np.float32] = {}
    for token_id, neighbors in cooccurrence.items():
        total = np.float32(0.0)
        for weight in neighbors.values():
            total = np.float32(total + weight)
        token_frequencies[token_id] = total
    return cooccurrence, token_frequencies, decoded_by_id


def find_candidate_pairs(
    cooccurrence: dict[int, dict[int, np.float32]],
    token_frequencies: dict[int, np.float32],
    decoded_by_id: dict[int, str],
    threshold: float = 0.3,
) -> list[TokenPair]:
    """Apply the asymmetric conditional-probability rule from the paper code."""
    candidates: list[TokenPair] = []
    for child_id, neighbors in cooccurrence.items():
        for parent_id, weight in neighbors.items():
            parent_given_child = weight / token_frequencies[child_id]
            child_given_parent = (
                cooccurrence[parent_id][child_id] / token_frequencies[parent_id]
            )
            if parent_given_child > threshold and child_given_parent < parent_given_child:
                candidates.append(
                    TokenPair(
                        child_token_id=child_id,
                        parent_token_id=parent_id,
                        child=_escaped(decoded_by_id[child_id]),
                        parent=_escaped(decoded_by_id[parent_id]),
                        probability=float(parent_given_child),
                    )
                )
    return candidates


def build_legacy_graph(candidates: list[TokenPair]) -> nx.DiGraph:
    """Reproduce the topology cleanup used by the final research script.

    The insertion-order-dependent two-parent rule and the bounded ancestor-edge
    cleanup are intentionally preserved. They are legacy semantics, not a new
    generalized tree algorithm.
    """
    graph = nx.DiGraph()
    conditional: dict[tuple[str, str], float] = {}
    for pair in candidates:
        graph.add_edge(pair.parent, pair.child)
        conditional[(pair.child, pair.parent)] = pair.probability

    # This mirrors emotion_tree_construction.py exactly, including its reversed
    # probability lookup. For two parents it consequently retains the first
    # inserted parent; nodes with more than two parents are left unchanged.
    for node in graph.nodes:
        parents = list(graph.predecessors(node))
        if len(parents) == 2:
            parent1, parent2 = parents
            probability1 = conditional.get((parent1, node), 0)
            probability2 = conditional.get((parent2, node), 0)
            if probability1 < probability2:
                graph.remove_edge(parent1, node)
            else:
                graph.remove_edge(parent2, node)

    # Preserve the original cleanup of redundant grandparent and
    # great-grandparent edges rather than applying full transitive reduction.
    for node in graph.nodes:
        for parent in list(graph.predecessors(node)):
            for grandparent in list(graph.predecessors(parent)):
                if graph.has_edge(grandparent, node):
                    graph.remove_edge(grandparent, node)
                for great_grandparent in list(graph.predecessors(grandparent)):
                    if graph.has_edge(great_grandparent, node):
                        graph.remove_edge(great_grandparent, node)
    return graph


def extract_hierarchy(
    artifact: TopTokenArtifact,
    taxonomy: Taxonomy,
    *,
    threshold: float = 0.3,
    rank_limit: int = 100,
) -> HierarchyResult:
    cooccurrence, token_frequencies, decoded_by_id = compute_cooccurrence(
        artifact, set(taxonomy.labels), rank_limit
    )
    candidates = find_candidate_pairs(
        cooccurrence, token_frequencies, decoded_by_id, threshold
    )
    graph = build_legacy_graph(candidates)
    return HierarchyResult(
        graph,
        cooccurrence,
        token_frequencies,
        candidates,
        rank_limit,
        threshold,
        artifact.metadata,
    )


def _all_paths_to_leaves(graph: nx.DiGraph, source: str) -> list[list[str]]:
    if graph.out_degree(source) == 0:
        return [[source]]
    paths: list[list[str]] = []
    for child in graph.successors(source):
        for path in _all_paths_to_leaves(graph, child):
            paths.append([source, *path])
    return paths


def hierarchy_metrics(graph: nx.DiGraph) -> dict[str, float | int]:
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Legacy graph contains a cycle; depth is undefined")
    roots = [node for node in graph if graph.in_degree(node) == 0]
    path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
    total_path_length = sum(sum(lengths.values()) for lengths in path_lengths.values())
    root_to_leaf_depths = [
        len(path) - 1
        for root in roots
        for path in _all_paths_to_leaves(graph, root)
    ]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "roots": len(roots),
        "all_pairs_path_length": int(total_path_length),
        "average_root_to_leaf_depth": (
            float(np.mean(root_to_leaf_depths)) if root_to_leaf_depths else 0.0
        ),
    }


def _layered_layout(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Deterministic clean layout; topology remains the legacy topology."""
    roots = sorted(node for node in graph if graph.in_degree(node) == 0)
    generations = list(nx.topological_generations(graph))
    depth = {
        node: generation
        for generation, nodes in enumerate(generations)
        for node in nodes
    }
    positions: dict[str, tuple[float, float]] = {}
    by_depth: dict[int, list[str]] = {}
    for node, node_depth in depth.items():
        by_depth.setdefault(node_depth, []).append(node)
    for node_depth, nodes in by_depth.items():
        for index, node in enumerate(sorted(nodes)):
            positions[node] = (float(index), -float(node_depth))
    if roots and len(by_depth.get(0, [])) == 1:
        positions[roots[0]] = (0.0, 0.0)
    return positions


def plot_hierarchy(
    graph: nx.DiGraph,
    taxonomy: Taxonomy,
    output: str | Path,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    width = max(8.0, min(24.0, graph.number_of_nodes() * 0.18))
    figure, axis = plt.subplots(figsize=(width, 6.0))
    if graph.number_of_nodes() == 0:
        axis.text(0.5, 0.5, "No edges at this threshold", ha="center")
        axis.axis("off")
        figure.savefig(output, bbox_inches="tight")
        plt.close(figure)
        return

    palette = plt.get_cmap("Spectral")
    color_positions = [0.4, 0.2, 0.5, 0.0, 0.8, 0.9]
    colors = {
        primary: palette(position)
        for primary, position in zip(taxonomy.primary_labels, color_positions)
    }
    node_colors = [
        colors[taxonomy.primary_by_label[node.strip()]] for node in graph
    ]
    positions = _layered_layout(graph)
    nx.draw_networkx_edges(graph, positions, ax=axis, edge_color="0.75", width=1.2)
    nx.draw_networkx_nodes(
        graph, positions, ax=axis, node_color=node_colors, node_size=360, alpha=0.7
    )
    nx.draw_networkx_labels(graph, positions, ax=axis, font_size=7)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def write_hierarchy_outputs(result: HierarchyResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "cooccurrence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["token_a", "token_b", "weight", "frequency_a"])
        for token_a, neighbors in result.cooccurrence.items():
            for token_b, weight in neighbors.items():
                writer.writerow(
                    [token_a, token_b, float(weight), float(result.token_frequencies[token_a])]
                )

    with (output_dir / "candidate_edges.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["parent", "child", "parent_token_id", "child_token_id", "probability"]
        )
        for pair in result.candidates:
            writer.writerow(
                [
                    pair.parent,
                    pair.child,
                    pair.parent_token_id,
                    pair.child_token_id,
                    pair.probability,
                ]
            )

    with (output_dir / "edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parent", "child"])
        writer.writerows(result.graph.edges())

    metadata = {
        "threshold": result.threshold,
        "rank_limit": result.rank_limit,
        "topology_semantics": "legacy-paper-code",
        "source_artifact": result.artifact_metadata,
        **hierarchy_metrics(result.graph),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
