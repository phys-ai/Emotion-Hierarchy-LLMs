import tempfile
import unittest
from pathlib import Path

import numpy as np

from emotion_hierarchy.hierarchy import (
    TokenPair,
    build_legacy_graph,
    compute_cooccurrence,
    extract_hierarchy,
)
from emotion_hierarchy.recognition import evaluate_recognition, predict_from_top_tokens
from emotion_hierarchy.scoring import (
    ARTIFACT_FORMAT,
    TopTokenArtifact,
    load_top_tokens,
    read_lines,
    save_top_tokens,
)
from emotion_hierarchy.taxonomy import load_taxonomy


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = load_taxonomy(ROOT / "data/emotion_taxonomy.json")


def artifact(
    probabilities,
    token_ids,
    decoded_tokens,
    ground_truth=None,
) -> TopTokenArtifact:
    return TopTokenArtifact(
        probabilities=np.asarray(probabilities, dtype=np.float32),
        token_ids=np.asarray(token_ids, dtype=np.int64),
        decoded_tokens=np.asarray(decoded_tokens, dtype=np.str_),
        metadata={"artifact_format": ARTIFACT_FORMAT, "top_k": len(token_ids[0])},
        ground_truth=ground_truth,
    )


class ArtifactTests(unittest.TestCase):
    def test_round_trip_without_pickle(self):
        expected = artifact([[0.5]], [[7]], [[" joy"]], ["joy"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.npz"
            save_top_tokens(path, expected)
            actual = load_top_tokens(path)
        np.testing.assert_array_equal(actual.probabilities, expected.probabilities)
        np.testing.assert_array_equal(actual.token_ids, expected.token_ids)
        np.testing.assert_array_equal(actual.decoded_tokens, expected.decoded_tokens)
        self.assertEqual(actual.ground_truth, ["joy"])

    def test_empty_input_rows_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.txt"
            path.write_text("one\n\nthree", encoding="utf-8")
            self.assertEqual(read_lines(path), ["one", "", "three"])


class HierarchyFidelityTests(unittest.TestCase):
    def setUp(self):
        self.tokens = artifact(
            probabilities=[[0.5, 0.2, 0.1], [0.4, 0.3, 0.1]],
            token_ids=[[1, 2, 9], [1, 3, 9]],
            decoded_tokens=[
                [" joy", " hope", " other"],
                [" joy", " fear", " other"],
            ],
        )

    def test_token_id_weighted_cooccurrence_matches_original_loop(self):
        cooccurrence, frequencies, _ = compute_cooccurrence(
            self.tokens, {"joy", "hope", "fear"}, rank_limit=3
        )
        self.assertAlmostEqual(float(cooccurrence[1][2]), 0.1, places=6)
        self.assertAlmostEqual(float(cooccurrence[1][3]), 0.12, places=6)
        self.assertAlmostEqual(float(frequencies[1]), 0.22, places=6)
        self.assertAlmostEqual(float(frequencies[2]), 0.1, places=6)

    def test_conditional_rule_points_from_general_to_specific(self):
        result = extract_hierarchy(
            self.tokens, TAXONOMY, threshold=0.3, rank_limit=3
        )
        self.assertIn((" joy", " hope"), result.graph.edges)
        self.assertIn((" joy", " fear"), result.graph.edges)

    def test_two_parent_insertion_order_quirk_is_preserved(self):
        candidates = [
            TokenPair(3, 1, " child", " first", 0.1),
            TokenPair(3, 2, " child", " second", 0.9),
        ]
        graph = build_legacy_graph(candidates)
        self.assertIn((" first", " child"), graph.edges)
        self.assertNotIn((" second", " child"), graph.edges)

    def test_more_than_two_parents_are_not_collapsed(self):
        candidates = [
            TokenPair(4, 1, " child", " first", 0.1),
            TokenPair(4, 2, " child", " second", 0.2),
            TokenPair(4, 3, " child", " third", 0.3),
        ]
        graph = build_legacy_graph(candidates)
        self.assertEqual(graph.in_degree(" child"), 3)

    def test_bounded_transitive_cleanup_is_preserved(self):
        candidates = [
            TokenPair(2, 1, " parent", " grandparent", 0.8),
            TokenPair(3, 2, " child", " parent", 0.8),
            TokenPair(3, 1, " child", " grandparent", 0.7),
            TokenPair(3, 4, " child", " unrelated", 0.6),
        ]
        graph = build_legacy_graph(candidates)
        self.assertIn((" grandparent", " parent"), graph.edges)
        self.assertIn((" parent", " child"), graph.edges)
        self.assertNotIn((" grandparent", " child"), graph.edges)


class RecognitionFidelityTests(unittest.TestCase):
    def test_first_matching_word_in_rank_order_is_used(self):
        decoded = np.asarray(
            [[" noise", " joy", " fear"], [" fear", " joy", " noise"]]
        )
        predictions = predict_from_top_tokens(
            decoded, set(TAXONOMY.labels), rank_limit=3
        )
        self.assertEqual(predictions, ["joy", "fear"])

    def test_exact_and_broad_accuracy(self):
        tokens = artifact(
            probabilities=[[0.5, 0.4], [0.5, 0.4]],
            token_ids=[[1, 2], [2, 1]],
            decoded_tokens=[[" joy", " fear"], [" fear", " joy"]],
            ground_truth=["joy", "joy"],
        )
        result = evaluate_recognition(tokens, TAXONOMY, rank_limit=2)
        self.assertEqual(result.summary["exact_accuracy"], 0.5)
        self.assertEqual(result.summary["broad_accuracy"], 0.5)
        self.assertEqual(result.summary["coverage"], 1.0)

    def test_paper_accuracy_excludes_unmatched_rows(self):
        tokens = artifact(
            probabilities=[[0.5], [0.5]],
            token_ids=[[1], [9]],
            decoded_tokens=[[" joy"], [" unknown"]],
            ground_truth=["joy", "joy"],
        )
        result = evaluate_recognition(tokens, TAXONOMY, rank_limit=1)
        self.assertEqual(result.summary["exact_accuracy"], 1.0)
        self.assertEqual(result.summary["exact_accuracy_all_examples"], 0.5)
        self.assertEqual(result.summary["coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
