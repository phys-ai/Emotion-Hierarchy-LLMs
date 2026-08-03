# Legacy-semantics map

This public implementation preserves the numerical and topology semantics of
the research scripts while separating inference, hierarchy extraction, and
recognition evaluation.

| Public function | Research-code source | Preserved behavior |
| --- | --- | --- |
| `score_with_nnsight` | `emotion_tree_get_logits.py`, `get_logits_neutral.py` | `LanguageModel.trace`, last-position logits, full softmax, ranked token IDs and probabilities |
| `score_with_transformers` | Public local/smoke-test backend | Equivalent last-position scoring without NNsight orchestration |
| `compute_cooccurrence` | `emotion_tree_construction.compute_co_occurence` | Individual token decoding, stripped exact-word filtering, token-ID-level `p_i * p_j` accumulation |
| `find_candidate_pairs` | `emotion_tree_construction.find_pairs` | Asymmetric conditional threshold and insertion order |
| `build_legacy_graph` | `emotion_tree_construction.plot_tree` | Parent-to-child edge direction, exactly-two-parent rule, and bounded ancestor-edge removal |
| `predict_from_top_tokens` | `plot_recognition.compute_confusion_matrix`, `plot_recognition.compute_prediction` | Escaped decoded-token concatenation, original regex, and first emotion match |

## Intentionally preserved quirks

- Emotion matching occurs after decoding individual token IDs. Tokens are not
  merged merely because they decode to the same stripped word.
- Self-co-occurrence is absent because only pairs at distinct ranks are
  accumulated.
- When a graph node has exactly two parents, the final research script performs
  a reversed probability lookup. Both lookups are normally zero, so the first
  inserted parent is retained. This behavior is regression-tested.
- Nodes with more than two parents are not collapsed by that rule.
- Redundant ancestor edges are checked only through grandparents and
  great-grandparents; full transitive reduction is not used.
- Recognition selects the first emotion produced by the original regex over
  the concatenated top-token strings, rather than taking an argmax over a
  pre-aggregated 135-label vector.
- Paper accuracies use `trace(confusion) / sum(confusion)`, excluding examples
  with no recognized emotion from the denominator. The public output also
  reports coverage and an all-examples accuracy for transparency.

## Organizational changes that do not alter the analyzed token sequence

- The public artifact stores only the ranked token IDs, their full-softmax
  probabilities, and decoded strings instead of the entire vocabulary tensor.
- Ground-truth labels are stored explicitly in JSONL and in recognition
  artifacts instead of being inferred from `row_index // 20` during analysis.
- Inputs, model revision, dependency versions, and SHA-256 hashes are recorded
  in artifact metadata.
- Plot layout and output file formats were simplified. Graph topology and
  reported metrics retain the legacy semantics.

## Rank-limit note

The paper specifies 100 ranked tokens. Some older exploratory scripts loop over
20 ranks while saving files with `100` in their names. The public paper command
therefore defaults to 100 and exposes `--rank-limit` so an archived top-20
variant can be reproduced without editing source code.
