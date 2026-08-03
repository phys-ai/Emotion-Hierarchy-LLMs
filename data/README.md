# Data

This directory contains the fixed inputs required by the two main experiments.

- `emotion_taxonomy.json`: the 135 labels and six broad groups used in the
  paper, transcribed from Shaver et al. (1987).
- `hierarchy_prompts.txt`: 5,000 source records generated with GPT-4o. The
  original file contains 4,997 non-empty rows (3 empty rows) and 3,813 unique
  non-empty sentences. Empty rows and duplicates are retained because the
  original inference script scored all 5,000 records.
- `recognition_scenarios.jsonl`: 20 GPT-4o-generated scenarios for each of the
  135 labels (2,700 records). The `label` field makes the ordering assumption
  in the research code explicit.
- `personas.json`: the prefixes inserted before the recognition question.

The scenario-generation calls are intentionally not part of the reproduction
pipeline: using the released, fixed inputs separates evaluation from API drift
and generation randomness. Before publication, the authors should confirm that
the dataset and taxonomy can be redistributed under the repository's chosen
license.

Raw user-study responses are not included. They require a separate
de-identification, consent, and data-sharing review.
