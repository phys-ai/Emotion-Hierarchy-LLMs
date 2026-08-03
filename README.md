# Emergence of Hierarchical Emotion Organization in Large Language Models

Maya Okawa\*, Bo Zhao\*, Eric J. Bigelow, Rose Yu, Tomer Ullman,
Ekdeep Singh Lubana, and Hidenori Tanaka

Proceedings of the 43rd International Conference on Machine Learning
(ICML 2026)

\* Equal contribution.

Minimal code and fixed inputs for reproducing the paper's two central
experiments:

1. extract a hierarchy from next-token emotion probabilities; and
2. evaluate 135-way and six-way emotion recognition across personas.

The release intentionally omits exploratory notebooks, intermediate full-vocabulary
logits, plotting variants, wine-aroma experiments, intervention experiments,
conversation simulations, and raw user-study data.

## Faithfulness to the research code

The implementation is reorganized into small functions and a CLI, but preserves
the computation used by the research scripts:

1. append the original task suffix to each input;
2. compute the full-softmax next-token probabilities;
3. retain the top-100 token IDs, probabilities, and individually decoded tokens;
4. filter decoded tokens against the 135 emotion words without aggregating
   token IDs that decode to the same word; and
5. accumulate `p_i * p_j` for every retained pair of distinct ranked tokens.

Here, `C[a, b]` denotes this accumulated token-level co-occurrence weight.

For an emotion token `a`, the original conditional rule attaches `a` below `b`
when

```text
C[a, b] / sum(C[a, :]) > threshold
C[b, a] / sum(C[b, :]) < C[a, b] / sum(C[a, :])
```

The topology cleanup also follows the final research script. In particular, its
insertion-order rule for exactly two parents and its bounded removal of redundant
ancestor edges are preserved. They have deliberately not been replaced by a
general transitive reduction or a new parent-selection rule.

Emotion recognition likewise preserves the original procedure: decode the
top-100 tokens, concatenate their escaped strings, apply the original regular
expression, and select the first recognized emotion word.

Only storage, validation, CLI argument handling, and figure layout were cleaned
up. The compressed intermediate artifact stores top-token data instead of the
much larger full-vocabulary probability tensor.

See [`docs/legacy_semantics.md`](docs/legacy_semantics.md) for a function-level
map to the research scripts and the intentionally preserved quirks.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[inference]'
```

For the exact dependency versions used to verify this cleaned pipeline, use
`pip install -r requirements-verified.txt`. This file is a verification
environment, not a recovered lockfile from the original NNsight runs.

Llama 3.1 models are gated on Hugging Face. Full 70B and 405B runs require
access to the weights and corresponding multi-GPU infrastructure. The GPT-2
command below is a small end-to-end check, not a reproduction of the Llama
results.

The paper's Llama runs used NNsight. Install that optional backend with
`pip install -e '.[inference,nnsight]'` and configure NNsight credentials before
using `--backend nnsight --remote`.

## Quick check

Run the complete hierarchy pipeline on the first 200 inputs:

```bash
emotion-hierarchy score-hierarchy \
  --model gpt2 \
  --prompts data/hierarchy_prompts.txt \
  --taxonomy data/emotion_taxonomy.json \
  --limit 200 \
  --output results/gpt2/top_tokens.npz

emotion-hierarchy hierarchy \
  --tokens results/gpt2/top_tokens.npz \
  --taxonomy data/emotion_taxonomy.json \
  --threshold 0.3 \
  --rank-limit 100 \
  --output-dir results/gpt2
```

The second command writes the weighted token-level co-occurrences, candidate
edges, final legacy-topology edges, metrics, and `hierarchy.pdf`.

Run the dependency-free unit tests with:

```bash
python -m unittest discover -s tests
```

## Full hierarchy experiment

Repeat scoring for the paper models, using all 5,000 inputs:

```bash
emotion-hierarchy score-hierarchy \
  --model meta-llama/Meta-Llama-3.1-8B \
  --prompts data/hierarchy_prompts.txt \
  --output results/llama-3.1-8b/top_tokens.npz \
  --backend nnsight \
  --remote \
  --batch-size 8

emotion-hierarchy hierarchy \
  --tokens results/llama-3.1-8b/top_tokens.npz \
  --threshold 0.3 \
  --rank-limit 100 \
  --output-dir results/llama-3.1-8b
```

Change `8B` to `70B` or `405B` and choose a batch size suitable for the
hardware. The inference command stores the ranked token IDs and probabilities,
not the full vocabulary logits.

## Persona recognition experiment

Score the same 2,700 labeled scenarios once per persona:

```bash
emotion-hierarchy score-recognition \
  --model meta-llama/Meta-Llama-3.1-405B \
  --scenarios data/recognition_scenarios.jsonl \
  --personas data/personas.json \
  --persona neutral \
  --output results/recognition/neutral_top_tokens.npz \
  --backend nnsight \
  --remote \
  --batch-size 8

emotion-hierarchy score-recognition \
  --model meta-llama/Meta-Llama-3.1-405B \
  --scenarios data/recognition_scenarios.jsonl \
  --personas data/personas.json \
  --persona female \
  --output results/recognition/female_top_tokens.npz \
  --backend nnsight \
  --remote \
  --batch-size 8

emotion-hierarchy recognition \
  --tokens neutral=results/recognition/neutral_top_tokens.npz \
  --tokens female=results/recognition/female_top_tokens.npz \
  --rank-limit 100 \
  --output-dir results/recognition/summary
```

Add further `PERSONA=PATH` arguments for the comparisons of interest. The
command writes predictions, 135-way and six-way confusion matrices, accuracies,
and an accuracy plot.

The primary accuracy columns reproduce the paper plotting code and divide by
the confusion-matrix sum (recognized examples). Coverage and additional
all-examples accuracy columns are also reported.

## Reproducibility notes

- The fixed generated datasets are released, so GPT-4o does not need to be
  called again.
- The hierarchy input preserves the source data exactly: 5,000 records include
  3 empty rows and duplicate sentences. See `data/README.md` for the audit
  counts.
- Inference is deterministic in evaluation mode, subject to ordinary numerical
  differences between model runtimes and hardware.
- Score metadata records the resolved model revision and library versions. For
  an archival run, also pass `--revision COMMIT_HASH` explicitly.
- Emotion words are matched to individually decoded token IDs. IDs are not
  merged even when they decode to the same stripped word.
- The default `--top-k 100` follows the paper's stated setup. It is recorded in
  every score file's metadata.
- Older exploratory scripts contain a top-20 loop despite filenames containing
  `100`. The public paper command uses `--rank-limit 100`; the option remains
  explicit so archived variants can be rerun without changing code.
- Each artifact records input and taxonomy SHA-256 hashes, prompt count, model
  revision, tokenizer class, and library versions.
- `reference/` contains small artifacts from the original runs for regression
  comparison; it is not consumed by the pipeline.

## Citation

```bibtex
@inproceedings{okawa2026emergence,
  title     = {Emergence of Hierarchical Emotion Organization in Large Language Models},
  author    = {Okawa, Maya and Zhao, Bo and Bigelow, Eric J. and Yu, Rose and
               Ullman, Tomer and Lubana, Ekdeep Singh and Tanaka, Hidenori},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

Maya Okawa and Bo Zhao contributed equally.

## License

This project is released under the [MIT License](LICENSE).
