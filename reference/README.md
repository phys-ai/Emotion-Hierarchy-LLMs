# Reference outputs

These lightweight files were copied from the original experiment directory for
comparison with a fresh run.

- `hierarchies/` contains the serialized trees for GPT-2 and Llama 3.1
  8B/70B/405B.
- `recognition/` contains selected 6 x 6 confusion matrices for the persona
  comparisons in the paper. They preserve the original script's transposed
  orientation: rows are predicted broad emotions and columns are ground-truth
  broad emotions, in the header order.

They are reference artifacts, not inputs to the analysis pipeline.
