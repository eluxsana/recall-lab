# Experimental protocol

## Task and partitions

There are eight key tokens A–H and eight value tokens 0–7, plus beginning and query markers. Four distinct keys are sampled without replacement. Values are sampled independently and may repeat. A query key is selected uniformly from the four stored keys. The target is its associated value. The target is never appended to the input.

The 11-token training sequence is `BOS key value key value key value key value QUERY key`. The loss is computed only at the final query-key position. Random-value accuracy in expectation is 1/8. Copying the final stored value is a stronger baseline because the query sometimes requests that pair and unrelated pairs may share its value.

SHA-256 of the token bytes determines membership: the first eight hex digits modulo ten assign buckets 0–7 to training, bucket 8 to development, and bucket 9 to test. A set prevents duplicate inputs inside each generated dataset. We use 12,000 training examples (generator seed 17), 256 development examples (seed 99), and 1,024 test examples (seed 9281). The full training/test input hashes are summarized in the result receipts. All three model seeds share these exact datasets.

Development accuracy is monitored every 100 steps. The fixed final checkpoint at step 1,400 is reported; we do not select a best development checkpoint. The initial test score is recorded as a baseline, and final test metrics are reported after the fixed schedule. This is a small, developer-observed synthetic test, not a blinded benchmark.

## Architecture and training

Two pre-layer-normalization blocks with residual self-attention and a GELU MLP. Width 64, four heads with head dimension 16, MLP width 128, no dropout. Learned token embeddings and absolute position embeddings (17 positions), layer normalization, and an eight-value classifier. Including biases, the total is 69,832 parameters.

Attention uses explicit scaled Q/K dot products and a strict causal mask. The classifier is applied at every token for prefix-invariance testing, but only final-token logits enter training and evaluation loss. Zeroed head outputs are removed after attention-weighted value aggregation and before the output projection; attention weights themselves are not zeroed or renormalized. The output-projection bias remains present.

AdamW with learning rate 0.002, weight decay 0.01, default beta parameters, batch size 96, gradient norm clipping at 1.0. Training samples are drawn with replacement from the fixed finite training set for 1,400 steps. Seeds 42, 43, 44 control initialization and minibatch order. CPU, two threads, deterministic algorithms enabled. All hyperparameters are in train.py; there is no external experiment service.

## Explorer and interventions

The explorer uses the **first eight** primary test examples in generator order, with no correctness-based example selection. For each example we increment one of its four value tokens modulo eight, producing 32 variants. Targets are recomputed from the changed mapping. The original and all variants store actual softmax probabilities and attention matrices from the trained checkpoint.

An edited input may move to another hash partition. The UI explicitly records whether it remains in the test partition and whether it occurred in the finite training set. These examples are interventions for exploration, not an additional 32-example held-out accuracy claim. The original ID shown is the base example's ID. No arbitrary browser inference occurs.

Attention matrices are rounded to five decimal places and probabilities to six for export. Matrix rows are source/query positions and columns are attended-to positions. Percentages in the interface may round to zero or 100. The routing graphic and matrix are descriptive; neither is a causal attribution method.

Separately, ten output-ablation settings are measured on the full 1,024-example test: each of eight individual heads and all four heads in each layer. The checkpoint is never retrained for an ablation. Accuracy and cross-entropy are both reported. These interventions do not uniquely isolate a circuit or account for all head interactions.

## Length and seed sensitivity

For each input length of 2–6 pairs, generate 512 examples with seed 9281 in the test hash partition. These are separate sets, not a paired comparison across lengths. Training uses only four pairs. Longer inputs change both the number of pairs and the position of the query; some absolute positions have never trained. Shorter inputs reuse trained positions in new roles. Results therefore do not isolate context length from positional generalization or task load.

The four-pair stress subset is the first 512 examples of the main 1,024-example test, because it uses the same generator, seed and partition rule. Other lengths generate different input sets.

Three initialization/minibatch seeds with fixed data give a small sensitivity check. The primary explorer remains seed 42; seed43.safetensors and seed44.safetensors are also published. seed-study.json records every run's four-pair, length-stress and ablation measurements. Three seeds do not establish reliable tail behavior or a precise population uncertainty estimate.

## Reproduction and safety

The checkpoint uses safetensors, not a pickled Python model object. verify.py checks source and checkpoint digests, loads only tensor data, recomputes primary results and attention matrices, and verifies the additional seeds. Hashes detect changes relative to the receipt, not authorship or a maliciously rewritten receipt. Numerical tolerance allows ordinary floating-point differences; bitwise reproduction across platforms is not asserted.

All data are generated locally. The report has no third-party fonts, scripts, analytics, API keys, employer code, or private data. GitHub Pages access logs are outside this repository's control. AI assistance is disclosed in the README and explorer.
