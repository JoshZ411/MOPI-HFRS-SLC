# MORL Main Pipeline

This document captures the execution flow of `morl_main.py` and the main behaviors you should expect during a normal run.

## Entry Point

The script entry point is `main()` in `morl_main.py`.

It performs five high-level stages:

1. Parse arguments, set seeds, choose device, and create the output directory.
2. Load frozen user and item embeddings from the earlier training stage.
3. Load the graph, reconstruct train/validation/test splits, and prepare lookup structures.
4. Train a conditional MORL policy using REINFORCE over a sequential recommendation environment.
5. Select the best trade-off weight on validation, evaluate on test, and save results.

## Stage 0: Argument Parsing and Setup

`morl_main.py` accepts these main arguments:

- `--checkpoint`: path to frozen embeddings produced by `main.py`
- `--graph_path`: path to the processed graph file
- `--epochs`: number of MORL training epochs
- `--batch_size`: users sampled per policy update
- `--K`: recommendation list length per episode
- `--M`: candidate pool size per user
- `--lr`: learning rate
- `--hidden_dim`: hidden size of the policy MLP
- `--seed`: random seed
- `--output_dir`: where checkpoints and results are saved
- `--val_weight_alpha`: validation-time trade-off between NDCG and health score

After parsing:

- `torch.manual_seed(args.seed)` and `np.random.seed(args.seed)` make the run repeatable.
- The device is chosen as `cuda` if available, otherwise `cpu`.
- The output directory is created if it does not already exist.

Typical console output:

```text
Device: cuda
```

or

```text
Device: cpu
```

## Stage 1: Load Frozen Embeddings

The script loads the checkpoint from `args.checkpoint` and expects it to contain:

- `user_emb`: shape `(num_users, d)`
- `item_emb`: shape `(num_items, d)`

These embeddings are treated as frozen features. The MORL stage does not fine-tune them.

Typical console output:

```text
[Phase 1] Loading embeddings from embeddings_checkpoint.pt ...
  user_emb: (8452, 128)
  item_emb: (3120, 128)
```

The exact numbers depend on the dataset and upstream pipeline.

## Stage 2: Load Graph and Rebuild Data Splits

The script loads the processed graph from `args.graph_path` and extracts:

- `graph['user'].tags`
- `graph['food'].tags`
- `graph[('user', 'eats', 'food')].edge_index`

The edge list is then split into train, validation, and test sets using:

- 80 percent train+val / 20 percent test
- then 75 percent train / 25 percent val on the remaining part

This reproduces a 60/20/20 split overall.

From those splits, the script builds:

- `train_pos`, `val_pos`, `test_pos`: mapping `user_id -> positive item list`
- `train_users`, `val_users`, `test_users`: active users in each split
- `exclude_val`: training positives masked during validation ranking
- `exclude_test`: training and validation positives masked during test ranking

Typical console output:

```text
Loading graph from ../processed_data/benchmark_macro.pt ...
```

This stage does not print split sizes, but these data structures are used by later training and evaluation.

## Stage 3: Build Candidate Pools

Candidate pool construction is implemented in `environment.py` through `build_candidate_pools(...)`.

For each user:

1. Compute scores for all items using `user_emb @ item_emb.T`.
2. Optionally mask excluded items by setting their scores to negative infinity.
3. Keep the top `M` item indices.

This gives each user a fixed top-`M` shortlist that the policy will operate over.

During training:

- candidate pools are built once with no exclusion mask
- only training users are retained

During validation and test:

- pools are rebuilt with exclusion masks so known positives are not recommended back

Typical console output:

```text
Building candidate pools ...
```

## Stage 4: Create the Sequential Recommendation Environment

The environment is `RecommendationEnv` in `environment.py`.

Each episode is a deterministic `K`-step recommendation process for one user.

### State Definition

At step `t`, the state is:

- user embedding
- mean embedding of items selected so far
- binary tag coverage accumulated so far
- normalized timestep `t / K`

So the state dimension is:

`2 * d + tag_dim + 1`

### Action Definition

An action chooses one item from the remaining entries in the current user candidate pool.

### Reward Definition

Each step returns a 3D reward vector:

- `r_pref`: cosine similarity between the user embedding and the selected item embedding
- `r_health`: Jaccard-style overlap between covered recommendation tags and the user's tag vector
- `r_div`: negative mean cosine similarity to previously selected items

Interpretation:

- higher `r_pref` means stronger preference relevance
- higher `r_health` means the current list better aligns with user health tags
- higher `r_div` means lower similarity to already selected items, so more diversity

The first selected item always gets `r_div = 0.0` because there are no earlier selections to compare against.

## Stage 5: Build the Conditional MORL Policy

The policy is `ConditionalPolicy` in `policy.py`.

It takes:

- the current environment state `s_t`
- a preference weight vector `w = [w_pref, w_health, w_div]`

The network concatenates `state` and `weight`, passes them through a 3-layer MLP, and produces logits over the `M` candidate slots.

Invalid actions are masked, then the logits are turned into log-probabilities.

During training:

- actions are sampled from the policy distribution

During evaluation:

- actions are chosen greedily using the highest-probability action

## Stage 6: Train the Policy with REINFORCE

Training is implemented in `train_morl(...)` in `training.py`.

Each epoch does the following:

1. Sample a random batch of training users.
2. For each user, sample a random preference vector from a symmetric Dirichlet distribution.
3. Run one recommendation episode for that user under that sampled preference.
4. Convert the 3D reward sequence into a scalar episodic return using:

   `R = sum_t (w dot r_t)`

5. Compute a batch mean-return baseline.
6. Apply REINFORCE with advantage `R - baseline`.
7. Update only the policy network parameters with Adam.

Important detail:

- frozen embeddings do not receive gradients
- only `ConditionalPolicy` is updated

The training loop logs progress every 10 epochs and saves checkpoints every `checkpoint_every` epochs.

Typical console output:

```text
[Phases 3-6] Training MORL policy ...
Building candidate pools ...
Starting MORL training: 200 epochs, batch_size=64, K=20, M=200
Epoch   10 | loss=0.1842 | mean_return=6.9125
Epoch   20 | loss=-0.0376 | mean_return=7.1889
Epoch   30 | loss=0.1021 | mean_return=7.5404
...
Epoch  200 | loss=-0.0914 | mean_return=8.3278
Training complete. Final checkpoint saved to morl_output/morl_policy_final.pt
```

Notes on the logged values:

- `loss` can be positive or negative in policy gradient training
- `mean_return` is the batch baseline, not a validation metric
- upward movement in `mean_return` is usually the more useful signal

## Stage 7: Build the Validation Weight Grid

After training, the script generates a small preference grid using `simplex_grid(n_points=15)`.

This grid includes four fixed weights:

- `[1.0, 0.0, 0.0]`
- `[0.0, 1.0, 0.0]`
- `[0.0, 0.0, 1.0]`
- `[1/3, 1/3, 1/3]`

and the remaining points are sampled randomly from the same Dirichlet distribution over the 3-objective simplex.

The purpose is to test how the trained conditional policy behaves under different trade-off preferences.

## Stage 8: Validation-Time Trade-off Selection

For every weight vector in the grid, the script runs `evaluate_morl(...)` on validation users.

This evaluation uses greedy action selection and computes four aggregate metrics:

- `ndcg`
- `health_score`
- `diversity`
- `recall`

### Metric Meaning

- `ndcg`: ranking quality against held-out validation positives
- `recall`: fraction of held-out positives recovered in the top `K`
- `health_score`: fraction of recommended items matching user health preferences through tag overlap
- `diversity`: average intra-list diversity measured as `1 - mean pairwise cosine similarity`

The script then computes the median validation diversity across all tested weights.

Selection rule:

- if a weight's diversity is below the median diversity, it is disqualified with score `-1.0`
- otherwise the score is:

  `score = alpha * ndcg + (1 - alpha) * health_score`

where `alpha = args.val_weight_alpha`

The best-scoring valid weight becomes `w*`.

Typical console output:

```text
[Phase 7] Selecting best trade-off weight w* via validation metrics ...
  w_pref  w_health   w_div |     NDCG   Health      Div |    score
--------------------------------------------------------------------
   1.000     0.000   0.000 |   0.2147   0.3810   0.1465 |  -1.0000
   0.000     1.000   0.000 |   0.1014   0.6125   0.1792 |   0.2547
   0.000     0.000   1.000 |   0.0721   0.2950   0.2614 |   0.1390
   0.333     0.333   0.333 |   0.1863   0.5040   0.2011 |   0.2816
   0.521     0.318   0.161 |   0.2235   0.4621   0.1888 |   0.2951
   0.114     0.701   0.185 |   0.1328   0.5986   0.2163 |   0.2725
   ...
```

Then a selection summary is printed:

```text
Selected w* = [0.521, 0.318, 0.161]  (val score=0.2951)
```

The values above are illustrative. Your exact values depend on the dataset, checkpoint, split, and training dynamics.

## Stage 9: Final Test Evaluation

Once `w*` is selected, the script evaluates the policy on test users using that fixed preference vector.

The same evaluation procedure is used as in validation, except the exclusion mask removes both training and validation positives from candidate pools before ranking.

Typical console output:

```text
[Final] Evaluating MORL on test split ...

=== Test Results (MORL, sequential, w*) ===
  ndcg: 0.22841
  health_score: 0.48765
  diversity: 0.20934
  recall: 0.17358
```

## Stage 10: Save Artifacts

At the end of execution, the script saves `test_results.pt` in `args.output_dir`.

That file contains:

- `best_w`: the selected validation weight
- `test_metrics`: final test metrics
- `val_grid_results`: all validation grid results, stored as `(weight, metrics)` pairs

Typical console output:

```text
Results saved to morl_output/test_results.pt
```

## Summary of What the Script Is Actually Learning

The script does not train a separate model for each objective weight.

Instead, it trains one conditional policy that learns to adapt its recommendation behavior based on the input preference vector. During training, the weight vector changes from episode to episode. During deployment selection, the script searches over candidate preference vectors and chooses one validation-approved operating point `w*`.

So the workflow is:

1. learn a preference-conditioned sequential recommender
2. find a practical trade-off weight on validation
3. report the final performance of that chosen trade-off on test

## Practical Reading of the Outputs

If you run the script and want to interpret the logs quickly:

- setup output confirms the device and loaded tensor shapes
- training output tells you whether policy-gradient optimization is progressing
- the validation table shows the relevance/health/diversity trade-off under different preference weights
- the selected `w*` is the deployment-time operating point chosen by the script
- the final test block is the main result you should compare against baselines

## Important Implementation Detail

The file-level docstring says Phase 2 builds candidate pools from frozen embeddings. That is conceptually correct, but in the actual code this happens inside helper functions during training and evaluation rather than as a standalone visible block inside `main()`.