# 1. Executive Summary
The current repository trains a graph-based recommender with SGSL-style graph masking/fusion and MGDA-style multi-objective loss balancing, then performs one-shot top-K ranking from learned user and food embeddings. The core training/eval flow is in [code/main.py](code/main.py#L8), model definitions are in [code/RCSYS_models.py](code/RCSYS_models.py#L14), and multi-objective + MGDA logic is in [code/RCSYS_utils.py](code/RCSYS_utils.py#L514) and [code/min_norm_solvers.py](code/min_norm_solvers.py#L6).

Your proposed pivot is feasible as an inference-time augmentation:
1. Keep SGSL + MGDA training unchanged.
2. Export frozen embeddings post-training.
3. Train a separate sequential MORL policy on those frozen embeddings.
4. Compare one-shot baseline vs sequential MORL under a standardized exclusion protocol.

I verified this against code and preprocessing artifacts, including how positive health-compatible edges are created in [preprocess/0.3_benchmark.ipynb](preprocess/0.3_benchmark.ipynb#L336).

# 2. Repository Map
Relevant locations and purpose:

1. [code/main.py](code/main.py#L8)
Current training entrypoint; loads benchmark graph, splits data, trains SGSL with pareto_loss, runs validation/test evaluation.

2. [code/RCSYS_models.py](code/RCSYS_models.py#L285)
Model stack:
LightGCN, SignedGCN, feature graph generator, graph channel fusion, SGSL wrapper.

3. [code/RCSYS_utils.py](code/RCSYS_utils.py#L48)
Data split, sampling, objective losses, evaluation metrics, one-shot ranking, pareto loss combination.

4. [code/min_norm_solvers.py](code/min_norm_solvers.py#L141)
Frank-Wolfe minimum-norm solver used to compute convex task weights for MGDA-style balancing.

5. [preprocess/0.3_benchmark.ipynb](preprocess/0.3_benchmark.ipynb#L315)
Constructs benchmark graph, creates edge_label_index from clean_score > 0, assigns user/food tags, saves benchmark_all and benchmark_macro.

6. [preprocess/0.1_user_tagging.ipynb](preprocess/0.1_user_tagging.ipynb#L328)
Builds user health tags from demographics/diet/labs rules.

7. [preprocess/0.2_food_tagging.ipynb](preprocess/0.2_food_tagging.ipynb#L264)
Builds food nutrition tags with thresholded low/high nutrition indicators.

Execution flow today:
1. Load graph and features in [code/main.py](code/main.py#L34).
2. Split interactions and health-labeled subsets via [code/RCSYS_utils.py](code/RCSYS_utils.py#L48).
3. Train SGSL and compute three losses through [code/main.py](code/main.py#L97).
4. Balance losses with MGDA in [code/RCSYS_utils.py](code/RCSYS_utils.py#L514).
5. Rank all items one-shot in [code/RCSYS_utils.py](code/RCSYS_utils.py#L319).

# 3. Deep Understanding of the Existing Codebase

## 3.1 Data Loading and Preprocessing
1. Runtime graph loaded from [code/main.py](code/main.py#L34) using benchmark_macro.pt.
2. Graph contains:
User and food node features in x_dict, user-food edge_index, health-compatible edge_label_index, and tags tensors read in [code/main.py](code/main.py#L60).
3. edge_label_index semantics are defined in preprocessing:
edge_labels are edges whose clean_score > 0 in [preprocess/0.3_benchmark.ipynb](preprocess/0.3_benchmark.ipynb#L333) and assigned in [preprocess/0.3_benchmark.ipynb](preprocess/0.3_benchmark.ipynb#L351).

## 3.2 Graph Construction / Structure Learning
1. Feature-based graph refinement:
GraphGenerator computes per-edge cosine similarity with learnable metric heads in [code/RCSYS_models.py](code/RCSYS_models.py#L237).
2. Semantic graph refinement:
SignedGCN produces signed relation prediction, used as semantic mask in [code/RCSYS_models.py](code/RCSYS_models.py#L134).
3. Graph fusion:
GraphChannelAttLayer normalizes/fuses masks and thresholds to boolean keep/drop in [code/RCSYS_models.py](code/RCSYS_models.py#L262).
4. Final propagation:
LightGCN runs on fused adjacency in [code/RCSYS_models.py](code/RCSYS_models.py#L307).

## 3.3 Embedding Generation / Propagation
1. SGSL forward returns full user/item final and initial embeddings from LightGCN in [code/RCSYS_models.py](code/RCSYS_models.py#L330).
2. Training loop then slices those to minibatches in [code/main.py](code/main.py#L79).
Important implication:
At loop end, in-scope tensors are minibatch slices, not guaranteed full embedding matrices.

## 3.4 Objective Functions
1. Preference objective:
BPR loss in [code/RCSYS_utils.py](code/RCSYS_utils.py#L100).
2. Health objective:
Tag-based Jaccard-modulated ranking loss in [code/RCSYS_utils.py](code/RCSYS_utils.py#L141).
3. Diversity objective:
Top-k item-feature similarity-based term in [code/RCSYS_utils.py](code/RCSYS_utils.py#L158).

## 3.5 Pareto / MGDA Logic
1. pareto_loss computes gradients per task and normalizes them in [code/RCSYS_utils.py](code/RCSYS_utils.py#L514).
2. MinNormSolver Frank-Wolfe computes convex coefficients in [code/min_norm_solvers.py](code/min_norm_solvers.py#L141).
3. Final scalar loss is weighted sum of task losses in [code/RCSYS_utils.py](code/RCSYS_utils.py#L553).

## 3.6 Training Loop
1. Forward SGSL on train split.
2. Sample triplets with sampling helper in [code/RCSYS_utils.py](code/RCSYS_utils.py#L78).
3. Compute Pareto-balanced loss and backprop in [code/main.py](code/main.py#L100).
4. Periodic eval every iters_per_eval in [code/main.py](code/main.py#L116).

## 3.7 Evaluation and Recommendation Generation
1. get_metrics does full user-item dense score matrix and one-shot top-k in [code/RCSYS_utils.py](code/RCSYS_utils.py#L319).
2. Exclusion mask is applied by passing exclude_edge_indices in [code/RCSYS_utils.py](code/RCSYS_utils.py#L334).
3. Metrics:
Recall, precision, ndcg, health score, avg health tags, coverage from [code/RCSYS_utils.py](code/RCSYS_utils.py#L371).

## 3.8 Checkpointing / Reusable Outputs
No model/embedding checkpoint is saved in current main flow. There is no torch.save call in [code/main.py](code/main.py#L8) beyond input load.

# 4. Paper-Concept-to-Code Mapping

1. Health/nutrition bipartite recommendation graph
Status: Clearly implemented.
Where: [preprocess/0.3_benchmark.ipynb](preprocess/0.3_benchmark.ipynb#L350), [code/main.py](code/main.py#L36).
Notes: user-eats-food edge graph.

2. User and food feature handling
Status: Clearly implemented.
Where: [code/main.py](code/main.py#L38), [code/RCSYS_models.py](code/RCSYS_models.py#L295).
Notes: heterogeneous mapping with per-type linear layer.

3. Feature-based graph structure learning
Status: Clearly implemented.
Where: [code/RCSYS_models.py](code/RCSYS_models.py#L237).

4. Health-aware or healthy-edge graph refinement
Status: Partially implemented under different mechanics.
Where: [preprocess/0.3_benchmark.ipynb](preprocess/0.3_benchmark.ipynb#L351), [code/RCSYS_utils.py](code/RCSYS_utils.py#L48), [code/RCSYS_models.py](code/RCSYS_models.py#L321).
Notes: health compatibility drives edge_label_index and pos/neg split for SignedGCN and losses, but there is no separate explicit healthy-edge pruning stage in training loop.

5. Structure pooling / graph fusion
Status: Implemented under different naming.
Where: [code/RCSYS_models.py](code/RCSYS_models.py#L262).
Notes: channel attention fusion of mask_ori, mask_feature, mask_semantic.

6. Embedding initialization and propagation
Status: Clearly implemented.
Where: [code/RCSYS_models.py](code/RCSYS_models.py#L33), [code/RCSYS_models.py](code/RCSYS_models.py#L55).

7. Multi-objective losses
Status: Clearly implemented.
Where: [code/RCSYS_utils.py](code/RCSYS_utils.py#L514).

8. Preference objective
Status: Clearly implemented.
Where: [code/RCSYS_utils.py](code/RCSYS_utils.py#L100).

9. Health objective
Status: Clearly implemented.
Where: [code/RCSYS_utils.py](code/RCSYS_utils.py#L141).

10. Diversity objective
Status: Clearly implemented with caveat.
Where: [code/RCSYS_utils.py](code/RCSYS_utils.py#L158).
Notes: computed from feature-space item similarity, not embedding-space list diversification.

11. Pareto / MGDA optimization
Status: Clearly implemented.
Where: [code/RCSYS_utils.py](code/RCSYS_utils.py#L543), [code/min_norm_solvers.py](code/min_norm_solvers.py#L141).

12. Current one-shot top-K recommendation
Status: Clearly implemented.
Where: [code/RCSYS_utils.py](code/RCSYS_utils.py#L349).

13. Evaluation metrics and split/filtering behavior
Status: Implemented, with important behavior caveat.
Where: [code/main.py](code/main.py#L119), [code/RCSYS_utils.py](code/RCSYS_utils.py#L334).
Notes: current main passes neg_train_edge_index as exclusion set, not pos_train edges. This affects strict seen-item exclusion semantics.

# 5. Current Inference / Recommendation Mechanism
Current pipeline is one-shot, non-sequential:
1. Compute final embeddings from SGSL for evaluated split in [code/RCSYS_utils.py](code/RCSYS_utils.py#L401).
2. Build dense rating matrix user_emb × item_emb^T in [code/RCSYS_utils.py](code/RCSYS_utils.py#L334).
3. Apply exclusion mask by overwriting selected entries with large negative value in [code/RCSYS_utils.py](code/RCSYS_utils.py#L346).
4. Select top-K once per user via topk in [code/RCSYS_utils.py](code/RCSYS_utils.py#L349).
5. Compute static list metrics in [code/RCSYS_utils.py](code/RCSYS_utils.py#L371).

There is no iterative state update, no action masking beyond one-shot exclusion, and no policy conditioned on trade-off weights.

# 6. How the Sequential MORL Pivot Would Fit
Minimal-disruption hook point:
1. Keep SGSL training and pareto_loss unchanged.
2. After training completes in [code/main.py](code/main.py#L152), run one full forward pass to get full embeddings.
3. Save frozen artifacts for MORL in one checkpoint package.

Important nuance:
Do not save users_emb_final or items_emb_final from inside minibatch code paths, because those are sliced tensors after [code/main.py](code/main.py#L79). Recompute full-matrix embeddings post-training for export.

Planned MORL integration architecture:
1. Separate module tree under code/morl (new files).
2. Load frozen embeddings and tags.
3. Build fixed top-M candidate pools by raw dot product.
4. Train conditional sequential policy π(a|s,w) with vector rewards and scalarized optimization.
5. Evaluate one-shot baseline and sequential MORL on same frozen embeddings with standardized seen-item exclusion.

Files to keep untouched during implementation:
1. [code/RCSYS_models.py](code/RCSYS_models.py#L285)
2. [code/RCSYS_utils.py](code/RCSYS_utils.py#L514)
3. [code/min_norm_solvers.py](code/min_norm_solvers.py#L6)

# 7. File-by-File Change Plan
Existing files:

1. [code/main.py](code/main.py#L8)
Action: minimally modified.
Plan: add post-training artifact export only; no change to training loop or pareto_loss invocation.

2. [code/RCSYS_models.py](code/RCSYS_models.py#L285)
Action: reused, inspect only.
Plan: no edits; reuse SGSL forward outputs.

3. [code/RCSYS_utils.py](code/RCSYS_utils.py#L319)
Action: reused, inspect only for baseline/eval parity.
Plan: no edits in training objective logic.

4. [code/min_norm_solvers.py](code/min_norm_solvers.py#L141)
Action: inspect only.
Plan: no edits.

5. [preprocess/0.3_benchmark.ipynb](preprocess/0.3_benchmark.ipynb#L315)
Action: inspect only.
Plan: no edits; used to verify edge_label and tag semantics.

Planned new modules (not currently present):
1. code/morl/environment.py
2. code/morl/policy.py
3. code/morl/training.py
4. code/morl/evaluation.py
5. code/morl/morl_main.py

Optional logging artifact requested by prior outline:
1. auto_logs.md (new, if you want phased implementation trace during coding)

# 8. Ambiguities, Risks, and Required Clarifications
Resolved from your answers:
1. Main can be minimally modified for post-training export.
2. Evaluation should use standard seen-item exclusion (train for val, train+val for test).
3. MORL training pools include training positives, only per-episode duplicate masking.
4. Preference reward should use raw dot-product affinity.

Remaining technical risks and caveats:
1. Current repository baseline numbers may not be directly comparable to corrected exclusion protocol because current main passes neg_train as exclusions in [code/main.py](code/main.py#L119).
2. edge_label_index encodes clean_score > 0 interactions, not generic implicit positives from all eats edges, verified in [preprocess/0.3_benchmark.ipynb](preprocess/0.3_benchmark.ipynb#L333).
3. Diversity objective in training is feature-space and pairwise-top-k based, while MORL diversity may be embedding-list based unless explicitly aligned.
4. No existing artifact schema for exporting splits/tags/ids with embeddings; schema drift risk unless standardized in initial implementation.

# 9. Recommended Implementation Sequence
Phase 1: Post-training export contract
1. Add single post-training export path in main.
2. Export full embeddings and metadata needed by MORL.

Phase 2: MORL data adapter
1. Loader for checkpoint artifacts.
2. Candidate pool builder with fixed top-M per user.

Phase 3: Sequential environment
1. Deterministic K-step episode logic.
2. Reward vector outputs for preference, health, diversity.

Phase 4: Conditional policy + trainer
1. Policy architecture with state+weight conditioning.
2. REINFORCE with Dirichlet-sampled weights.

Phase 5: Evaluation bridge
1. One-shot baseline using same frozen artifacts and corrected filtering.
2. MORL validation grid over weights and test evaluation at selected operating point.

Phase 6: Reproducibility + reporting
1. Save training curves and per-objective returns.
2. Produce side-by-side baseline vs MORL report with explicit protocol notes.

# 10. Questions for the User Before Coding
1. Do you want implementation to evaluate both benchmark_macro and benchmark_all, or benchmark_macro only for the first deliverable?
2. For comparison reporting, should we include both:
current-repo baseline behavior and corrected-exclusion baseline,
or only corrected-exclusion baseline?

# 11. New-Code Scaffolding Blueprint (Generation-Ready)
This section specifies concrete contracts for new modules so implementation can proceed without re-deciding architecture details.

## 11.1 New Directory Layout
1. code/morl/__init__.py
2. code/morl/config.py
3. code/morl/io.py
4. code/morl/candidates.py
5. code/morl/environment.py
6. code/morl/policy.py
7. code/morl/training.py
8. code/morl/evaluation.py
9. code/morl/morl_main.py

## 11.2 Artifact Contract (Produced Post-Training)
Saved artifact path (default): processed_data/embeddings_checkpoint.pt

Required keys:
1. user_emb: float32 tensor [num_users, d]
2. item_emb: float32 tensor [num_items, d]
3. user_tags: float32 or int tensor [num_users, tag_dim]
4. food_tags: float32 or int tensor [num_items, tag_dim]
5. train_edge_index: int64 tensor [2, n_train]
6. val_edge_index: int64 tensor [2, n_val]
7. test_edge_index: int64 tensor [2, n_test]
8. pos_train_edge_index: int64 tensor [2, n_pos_train]
9. pos_val_edge_index: int64 tensor [2, n_pos_val]
10. pos_test_edge_index: int64 tensor [2, n_pos_test]
11. meta: dict with dataset_name, K_default, feature_threshold, hidden_dim, layers, seed, schema_version

Rationale:
This avoids hidden coupling to main runtime state and makes MORL pipeline reproducible and restartable.

## 11.3 Minimal Main Hook (No Training Logic Change)
1. Add a post-training forward pass to recover full embeddings (not minibatch slices).
2. Save the artifact schema above.
3. Keep all existing training, loss, and scheduler mechanics unchanged.

## 11.4 Module Responsibilities and Public Interfaces
1. code/morl/config.py
Defines dataclasses or argparse config groups.
Primary fields: device, seed, K, M, batch_users, lr, episodes, entropy_coef, gamma, eval_weight_grid_size.

2. code/morl/io.py
Functions:
- load_checkpoint(path) -> dict
- validate_checkpoint_schema(ckpt) -> None
- save_policy_checkpoint(path, state_dict, optimizer_state, stats, config) -> None

3. code/morl/candidates.py
Functions:
- build_candidate_pools(user_emb, item_emb, top_m, chunk_size=None) -> LongTensor [num_users, M]
- apply_exclusion_mask(scores, user_to_exclude_items) -> scores

4. code/morl/environment.py
Class: SequentialMORLEnv
Core methods:
- reset(user_id, weight_vector) -> state_dict
- step(action_item_id) -> next_state_dict, reward_vec, done, info
- valid_action_mask() -> BoolTensor [M]

State fields:
1. user_emb
2. agg_item_emb (running mean)
3. tag_union
4. t_norm
5. weight_vector (if conditioned in-state)

Reward vector definition:
1. r_pref_t = dot(user_emb, item_emb[action])
2. r_health_t = jaccard(tag_union_after_step, user_tags[user])
3. r_div_t = negative mean cosine similarity against already selected items

5. code/morl/policy.py
Class: ConditionalPolicy
Input: concatenated state + weight vector (if not already included in state object)
Output: logits over M candidate positions; masking applied before sampling.

6. code/morl/training.py
Functions:
- sample_dirichlet_weights(batch_size, alpha=(1,1,1)) -> FloatTensor [B, 3]
- rollout_episode(env, policy, user_id, w, K) -> trajectory dict
- compute_scalar_returns(reward_vectors, w) -> returns
- reinforce_update(policy, optimizer, trajectories, baseline_mode) -> stats

7. code/morl/evaluation.py
Functions:
- evaluate_one_shot_baseline(...) -> metrics dict
- evaluate_morl_policy(...) -> metrics dict
- select_weight_on_val(weight_grid_metrics, strategy="alpha_beta_with_div_constraint") -> w_star

8. code/morl/morl_main.py
CLI modes:
- train
- eval
- train_eval

## 11.5 Candidate and Masking Policy
Training-time candidate pool:
1. built once from frozen dot-product top-M
2. includes seen interactions
3. per-episode duplicate masking only

Validation/Test masking:
1. validation excludes train positives
2. test excludes train + val positives

## 11.6 Baseline-vs-MORL Evaluation Protocol
To ensure fair comparison:
1. Same frozen embeddings
2. Same user set per split
3. Same exclusion policy
4. Same K
5. Report preference (NDCG/Recall/Precision), health score, diversity, coverage

## 11.7 Generation Order (Concrete)
1. Implement io.py + config.py
2. Implement candidate builder
3. Implement environment dynamics and reward vector
4. Implement policy network with action masking
5. Implement trainer loop with REINFORCE baseline
6. Implement evaluation and w* selection
7. Wire morl_main.py CLI
8. Add post-training export hook in main.py

## 11.8 Immediate Acceptance Checks
1. Shape checks pass for exported embeddings and tags.
2. Candidate pools have shape [num_users, M] and no out-of-range ids.
3. Environment terminates correctly at K or exhaustion.
4. Policy never samples duplicate items within an episode.
5. MORL train and eval run end-to-end from checkpoint without touching SGSL training path.

# 12. Quick-Access Original Code Map for Generation Phase
These existing methods are the highest leverage references to keep open while implementing new MORL code.

## 12.1 Splits, Positives, and Filtering
1. split_data_new in [code/RCSYS_utils.py](code/RCSYS_utils.py#L48)
Use for train/val/test and pos/neg edge conventions.

2. get_user_positive_items in [code/RCSYS_utils.py](code/RCSYS_utils.py#L206)
Use for building exclusion maps and per-user seen sets.

3. get_metrics in [code/RCSYS_utils.py](code/RCSYS_utils.py#L319)
Reference for one-shot scoring, exclusion masking, and top-k extraction.

## 12.2 Embeddings and Forward Contracts
1. SGSL.forward in [code/RCSYS_models.py](code/RCSYS_models.py#L307)
Returns users_emb_final, users_emb_0, items_emb_final, items_emb_0.

2. main training and test evaluation blocks in [code/main.py](code/main.py#L74)
Reference insertion point for post-training export.

## 12.3 Objective and Metric Components Worth Reusing
1. jaccard_similarity in [code/RCSYS_utils.py](code/RCSYS_utils.py#L134)
Can be reused or mirrored for MORL health reward.

2. calculate_health_score in [code/RCSYS_utils.py](code/RCSYS_utils.py#L269)
Align MORL health metric with existing repo semantics.

3. NDCGatK_r and RecallPrecision_ATk in [code/RCSYS_utils.py](code/RCSYS_utils.py#L239)
Reuse for consistent preference reporting.

## 12.4 Data-Semantics References
1. benchmark graph_construction in [preprocess/0.3_benchmark.ipynb](preprocess/0.3_benchmark.ipynb#L315)
Defines edge_label_index and tag tensor construction.

2. clean_score semantics in [preprocess/0.3_benchmark.ipynb](preprocess/0.3_benchmark.ipynb#L211)
Clarifies what "positive" means for health-compatible edges.

## 12.5 Pitfalls to Avoid During Generation
1. Do not serialize minibatch-sliced embeddings from inside training loop.
2. Do not alter pareto_loss or MinNormSolver path.
3. Keep exclusion policy explicit and split-dependent in MORL evaluation.
4. Keep reward vector decomposition logged before scalarization.

# 13. Reward Spec Contract (Locked for Generation)
This contract converts the current high-level MORL reward idea into precise step-wise definitions compatible with sequential list construction.

## 13.1 Core Principle
Each action adds one item to a partial list, updates the environment state, and contributes an immediate vector reward.

At time step t, with selected item i_t and user u:
1. state before action: s_t
2. state after action: s_{t+1}
3. reward vector: r_t = [r_pref_t, r_health_t, r_div_t]

Episode objective:
R_episode = sum_{t=1..T} (w dot r_t) + r_terminal
where T <= K and w is sampled from Dirichlet(1,1,1).

## 13.2 Preference Reward (Per-Step)
Base form:
r_pref_t_raw = dot(user_emb[u], item_emb[i_t])

Normalization (locked):
1. compute per-user candidate mean mu_u and std sigma_u over fixed top-M pool
2. z-score and clip:
r_pref_t = clip((r_pref_t_raw - mu_u) / (sigma_u + 1e-8), -3, 3) / 3

Resulting range: approximately [-1, 1].

## 13.3 Health Reward (Incremental, Not Absolute)
Let tag_union_t be cumulative OR of tags up to step t.
Let J(a,b) be Jaccard similarity.

Incremental health gain:
r_health_t = J(tag_union_t, user_tags[u]) - J(tag_union_{t-1}, user_tags[u])

This ensures the policy is rewarded for new health coverage rather than repeatedly receiving the same absolute score.

## 13.4 Diversity Reward (Marginal Redundancy Penalty)
For t = 1:
r_div_1 = 0

For t > 1:
r_div_t = - mean_{j < t}( cosine(item_emb[i_t], item_emb[i_j]) )

Clamp to [-1, 1] after computation.

## 13.5 Scalarization and Optimization
Per-step scalar reward:
r_scalar_t = w dot [r_pref_t, r_health_t, r_div_t]

Return for policy gradient:
G_t = sum_{tau=t..T} gamma^(tau-t) * r_scalar_tau + terminal contribution

Locked defaults:
1. gamma = 1.0 initially (finite horizon K)
2. optional tuning: gamma in [0.95, 1.0]

## 13.6 Terminal Shaping (Alignment with Final Metrics)
To reduce mismatch between step rewards and final list metrics, add terminal shaping once at episode end:

r_terminal = lambda_pref * ndcg_proxy + lambda_health * final_health + lambda_div * final_div

Locked initial defaults:
1. lambda_pref = 0.2
2. lambda_health = 0.2
3. lambda_div = 0.2

Note:
Terminal shaping is a small additive term and should not dominate dense per-step rewards.

## 13.7 Credit Assignment Stabilization
Policy optimization method:
1. REINFORCE with baseline (required)
2. entropy regularization (small, optional)

Locked baseline strategy (phase 1):
1. moving-average baseline over episode returns
2. advantage = return - baseline

Optional phase 2 upgrade:
1. learned value head baseline

## 13.8 Reward Logging Contract
Store and report both decomposed and scalarized values:
1. per-step vectors r_pref_t, r_health_t, r_div_t
2. per-episode sums of each component
3. scalarized return used for gradient update
4. sampled w used in each episode

This is mandatory for debugging objective collapse and verifying trade-off behavior.

# 14. Risk Closure Checklist (Addressing Known Ambiguities)
This section explicitly closes the key issues identified during planning.

## 14.1 Reward Ambiguity Closed
Status: resolved by Section 13.
Implementation requirement:
1. use incremental health and marginal diversity terms
2. normalize preference scale

## 14.2 Objective Scale Mismatch Closed
Status: resolved.
Implementation requirement:
1. z-score preference by user candidate pool
2. clamp all reward components to bounded ranges

## 14.3 High-Variance Policy Gradient Closed
Status: resolved.
Implementation requirement:
1. REINFORCE must use a baseline
2. log variance of returns for training diagnostics

## 14.4 Training-Evaluation Mismatch Closed
Status: resolved.
Implementation requirement:
1. include terminal shaping tied to final-list quality
2. evaluate with same K and split filtering policy

## 14.5 Action Mask and Candidate Semantics Closed
Status: resolved.
Implementation requirement:
1. fixed top-M candidate pool per user
2. no duplicate selections in episode
3. early termination if no valid actions remain

## 14.6 Embedding Export Correctness Closed
Status: resolved.
Implementation requirement:
1. run full forward pass post-training for export
2. never export minibatch-sliced tensors from loop scope

## 14.7 Baseline Comparison Fairness Closed
Status: resolved.
Implementation requirement:
1. validation excludes train positives
2. test excludes train+val positives
3. report protocol explicitly in final results

# 15. Definition of Done for Generation Phase
Generation is considered complete only if all are true:
1. checkpoint schema validates and loads in MORL pipeline
2. MORL train mode runs end-to-end from frozen checkpoint
3. MORL eval mode outputs NDCG/Recall/Precision + health + diversity + coverage
4. baseline and MORL both use identical split exclusion protocol
5. logs include decomposed rewards, scalar returns, and sampled weight vectors
6. SGSL training path behavior remains unchanged except post-training artifact save
