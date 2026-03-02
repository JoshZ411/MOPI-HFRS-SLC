# MO-DQN Sequential MORL Pivot Automation Plan

You are an AI coding agent operating inside the MOPI-HFRS repository.

Your objective is to augment the MOPI-HFRS pipeline by adding a sequential MORL recommendation stage that operates on frozen embeddings post-training, without modifying the SGSL training phase.

CRITICAL CONSTRAINTS:
- DO NOT modify SGSL backbone, graph preprocessing, or health tag enrichment.
- DO NOT alter GNN training loop or pareto_loss (MGDA gradient balancing).
- DO NOT modify main.py training mechanics.
- SGSL + MGDA training completes unchanged; embeddings are frozen for RL.
- MORL replaces the inference-time recommendation mechanism (one-shot Pareto ranking → sequential list construction).

Pipeline architecture:
```
SGSL Training (MGDA multi-obj balancing) 
    → Frozen user/item embeddings 
    → MORL sequential list construction
    → Evaluation: baseline (one-shot) vs MORL (sequential)
```

---

# IMPLEMENTATION ARCHITECTURE: Inference-Time Augmentation

**Goal:** Replace one-shot Pareto ranking with sequential MORL policy at recommendation time; leave training phase untouched.

**Workflow:**
1. Run `python main.py` normally: SGSL trains with MGDA multi-objective loss balancing (unchanged).
2. After training completes: extract frozen embeddings to checkpoint.
3. Create separate MORL training pipeline in `/code/morl/` that loads frozen embeddings.
4. MORL trains conditional policy π(a|s,w) on frozen embeddings.
5. Evaluation: 
   - Baseline: one-shot Pareto-optimal ranking from main.py eval
   - MORL: sequential K-step list construction using trained policy
   - Compare metrics on same frozen embeddings.

**No modifications to main.py training loop or pareto_loss.**

**Code organization:**
```
code/
  main.py (unchanged; adds one line: embedding checkpoint save)
  morl/
    environment.py (RL environment)
    policy.py (conditional policy network)
    training.py (MORL training loop)
    morl_main.py (entry point)
  RCSYS_*.py (unchanged)
  utils.py (reused for evaluation)
```

**Contribution framing:**
- **MGDA:** gradient-space multi-objective compromise during GNN training (learning phase).
- **MORL:** policy-space horizon-aware trade-off allocation during recommendation (inference phase).
Both operate on identical frozen embeddings.

---

# KEY ASSUMPTIONS & DESIGN CHOICES

Before implementation, lock these decisions:

**Embedding freezing:** GNN training completes; user/item embeddings are frozen (no backprop during RL).

**Candidate pool size M:** Top-M items per user governs action space size.
- Current baseline: TBD (recommend 100–200 items; will tune based on memory).
- Pools computed once at start; reused across entire RL training.

**Reward aggregation:** Per-step rewards [r_pref, r_health, r_div] are scalarized only for policy gradient updates.
- Storage: environment returns full 3-dim vector.
- Validation: evaluate all three objectives independently.

**Tag encoding:** Binary union vector (|food_tags| dimensions); cumulative OR operation during episode.

**State dimensionality:** d_user + d_user + |food_tags| + 1 = 2·d_aggregated + tag_dim + 1.

**Batch sampling:** Small batch (32–64 users) sufficient for discrete action space; no memory bottleneck on A100s expected.

**Evaluation protocol:** Standard train/val/test split (60/20/20); exclude seen training interactions from ranking at test time.

---

MGDA provides a locally Pareto-feasible gradient direction (myopic compromise). We aim to learn long-horizon policies that approximate the achievable trade-offs of sequential recommenders.

Instead of selecting a single compromise during training (as MGDA does), we train a conditional policy that can model different trade-offs over full K-step recommendation trajectories.

Final operating point selection will be done via validation metrics.

---

# PHASE 1: Embedding Extraction (Post-Training)

Goal: Save frozen embeddings after SGSL training completes; prepare for MORL.

Step 1: Verify SGSL/MGDA training completes normally.
- Run standard `python main.py`.
- No modifications to training loop or pareto_loss.
- MGDA gradient balancing operates as designed.

Step 2: Add minimal checkpoint save.
- After training completes (after test eval), save frozen embeddings.
- One-line addition to main.py (post-training, no loop changes):
  ```python
  torch.save({'user_emb': users_emb_final.detach().cpu(), 
              'item_emb': items_emb_final.detach().cpu()}, 
             'embeddings_checkpoint.pt')
  ```

Step 3: Create MORL entry point in `morl/morl_main.py`.
- Load frozen embeddings from checkpoint.
- Load user/item health tags.
- Proceed to Phases 2–6 (environment, policy, training).

SGSL training pipeline completely untouched. MORL operates entirely post-hoc on frozen embeddings.

Deliverable:
    - `embeddings_checkpoint.pt` saved after SGSL training.
    - MORL module loads checkpoint; trains policy independently.
    - main.py remains functionally equivalent (one save call added at end).

---

# PHASE 2: Candidate Pool Construction

Goal: Define discrete action space for sequential MDP.

Step 1: Generate candidate pools per user.
- Use baseline scoring: score = user_emb · item_emb^T
- Select top-M items per user (M ∈ {100, 200, 500}; TBD based on memory/runtime).
- Clarification: Do NOT exclude training positives from pools during RL training
  (GNN is frozen; RL explores all top-M rankings).
- At evaluation/test time, exclude training/validation edges per standard protocol.

Step 2: Store candidate pools.
- Pools remain fixed during RL training.
- No dynamic re-ranking at this stage.

Deliverable:
    candidate_pools[user_id] → list of M item_ids

---

# PHASE 3: MDP Definition (Sequential List Construction)

Goal: Formalize deterministic environment.

State s_t:
    - user embedding (d-dim)
    - aggregated embedding of selected items: mean pooling over [i_1, ..., i_{t-1}] (initialized to zero vector at t=0)
    - cumulative health tag coverage: binary vector encoding union of selected items' tags
    - normalized timestep t/K

Action:
    - select next item from candidate pool (masked to prevent duplicates).
    - Terminal state masking: if candidate pool exhausted (|remaining| = 0), 
      terminate episode early.

Transition:
    - append selected item to list
    - update aggregated embedding via mean update
    - update tag union (bitwise OR of item tags)
    - increment timestep t → t+1

Episode ends after K selections (K=20 per baseline).

Deliverable:
    Deterministic K-step list-construction environment with state/action/transition specs.

---

# PHASE 4: Multi-Objective Reward Structure

Goal: Define interpretable, decomposed reward signals.

Per-step reward components:
    r_pref_t:
        Based on preference loss (BPR-style ranking).
        Computed from interactions between user_emb and selected item_emb.
    r_health_t:
        Jaccard similarity between cumulative tag coverage and user profile tags.
    r_div_t:
        Penalize selecting items with high embedding similarity to already-selected items.
        Compute as negative mean cosine similarity within current list.

Return reward vector per step:
    r_t = [r_pref_t, r_health_t, r_div_t]

Scalarized episodic return (for optimization):
    R = ∑_t w · r_t  (where w is sampled weight vector).

Deliverable:
    Environment returns multi-objective reward vector per step; scalarization happens in training loop.

---

# PHASE 5: Conditional MORL Policy

Goal: Learn a trade-off-aware policy.

Approach:
    Train conditional policy π(a | s, w)

Where:
    w = preference vector sampled from simplex
    w ∈ ℝ^3, w_i ≥ 0, sum(w_i) = 1

Step 1: Extend state:
    s'_t = concat(s_t, w)

Step 2: During training:
    - Sample weight vector w (Dirichlet distribution)
    - Run K-step episode conditioned on w
    - Compute scalar reward for optimization using:
          R_t = w · r_t
    - Apply standard policy gradient update
      (REINFORCE with baseline is sufficient)

Deliverable:
    Single conditional policy capable of producing different trade-offs.

---

# PHASE 6: Training Loop

Goal: Stable episodic training across trade-offs.

Configuration:
    - Batch size: 32–64 users per gradient step (adjust per GPU memory on A100s).
    - Weight sampling: Dirichlet(α = [1,1,1]) for symmetric preference distribution.

Step 1: Sample batch of users (size B).
Step 2: For each user:
    - Sample weight vector w from Dirichlet simplex
    - Run K-step episode with sampled w
    - Collect trajectory {s_t, a_t, r_t} for t=0..K-1
Step 3: Aggregate rewards:
    R_episode = ∑_t w · r_t (scalar per episode)
Step 4: Update policy via REINFORCE:
    ∇ log π(a_t | s_t, w) * R_episode

No gradients flow into GNN.
No retraining of embeddings.

Checkpoint:
    - policy parameters (network weights)
    - training statistics (per-objective returns, policy loss trajectories)

Deliverable:
    Trained conditional MORL policy π(a | s, w).

---

# PHASE 7: Trade-Off Selection via Validation Metrics

Goal: Select final operating point empirically.

Step 1: Define small grid of weight vectors W_eval.
    - (10–20 evenly spaced preference vectors sampled from simplex).
    - Include corner points (1,0,0), (0,1,0), (0,0,1) + uniform weights.

Step 2: For each w in W_eval:
    - Generate Top-K lists on validation split (using π(· | s, w))
    - Compute metrics:
        - NDCG@K (preference quality)
        - Health score (tag coverage alignment)
        - Diversity score (mean pairwise dissimilarity in K-list)

Step 3: Select operating weight w* using one of:
    - Option A: Maximize (α·NDCG + β·Health) subject to Diversity ≥ threshold
    - Option B: Lexicographic: prioritize NDCG, then break ties on Health
    - Option C: Extract Pareto front; select median point
    (Decision: TBD after Phase 6 validation results; default Option A with α=0.7, β=0.3)

Final step: Evaluate π(· | s, w*) on test split.

Deliverable:
    Selected weight vector w*; test-set evaluation metrics under w*.

---

# CONTRIBUTION STATEMENT ALIGNMENT

**SGSL + MGDA (Training):**
- Multi-objective gradient balancing in embedding space during GNN training.
- Balances preference, health, and diversity objectives at each epoch.
- Produces a single set of compromised embeddings.

**MORL (Inference):**
- Takes frozen embeddings; constructs recommendations sequentially.
- Learns a conditional policy π(a|s,w) that adapts to preference weights.
- Performs horizon-aware allocation of objectives across K recommendation steps.
- Defers trade-off decisions to inference time (validation metric selection).

**Key difference:**
- MGDA: myopic, parameter-space compromise during training.
- MORL: foresighted, policy-space allocation during recommendation.

Both use identical frozen embeddings. MORL does not replace MGDA training; it augments the inference mechanism to exploit learned representations more effectively via sequential decision-making.

---

# END CONDITION

System produces:
    - SGSL-trained GNN with frozen embeddings (MGDA-balanced).
    - Trained conditional MORL policy π(a | s, w).
    - Validation-selected trade-off weight w*.
    
Evaluation compares:
    - Baseline: one-shot Pareto ranking (main.py inference) on same embeddings.
    - MORL: sequential list construction using trained policy w*.
    - Metrics: NDCG, health score, diversity, coverage on test set.

Proceed phase-by-phase.
At the end of each phase, summarize implementation decisions and ambiguities before continuing.