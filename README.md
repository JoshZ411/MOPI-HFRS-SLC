# MOPI-HFRS
The implementation code of paper MOPI-HFRS: A Multi-objective Personalized Health-aware Food Recommendation System with LLM-enhanced Interpretation. The author information is redacted since the paper is under review. 

To use reproduce the model, please first download data [here](https://drive.google.com/drive/folders/1u_YC3Z5p6geUSyEKMvjSqtnj8aKPv45r?usp=sharing), and put the data and processed_data directory under the root. Specifically, the benchmark_all.pt and benchmark_macro.pt are the two benchmarks proposed in the paper.  

To run the model, please install the environment requirements and go to the code directory, then use the following command: 

```
python main.py
```

## Architecture: Advantage Actor-Critic (A2C) Reranker

The MORL stage is an **A2C reinforcement learning reranker** built on top of frozen LightGCN embeddings. It does not retrain the GNN — it learns a sequential selection policy that reranks the full item catalog to jointly optimize relevance and health alignment.

### How It Works

The policy operates as a K-step episode per user. At each step it selects one food item from all 6,769 candidates, receives a per-step reward, and updates its internal state before the next selection.

**State vector** at step `t`:
```
s_t = concat(user_emb[u], agg_emb, tag_coverage, t/K)
       ↑ frozen (128d)   ↑ mean of  ↑ health tags  ↑ normalized
                           selected   covered so far   timestep
```

**Policy network (`ConditionalPolicy`):**
```
state_encoder:    Linear(321→256)→ReLU → Linear(256→256)→ReLU  → state_hidden (256d)
candidate_encoder: Linear(128→256)→ReLU                         → cand_hidden  (6769×256d)
logit_i = dot(state_hidden, cand_hidden_i)  →  softmax over all 6769 items
```

**Critic network (`ValueHead`):**
```
Linear(321→256)→ReLU → Linear(256→128)→ReLU → Linear(128→1)  →  V(s_t)
```

**Reward per step:**
```
r_t = r_rel + β · r_health
r_rel    = 1.0 if selected item ∈ train_pos_items[user], else 0.0
r_health = 1.0 if item_tags[item] ∩ user_tags[user] ≠ ∅,  else 0.0
```

**A2C update:**
```
G_t       = discounted return from step t (γ=0.99)
A_t       = G_t − V(s_t)                           ← advantage (low variance)
loss      = −Σ A_t·log π(a_t|s_t)                  ← policy loss
          + 0.5 · MSE(V(s_t), G_t)                 ← critic loss
          − 0.01 · H(π)                             ← entropy bonus
```

**β** controls the relevance–health trade-off without retraining: lower β preserves NDCG, higher β maximizes health alignment.

### Imitation Pretraining

Before RL begins, the policy is pretrained for 50 epochs to replicate the GNN's top-K rankings via cross-entropy loss (behavioral cloning). This warm-starts the policy at ~0.222 NDCG so RL fine-tuning improves health from a strong relevance floor rather than from random.

### Full Item Space

The policy acts over all 6,769 items at every step — there is no candidate pool ceiling. The previous M=500 pooling design imposed a structural recall ceiling of ~0.498 (maximum recoverable NDCG ~0.52) and prevented health-compatible items outside the top-500 from ever being recommended. Removing the pool eliminates this constraint entirely.

---

## MORL Training

The MORL stage depends on the frozen embeddings checkpoint produced by `main.py`. After running base model training, stay in the `code` directory and run:

```bash
WANDB_MODE=offline python -m morl.morl_main \
    --checkpoint embeddings_checkpoint.pt \
    --graph_path ../processed_data/benchmark_macro.pt \
    --device cuda \
    --epochs 5000 \
    --K 20 \
    --M 500 \
    --batch_size 64 \
    --lr 1e-4 \
    --hidden_dim 256 \
    --gamma 0.99 \
    --beta 0.5 \
    --entropy_coef 0.3 \
    --value_coef 0.5 \
    --pretrain_epochs 50 \
    --pretrain_lr 1e-3 \
    --log_every 10 \
    --val_eval_every 100 \
    --seed 42 \
    --use_wandb \
    --wandb_mode offline \
    --output_dir ../morl_output
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--epochs` | Number of RL training epochs (5000 recommended) |
| `--K` | Recommendation list length per user |
| `--M` | Candidate pool size for periodic val eval (training always uses full item space) |
| `--beta` | Health–relevance trade-off weight. Lower = more NDCG, higher = more health |
| `--pretrain_epochs` | Behavioral cloning epochs before RL (50 recommended) |
| `--entropy_coef` | Entropy bonus coefficient — prevents policy from collapsing to one item |
| `--value_coef` | Critic loss weight in total loss |
| `--gamma` | Discount factor for episode returns |

### β Trade-off Guide

| β | Expected NDCG | Expected Health |
|---|--------------|----------------|
| 0.5 | ~0.206 (near GNN baseline) | ~0.696 |
| 1.5 | ~0.163 | ~0.70 |
| 4.0 | ~0.160 | ~0.920 |
| 20.0 | ~0.172 | ~0.963 |

### Output Artifacts

- `morl.log` — full console log
- `run_config.json` — resolved run configuration
- `train_metrics.jsonl` — per-epoch training metrics
- `eval_metrics.jsonl` — periodic val evaluation rows
- `morl_policy_epoch*.pt` and `morl_policy_final.pt` — checkpoints containing `policy_state_dict`, `value_head_state_dict`, `optimizer_state_dict`, and full training `stats`
- `test_results.pt` — final test split metrics

### Terminal Monitoring

Launch the W&B terminal dashboard in a second terminal from the run directory:

```bash
wandb beta leet run <path/to/run-*.wandb>
```

---

## Baseline Metrics

### Original Paper Evaluation (Buggy — Edge-Indexed)

The original `main.py` evaluation contained three bugs that computed NDCG over test *edges* instead of per *user*, producing an uninterpretable metric. These numbers correspond to the buggy evaluation:

| Metric | Value |
|--------|-------|
| test_ndcg@20 | 0.10252 |
| test_recall@20 | 0.12731 |
| test_precision@20 | 0.04667 |
| test_health_score | 0.39399 |

### Corrected GNN Baseline (Per-User Evaluation)

After fixing the evaluation to standard per-user NDCG@K (`scores = user_emb @ item_emb.T`, one row per user):

| Metric | Value |
|--------|-------|
| test_ndcg@20 | **0.22200** |
| test_recall@20 | **0.23800** |
| test_health_score | **0.460** |

The GNN model weights are identical — only the evaluation logic changed.

### MORL Results (β=0.5, 5000 epochs)

| Metric | GNN Baseline | MORL Policy | Change |
|--------|-------------|-------------|--------|
| ndcg@20 | 0.222 | **0.206** | −7% |
| recall@20 | 0.238 | **0.236** | −1% |
| health_score | 0.460 | **0.696** | **+51%** |
| diversity | — | 0.162 | — |

Health alignment improves by 51% with a minimal 7% NDCG trade-off. The β parameter allows practitioners to tune the operating point without retraining.

