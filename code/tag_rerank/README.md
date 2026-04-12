# TARS — Tag-Augmented Inference Re-scoring

**Author:** Harshit | **Branch:** `harshit-develop`

---

## Motivation

SGSL trains with a Jaccard-based **health loss** that teaches the embedding space
to align with nutritional health tags. But at inference time, `get_metrics()` in
`RCSYS_utils.py` scores items using **only** the dot product:

```python
rating = user_emb @ food_emb.T   # tags completely ignored
```

A food with **perfect nutritional alignment** (Jaccard = 1.0) but a slightly
lower embedding score always loses to a tag-mismatched food. TARS closes this gap.

---

## Formula

```
score(u, i) = (1 - α) × norm(u_emb · i_emb)  +  α × J(tags_u, tags_i)
```

| Component | Description |
|-----------|-------------|
| `norm(u_emb · i_emb)` | Per-user min-max normalized dot-product score, ∈ [0, 1] |
| `J(tags_u, tags_i)` | Jaccard similarity between user and food binary health tags, ∈ [0, 1] |
| `α` | Tag blending coefficient. `α=0` = pure SGSL baseline. `α=1` = pure tag ranking |

The Jaccard matrix is computed efficiently as:

```
intersection(u, i) = tags_u · tags_i       (matrix multiply of binary tensors)
union(u, i)        = ‖tags_u‖₁ + ‖tags_i‖₁ - intersection(u, i)
J(u, i)            = intersection / (union + ε)
```

---

## How It Differs from the Team's Constrained Reranker

| | Constrained Reranker (auto_implement_plan.md) | TARS (this module) |
|---|---|---|
| **When** | Post-ranking (modifies the already-ranked list) | Pre-ranking (modifies the score matrix) |
| **Signal** | Embedding score margins + position locks | Health tag Jaccard similarity |
| **Requires retraining** | No | No |
| **Hyperparameter** | `epsilon`, `lock_positions`, `max_swaps` | Single `alpha` |
| **Fallback** | Forces anchor item | `alpha=0` is exact baseline |

The two approaches are **complementary** — TARS could be applied first (to boost
health-aligned items in the score), then the constrained reranker applied on top.

---

## Package Structure

```
code/tag_rerank/
  __init__.py         Package exports
  scorer.py           jaccard_matrix(), tag_augmented_scores()
  evaluate.py         tars_get_metrics(), tars_eval(), assert_baseline_parity()
  alpha_sweep.py      Sweeps alpha ∈ [0, 1], prints comparison table
  main.py             CLI entrypoint (train + evaluate + optional sweep)
  README.md           This file
```

---

## Quick Start

> **Requires:** Conda environment `FRS` active and data at `processed_data/`

### 1. Parity check (verifies alpha=0 ≡ baseline)
```bash
cd code/tag_rerank
python evaluate.py --parity_check --user_limit 50
```

### 2. Single alpha evaluation (with a trained checkpoint)
```bash
python main.py --alpha 0.1 --checkpoint ../../processed_data/sgsl_checkpoint.pt
```

### 3. Train from scratch and then evaluate
```bash
python main.py --alpha 0.1 --epochs 300 --output_json tars_results.json
```

### 4. Full alpha sweep (fastest way to find optimal alpha)
```bash
python alpha_sweep.py --checkpoint ../../processed_data/sgsl_checkpoint.pt --user_limit 200
```

### 5. Dry-run (random embeddings, no data needed)
```bash
python main.py --skip_training --alpha 0.1 --user_limit 50
```

---

## Expected Results

Based on the design:

| Alpha | NDCG | Health Score | Coverage |
|-------|------|-------------|----------|
| 0.00 (baseline) | highest | lowest | moderate |
| 0.05–0.15 | slight drop | **improved** | likely improved |
| 0.30–0.50 | moderate drop | higher | higher |
| 1.00 (pure Jaccard) | lowest | highest | highest |

The "sweet spot" is the alpha where **health_score improves without NDCG
dropping more than 5%** relative to the baseline. The `alpha_sweep.py` script
auto-selects this value.

---

## Evaluation Protocol

All metrics are computed using the **exact same functions** from `RCSYS_utils.py`:

| Metric | Function |
|--------|----------|
| recall@K | `RecallPrecision_ATk` |
| precision@K | `RecallPrecision_ATk` |
| ndcg@K | `NDCGatK_r` |
| health_score | `calculate_health_score` |
| avg_health_tags | `calculate_average_health_tags` |
| pct_foods | `calculate_percentage_recommended_foods` |

The exclusion masking (train edges masked during val, train+val during test)
is also **identical** to the baseline protocol. `assert_baseline_parity()` in
`evaluate.py` formally verifies this before any experiment runs.

---

## Connection to the Paper

This experiment directly addresses the **training-inference gap** in MOPI-HFRS:
the model is health-aware during training (via the Jaccard health loss in
`RCSYS_utils.py`) but health-blind during inference (pure dot-product scoring
in `get_metrics()`). TARS is a lightweight, **zero-retraining** fix for this gap.

If the sweep finds an alpha > 0 where health_score improves without degrading
NDCG, it demonstrates that:

1. The SGSL embedding space does **not** fully internalize the health tag signal
2. Explicit tag injection at inference time is **complementary** to embedding learning
3. A simple convex blend is sufficient to recover meaningful health improvements

This is the paper's **key experimental result** for this branch.
