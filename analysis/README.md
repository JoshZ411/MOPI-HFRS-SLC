# MOPI-HFRS Analysis Module

This directory contains exploratory data analysis (EDA) and visualization notebooks for the MOPI-HFRS food recommendation system.

## Author
Harshit — `harshit-develop` branch contribution.

## Contents

| File | Description |
|------|-------------|
| `graph_and_model_analysis.ipynb` | Main EDA notebook — 7-section analysis of the graph, health tags, embeddings, metrics, and reranker hyperparameter guidance |

## How to Run

### Prerequisites
Make sure the Conda environment is active:
```bash
conda activate FRS
```

Download the data from the link in the root README and place:
- `processed_data/benchmark_macro.pt` — macro benchmark graph
- `processed_data/benchmark_all.pt` — full all-item benchmark graph

Both files should be at the **repo root** (one level above `analysis/`).

### Run the Notebook
```bash
cd analysis/
jupyter notebook graph_and_model_analysis.ipynb
```

You can run sections independently — each section loads the graph fresh and is standalone.

## Sections Overview

1. **Graph Structure Analysis** — node counts, edge density, degree distribution
2. **Health Tag Distribution** — user vs. food tag sparsity, tag co-occurrence
3. **User-Food Interaction Patterns** — food popularity, coverage, long-tail analysis
4. **Embedding Space Visualization** — PCA and t-SNE on user/food embeddings post-training
5. **Metric Sensitivity vs. K** — how recall, NDCG, health score change with top-K cutoff
6. **Multi-Objective Loss Analysis** — BPR vs. diversity vs. health loss behavior
7. **Constrained Reranker Parameter Guidance** — recommendations for `epsilon`, `lock_positions`, and `max_swaps` derived from this analysis

## Connection to the Constrained Reranker

Section 7 directly informs the `auto_implement_plan.md` reranker by:
- Using the score distribution from Section 1 to recommend a good `anchor_epsilon` value
- Using coverage data from Section 3 to recommend a `max_swaps_per_list` budget
- Using health score vs. K curves from Section 5 to set the `anchor_lock_positions` floor
