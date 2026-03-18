# MOPI-HFRS
The implementation code of paper MOPI-HFRS: A Multi-objective Personalized Health-aware Food Recommendation System with LLM-enhanced Interpretation. The author information is redacted since the paper is under review. 

To use reproduce the model, please first download data [here](https://drive.google.com/drive/folders/1u_YC3Z5p6geUSyEKMvjSqtnj8aKPv45r?usp=sharing), and put the data and processed_data directory under the root. Specifically, the benchmark_all.pt and benchmark_macro.pt are the two benchmarks proposed in the paper.  

To run the model, please install the environment requirements and go to the code directory, then use the following command: 

```
python main.py
```

## MORL training loop

The MORL stage depends on the frozen embeddings checkpoint produced by `main.py`. After running the base model training above, stay in the `code` directory and run:

```bash
python -m morl.morl_main \
	--checkpoint embeddings_checkpoint.pt \
	--graph_path ../processed_data/benchmark_macro.pt \
	--device cuda \
	--epochs 200 \
	--batch_size 64 \
	--K 20 \
	--M 200 \
	--wandb_mode offline \
	--output_dir morl_output
```

Common optional arguments:

```bash
python -m morl.morl_main \
	--checkpoint embeddings_checkpoint.pt \
	--graph_path ../processed_data/benchmark_macro.pt \
	--device cuda \
	--lr 1e-3 \
	--hidden_dim 256 \
	--seed 42 \
	--log_every 10 \
	--probe_every 10 \
	--num_probe_users 4 \
	--use_wandb \
	--wandb_project mopi-morl \
	--val_weight_alpha 0.7 \
	--output_dir morl_output
```

This command trains the MORL policy, selects the best trade-off weight on the validation split, and evaluates the final policy on the test split.

The MORL run now emits structured logs and artifacts into the output directory:

- `morl.log`: full console log persisted to disk
- `run_config.json`: resolved run configuration and device info
- `train_metrics.jsonl`: per-epoch training metrics and diagnostics
- `eval_metrics.jsonl`: validation-grid rows and final test metrics
- `validation_weight_grid.csv`: all validation weight vectors with metrics and selected score
- `morl_policy_epoch*.pt` and `morl_policy_final.pt`: checkpoints with embedded stats

Training diagnostics now include:

- mean and standard deviation of episodic return
- reward decomposition (`pref`, `health`, `div`)
- mean episode length
- action entropy and action-position usage rates
- gradient norm
- weight-sampling averages
- fixed-user probe statistics that help detect whether different weight vectors actually change policy behavior

Warnings are emitted when:

- candidate pools are smaller than `K`
- action entropy collapses
- reward variance is near zero
- validation metrics are almost identical across weight vectors

## Terminal monitoring

If you enable W&B logging, you can watch the run live in the terminal using the W&B TUI:

```bash
python -m morl.morl_main \
	--checkpoint embeddings_checkpoint.pt \
	--graph_path ../processed_data/benchmark_macro.pt \
	--device cuda \
	--use_wandb \
	--wandb_mode offline \
	--output_dir morl_output
```

In a second terminal, from the same directory or the run directory, launch:

```bash
wandb beta leet
```

This provides a live terminal dashboard with metric plots, config details, and system stats without needing the browser UI.

k == lengtth of recomendation list

m == full canidate pool for the RL training loop 

batch_size = number of users sampled for each training step


### Key Benchmarks for a sucessful implementation:
- mean_return should improve over epochs, and ideally converge to a stable value.
- different weights should lead to different scores for validation metrics
- if weights give identical results, the policy is not learning the trade-offs 
- some weights should improve NDCG, improve health, improve diversity, vice versa 

#### Final Metrics: Proving our system is better than baseline
- The score of ndcg, health, diversity should all be better than baseline
- or an equal NDCG but better health or diversity
- equal health but better NDCG or diversity
- equal diversity but better NDCG or health
- or better diversitry with a small drop off in NDCG or health, etc.

# Baseline metrics

- test_recall@20: 0.12731, 
- test_precision@20: 0.04667, 
- test_ndcg@20: 0.10252, 
- test_health_score: 0.39399, 
- avg_health_tags_ratio: 6.1568, 
- percentage_recommended_foods: 0.13946


### Better Logging metrics

- | WARNING | Epoch x: probe suggests weeal weight sensitivity
- This means that a small test set in weights aren't actually yielding diff reccomendations 
- 3/18/2026 logging indicates that the model isn't actually leveraging weights proberly since making a change to the weight leads to an irrelevant change`


