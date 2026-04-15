# Auto Implementation Logs


## Phase: Constrained Rerank (test)
**Timestamp**: 2026-03-25T22:51:01.832644

**Commands Run**:
- `python -m constrained_rerank.main --output_dir ../constrained_rerank_results --K 20 --M 200 --anchor_lock_positions 6 --anchor_epsilon 0.05 --max_swaps_per_list 4`

**Metrics**:
- recall: 0.13460
- precision: 0.04942
- ndcg: 0.10789
- health_score: 0.38543
- avg_health_tags_ratio: 6.14642
- percentage_recommended_foods: 0.18422
- reranked_recall: 0.11547
- reranked_precision: 0.04228
- reranked_ndcg: 0.09653
- reranked_health_score: 0.54059
- reranked_avg_health_tags_ratio: 6.13913
- reranked_percentage_recommended_foods: 0.18112

**Gate Status**: FAIL

**Next Phase Decision**: Tighten constraints
---
