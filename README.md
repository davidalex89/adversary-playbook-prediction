# Adversary Playbook Prediction

Given that a threat actor uses technique X, predict what other techniques they're likely to use. Built on MITRE ATT&CK Enterprise group-technique attribution data.

## What it does

- **Phase 1** — Fetches the full ATT&CK STIX bundle, extracts techniques, threat groups, and their relationships into clean CSVs
- **Phase 2** — Builds technique co-occurrence matrices (Jaccard similarity and Lift scores) to power a "given technique X, predict technique Y" recommender
- **Phase 3** — Clusters threat groups by TTP similarity (K-Means + UMAP) and techniques by co-usage patterns
- **Phase 4** — Embeds technique descriptions with sentence-transformers and tests whether semantic similarity correlates with behavioral co-occurrence

## Running it

```bash
pip install -r requirements.txt
python phase1_data_pipeline.py   # ~3s, fetches STIX data
python phase2_cooccurrence.py    # ~1s, builds co-occurrence models
python phase3_clustering.py      # ~20s, clustering + UMAP
python phase4_nlp_enrichment.py  # ~15s, requires sentence-transformers model download on first run
```

## Output

All results go to `output/`:
- `techniques.csv`, `groups.csv`, `group_technique_matrix.csv` — raw parsed data
- `cooccurrence_jaccard.csv`, `cooccurrence_lift.csv` — full similarity matrices
- `cooccurrence_top10_jaccard.json`, `cooccurrence_top10_lift.json` — pre-computed top-10 for dashboard
- `group_clusters.csv`, `group_clusters.json`, `group_clusters_umap.png` — group clustering results
- `technique_clusters.csv`, `technique_clusters.json`, `technique_clusters_umap.png` — technique clustering
- `semantic_similarity.csv`, `nlp_correlation_stats.json`, `predicted_expansions.csv` — NLP enrichment

## Dashboard

The `dashboard/` directory contains a standalone Chart.js interactive page. Serve it locally:

```bash
cd dashboard && python -m http.server 8080
```

## Development

`main` is branch-protected. Use feature branches and pull requests — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Data Source

[MITRE ATT&CK Enterprise](https://github.com/mitre/cti) (STIX 2.1 bundle)

## Dependencies

- pandas, numpy, scikit-learn — core ML
- umap-learn — dimensionality reduction
- sentence-transformers — NLP embeddings (Phase 4)
- matplotlib, seaborn — static visualizations
- requests — data fetching
