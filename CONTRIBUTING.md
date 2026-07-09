# Contributing

## Branch workflow

`main` is protected. Do not push code directly to `main`.

1. Branch from `main`:
   ```bash
   git checkout main && git pull
   git checkout -b feature/short-description
   ```
2. Push and open a pull request.
3. Wait for the **CI** workflow (`validate` job) to pass.
4. Merge (squash is fine).

## CI checks

Pull requests run `.github/workflows/ci.yml`:

- Python syntax check on `phase*.py`
- Phase 1 pipeline smoke test (fetches public MITRE ATT&CK STIX data)
- Basic scan for hardcoded secrets in source files

Phases 2–4 are not run in CI (heavier compute / optional NLP models).

## Local pipeline

```bash
pip install -r requirements.txt
python phase1_data_pipeline.py
python phase2_cooccurrence.py
python phase3_clustering.py
python phase4_nlp_enrichment.py  # optional; downloads embedding models
```

Generated artifacts land in `output/`. Open `dashboard/index.html` for the static view.
