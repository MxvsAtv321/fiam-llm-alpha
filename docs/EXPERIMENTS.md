# Experiments

## Reproduce main OOS results

1. Ensure raw data is placed under `data/raw/` as per DATA_DICTIONARY.
2. Run Make targets in order.
3. MLflow logs (params/metrics/artifacts) are stored under `mlruns/`.

## Notes
- Stubs are used for CI speed; swap to FinBERT and full features for real runs.

## MLflow
- Experiment: `fiam_llm_alpha`
- Runs:
  - `phase2_text_models`: logs composite weights and validation metrics.
  - `phase2_blend`: logs validation IC for text vs blended predictions.
