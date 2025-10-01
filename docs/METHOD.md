# Method

- Sections: MD&A and Risk Factors processed separately.
- Features: LM lexicon counts per 1k words, readability, simple novelty proxy.
- Section scores: cross-sectional z then tanh: `s_mdna = tanh(z_mdna)`, `s_risk = -tanh(z_risk)`.
- Composite: average for stub; in full, learn weights on 2013–2014 then freeze.
- Supervised: Elastic Net or LightGBM on 2005–2012, validate 2013–2014, OOS 2015–2025:05.
- Mapping to [-1,1]: rank → z → tanh monthly.
- Blending: linear blender on validation.

## Numeric model and blending (Phase 3)
- Numeric model: ElasticNet on a config-driven whitelist (147-like factors). Features selected via stability selection (seeds [11,13,17,19,23], keep features appearing in ≥ ceil(K/2), cap to `max_features`).
- Scores→weights: cross-sectional rank → Gaussian z (erfinv approx) → tanh to map into [-1,1].
- Blend: `w_blend = 0.8*w_numeric + 0.2*combined_score` (configurable). Then enforce holdings count [min,max] by |weight| and normalize exposures so sum(long)=+1 and sum(short)=−1.
- Risk adjustments: optional multiplicative/additive adjustments applied per month, followed by re-normalization.
