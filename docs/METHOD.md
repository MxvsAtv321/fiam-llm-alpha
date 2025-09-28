# Method

- Sections: MD&A and Risk Factors processed separately.
- Features: LM lexicon counts per 1k words, readability, simple novelty proxy.
- Section scores: cross-sectional z then tanh: `s_mdna = tanh(z_mdna)`, `s_risk = -tanh(z_risk)`.
- Composite: average for stub; in full, learn weights on 2013–2014 then freeze.
- Supervised: Elastic Net or LightGBM on 2005–2012, validate 2013–2014, OOS 2015–2025:05.
- Mapping to [-1,1]: rank → z → tanh monthly.
- Blending: linear blender on validation.
