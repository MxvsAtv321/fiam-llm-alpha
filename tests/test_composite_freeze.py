import json
import numpy as np
import pandas as pd
from fiam_llm.calibrate import learn_composite_weights, combine_score


def test_composite_weights_freeze(tmp_path):
    # Build simple validation df (2013-2014)
    val = pd.DataFrame({
        "s_mdna": [0.1, -0.2, 0.3, 0.0],
        "s_risk": [-0.1, 0.2, -0.1, 0.0],
        "tone_delta": [0.0, 0.1, -0.1, 0.0],
        "novelty": [0.2, 0.1, 0.0, 0.0],
        "hedge_ratio": [0.05, 0.0, 0.1, 0.0],
        "exret_next": [0.0, 0.0, 0.0, 0.0],
    })
    w1 = learn_composite_weights(val)
    wpath = tmp_path / "w.json"
    wpath.write_text(json.dumps(w1))
    w2 = json.loads(wpath.read_text())
    # Apply to OOS rows
    oos = pd.DataFrame({
        "s_mdna": [0.2, -0.2],
        "s_risk": [0.1, -0.1],
        "tone_delta": [0.0, 0.0],
        "novelty": [0.1, 0.1],
        "hedge_ratio": [0.0, 0.0],
    })
    s1 = oos.apply(lambda r: combine_score(r, w1), axis=1)
    s2 = oos.apply(lambda r: combine_score(r, w2), axis=1)
    assert np.allclose(s1.values, s2.values)
