from fiam_llm.model_text_en import make_windows


def test_make_windows_exact():
    wins = make_windows(train_start="2005-01", first_oos_year=2015, last_oos="2025-05")
    w2015 = [w for w in wins if w["oos_year"] == 2015][0]
    assert w2015["train"][0] == "2005-01" and w2015["train"][-1] == "2012-12"
    assert w2015["val"][0] == "2013-01" and w2015["val"][-1] == "2014-12"
    assert w2015["oos"][0] == "2015-01" and w2015["oos"][-1] == "2015-12"
