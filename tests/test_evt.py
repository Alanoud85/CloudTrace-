import numpy as np

from cloudtrace_mvad.evt import EVTModel


def test_archived_evt_thresholds():
    model = EVTModel(0.98, 13.400282392621353, -0.29060146758924, 6.630643419671944, 130, 6486)
    assert np.isclose(model.threshold(0.001), 26.663436101404606)
    assert np.isclose(model.threshold(0.005), 20.96624594653136)
    assert np.isclose(model.threshold(0.01), 17.562983377082237)
