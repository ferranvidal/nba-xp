import numpy as np
from xgboost import XGBRegressor

from nbaxp.model import core, tracking


def tiny_model(n_cols: int) -> XGBRegressor:
    rng = np.random.default_rng(0)
    return XGBRegressor(n_estimators=2).fit(rng.normal(size=(20, n_cols)), rng.normal(size=20))


def test_champion_moves_only_when_feature_list_changes(tmp_path, monkeypatch):
    tracking.setup_mlflow(tmp_path)
    assert not tracking.has_champion(tmp_path)

    monkeypatch.setattr(core, "FEATURE_COLS", ["a", "b"])
    assert tracking.register(tiny_model(2), 1.0) == 1  # no champion yet: v1 becomes champion
    assert tracking.register(tiny_model(2), 1.0) == 2  # same features: v1 stays champion
    assert tracking.has_champion(tmp_path)
    assert tracking.load_champion(tmp_path)[2] == ["a", "b"]

    monkeypatch.setattr(core, "FEATURE_COLS", ["a", "b", "c"])
    assert not tracking.has_champion(tmp_path)  # v1 cannot score the new feature table
    tracking.register(tiny_model(3), 1.0)
    assert tracking.load_champion(tmp_path)[2] == ["a", "b", "c"]
