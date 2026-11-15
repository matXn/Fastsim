"""Regressions for the imported checkpoint, portable imports and training."""

import shutil

import numpy as np
import pytest

from controllers.neural import Controller, Features, Net
from fastsim import CarParams, State, load_track, run_lap, start_pose


@pytest.mark.parametrize('name', ['A', 'B', 'C', 'D'])
@pytest.mark.parametrize('seed', [None, 90001])
def test_saved_controller_finishes_with_current_physics(name, seed):
    track = load_track(name)
    if seed is not None:
        track = track.perturbed(seed)
    result = run_lap(track, Controller(track, CarParams()), noise_seed=seed)
    assert result.finished, result


def test_model_round_trip_and_features(tmp_path):
    track, params = load_track('A'), CarParams()
    observation, label = Features(track, params).query(State(0, *start_pose(track), 0, 0, 0, 0))
    assert observation.shape == (21,) and label.shape == (2,)
    assert np.isfinite(observation).all() and np.isfinite(label).all()
    net = Net(42)
    model = tmp_path / 'model.npz'
    net.save(model, 2.5)
    restored, cap = Net.load(model)
    assert cap == 2.5
    np.testing.assert_array_equal(net.predict(observation), restored.predict(observation))


def test_neural_package_runs_in_isolated_worker(tmp_path):
    import accept

    package = tmp_path / 'reference'
    package.mkdir()
    (package / 'controller.py').write_text('from controllers.neural import Controller\n', encoding='utf-8')
    workdir = accept.workspace(package)
    try:
        elapsed, reason, _ = accept.run_one(workdir, load_track('B'), CarParams())
        assert reason == 'finished' and elapsed > 0
    finally:
        shutil.rmtree(workdir)


def test_training_smoke(tmp_path):
    from training.train import smoke

    smoke(tmp_path)
    assert (tmp_path / 'smoke.npz').is_file()
