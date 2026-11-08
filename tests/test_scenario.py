import math

import numpy as np
import pytest

from optionlab.simulation.scenario import simulate_scenario
from optionlab.simulation.stats import gbm_log_return_theory, log_return_stats

ARGS = dict(S0=100.0, drift=0.08, sigma=0.2, T=1.0)


def test_shapes_and_start_values():
    sc = simulate_scenario(**ARGS, n_steps=50, n_paths=1000, n_draw=20, seed=1)
    assert sc.sample_paths.shape == (20, 51)
    assert sc.terminal_prices.shape == (1000,)
    assert sc.times.shape == (51,) and sc.times[0] == 0.0 and sc.times[-1] == 1.0
    assert (sc.sample_paths[:, 0] == 100.0).all()
    assert (sc.terminal_prices > 0).all()


def test_drawn_paths_are_part_of_the_terminal_sample():
    sc = simulate_scenario(**ARGS, n_steps=10, n_paths=500, n_draw=30, seed=2)
    assert np.array_equal(sc.terminal_prices[:30], sc.sample_paths[:, -1])


def test_n_draw_is_capped_at_n_paths():
    sc = simulate_scenario(**ARGS, n_steps=10, n_paths=15, n_draw=100, seed=3)
    assert sc.sample_paths.shape[0] == 15
    assert sc.terminal_prices.shape == (15,)


def test_reproducible_with_a_seed():
    a = simulate_scenario(**ARGS, n_paths=500, seed=5)
    b = simulate_scenario(**ARGS, n_paths=500, seed=5)
    c = simulate_scenario(**ARGS, n_paths=500, seed=6)
    assert np.array_equal(a.terminal_prices, b.terminal_prices)
    assert not np.array_equal(a.terminal_prices, c.terminal_prices)


def test_terminal_prices_follow_the_gbm_law():
    n = 100_000
    sc = simulate_scenario(**ARGS, n_steps=20, n_paths=n, n_draw=100, seed=7)
    sample = log_return_stats(sc.terminal_prices, 100.0)
    theory = gbm_log_return_theory(0.08, 0.2, 1.0)
    assert abs(sample["mean"] - theory["mean"]) < 4 * math.sqrt(theory["variance"] / n)
    assert sample["variance"] == pytest.approx(theory["variance"], rel=0.03)


def test_many_paths_do_not_store_every_full_path():
    sc = simulate_scenario(**ARGS, n_steps=252, n_paths=200_000, n_draw=100, seed=8)
    assert sc.sample_paths.size == 100 * 253          # only the drawn paths are kept in full


@pytest.mark.parametrize("bad", [dict(n_draw=0), dict(n_paths=0), dict(sigma=-0.1), dict(T=0.0)])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        simulate_scenario(**{**ARGS, "n_paths": 100, **bad})