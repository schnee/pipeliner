# tests/test_simulation.py
import numpy as np
import pandas as pd
from app.simulation import est_beta_params, est_booking, est_revenue, est_booking_revenue, run_simulations

def test_est_beta_params_valid():
    """Test that est_beta_params returns valid parameters."""
    params = est_beta_params(0.5, 0.01)
    assert params['alpha'] > 0
    assert params['beta'] > 0

def test_est_beta_params_invalid():
    """Test that est_beta_params handles invalid variance."""
    params = est_beta_params(0.5, 0.5)
    assert params['alpha'] == 1.0
    assert params['beta'] == 1.0

def test_est_booking():
    """Test that est_booking returns an array of 0s and 1s."""
    bookings = est_booking(2, 3, 100)
    assert all(x in [0, 1] for x in bookings)

def test_est_revenue():
    """Test that est_revenue returns an array of expected values."""
    revenues = est_revenue(100, 0.6, 200, 0.4, 100)
    assert all(x in [100, 200] for x in revenues)

def test_est_booking_revenue():
    """Test that est_booking_revenue returns an array of expected values."""
    revenues = est_booking_revenue(100, 0.6, 200, 0.4, 2, 3, 100)
    assert all(x in [0, 100, 200] for x in revenues)

def test_run_simulations():
    """Test that run_simulations returns an array of revenue simulations."""
    data = {
        'name': ['Deal 1', 'Deal 2'],
        'revenue_lo': [100, 200],
        'rev_lo_prob': [0.6, 0.4],
        'revenue_hi': [150, 250],
        'rev_hi_prob': [0.4, 0.6],
        'booking_mean': [0.8, 0.9],
        'booking_var': [0.01, 0.02]
    }
    df = pd.DataFrame(data)
    sims = run_simulations(df, 100)
    assert len(sims) == 100
