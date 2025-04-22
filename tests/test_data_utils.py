# tests/test_data_utils.py
import pandas as pd
import numpy as np
from app.data_utils import validate_deals

def test_validate_deals_valid():
    """Test that validate_deals returns an empty DataFrame for valid data."""
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
    result = validate_deals(df)
    assert result.empty

def test_validate_deals_invalid_booking_range():
    """Test that validate_deals identifies invalid booking probability ranges."""
    data = {
        'name': ['Deal 1', 'Deal 2'],
        'revenue_lo': [100, 200],
        'rev_lo_prob': [0.6, 0.4],
        'revenue_hi': [150, 250],
        'rev_hi_prob': [0.4, 0.6],
        'booking_mean': [0.8, 0.9],
        'booking_var': [0.5, 0.2]  # Invalid variance
    }
    df = pd.DataFrame(data)
    result = validate_deals(df)
    assert not result.empty
    assert "Booking probability range" in result['reason'].iloc[0]

def test_validate_deals_invalid_revenue_probs():
    """Test that validate_deals identifies invalid revenue probabilities."""
    data = {
        'name': ['Deal 1', 'Deal 2'],
        'revenue_lo': [100, 200],
        'rev_lo_prob': [0.6, 0.5],  # Invalid probabilities
        'revenue_hi': [150, 250],
        'rev_hi_prob': [0.5, 0.6],  # Invalid probabilities
        'booking_mean': [0.8, 0.9],
        'booking_var': [0.01, 0.02]
    }
    df = pd.DataFrame(data)
    result = validate_deals(df)
    assert not result.empty
    assert "Sum of rev_lo_prob and rev_hi_prob is not 1" in result['reason'].iloc[0]
