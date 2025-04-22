# app/data_utils.py
import pandas as pd
import numpy as np
import streamlit as st
import io

REQUIRED_COLS = ['name', 'revenue_lo', 'rev_lo_prob', 'revenue_hi', 'rev_hi_prob', 'booking_mean', 'booking_var']

def load_deals_data(file_path_or_buffer):
    """
    Loads deals data from a CSV file or a file-like buffer.

    Args:
        file_path_or_buffer: Either a file path (string) or a file-like buffer (e.g., from Streamlit's file_uploader).

    Returns:
        A pandas DataFrame containing the deals data, or None if there's an error.
    """
    try:
        if isinstance(file_path_or_buffer, str):
            deals_df = pd.read_csv(file_path_or_buffer)
        elif isinstance(file_path_or_buffer, io.StringIO):
            deals_df = pd.read_csv(file_path_or_buffer)
        elif hasattr(file_path_or_buffer, 'getvalue'):
            stringio = io.StringIO(file_path_or_buffer.getvalue().decode("utf-8"))
            deals_df = pd.read_csv(stringio)
        else:
            raise ValueError("Invalid input type. Expected file path or file-like buffer.")

        # Basic Column Check
        if not all(col in deals_df.columns for col in REQUIRED_COLS):
            if isinstance(file_path_or_buffer, str):
                raise ValueError(f"CSV must contain the following columns: {', '.join(REQUIRED_COLS)}")
            else:
                st.error(f"CSV must contain the following columns: {', '.join(REQUIRED_COLS)}")
                return None

        # Type Conversion and NaN Check
        for col in REQUIRED_COLS:
            if col != 'name':
                deals_df[col] = pd.to_numeric(deals_df[col], errors='coerce')

        if deals_df.isnull().any().any():
            if isinstance(file_path_or_buffer, str):
                print("Warning: Some non-numeric values were found in numeric columns and were converted to NaN. Please check your CSV.")
            else:
                st.warning("Some non-numeric values were found in numeric columns and were converted to NaN. Please check your CSV.")
                st.dataframe(deals_df[deals_df.isnull().any(axis=1)])
            # Optionally stop execution or proceed with NaN handling later

        return deals_df

    except Exception as e:
        if isinstance(file_path_or_buffer, str):
            raise e
        else:
            st.error(f"Error reading or processing CSV file: {e}")
            return None

def validate_deals(deals_df):
    """
    Validates the input deals DataFrame.

    Checks:
    1. If booking_mean +/- booking_var goes outside [0, 1].
    2. If rev_lo_prob + rev_hi_prob does not sum to 1.

    Args:
        deals_df: The deals DataFrame.

    Returns:
        A DataFrame of invalid deals with the reason, or an empty DataFrame if all are valid.
    """
    invalid_deals = []

    # Check booking probability range
    check_booking = deals_df[(deals_df['booking_mean'] < 1) & (deals_df['booking_var'] > 0)].copy()
    if not check_booking.empty:
        check_booking['high_prob'] = check_booking['booking_mean'] + check_booking['booking_var']
        check_booking['low_prob'] = check_booking['booking_mean'] - check_booking['booking_var']
        oob_booking = check_booking[
            (check_booking['high_prob'] > 1.0) | (check_booking['low_prob'] < 0.0)
        ]
        if not oob_booking.empty:
            oob_booking['reason'] = "Booking probability range (mean +/- var) extends outside [0, 1]"
            invalid_deals.append(oob_booking[['name', 'booking_mean', 'booking_var', 'reason']])

    # Check revenue probabilities sum to 1
    check_revenue = deals_df.copy()
    check_revenue['prob_sum'] = check_revenue['rev_lo_prob'] + check_revenue['rev_hi_prob']
    oob_revenue = check_revenue[~np.isclose(check_revenue['prob_sum'], 1.0)]

    if not oob_revenue.empty:
        oob_revenue['reason'] = "Sum of rev_lo_prob and rev_hi_prob is not 1"
        existing_invalid_names = set(pd.concat(invalid_deals)['name']) if invalid_deals else set()
        new_invalid_revenue = oob_revenue[~oob_revenue['name'].isin(existing_invalid_names)]
        if not new_invalid_revenue.empty:
            invalid_deals.append(new_invalid_revenue[['name', 'rev_lo_prob', 'rev_hi_prob', 'reason']])

    if not invalid_deals:
        return pd.DataFrame()
    else:
        all_invalid = pd.concat(invalid_deals).drop_duplicates(subset=['name'])
        return all_invalid
