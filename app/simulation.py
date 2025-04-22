# app/simulation.py
import numpy as np
from scipy.stats import beta as beta_dist, bernoulli
import streamlit as st

def est_beta_params(mu, var):
    """
    Estimates Alpha and Beta parameters for a Beta distribution.
    """
    epsilon = 1e-9
    mu = np.clip(mu, epsilon, 1 - epsilon)
    var = np.clip(var, epsilon, None)

    alpha = ((1 - mu) / var - 1 / mu) * mu**2
    beta = alpha * (1 / mu - 1)

    if alpha <= 0 or beta <= 0:
        st.warning(f"Calculated invalid Beta parameters (alpha={alpha:.2f}, beta={beta:.2f}) for mu={mu:.2f}, var={var:.2f}. "
                   f"Variance might be too high. Check input data.")
        return {'alpha': 1.0, 'beta': 1.0}

    return {'alpha': alpha, 'beta': beta}

def est_booking(alpha, beta, N):
    """
    Simulates N booking events.
    """
    p_booking = beta_dist.rvs(a=alpha, b=beta, size=N)
    bookings = bernoulli.rvs(p=p_booking, size=N)
    return bookings

def est_revenue(revenue_lo, rev_lo_prob, revenue_hi, rev_hi_prob, N):
    """
    Simulates N revenue outcomes for a single deal.
    """
    prob_sum = rev_lo_prob + rev_hi_prob
    if not np.isclose(prob_sum, 1.0):
        st.warning(f"Low ({rev_lo_prob:.2f}) and High ({rev_hi_prob:.2f}) revenue probabilities for deal "
                   f"(low: {revenue_lo}, high: {revenue_hi}) do not sum to 1. Normalizing.")
        if prob_sum > 0:
            rev_lo_prob = rev_lo_prob / prob_sum
            rev_hi_prob = 1.0 - rev_lo_prob
        else:
            rev_lo_prob = 0.5
            rev_hi_prob = 0.5

    revenues = np.random.choice(
        a=[revenue_lo, revenue_hi],
        size=N,
        p=[rev_lo_prob, rev_hi_prob],
        replace=True
    )
    return revenues

def est_booking_revenue(revenue_lo, rev_lo_prob, revenue_hi, rev_hi_prob, alpha, beta, N):
    """
    Simulates N scenarios for a deal, considering booking probability and revenue amount.
    """
    booked_sims = est_booking(alpha, beta, N)
    revenue_sims = est_revenue(revenue_lo, rev_lo_prob, revenue_hi, rev_hi_prob, N)
    return booked_sims * revenue_sims

def run_simulations(deals_df, N_SIMULATIONS):
    """
    Runs simulations for all deals and aggregates the results.

    Args:
        deals_df: The deals DataFrame.
        N_SIMULATIONS: The number of simulations to run.

    Returns:
        A NumPy array of total revenue simulations.
    """
    sure_things = deals_df[deals_df['booking_mean'] == 1.0].copy()
    not_sure_things = deals_df[deals_df['booking_mean'] < 1.0].copy()

    all_simulated_revenues = []

    if not not_sure_things.empty:
        beta_params = not_sure_things.apply(
            lambda row: est_beta_params(row['booking_mean'], row['booking_var']),
            axis=1
        )
        not_sure_things['alpha'] = [p['alpha'] for p in beta_params]
        not_sure_things['beta'] = [p['beta'] for p in beta_params]

        if not_sure_things['alpha'].le(0).any() or not_sure_things['beta'].le(0).any():
            st.error("Simulation stopped: Invalid Beta parameters calculated. Check booking_mean and booking_var values.")
            st.dataframe(not_sure_things[not_sure_things['alpha'].le(0) | not_sure_things['beta'].le(0)])
            st.stop()

        for _, row in not_sure_things.iterrows():
            sim_rev = est_booking_revenue(
                row['revenue_lo'], row['rev_lo_prob'], row['revenue_hi'], row['rev_hi_prob'],
                row['alpha'], row['beta'], N_SIMULATIONS
            )
            all_simulated_revenues.append(sim_rev)

    if not sure_things.empty:
        for _, row in sure_things.iterrows():
            sim_rev = est_revenue(
                row['revenue_lo'], row['rev_lo_prob'], row['revenue_hi'], row['rev_hi_prob'],
                N_SIMULATIONS
            )
            all_simulated_revenues.append(sim_rev)

    if not all_simulated_revenues:
        st.warning("No deals found or processed.")
        total_revenue_sims = np.zeros(N_SIMULATIONS)
    else:
        total_revenue_sims = np.sum(np.array(all_simulated_revenues), axis=0)

    return total_revenue_sims
