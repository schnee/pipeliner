# app/main.py
import streamlit as st
import pandas as pd
import numpy as np
from app.simulation import run_simulations
from app.data_utils import load_deals_data, validate_deals
from app.visualization import plot_revenue_distribution

# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("Booking Target Revenue Probability")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Inputs")

    uploaded_file = st.file_uploader(
        "Upload Deals CSV File",
        type=["csv"],
        help="CSV must have columns: name, revenue_lo, rev_lo_prob, revenue_hi, rev_hi_prob, booking_mean, booking_var"
    )

    target_rev = st.number_input(
        "Target Revenue ($)",
        min_value=0.0,
        value=1000000.0,
        step=50000.0,
        format="%f"
    )

    N_SIMULATIONS = st.number_input(
        "Number of Simulations",
        min_value=1000,
        max_value=5000000,
        value=100000,
        step=10000,
        help="Higher numbers increase accuracy but take longer to compute."
    )

    submitted = st.button("Run Simulation")

# --- Main Panel ---

if 'deals_df' not in st.session_state:
    st.session_state.deals_df = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Load data when file is uploaded
if uploaded_file is not None:
    st.session_state.deals_df = load_deals_data(uploaded_file)

# --- Simulation Logic ---
if submitted and st.session_state.deals_df is not None:
    deals = st.session_state.deals_df.copy()

    # 1. Validate Deals
    invalid_deals = validate_deals(deals)
    if not invalid_deals.empty:
        st.error("Invalid deal configurations found. Please correct the data:")
        st.dataframe(invalid_deals)
        st.stop()

    # 2. Run Simulations
    total_revenue_sims = run_simulations(deals, N_SIMULATIONS)

    # 3. Calculate Probability of Success & Quantiles
    prob_of_success = np.mean(total_revenue_sims >= target_rev)
    probs_to_calc = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    revenue_quantiles = np.quantile(total_revenue_sims, probs_to_calc)

    prob_df = pd.DataFrame({
        'probability': [f"{100*(1-p):.0f}%" for p in probs_to_calc],
        'revenue': revenue_quantiles
    })
    prob_df = prob_df.iloc[::-1]

    # Store results in session state
    st.session_state.results = {
        'total_revenue_sims': total_revenue_sims,
        'prob_of_success': prob_of_success,
        'prob_df': prob_df,
        'target_rev': target_rev
    }

# --- Display Results ---
if st.session_state.results:
    results = st.session_state.results
    deals = st.session_state.deals_df

    tab1, tab2, tab3 = st.tabs(["📈 Simulation Results", "📊 Input Deals", "❓ Help"])

    with tab1:
        st.subheader("Revenue Distribution Plot")
        plot_revenue_distribution(
            results['total_revenue_sims'],
            results['target_rev'],
            results['prob_of_success']
        )
        st.markdown(f"**Target Revenue:** `${results['target_rev']:,.0f}`")
        st.markdown(f"**Estimated Probability of Exceeding Target:** `{results['prob_of_success']:.1%}`")

        st.divider()

        st.subheader("Revenue Probability Estimates")
        st.markdown("The table shows the estimated minimum revenue you have a certain probability of achieving or exceeding.")
        prob_df_display = results['prob_df'].copy()
        prob_df_display['revenue'] = prob_df_display['revenue'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(prob_df_display, hide_index=True)

    with tab2:
        st.subheader("Input Deals Data")
        if deals is not None:
            st.dataframe(deals)
        else:
            st.info("Upload a CSV file to see the deals data here.")

    with tab3:
        st.subheader("Help Information")
        st.markdown("""
        This application helps estimate the probability of attaining a target revenue from a 'basket' of sales deals **within a given time period**.

        **How to Use:**
        1.  Set the desired **Target Revenue** in the sidebar.
        2.  Prepare a **CSV file** containing your deals. You can use this Google Sheet template - make a copy, edit your deals, ensure column names are unchanged, and export as CSV.
            * **Required Columns:** `name`, `revenue_lo`, `rev_lo_prob`, `revenue_hi`, `rev_hi_prob`, `booking_mean`, `booking_var`.
            * `name`: Unique identifier for the deal.
            * `revenue_lo`: Low-end estimated revenue if the deal closes.
            * `rev_lo_prob`: Probability of the revenue being the low-end amount (if the deal closes).
            * `revenue_hi`: High-end estimated revenue if the deal closes.
            * `rev_hi_prob`: Probability of the revenue being the high-end amount (must be `1 - rev_lo_prob`).
            * `booking_mean`: The average probability (0 to 1) that the deal will close (book).
            * `booking_var`: The variance around the `booking_mean`. Represents uncertainty in the booking probability. Use 0 variance for deals with fixed probabilities (e.g., 1.0 mean and 0 var for a 100% certain deal).
        3.  Upload the CSV file using the **Upload Deals CSV File** button in the sidebar.
        4.  Adjust the **Number of Simulations** if desired (more simulations = potentially more accurate but slower).
        5.  Click the **Run Simulation** button.

        **Output:**
        * **Simulation Results Tab:**
            * A **density plot** showing the distribution of possible total revenues based on the simulations. The shaded area represents simulations exceeding the target revenue.
            * The overall **probability of exceeding the target revenue**.
            * A **table** showing specific revenue levels and the estimated probability of achieving *at least* that much revenue.
        * **Input Deals Tab:** Shows the data you uploaded.
        * **Help Tab:** This information.

        **Important Considerations:**
        * **Data Validity:** Ensure `rev_lo_prob + rev_hi_prob = 1` for each deal. Also, ensure the booking probability range (`booking_mean ± booking_var`) stays within [0, 1]. The app performs basic checks.
        * **Variance:** The `booking_var` must be mathematically possible for the given `booking_mean` (specifically, `variance < mean * (1 - mean)`). If the variance is too high, the calculated Beta distribution parameters (alpha, beta) can become invalid, and the simulation may warn or stop.
        * **Model Assumptions:** This model assumes deal outcomes are independent and that the Beta distribution is appropriate for modeling booking uncertainty.

        **GitHub:** Original R code source (if applicable)
        """)

elif submitted and st.session_state.deals_df is None:
    st.warning("Please upload a CSV file first.")

else:
    if st.session_state.deals_df is not None:
        st.info("Deals file loaded. Adjust target revenue and click 'Run Simulation' in the sidebar.")
        st.dataframe(st.session_state.deals_df)
    else:
        st.info("Upload a deals CSV file using the sidebar to get started.")
