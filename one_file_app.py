# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import beta as beta_dist, bernoulli, gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import io # Needed for reading file uploader buffer

# --- Helper Functions ---

def est_beta_params(mu, var):
    """
    Estimates Alpha and Beta parameters for a Beta distribution given mean (mu)
    and variance (var).

    Note: This uses the formula provided in the R code. Be aware that this
    parameterization might differ from other standard calculations and requires
    var < mu * (1 - mu). It can also potentially yield negative alpha/beta
    values if variance is too high relative to the mean.
    """
    # Added small epsilon to prevent division by zero or invalid values
    epsilon = 1e-9
    mu = np.clip(mu, epsilon, 1 - epsilon)
    var = np.clip(var, epsilon, None) # Ensure variance is positive

    # Calculate alpha using the formula from the R code
    alpha = ((1 - mu) / var - 1 / mu) * mu**2

    # Calculate beta using the formula from the R code
    beta = alpha * (1 / mu - 1)

    # Ensure alpha and beta are positive
    if alpha <= 0 or beta <= 0:
        # Fallback or raise error: If parameters are invalid,
        # maybe return uniform distribution parameters or handle error
        # For now, returning parameters that might lead to errors downstream
        # or could be caught later. A more robust solution might be needed.
        st.warning(f"Calculated invalid Beta parameters (alpha={alpha:.2f}, beta={beta:.2f}) for mu={mu:.2f}, var={var:.2f}. "
                   f"Variance might be too high. Check input data.")
        # Return dummy positive values to avoid immediate crashes in rvs,
        # although the simulation result will be meaningless for this deal.
        return {'alpha': 1.0, 'beta': 1.0}


    return {'alpha': alpha, 'beta': beta}

def est_booking(alpha, beta, N):
    """
    Simulates N booking events (0 or 1) based on drawing probabilities
    from a Beta distribution.
    """
    # Draw N probabilities from the Beta distribution
    p_booking = beta_dist.rvs(a=alpha, b=beta, size=N)
    # Simulate Bernoulli trials based on these probabilities
    bookings = bernoulli.rvs(p=p_booking, size=N)
    return bookings # Returns array of 0s and 1s

def est_revenue(revenue_lo, rev_lo_prob, revenue_hi, rev_hi_prob, N):
    """
    Simulates N revenue outcomes for a *single* deal based on low/high
    revenue estimates and their probabilities. Assumes the deal *does* book.
    """
    # Ensure probabilities sum to 1 (or close enough)
    prob_sum = rev_lo_prob + rev_hi_prob
    if not np.isclose(prob_sum, 1.0):
        st.warning(f"Low ({rev_lo_prob:.2f}) and High ({rev_hi_prob:.2f}) revenue probabilities for deal "
                   f"(low: {revenue_lo}, high: {revenue_hi}) do not sum to 1. Normalizing.")
        if prob_sum > 0:
             rev_lo_prob = rev_lo_prob / prob_sum
             rev_hi_prob = 1.0 - rev_lo_prob # Adjust hi probability
        else: # If both are 0, assign equal probability (or handle as error)
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
    Simulates N scenarios for a deal, considering both booking probability
    (from Beta distribution) and revenue amount (low/high if booked).
    Result is revenue (if booked) or 0 (if not booked).
    """
    booked_sims = est_booking(alpha, beta, N) # Array of 0s and 1s
    revenue_sims = est_revenue(revenue_lo, rev_lo_prob, revenue_hi, rev_hi_prob, N) # Array of potential revenues
    # Multiply: Revenue is kept if booked (1), zeroed out if not booked (0)
    return booked_sims * revenue_sims

def validate_deals(deals_df):
    """
    Validates the input deals DataFrame.
    Checks:
    1. If booking_mean +/- booking_var goes outside [0, 1].
    2. If rev_lo_prob + rev_hi_prob does not sum to 1.
    Returns a DataFrame of invalid deals with the reason.
    """
    invalid_deals = []

    # Check booking probability range
    # Only check for deals where booking is not certain (mean < 1) and var > 0
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
    # Use np.isclose for floating point comparison
    oob_revenue = check_revenue[~np.isclose(check_revenue['prob_sum'], 1.0)]

    if not oob_revenue.empty:
            oob_revenue['reason'] = "Sum of rev_lo_prob and rev_hi_prob is not 1"
            # Avoid adding duplicates if already marked invalid by booking check
            existing_invalid_names = set(pd.concat(invalid_deals)['name']) if invalid_deals else set()
            new_invalid_revenue = oob_revenue[~oob_revenue['name'].isin(existing_invalid_names)]
            if not new_invalid_revenue.empty:
                invalid_deals.append(new_invalid_revenue[['name', 'rev_lo_prob', 'rev_hi_prob', 'reason']])

    if not invalid_deals:
        return pd.DataFrame() # Return empty dataframe if all valid
    else:
        # Combine invalid deals from both checks, remove potential duplicates again
        all_invalid = pd.concat(invalid_deals).drop_duplicates(subset=['name'])
        return all_invalid

def plot_revenue_distribution(rev_array, target_rev, prob_of_success):
    """
    Generates the density plot of simulated revenues using Matplotlib/Seaborn.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use Seaborn for density plot
    sns.kdeplot(rev_array, ax=ax, fill=False, color='black', linewidth=1.5, label='Revenue Density')

    # Create data for the shaded ribbon (area >= target_rev)
    kde = gaussian_kde(rev_array)
    x_vals = ax.lines[0].get_xdata() # Get x values from the kdeplot line
    y_vals = ax.lines[0].get_ydata() # Get y values from the kdeplot line

    # Ensure x_vals cover the range including the target
    min_x, max_x = min(x_vals), max(x_vals)
    plot_x = np.linspace(min_x, max_x, 512)
    plot_y = kde(plot_x)

    # Fill area where x >= target_rev
    ax.fill_between(plot_x, 0, plot_y, where=(plot_x >= target_rev),
                    color="#A47AA9", alpha=0.3, label=f'Revenue ≥ Target ({prob_of_success:.1%})')

    # Formatting
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('$%.0f'))
    plt.xticks(rotation=45, ha='right')
    ax.set_xlabel("Simulated Total Revenue")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Probability of Reaching Target: {prob_of_success:.1%}")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine() # Remove top and right spines

    st.pyplot(fig)

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
        format="%f" # Use %f for float formatting
    )

    N_SIMULATIONS = st.number_input(
        "Number of Simulations",
        min_value=1000,
        max_value=5000000, # Limit for performance
        value=100000, # Reduced default from R for quicker interaction
        step=10000,
        help="Higher numbers increase accuracy but take longer to compute."
    )

    # Use a form for the submit button to control execution
    submitted = st.button("Run Simulation")

# --- Main Panel ---

# Initialize session state variables if they don't exist
if 'deals_df' not in st.session_state:
    st.session_state.deals_df = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Load data when file is uploaded
if uploaded_file is not None:
    try:
        # Read the file from the buffer
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        deals_df = pd.read_csv(stringio)

        # Basic Column Check (adjust column names as needed)
        required_cols = ['name', 'revenue_lo', 'rev_lo_prob', 'revenue_hi', 'rev_hi_prob', 'booking_mean', 'booking_var']
        if not all(col in deals_df.columns for col in required_cols):
            st.error(f"CSV must contain the following columns: {', '.join(required_cols)}")
            st.session_state.deals_df = None # Reset if columns are wrong
        else:
             # Attempt basic type conversion and check for non-numeric errors
            for col in required_cols:
                if col != 'name': # Skip name column
                    deals_df[col] = pd.to_numeric(deals_df[col], errors='coerce') # Convert to numeric, invalid parsing will be NaT or NaN

            if deals_df.isnull().any().any():
                 st.warning("Some non-numeric values were found in numeric columns and were converted to NaN. Please check your CSV.")
                 st.dataframe(deals_df[deals_df.isnull().any(axis=1)]) # Show rows with NaNs
                 # Optionally stop execution or proceed with NaN handling later

            st.session_state.deals_df = deals_df # Store in session state

    except Exception as e:
        st.error(f"Error reading or processing CSV file: {e}")
        st.session_state.deals_df = None # Reset on error

# --- Simulation Logic (runs when Submit is clicked and data is ready) ---
if submitted and st.session_state.deals_df is not None:
    deals = st.session_state.deals_df.copy()

    # 1. Validate Deals
    invalid_deals = validate_deals(deals)
    if not invalid_deals.empty:
        st.error("Invalid deal configurations found. Please correct the data:")
        st.dataframe(invalid_deals)
        st.stop() # Stop execution if deals are invalid

    # 2. Separate Deals
    sure_things = deals[deals['booking_mean'] == 1.0].copy()
    not_sure_things = deals[deals['booking_mean'] < 1.0].copy()

    all_simulated_revenues = [] # List to hold revenue arrays for each deal

    # 3. Simulate "Not Sure Things"
    if not not_sure_things.empty:
        # Calculate Beta parameters
        beta_params = not_sure_things.apply(
            lambda row: est_beta_params(row['booking_mean'], row['booking_var']),
            axis=1
        )
        not_sure_things['alpha'] = [p['alpha'] for p in beta_params]
        not_sure_things['beta'] = [p['beta'] for p in beta_params]

        # Check for invalid parameters again after calculation
        if not_sure_things['alpha'].le(0).any() or not_sure_things['beta'].le(0).any():
             st.error("Simulation stopped: Invalid Beta parameters calculated. Check booking_mean and booking_var values.")
             st.dataframe(not_sure_things[not_sure_things['alpha'].le(0) | not_sure_things['beta'].le(0)])
             st.stop()


        # Simulate revenue for each "not sure" deal
        for _, row in not_sure_things.iterrows():
            sim_rev = est_booking_revenue(
                row['revenue_lo'], row['rev_lo_prob'], row['revenue_hi'], row['rev_hi_prob'],
                row['alpha'], row['beta'], N_SIMULATIONS
            )
            all_simulated_revenues.append(sim_rev)

    # 4. Simulate "Sure Things" (only revenue uncertainty)
    if not sure_things.empty:
        for _, row in sure_things.iterrows():
            sim_rev = est_revenue(
                row['revenue_lo'], row['rev_lo_prob'], row['revenue_hi'], row['rev_hi_prob'],
                N_SIMULATIONS
            )
            all_simulated_revenues.append(sim_rev)

    # 5. Aggregate Results
    if not all_simulated_revenues:
        st.warning("No deals found or processed.")
        total_revenue_sims = np.zeros(N_SIMULATIONS)
    else:
        # Sum the revenue arrays element-wise (axis=0)
        total_revenue_sims = np.sum(np.array(all_simulated_revenues), axis=0)

    # 6. Calculate Probability of Success & Quantiles
    prob_of_success = np.mean(total_revenue_sims >= target_rev)
    probs_to_calc = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] # Probabilities OF AT LEAST this revenue
    revenue_quantiles = np.quantile(total_revenue_sims, probs_to_calc)

    prob_df = pd.DataFrame({
        'probability': [f"{100*(1-p):.0f}%" for p in probs_to_calc], # Probability of exceeding the quantile revenue
        'revenue': revenue_quantiles
    })
    # Sort by probability descending (highest revenue first)
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
        # Format revenue column as currency
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
        2.  Prepare a **CSV file** containing your deals. You can use [this Google Sheet template](https://docs.google.com/spreadsheets/d/1kNbJVZURMRdG6WAOzxrXuZ3-J6iYaF0q1e3Gi3U2qEk/edit?usp=sharing) - make a copy, edit your deals, ensure column names are unchanged, and export as CSV.
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

        **GitHub:** [Original R code source](https://github.com/schnee/salesian) (if applicable)
        """)

elif submitted and st.session_state.deals_df is None:
    st.warning("Please upload a CSV file first.")

else:
    # Initial state or after file upload but before submission
    if st.session_state.deals_df is not None:
         st.info("Deals file loaded. Adjust target revenue and click 'Run Simulation' in the sidebar.")
         st.dataframe(st.session_state.deals_df.head()) # Show preview
    else:
         st.info("Upload a deals CSV file using the sidebar to get started.")