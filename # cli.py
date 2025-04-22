# cli.py
import argparse
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from app.data_utils import load_deals_data, validate_deals
from app.simulation import run_simulations
from app.visualization import plot_revenue_distribution

def main():
     # Print the execution string
    print(f"Execution string: {' '.join(sys.argv)}")

    parser = argparse.ArgumentParser(description="Run revenue simulations from a deals CSV file.")
    parser.add_argument("deals_file", help="Path to the deals CSV file")
    parser.add_argument("target_revenue", type=float, help="Target revenue amount")
    parser.add_argument("--num_simulations", type=int, default=100000, help="Number of simulations to run (default: 100000)")
    parser.add_argument("--output_image", default="revenue_distribution.png", help="Path to save the output image (default: revenue_distribution.png)")

    args = parser.parse_args()

    # Load deals data
    try:
        deals_df = load_deals_data(args.deals_file)
        if deals_df is None:
            print("Error: Could not load deals data.")
            return
    except FileNotFoundError:
        print(f"Error: Deals file not found at {args.deals_file}")
        return
    except Exception as e:
        print(f"Error loading deals data: {e}")
        return

    # Validate deals
    invalid_deals = validate_deals(deals_df)
    if not invalid_deals.empty:
        print("Error: Invalid deal configurations found:")
        print(invalid_deals)
        return

    # Run simulations
    total_revenue_sims = run_simulations(deals_df, args.num_simulations)

    # Calculate probability of success and quantiles
    prob_of_success = np.mean(total_revenue_sims >= args.target_revenue)
    probs_to_calc = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    revenue_quantiles = np.quantile(total_revenue_sims, probs_to_calc)

    prob_df = pd.DataFrame({
        'probability': [f"{100*(1-p):.0f}%" for p in probs_to_calc],
        'revenue': revenue_quantiles
    })
    prob_df = prob_df.iloc[::-1]

    # Output results
    print(f"Target Revenue: ${args.target_revenue:,.0f}")
    print(f"Estimated Probability of Exceeding Target: {prob_of_success:.1%}")
    print("\nRevenue Probability Estimates:")
    print(prob_df.to_string(index=False))

    # Save the plot to a file
    fig = plot_revenue_distribution(total_revenue_sims, args.target_revenue, prob_of_success)
    fig.savefig(args.output_image)
    print(f"\nRevenue distribution plot saved to {args.output_image}")

if __name__ == "__main__":
    main()
