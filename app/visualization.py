# app/visualization.py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.stats import gaussian_kde
import numpy as np
import streamlit as st

def plot_revenue_distribution(rev_array, target_rev, prob_of_success):
    """
    Generates the density plot of simulated revenues.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(rev_array, ax=ax, fill=False, color='black', linewidth=1.5, label='Revenue Density')

    kde = gaussian_kde(rev_array)
    x_vals = ax.lines[0].get_xdata()
    y_vals = ax.lines[0].get_ydata()

    min_x, max_x = min(x_vals), max(x_vals)
    plot_x = np.linspace(min_x, max_x, 512)
    plot_y = kde(plot_x)

    ax.fill_between(plot_x, 0, plot_y, where=(plot_x >= target_rev),
                    color="#A47AA9", alpha=0.3, label=f'Revenue ≥ Target ({prob_of_success:.1%})')

    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('$%.0f'))
    plt.xticks(rotation=45, ha='right')
    ax.set_xlabel("Simulated Total Revenue")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Probability of Reaching Target: {prob_of_success:.1%}")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine()

    return fig
