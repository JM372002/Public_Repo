import numpy as np
from math import log2
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from scipy.special import erfc
from matplotlib.lines import Line2D

# Set UTF-8 encoding for stdout
import sys
sys.stdout.reconfigure(encoding='utf-8')

plt.style.use('bmh')

# Simulation functions
def simulate_network_gaussian_input(mu_X, sigma_X, N, sigma_noise, num_samples):
    """
    Simulate the summing network with Gaussian input and Gaussian noise.

    Parameters:
    - mu_X: Mean of the input signal X.
    - sigma_X: Standard deviation of the input signal X.
    - N: Number of junctions in the network.
    - sigma_noise: Standard deviation of the noise added at each junction.
    - num_samples: Number of samples to simulate.

    Returns:
    - X_samples: Array of input signal samples.
    - Y_samples: Array of network output samples (sum of junction outputs).
    """
    # Generate input signal samples from Gaussian distribution
    X_samples = np.random.normal(mu_X, sigma_X, num_samples)

    # Initialize array to store summed outputs
    Y_samples = np.zeros(num_samples)

    # Simulate the network
    for i in range(num_samples):
        X = X_samples[i]

        # Generate noise for all junctions
        noise = np.random.normal(0, sigma_noise, N)

        # Total input at each junction
        total_input = X + noise

        # Apply thresholds (zero in this case)
        outputs = (total_input > 0).astype(int)

        # Sum the outputs
        Y_samples[i] = outputs.sum()

    return X_samples, Y_samples

def calculate_mutual_information(X_samples, Y_samples, num_bins):
    """
    Calculate the mutual information I(X; Y) between the input signal X and the output Y.
    (NOTE: Symmetric nature of mutual information)

    Parameters:
    - X_samples: Array of input signal samples.
    - Y_samples: Array of network output samples.
    - num_bins: Number of bins to discretize the input signal X.

    Returns:
    - I_XY: Mutual information I(X; Y) in bits.
    """
    # Discretize the input signal X
    hist, bin_edges = np.histogram(X_samples, bins=num_bins)
    bin_indices = np.digitize(X_samples, bin_edges[:-1], right=True) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Possible values of Y
    y_values = np.unique(Y_samples)
    num_y_values = len(y_values)

    # Initialize joint count matrix
    joint_counts = np.zeros((num_bins, num_y_values))

    # Map Y values to indices
    y_value_to_index = {y: idx for idx, y in enumerate(y_values)}

    # Count joint occurrences
    for i in range(len(X_samples)):
        x_idx = bin_indices[i]
        y_idx = y_value_to_index[Y_samples[i]]
        joint_counts[x_idx, y_idx] += 1

    # Convert counts to probabilities
    total_samples = len(X_samples)
    joint_probs = joint_counts / total_samples
    marginal_probs_X = joint_probs.sum(axis=1)  # Sum over Y
    marginal_probs_Y = joint_probs.sum(axis=0)  # Sum over X

    # Calculate entropies
    H_X = -np.sum([p * log2(p) for p in marginal_probs_X if p > 0])
    H_Y = -np.sum([p * log2(p) for p in marginal_probs_Y if p > 0])
    H_XY = -np.sum([p * log2(p) for row in joint_probs for p in row if p > 0])

    # Mutual Information
    I_XY = H_X + H_Y - H_XY

    return I_XY

# Theoretical functions
def P_x(x, mean, std):
    return norm.pdf(x, mean, std)

def P_conditional(x, threshold, noise_std):
    if noise_std == 0:
        # With zero noise, conditional probability is a step function at the threshold
        return np.where(x >= threshold, 1.0, 0.0)
    else:
        return 0.5 * erfc((threshold - x) / (np.sqrt(2) * noise_std))

def P_y_vectorized(N, threshold, noise_std, x, signal_mean, signal_std):
    p_x = P_x(x, mean=signal_mean, std=signal_std)
    p_cond = P_conditional(x, threshold, noise_std)
    # Clip p_cond to avoid probabilities of exactly 0 or 1
    p_cond = np.clip(p_cond, 1e-12, 1 - 1e-12)
    n_values = np.arange(N + 1)
    binom_probs = binom.pmf(n_values[:, None], N, p_cond)
    integrand = p_x * binom_probs
    P_y_n = np.trapezoid(integrand, x, axis=1)
    return P_y_n

def H_y_vectorized(N, threshold, noise_std, x, signal_mean, signal_std):
    P_y_n = P_y_vectorized(N, threshold, noise_std, x, signal_mean, signal_std)
    P_y_n = np.clip(P_y_n, 1e-12, 1)  # Avoid log(0)
    return -np.sum(P_y_n * np.log2(P_y_n))

def H_binomial_vectorized(N, p_array):
    n_values = np.arange(N + 1)
    binom_probs = binom.pmf(n_values[:, None], N, p_array)
    binom_probs = np.clip(binom_probs, 1e-12, 1)
    H_bin = -np.sum(binom_probs * np.log2(binom_probs), axis=0)
    return H_bin

def H_y_given_x_vectorized(N, threshold, noise_std, x, signal_mean, signal_std):
    p_x = P_x(x, mean=signal_mean, std=signal_std)
    p_cond = P_conditional(x, threshold, noise_std)
    # Clip p_cond to avoid probabilities of exactly 0 or 1
    p_cond = np.clip(p_cond, 1e-12, 1 - 1e-12)
    H_bin = H_binomial_vectorized(N, p_cond)
    integrand = p_x * H_bin
    H_Y_given_X = np.trapezoid(integrand, x)
    return H_Y_given_X

def calc_theoretical_mutual_information(N, threshold, noise_std, x, signal_mean, signal_std):
    H_Y = H_y_vectorized(N, threshold, noise_std, x, signal_mean, signal_std)
    H_Y_given_X = H_y_given_x_vectorized(N, threshold, noise_std, x, signal_mean, signal_std)
    I_XY = H_Y - H_Y_given_X
    return I_XY

if __name__ == '__main__':
    # Common parameters
    mu_X = 0.0             # Mean of the input signal
    sigma_X = 1.0          # Standard deviation of the input signal
    threshold = 0
    N_values = [1, 2, 3, 7, 15, 31]
    num_samples = 100000   # Number of samples for simulation
    num_bins = 400         # Number of bins for discretising X

    # x values for integration (theoretical calculation)
    x = np.linspace(-10, 10, 1000)

    # Noise standard deviation values from 0 to 1.5
    sigma_noise_values = np.linspace(0, 1.5, 31)

    plt.figure(figsize=(15, 8))

    for N in N_values:
        sim_mutual_info_values = []
        theoretical_mutual_info_values = []

        print(f"Processing N = {N}", flush=True)
        for sigma_noise in sigma_noise_values:
            print(f"  Calculating for σ_noise = {sigma_noise:.4f}", flush=True)

            # Simulation
            X_samples, Y_samples = simulate_network_gaussian_input(
                mu_X, sigma_X, N, sigma_noise, num_samples
            )
            sim_I_XY = calculate_mutual_information(X_samples, Y_samples, num_bins)
            sim_mutual_info_values.append(sim_I_XY)

            # Theoretical calculation
            theoretical_I_XY = calc_theoretical_mutual_information(
                N, threshold, sigma_noise, x, mu_X, sigma_X
            )
            theoretical_mutual_info_values.append(theoretical_I_XY)

        # Convert lists to numpy arrays
        sim_mutual_info_values = np.array(sim_mutual_info_values)
        theoretical_mutual_info_values = np.array(theoretical_mutual_info_values)

        # Plot theoretical results
        plt.plot(sigma_noise_values, theoretical_mutual_info_values, color='k', label=f'Theoretical N = {N}')

        # Plot sim results
        plt.plot(sigma_noise_values, sim_mutual_info_values, color='k', marker='o', linestyle='')

        # Place label at 75% place of sim data
        index_75 = int(0.75 * len(sigma_noise_values))
        x_pos = sigma_noise_values[index_75]
        y_pos = sim_mutual_info_values[index_75]
        plt.text(x_pos, y_pos + 0.05, f'N={N}', horizontalalignment='center', fontsize=10)

    # Create custom legend entries
    legend_elements = [
        Line2D([0], [0], color='k', lw=2, label='Theoretical Results'),
        Line2D([0], [0], marker='o', color='k', linestyle='', label='Simulation Results')
    ]

    # Add the legend to the plot
    plt.legend(handles=legend_elements)

    plt.title('Mutual Information I(Y; X) vs. Noise Standard Deviation (σ_noise)')
    plt.xlabel('Noise Standard Deviation (σ_noise)')
    plt.ylabel('Mutual Information I [bits]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
