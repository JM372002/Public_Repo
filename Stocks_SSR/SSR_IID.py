import numpy as np
from math import log2
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from scipy.special import erfc
from matplotlib.lines import Line2D
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')
plt.style.use('bmh')

# ---------------------
# Simulation
# ---------------------
def simulate_network_gaussian_input(mu_X, sigma_X, N, sigma_noise, num_samples):
    """
    Simulate summing network with Gaussian input and additive Gaussian noise.
    """
    X_samples = np.random.normal(mu_X, sigma_X, num_samples)
    Y_samples = np.zeros(num_samples)

    for i in range(num_samples):
        noise = np.random.normal(0, sigma_noise, N)
        total_input = X_samples[i] + noise
        outputs = (total_input > 0).astype(int)
        Y_samples[i] = outputs.sum()

    return X_samples, Y_samples

def calculate_mutual_information(X_samples, Y_samples, num_bins):
    """
    Estimate mutual information I(X; Y) from discretized samples.
    """
    hist, bin_edges = np.histogram(X_samples, bins=num_bins)
    bin_indices = np.digitize(X_samples, bin_edges[:-1], right=True) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    y_values = np.unique(Y_samples)
    joint_counts = np.zeros((num_bins, len(y_values)))
    y_index_map = {y: idx for idx, y in enumerate(y_values)}

    for i in range(len(X_samples)):
        x_idx = bin_indices[i]
        y_idx = y_index_map[Y_samples[i]]
        joint_counts[x_idx, y_idx] += 1

    joint_probs = joint_counts / len(X_samples)
    p_x = joint_probs.sum(axis=1)
    p_y = joint_probs.sum(axis=0)

    H_X = -np.sum(np.fromiter((p * log2(p) for p in p_x if p > 0), dtype=float))
    H_Y = -np.sum(np.fromiter((p * log2(p) for p in p_y if p > 0), dtype=float))
    H_XY = -np.sum(np.fromiter((p * log2(p) for row in joint_probs for p in row if p > 0), dtype=float))

    return H_X + H_Y - H_XY

# ---------------------
# Theoretical Functions
# ---------------------
def P_x(x, mean, std):
    return norm.pdf(x, mean, std)

def P_conditional(x, threshold, noise_std):
    if noise_std == 0:
        return np.where(x >= threshold, 1.0, 0.0)
    return 0.5 * erfc((threshold - x) / (np.sqrt(2) * noise_std))

def P_y_vectorized(N, threshold, noise_std, x, signal_mean, signal_std):
    p_x = P_x(x, signal_mean, signal_std)
    p_cond = np.clip(P_conditional(x, threshold, noise_std), 1e-12, 1 - 1e-12)
    n_values = np.arange(N + 1)
    binom_probs = binom.pmf(n_values[:, None], N, p_cond)
    integrand = p_x * binom_probs
    return np.trapezoid(integrand, x, axis=1)

def H_y_vectorized(N, threshold, noise_std, x, signal_mean, signal_std):
    P_y = np.clip(P_y_vectorized(N, threshold, noise_std, x, signal_mean, signal_std), 1e-12, 1)
    return -np.sum(P_y * np.log2(P_y))

def H_binomial_vectorized(N, p_array):
    n_values = np.arange(N + 1)
    probs = np.clip(binom.pmf(n_values[:, None], N, p_array), 1e-12, 1)
    return -np.sum(probs * np.log2(probs), axis=0)

def H_y_given_x_vectorized(N, threshold, noise_std, x, signal_mean, signal_std):
    p_x = P_x(x, signal_mean, signal_std)
    p_cond = np.clip(P_conditional(x, threshold, noise_std), 1e-12, 1 - 1e-12)
    H_bin = H_binomial_vectorized(N, p_cond)
    return np.trapezoid(p_x * H_bin, x)

def calc_theoretical_mutual_information(N, threshold, noise_std, x, signal_mean, signal_std):
    H_Y = H_y_vectorized(N, threshold, noise_std, x, signal_mean, signal_std)
    H_Y_given_X = H_y_given_x_vectorized(N, threshold, noise_std, x, signal_mean, signal_std)
    return H_Y - H_Y_given_X

# ---------------------
# Main Execution
# ---------------------
if __name__ == '__main__':
    mu_X = 0.0
    sigma_X = 1.0
    threshold = 0
    N_values = [1, 2, 3, 7, 15, 31]
    num_samples = 100_000
    num_bins = 400
    sigma_noise_values = np.linspace(0, 1.5, 31)
    x = np.linspace(-10, 10, 1000)

    plt.figure(figsize=(15, 8))

    for N in N_values:
        sim_MI, theo_MI = [], []
        print(f"Processing N = {N}", flush=True)

        for sigma_noise in sigma_noise_values:
            print(f"  σ_noise = {sigma_noise:.4f}", flush=True)

            X_samples, Y_samples = simulate_network_gaussian_input(
                mu_X, sigma_X, N, sigma_noise, num_samples
            )
            sim_I = calculate_mutual_information(X_samples, Y_samples, num_bins)
            theo_I = calc_theoretical_mutual_information(N, threshold, sigma_noise, x, mu_X, sigma_X)

            sim_MI.append(sim_I)
            theo_MI.append(theo_I)

        sim_MI = np.array(sim_MI)
        theo_MI = np.array(theo_MI)

        plt.plot(sigma_noise_values, theo_MI, color='k', label=f'Theoretical N={N}')
        plt.plot(sigma_noise_values, sim_MI, color='k', marker='o', linestyle='')

        idx = int(0.75 * len(sigma_noise_values))
        plt.text(sigma_noise_values[idx], sim_MI[idx] + 0.05, f'N={N}', ha='center', fontsize=10)

    legend_elements = [
        Line2D([0], [0], color='k', lw=2, label='Theoretical'),
        Line2D([0], [0], marker='o', color='k', linestyle='', label='Simulation')
    ]
    plt.legend(handles=legend_elements)
    plt.title('Mutual Information I(Y; X) vs. Noise Std Dev (σ_noise)')
    plt.xlabel('σ_noise')
    plt.ylabel('Mutual Information [bits]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
