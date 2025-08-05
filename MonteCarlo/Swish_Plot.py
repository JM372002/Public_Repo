import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def compute_c(theta: float, sigma: float, u_samples: np.ndarray) -> float:
    """
    Compute c(σ) = E[(Φ'(u))²] for a given sigma, using Monte Carlo samples u ~ N(0,1).

    Parameters:
        theta (float): Threshold parameter.
        sigma (float): Noise standard deviation.
        u_samples (np.ndarray): Pre-sampled standard normal array.

    Returns:
        float: Estimated value of c(σ).
    """
    z = (theta - u_samples) / (np.sqrt(2) * sigma)
    A = 0.5 * erfc(z)
    B = (u_samples / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-u_samples**2 / (2 * sigma**2))
    phi_prime = A + B
    return np.mean(phi_prime**2)

if __name__ == "__main__":
    # Monte Carlo samples
    N = 1_000_000
    u = np.random.randn(N)

    # Sigma grid
    sigma_values = np.linspace(0.01, 2.0, 200)
    theta = 0.0

    # Compute c(σ) for all sigma
    c_values = [compute_c(theta, sig, u) for sig in sigma_values]

    # Identify peak
    peak_idx = np.argmax(c_values)
    peak_sigma = sigma_values[peak_idx]
    peak_c = c_values[peak_idx]

    # Compute value at σ = 1
    c_at_one = compute_c(theta, 1.0, u)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(sigma_values, c_values, lw=2, label=r'$c(\sigma)$')
    plt.axhline(0.5, ls='--', color='red', label='ReLU reference (0.5)')
    plt.fill_between(sigma_values, c_values, 0.5, where=np.array(c_values) > 0.5, alpha=0.1)

    # Peak point
    plt.scatter([peak_sigma], [peak_c], color='red', s=100, label='Peak')
    plt.annotate(f'Peak: σ={peak_sigma:.2f}, c={peak_c:.3f}',
                 xy=(peak_sigma, peak_c),
                 xytext=(peak_sigma + 0.2, peak_c),
                 arrowprops=dict(arrowstyle='->'))

    # Sigma = 1 annotation
    plt.scatter([1.0], [c_at_one], color='blue', marker='D', s=80, label=f'c at σ=1 ({c_at_one:.6f})')
    plt.annotate(f'σ=1, c={c_at_one:.6f}',
                 xy=(1.0, c_at_one),
                 xytext=(1.1, c_at_one - 0.02),
                 color='blue',
                 arrowprops=dict(arrowstyle='->', color='blue'))

    plt.xlabel('σ (Noise Standard Deviation)')
    plt.ylabel(r'$c(\sigma) = \chi$')
    plt.title(r'$\chi$ vs Noise $\sigma$ with Peak and Reference')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
