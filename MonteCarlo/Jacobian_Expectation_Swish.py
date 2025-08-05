import numpy as np
from scipy.special import erfc
import sys

# Unicode output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def estimate_c(theta: float, sigma: float, N: int = 2_000_000) -> tuple[float, float]:
    """
    Monte Carlo estimate of c = E[(Φ'(u))²], where Φ' is the smoothed threshold function.

    Parameters:
        theta (float): Threshold parameter θ.
        sigma (float): Standard deviation σ of the additive Gaussian noise.
        N (int): Number of Monte Carlo samples.

    Returns:
        est (float): Estimated expectation.
        se  (float): Standard error of the estimate.
    """
    u = np.random.randn(N)
    z = (theta - u) / (np.sqrt(2) * sigma)
    
    # Φ'(u): smoothed derivative of binary threshold via Gaussian noise
    phi_prime = 0.5 * erfc(z) + (u / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-z**2)
    
    squared = phi_prime ** 2
    est = np.mean(squared)
    se = np.std(squared, ddof=1) / np.sqrt(N)
    
    return est, se

if __name__ == "__main__":
    theta = 0.0
    sigma = 1.0
    N = 100_000_000

    estimate, stderr = estimate_c(theta, sigma, N)
    print(f"Monte Carlo estimate of c for θ = {theta}, σ = {sigma}: {estimate:.6f} ± {stderr:.6f}")
