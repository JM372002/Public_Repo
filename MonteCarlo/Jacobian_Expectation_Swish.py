import numpy as np
from scipy.special import erfc
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def estimate_c(theta, sigma, N=2_000_000):
    """
    Estimate c = E[(Phi'(u))^2] by Monte Carlo sampling u ~ N(0,1).
    
    Parameters:
    - theta: threshold parameter θ
    - sigma: noise std σ
    - N: number of Monte Carlo samples
    
    Returns:
    - est: Monte Carlo estimate of c
    - se: standard error of the estimate
    """
    u = np.random.randn(N)
    z = (theta - u) / (np.sqrt(2) * sigma)
    phi_prime = 0.5 * erfc(z) + (u / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-z**2)
    est = np.mean(phi_prime**2)
    se = np.std(phi_prime**2, ddof=1) / np.sqrt(N)
    return est, se

if __name__ == "__main__":
    theta = 0.0
    sigma = 1.0

    estimate, stderr = estimate_c(theta, sigma, N=100_000_000)
    print(f"Monte Carlo estimate of c for θ={theta}, σ={sigma}: {estimate:.6f} ± {stderr:.6f}")
