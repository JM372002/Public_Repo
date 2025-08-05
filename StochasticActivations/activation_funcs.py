import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm

# --- Functions ---
def h_bar_sigmoid(threshold, linear_output, sigma, gain=1):
    lower = (threshold - (linear_output * gain)) / sigma
    return norm.sf(lower)

def SRA_sigmoid(threshold, linear_output, sigma, T=100):
    N = len(linear_output)
    noise = np.random.normal(loc=0, scale=sigma, size=(N, T))
    noisy_values = linear_output[:, None] + noise
    above_threshold = noisy_values > threshold
    return above_threshold.mean(axis=1)

if __name__ == "__main__":
    # --- Parameters ---
    x = np.linspace(-3, 3, 500)
    threshold_values = np.linspace(-1.5, 1.5, 50)
    sigma_values = np.linspace(0.1, 2.0, 50)
    T_values = np.linspace(1, 50, 50, dtype=int)

    fixed_sigma = 0.7
    fixed_T = 100
    fixed_threshold = 0

    # --- Setup figure and axes ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    lines = []

    titles = [
        "Changing T (SRA_sigmoid)",
        "Changing Threshold (θ)",
        "Changing σ (noise std)",
        "Fixed Parameters Comparison"
    ]

    for ax, title in zip(axs.flat, titles):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.grid(True)
        line, = ax.plot([], [])
        lines.append(line)

    axs[1, 1].plot(x, h_bar_sigmoid(fixed_threshold, x, fixed_sigma), label='h_bar_sigmoid')
    axs[1, 1].plot(x, SRA_sigmoid(fixed_threshold, x, fixed_sigma, fixed_T), label='SRA_sigmoid')
    axs[1, 1].legend()

    # --- Update function ---
    def update(frame):
        T = T_values[frame]
        theta = threshold_values[frame]
        sigma = sigma_values[frame]

        y_T = SRA_sigmoid(fixed_threshold, x, fixed_sigma, T)
        y_theta = SRA_sigmoid(theta, x, fixed_sigma, fixed_T)
        y_sigma = SRA_sigmoid(fixed_threshold, x, sigma, fixed_T)

        lines[0].set_data(x, y_T)
        lines[1].set_data(x, y_theta)
        lines[2].set_data(x, y_sigma)

        axs[0, 0].set_title(f"Changing T (T={T})")
        axs[0, 1].set_title(f"Changing Threshold (θ={theta:.2f})")
        axs[1, 0].set_title(f"Changing σ (σ={sigma:.2f})")

        return lines

    # --- Animate ---
    ani = FuncAnimation(fig, update, frames=len(T_values), interval=300, blit=True)
    plt.close()

    ani.save("sra_vs_hbar.gif", fps=10, writer='pillow')


