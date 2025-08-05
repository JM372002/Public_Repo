import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(1234)

def gaussian_mixture_em(data, n_components=2, n_iterations=100):
    N = len(data)
    means = np.random.choice(data, n_components)
    stds = np.ones(n_components)
    weights = np.ones(n_components) / n_components

    for _ in range(n_iterations):
        resp = np.array([
            weights[k] * norm.pdf(data, means[k], stds[k])
            for k in range(n_components)
        ])
        resp /= resp.sum(0)

        Nk = resp.sum(axis=1)
        weights = Nk / N
        means = (resp @ data) / Nk
        stds = np.sqrt((resp @ data**2) / Nk - means**2)

    return means, stds, weights

if __name__ == '__main__':
    data = np.concatenate([
        np.random.normal(0, 1, 150),
        np.random.normal(5, 1.5, 150)
    ])

    means, stds, weights = gaussian_mixture_em(data)

    x = np.linspace(min(data), max(data), 500)
    for k in range(len(means)):
        plt.plot(x, weights[k] * norm.pdf(x, means[k], stds[k]), label=f"Component {k+1}")

    plt.hist(data, bins=30, density=True, alpha=0.5)
    plt.legend()
    plt.title("Gaussian Mixture Model (EM)")
    plt.show()
