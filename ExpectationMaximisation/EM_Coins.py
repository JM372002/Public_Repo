import numpy as np

def count_heads_tails(matrix):
    """
    Count the number of 1s (heads) and 0s (tails) in each row of the matrix.
    Returns a (N, 2) array where each row is [heads, tails].
    """
    counts = []
    for row in matrix:
        heads = np.sum(row == 1)
        tails = np.sum(row == 0)
        counts.append([heads, tails])
    return np.array(counts)

def expectation_step(counts, estimate_A, estimate_B):
    """
    Perform the E-step of the EM algorithm.

    Returns:
        - Expected counts attributed to Coin A
        - Expected counts attributed to Coin B
    """
    heads, tails = counts[:, 0], counts[:, 1]
    comp_A, comp_B = 1 - estimate_A, 1 - estimate_B

    # Likelihood of data under each coin
    likelihood_A = estimate_A ** heads * comp_A ** tails
    likelihood_B = estimate_B ** heads * comp_B ** tails

    # Normalize to get probabilities
    total = likelihood_A + likelihood_B
    prob_A = likelihood_A / total
    prob_B = likelihood_B / total

    # Expected counts
    A_attrib = np.stack([heads * prob_A, tails * prob_A], axis=1)
    B_attrib = np.stack([heads * prob_B, tails * prob_B], axis=1)

    return A_attrib, B_attrib

def maximization_step(A_attrib, B_attrib):
    """
    Perform the M-step: update probability estimates for each coin.
    """
    sum_A = np.sum(A_attrib, axis=0)
    sum_B = np.sum(B_attrib, axis=0)

    new_estimate_A = sum_A[0] / np.sum(sum_A)
    new_estimate_B = sum_B[0] / np.sum(sum_B)

    return new_estimate_A, new_estimate_B

def run_em(trials, n_iterations=100, init_A=0.7, init_B=0.2):
    """
    Run the EM algorithm on binary coin flip data.
    """
    counts = count_heads_tails(trials)
    estimate_A, estimate_B = init_A, init_B

    for _ in range(n_iterations):
        A_attrib, B_attrib = expectation_step(counts, estimate_A, estimate_B)
        estimate_A, estimate_B = maximization_step(A_attrib, B_attrib)

    return estimate_A, estimate_B

if __name__ == '__main__':
    # 1 = Heads, 0 = Tails
    trials = np.array([
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    ])

    print("\n--- EM Algorithm Started ---")
    final_A, final_B = run_em(trials, n_iterations=100)
    print("\n--- EM Algorithm Finished ---")
    print(f"Final Estimate for Coin A (P=1): {final_A:.4f}")
    print(f"Final Estimate for Coin B (P=1): {final_B:.4f}")
