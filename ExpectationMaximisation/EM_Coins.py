import numpy as np

def count_events(matrix):
    counts_matrix = []
    for row in matrix:
        ones_count = np.sum(row == 1)
        zeros_count = np.sum(row == 0)
        counts_matrix.append([ones_count, zeros_count])
    counts_matrix = np.array(counts_matrix)
    return counts_matrix

if __name__ == '__main__':
    # Two coins used: 1 = Heads, 0 = Tails
    trials = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
                      [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
                      [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]])

    counts = count_events(trials)
    n_iterations = 100  # Set the number of iterations

    # Initial Coin Estimates (starting point for the loop)
    estimate_A = 0.7
    estimate_B = 0.2

    print("\n--- Vectorized EM Algorithm started ---")

    for iteration in range(n_iterations):
    
        comp_A = 1 - estimate_A
        comp_B = 1 - estimate_B

        # E-step (Expectation) - Vectorized Calculation

        # Extract counts of 1s (Heads) and 0s (Tails) for all trials
        ones_counts = counts[:, 0]  # Shape: (n_trials,)
        zeros_counts = counts[:, 1] # Shape: (n_trials,)

        # Vectorized Likelihood Calculation for all trials at once
        like_A = estimate_A ** ones_counts * comp_A ** zeros_counts  # Shape: (n_trials,)
        like_B = estimate_B ** ones_counts * comp_B ** zeros_counts  # Shape: (n_trials,)

        # Vectorized Probability Calculation for all trials at once
        prob_denominator = like_A + like_B  # Shape: (n_trials,)
        prob_A = like_A / prob_denominator      # Shape: (n_trials,)
        prob_B = like_B / prob_denominator      # Shape: (n_trials,)

        # Vectorized Expected Counts (Attributions)
        H_Artib_A = ones_counts * prob_A  # Shape: (n_trials,)
        T_Artib_A = zeros_counts * prob_A # Shape: (n_trials,)
        H_Artib_B = ones_counts * prob_B  # Shape: (n_trials,)
        T_Artib_B = zeros_counts * prob_B # Shape: (n_trials,)

        # Stack the attributed Heads and Tails for each coin into matrices (for summing)
        Atrib_results_A = np.stack([H_Artib_A, T_Artib_A], axis=1) # Shape: (n_trials, 2)
        Atrib_results_B = np.stack([H_Artib_B, T_Artib_B], axis=1) # Shape: (n_trials, 2)

        # M-step (Maximization) - Update estimates based on summed attributions
        sum_Atrib_A = np.sum(Atrib_results_A, axis=0) # Sum along trials, shape: (2,)
        sum_Atrib_B = np.sum(Atrib_results_B, axis=0) # Sum along trials, shape: (2,)

        estimate_A = sum_Atrib_A[0] / sum_Atrib_A.sum()
        estimate_B = sum_Atrib_B[0] / sum_Atrib_B.sum()

    print("\n--- Vectorized EM Algorithm finished after", n_iterations, "iterations ---")
    print("\nFinal Estimates:")
    print(f"Estimate Coin A (P=1): {estimate_A:.4f}")
    print(f"Estimate Coin B (P=1): {estimate_B:.4f}")