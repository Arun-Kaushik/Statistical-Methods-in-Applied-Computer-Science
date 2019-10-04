import numpy as np
import train_generator

# Importing the sampling functions
from metropolis_hastings_within_gibbs_sampler import metropolis_hastings_gibbs_sampler
from gibbs_sampler import gibbs_sampler
from blocked_gibbs_sampler import blocked_gibbs_sampler

from evaluate_performance import histogram_plotter, calculate_accuracy, apply_burn_in, apply_lag


def main():
    # A seed used during data generation.
    data_seed = 9

    # The size of the train lattice: n_lattice x n_lattice
    n_lattice = 3

    # Number of time steps.
    T = 10  # 100

    # The probability of generating an inccorect observation from the train.
    error_prob = 0.1

    # Data generation in terms of a train lattice grap h, the true switch settings,
    # the true positions of the train and the errorneous train observations.
    G, X_truth, s_truth, o = train_generator.generate_data(
        data_seed, n_lattice, T, error_prob)

    # The number of iterations in our sampling procedure.
    num_iter = 1000

    # We try different seeds to make sure that we get to an equivalent result.
    # ,11,2222] # Different seeds used to run different chains
    seeds_chains = [225]

    for seed in seeds_chains:

        np.random.seed(seed)

        """ Running the sampling methods to infer the trains position and the switch settings."""
        # Blocked gibbs sampling.
        start_positions_blocked_gibbs, X = blocked_gibbs_sampler(o, G, num_iter)

        # A Metropolis-Hastings sampler (for the swith states) within a Gibbs sampler (for s1).
        start_positions_metropolis_hastings_gibbs, X = metropolis_hastings_gibbs_sampler(o, G, num_iter, error_prob)

        # Gibbs sampling.
        start_positions_gibbs, X = gibbs_sampler(o, G, num_iter, error_prob)

        burn_in = 0
        lag = 5

        start_positions_blocked_gibbs = apply_burn_in(start_positions_blocked_gibbs, burn_in)
        start_positions_metropolis_hastings_gibbs = apply_burn_in(start_positions_metropolis_hastings_gibbs, burn_in)
        start_positions_gibbs = apply_burn_in(start_positions_gibbs, burn_in)

        start_positions_blocked_gibbs = apply_lag(start_positions_blocked_gibbs, lag)
        start_positions_metropolis_hastings_gibbs = apply_lag(start_positions_metropolis_hastings_gibbs, lag)
        start_positions_gibbs = apply_lag(start_positions_gibbs, lag)


        histogram_plotter( start_positions_blocked_gibbs[0:200], s_truth[0], "Blocked Gibbs" )
        histogram_plotter( start_positions_metropolis_hastings_gibbs[0:200], s_truth[0], "Metropolis Hastings within Gibbs")
        histogram_plotter( start_positions_gibbs[0:200], s_truth[0], "Gibbs")


if __name__ == '__main__':
    main()
