import numpy as np
import train_generator

# Importing the sampling functions
from metropolis_hastings_within_gibbs_sampler import metropolis_hastings_gibbs_sampler
from gibbs_sampler import gibbs_sampler
from blocked_gibbs_sampler import blocked_gibbs_sampler

from evaluate_performance import convergence_histogram_plotter

def main():
    # A seed used during data generation.
    data_seed = 9

    # The size of the train lattice: n_lattice x n_lattice
    n_lattice = 3

    # Number of time steps.
    T = 10 #100

    # The probability of generating an inccorect observation from the train.
    error_prob = 0.1

    # Data generation in terms of a train lattice grap h, the true switch settings,
    # the true positions of the train and the errorneous train observations.
    G, X_truth, s_truth, o = train_generator.generate_data(data_seed, n_lattice, T, error_prob)

    # The number of iterations in our sampling procedure.
    num_iter = 1000

    # We try different seeds to make sure that we get to an equivalent result.
    seeds_chains=[225]#,11,2222] # Different seeds used to run different chains

    for seed in seeds_chains:

        np.random.seed(seed)

        """ Running the sampling methods to infer the trains position and the switch settings."""
        # # Blocked gibbs sampling.
        sbg, X = blocked_gibbs_sampler(o, G, num_iter)
        #
        # # A Metropolis-Hastings sampler (for the swith states) within a Gibbs sampler (for s1).
        smhg, X = metropolis_hastings_gibbs_sampler(o, G, num_iter, error_prob)

        # Gibbs sampling.
        sg, X = gibbs_sampler(o, G, num_iter, error_prob)

        s_list= [smhg, sg, sbg]
        s_list_check_convergence = [smhg[0:200],sg[0:200],sbg[0:200]]
        s_list_check_convergence_second_half = [smhg[500:-1],sg[500:-1],sbg[500:-1]]

        burn_in=100
        lag=5

        # Computing accuracy for the algorithms
        # for s_s in s_list:
        #      calc_acc(burn_in,lag,s_s,s_truth[0])

        # Creating histograms for the startpositions
        #convergence_histogram_plotter(burn_in,lag,s_list,s_truth[0])

        #convergence_histogram_plotter(burn_in,lag,s_list_check_convergence,s_truth[0])


        #convergence_histogram_plotter( 0, lag, s_list_check_convergence_second_half, s_truth[0])


if __name__ == '__main__':
    main()
