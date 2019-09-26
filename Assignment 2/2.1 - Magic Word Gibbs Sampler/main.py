import magic_word_gibbs_sampler
import numpy as np
import math
import matplotlib.pyplot as plt


def plot_start_position_distribution( startposition_samples, true_startpositions ):
    """ Plots the result of the gibbs sampling.
    """

    startpositions = [startposition_samples.count(0), startposition_samples.count(1), startposition_samples.count(2), startposition_samples.count(3), startposition_samples.count(4), startposition_samples.count(5)]
    star_position_indicies = ['0', '1', '2', '3', '4', '5']
    no_possible_start_positions = len(star_position_indicies)
    startposition_counts = np.arange(no_possible_start_positions)

    plt.bar( startposition_counts, startpositions, color = 'g')

    title_str = 'Start position distribution - Gibbs samples. True start position:' + str(true_startpositions)
    plt.title(title_str)
    plt.xlabel('Possible Start positions')
    plt.ylabel('No. Samples')
    plt.xticks( startposition_counts, star_position_indicies )
    plt.show()


def remove_burn_in( sampled_startpositions, burn_in ):
    """ Removes the samples results acquired during the burn in phase.
    """

    results_without_burnin = sampled_startpositions[burn_in:-1]

    return results_without_burnin


def remove_lag( sampled_startpositions, lag ):
    """ Since consecutive samples tend to be dependent we use a lag factor to remove
        close samples.
    """

    results_with_lag = sampled_startpositions[0::lag]

    return results_with_lag


def main():

    """ Define parameters for the Gibbs Sampling.
    """
    seed = 123

    # The number of word sequences.
    N = 3 # 20

    # The length of a word sequence
    M = 10

    # The length of the alphabet.
    K = 4

    # The length of a magic word.
    W = 5


    alpha_bg = [ 7, 13, 1, 5 ]
    alpha_mw = np.ones(K)

    # Number of samplings.
    num_iter = 1000

    # Generate synthetic data.
    word_sequences, true_startpositions, prior_bg, prior_mw = magic_word_gibbs_sampler.generator( seed, N, M, K, W, alpha_bg, alpha_mw )

    """ Print the generated data for reference during the sampling.
    """
    print( "\n Word Sequences containing background letters and a magic word each: " )
    print( word_sequences )
    print( "\n Ground Truth Magic Word Start positions in the above word sequences: " )
    print( true_startpositions )

    """ Inferring start positions of Magic Words via Gibbs Sampling.
    """
    # Use D, alpha_bg and alpha_mw to infer the start positions of magic words.
    sampled_startpositions = magic_word_gibbs_sampler.gibbs( word_sequences, alpha_bg, alpha_mw, num_iter, W, N, M, K )


    burn_in_period = 100
    lag = 2


    for word_sequence in range( N ):

        # Pre-processing the sampled start positions.
        sampled_startposition = sampled_startpositions[ :, word_sequence ]

        # Removing the burn in phase.
        sampled_startposition = remove_burn_in( sampled_startposition, burn_in_period )

        # Applying a lag inbetween samples to account for that consecutive samples are not independent.
        sampled_startposition = remove_lag( sampled_startposition, lag )

        # Calculating the accuracy of the sampled start positions for the current word sequence. I.e. how many times
        # have we accurately sampled the correct start position for the current word sequence.
        accuracy = sampled_startposition.tolist().count( true_startpositions[word_sequence] ) / len( sampled_startposition )
        print('The Accuracy was: ', accuracy ,' for word sequence number:', word_sequence + 1 )

        # Plot a histogram of the sampled startpositions for the current word sequence to get an estimate of its
        # distribution.
        plot_start_position_distribution( sampled_startposition.tolist(), true_startpositions[ word_sequence ] )

        input()


if __name__ == '__main__':
    main()
