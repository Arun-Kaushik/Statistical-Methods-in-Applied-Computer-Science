import numpy as np
import math
import matplotlib.pyplot as plt

alphabet = [1, 2, 3, 4]


def generator(seed, N, M, K, W, alpha_bg, alpha_mw):
    """ Data initilization for ..
    """
    # Data generator.
    # Input: seed: int, N: int, M: int, K: int, W: int, alpha_bg: numpy array with shape(K), alpha_mw: numpy array with shape(K)
    # Output: D: numpy array with shape (N,M), R_truth: numpy array with shape(N), prior_bg: numpy array with shape (K), prior_mw: numpy array with shape (W,K)

    np.random.seed(seed)        # Set the seed

    D = np.zeros((N, M))         # Sequence matrix of size NxM
    R_truth = np.zeros(N)       # Start position of magic word of each sequence

    # All backgorund letters within each sequence K are drawn from the same distribution
    prior_bg = np.zeros(K)
    # Every possible startposition W within each sequence K is drawn from an individual distribution
    prior_mw = np.zeros((W, K))

    # A dirichlet prior is used
    prior_bg = np.random.dirichlet(alpha_bg)
    prior_mw = np.random.dirichlet(alpha_mw, W)

    # Initially, for each sequence the a start position r_n for a magic word is sampled from a uniform distribution
    R_truth = np.random.randint(0, M - W + 1, N)


    for n in range(N):
        it = 0
        for m in range(M):

            # Next every j:th position in the magic words are sampled from a categorical distribution with a dirichlet prior
            if m in range(R_truth[n], R_truth[n] + W):
                D[n, m] = sample_letter_categorical_dirichlet_prior(prior_mw[it])
                it += 1

            else:
            # Every non magic word position is sampled from another categorical distribution with another prior
                D[n, m] =  sample_letter_categorical_dirichlet_prior(prior_bg)

    return D, R_truth, prior_bg, prior_mw

def sample_letter_categorical_dirichlet_prior(prior):
    """Draws a letter from a categorical distribution using a given prior
    """
    return alphabet[ np.argmax( np.random.multinomial( 1,prior)) ]


def marginal_likelihood_mw(alpha_mw, N_vec_j, N):  # where j is the current position in the word
    """ Computes the marginal likelihood for the j:th position of a magic word p(D_j|R)
    """

    ratio1 = math.log( math.gamma(np.sum(alpha_mw)) ) - math.log( math.gamma(N + np.sum(alpha_mw)) )

    prod_sums = alpha_mw + N_vec_j


    for k in range(len(alpha_mw)):
        prod_sums[k] = math.log(math.gamma(prod_sums[k])) - math.log(math.gamma(alpha_mw[k]))


    ratio2 = np.sum(prod_sums)

    marginal_likelihood = ratio1 + ratio2

    return marginal_likelihood


def marginal_likelihood_bw(alpha_bg, B_vec, B):
    """ Computes the marginal likelihood for the background positions: p(D_B|R)
        see equation # XXX:
    """
    ratio1 = math.log( math.gamma(np.sum(alpha_bg)) ) - math.log( math.gamma(B + np.sum(alpha_bg)) )

    prod_sums = alpha_bg + B_vec

    for k in range(len(alpha_bg)):
        prod_sums[k] = math.log(math.gamma(prod_sums[k])) -  math.log(math.gamma(alpha_bg[k]))

    ratio2 = np.sum(prod_sums)

    marginal_likelihood = ratio1 + ratio2

    return marginal_likelihood


def posterior_full_conditional(D, alpha_bg, alpha_mw, W, R_prev, seq, N, M, K, B, N_mw):
    # Estimates the full conditional distribution: p(r_n | R_-n, D)
    # See equation x

    no_seqs = D.shape[0]
    len_seq = D.shape[1]

    position_range_mw = M - W + 1

    cond_probs_start_pos = []

    # We explore the posterior by drilling through all possible values of r_n
    # conditioned on R_n and D
    for r_n in range(position_range_mw):

        R_curr = R_prev  # going through all possible start positions for the current sequence
        R_curr[seq] = r_n

        N_vec = np.zeros((W, K))
        B_vec = np.zeros(K)

        cond_probs_mw = []


        # To calculate the marginal likelihoods (see eq x) we need to calculate Nkj
        # which is the count of symbol k in the j:th column of the magic words, induced by R.
        # Also we need Bk, which is the count of symbol k in the background, induced by R.

        for row in range(no_seqs):
            index_mw = 0
            for col in range(len_seq):

                k = int(D[row, col]) - 1

                if is_magic_word(R_curr, row, col, W):

                    N_vec[index_mw, k] = update_Nkj(N_vec, index_mw, k)
                    index_mw += 1

                else:
                    # If the current position in the current sequence is a backgorund word we calculate
                    # the marginal likelihood using equation x

                    B_vec[k] = update_Bk(B_vec , k)


        # The log likelihoods: log(p(D_j| R_-n U r_n))
        for j in range(W):
        # For a certain column j in a magic word of length W

            marg_likelihood_mw_j = marginal_likelihood_mw(alpha_mw, N_vec[j], N_mw)
            cond_probs_mw.append(marg_likelihood_mw_j)

        marg_likelihood_bg = marginal_likelihood_bw(alpha_bg, B_vec, B)

        # The sum log(p( D_B | R_-n U r_n) * ∏ p(D_j| R_-n U r_n))
        full_cond = marg_likelihood_bg + sum(cond_probs_mw)

        cond_probs_start_pos.append(full_cond)

    return cond_probs_start_pos


def is_magic_word(R_curr, row, col, W ):
    """ Checks if the current letter in R is a strat position
    """

    if col in range(int(R_curr[row]), int(R_curr[row]) + W):
        return True
    else:
        return False


def update_Nkj(N_vec, index_mw, k):
    """ Updates the count for the k:th sequence at the j:th position
    """
    return N_vec[index_mw, k] + 1


def update_Bk(B_vec , k):
    """ updates the count for the k:th sequence
    """
    return B_vec[k] + 1


def gibbs(D, alpha_bg, alpha_mw, num_iter, W, N, M, K):
    """
    A collapsed Gibbs sampler that can be used for estimating the posterior over
    start positions after having observed the N word sequences.

    Input: D: numpy array with shape (N,M),  alpha_bg: numpy array with shape(K), alpha_mw: numpy array with shape(K), num_iter: int
    Output: R: numpy array with shape(num_iter, N)
    """

    # Used to store samples for start positions of magic word of each sequence
    R = np.zeros((num_iter, N))

    # p.162 murphy, eq 5.26

    # Firsy start positions for magic words are drawn from an arbitary distribution
    R_start = (np.random.randint(0, M - W + 1, N))
    R = np.insert(R, 0, R_start, 0)

    # N and B (see equation X) used to calculate the marginal likelihoods for The
    # start positions and the background positions
    B = N * (M - W)
    N_mw = N * W


    for it in range(num_iter):
        R_curr = np.copy(R[it]) # Create a deepcopy

        for seq in range(N):
            prob_pos = posterior_full_conditional( D, alpha_bg, alpha_mw, W, R_curr, seq, N, M, K, B, N_mw )

            prob_pos = np.exp(prob_pos - np.max(prob_pos))
            # normalize to get categorical
            prob_pos = prob_pos / np.sum(prob_pos)

            index_rand = np.random.multinomial(1, prob_pos)

            r_n_new = np.argmax(index_rand)  # categorical sampling

            R_curr[seq] = r_n_new
        R[it + 1] = R_curr

    return R


def main():

    """ Define parameters for the Gibbs Sampling.
    """
    seed = 123

    # The length of a word sequence.
    N = 20

    # The length of a magic word.
    M = 10

    # The lenght of the alphabet.
    K = 4

    # The lenght of a magic word.
    W = 5

    alpha_bg = [7, 13, 1, 5]
    alpha_mw = np.ones(K)

    # Number of samplings.
    num_iter = 1000

    # Generate synthetic data.
    D, R_truth, prior_bg, prior_mw = generator( seed, N, M, K, W, alpha_bg, alpha_mw )

    """ Print the generated data for reference during the sampling.
    """
    print("\nSequences: ")
    print(D)
    print("\nStart positions (truth): ")
    print(R_truth)

    """ Inferring start positions of Magic Words via Gibbs Sampling.
    """
    # Use D, alpha_bg and alpha_mw to infer the start positions of magic words.
    R = gibbs( D, alpha_bg, alpha_mw, num_iter, W, N, M, K )

    r0 = R[0, :]

    # input()
    print("\nStart positions (sampled): ")
    print(R[0, :])
    print(R[1, :])
    print(R[-1, :])

    for i in range(20):
        print(R_truth[i])
        print(R[-1, i])

    # YOUR CODE:
    # Analyze the results. Check for the convergence.

    norm = num_iter / N

    plot_r_0 = []
    plot_r_1 = []
    plot_r_2 = []
    plot_r_3 = []

    plot_r0_val = []
    plot_r1_val = []
    plot_r2_val = []
    plot_r3_val = []
    plot_r4_val = []
    plot_r5_val = []
    plot_r6_val = []
    plot_r7_val = []
    plot_r8_val = []
    plot_r9_val = []

    for it in range(len(R[:, 0])):
        plot_r0_val.append(R[it, 0])
        plot_r1_val.append(R[it, 1])
        plot_r2_val.append(R[it, 2])
        plot_r3_val.append(R[it, 3])
        plot_r4_val.append(R[it, 4])
        plot_r5_val.append(R[it, 5])
        plot_r6_val.append(R[it, 6])
        plot_r7_val.append(R[it, 7])
        plot_r8_val.append(R[it, 8])
        plot_r9_val.append(R[it, 9])

    r0_accuracy = plot_r0_val.count(R_truth[0]) / len(plot_r0_val)
    print('Accuracy r0: ', r0_accuracy)
    burn_in = 100
    plot_r0_val = plot_r0_val[burn_in:-1]
    plot_r1_val = plot_r1_val[burn_in:-1]
    plot_r2_val = plot_r2_val[burn_in:-1]
    plot_r3_val = plot_r3_val[burn_in:-1]
    plot_r4_val = plot_r4_val[burn_in:-1]
    plot_r5_val = plot_r5_val[burn_in:-1]
    plot_r6_val = plot_r6_val[burn_in:-1]
    plot_r7_val = plot_r7_val[burn_in:-1]
    plot_r8_val = plot_r8_val[burn_in:-1]
    plot_r9_val = plot_r9_val[burn_in:-1]

    lag = 2
    plot_r0_val = plot_r0_val[0::lag]
    plot_r1_val = plot_r1_val[0::lag]
    plot_r2_val = plot_r2_val[0::lag]
    plot_r3_val = plot_r3_val[0::lag]
    plot_r4_val = plot_r4_val[0::lag]
    plot_r5_val = plot_r5_val[0::lag]
    plot_r6_val = plot_r6_val[0::lag]
    plot_r7_val = plot_r7_val[0::lag]
    plot_r8_val = plot_r8_val[0::lag]
    plot_r9_val = plot_r9_val[0::lag]

################################################################################
    height0 = [plot_r0_val.count(0), plot_r0_val.count(1), plot_r0_val.count(
        2), plot_r0_val.count(3), plot_r0_val.count(4), plot_r0_val.count(5)]
    bars0 = ['0', '1', '2', '3', '4', '5']
    y0_pos = np.arange(len(bars0))
    plt.bar(y0_pos, height0, color='r')
    title_str = 'True r0 =' + str(R_truth[0])
    plt.title(title_str)
    plt.xlabel('Start positions')
    plt.ylabel('Occurances')
    plt.xticks(y0_pos, bars0)
    plt.show()
    r0_accuracy = plot_r0_val.count(R_truth[0]) / len(plot_r0_val)
    print('Accuracy r0: ', r0_accuracy)
################################################################################
    height1 = [plot_r1_val.count(0), plot_r1_val.count(1), plot_r1_val.count(
        2), plot_r1_val.count(3), plot_r1_val.count(4), plot_r1_val.count(5)]
    bars1 = ['0', '1', '2', '3', '4', '5']
    y1_pos = np.arange(len(bars1))
    plt.bar(y1_pos, height1, color='r')
    title_str = 'True r1 =' + str(R_truth[1])
    plt.title(title_str)
    plt.xlabel('Start positions')
    plt.ylabel('Occurances')
    plt.xticks(y1_pos, bars1)
    plt.show()
    r1_accuracy = plot_r1_val.count(R_truth[1]) / len(plot_r1_val)
    print('Accuracy r1: ', r1_accuracy)
################################################################################
    height2 = [plot_r2_val.count(0), plot_r2_val.count(1), plot_r2_val.count(
        2), plot_r2_val.count(3), plot_r2_val.count(4), plot_r2_val.count(5)]
    bars2 = ['0', '1', '2', '3', '4', '5']
    y2_pos = np.arange(len(bars2))
    plt.bar(y2_pos, height1, color='r')
    title_str = 'True r2 =' + str(R_truth[2])
    plt.title(title_str)
    plt.xlabel('Start positions')
    plt.ylabel('Occurances')
    plt.xticks(y2_pos, bars2)
    plt.show()
    r2_accuracy = plot_r2_val.count(R_truth[2]) / len(plot_r2_val)
    print('Accuracy r2: ', r2_accuracy)
################################################################################
    height2 = [plot_r3_val.count(0), plot_r3_val.count(1), plot_r3_val.count(
        2), plot_r3_val.count(3), plot_r3_val.count(4), plot_r3_val.count(5)]
    bars2 = ['0', '1', '2', '3', '4', '5']
    y3_pos = np.arange(len(bars2))
    plt.bar(y3_pos, height1, color='r')
    title_str = 'True r3 =' + str(R_truth[3])
    plt.title(title_str)
    plt.xlabel('Start positions')
    plt.ylabel('Occurances')
    plt.xticks(y2_pos, bars2)
    plt.show()

    r3_accuracy = plot_r3_val.count(R_truth[3]) / len(plot_r3_val)
    print('Accuracy r3: ', r3_accuracy)

    r4_accuracy = plot_r4_val.count(R_truth[4]) / len(plot_r4_val)
    print('Accuracy r4: ', r4_accuracy)

    r5_accuracy = plot_r5_val.count(R_truth[5]) / len(plot_r5_val)
    print('Accuracy r5: ', r5_accuracy)

    r6_accuracy = plot_r6_val.count(R_truth[6]) / len(plot_r6_val)
    print('Accuracy r6: ', r6_accuracy)

    r7_accuracy = plot_r7_val.count(R_truth[7]) / len(plot_r7_val)
    print('Accuracy r7: ', r7_accuracy)

    r8_accuracy = plot_r8_val.count(R_truth[8]) / len(plot_r8_val)
    print('Accuracy r8: ', r8_accuracy)

    r9_accuracy = plot_r9_val.count(R_truth[9]) / len(plot_r9_val)
    print('Accuracy r9: ', r9_accuracy)

    # r plot over sequence
    # plt.plot(plot_r0_val)
    # text0='value r0, r0_truth:'+str(R_truth[0])
    # plt.title(text0)
    # plt.show()
    #
    # plt.plot(plot_r1_val)
    # text1='value r1, r1_truth:'+str(R_truth[1])
    # plt.title(text1)
    # plt.show()
    #
    # plt.plot(plot_r2_val)
    # text2='value r2, r2_truth:'+str(R_truth[2])
    # plt.title(text2)
    # plt.show()
    #
    # plt.plot(plot_r3_val)
    # text3='value r3, r3_truth:'+str(R_truth[3])
    # plt.title(text3)
    # plt.show()
    #
    #
    # #probability plots
    # plt.plot(plot_r_0)
    # #title('prob max(p(rn|R))')
    # plt.show()
    #
    # plt.plot(plot_r_1)
    # plt.show()
    #
    # plt.plot(plot_r_2)
    # plt.show()
    #
    # plt.plot(plot_r_3)
    # plt.show()
    #
    # # plt.plot(plot_r_2)
    # # plt.show()
    # #
    # # plt.plot(plot_r_3)
    # # plt.show()


if __name__ == '__main__':
    main()
