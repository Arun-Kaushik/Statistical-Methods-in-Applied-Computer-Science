import numpy as np
import math

alphabet = [1, 2, 3, 4]


def generator(seed, N, M, K, W, alpha_bg, alpha_mw):
    """ Data initilization for
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


def gibbs( D, alpha_bg, alpha_mw, num_iter, W, N, M, K ):
    """
    A collapsed Gibbs sampler that can be used for estimating the posterior over
    start positions after having observed the N word sequences.

    Input: D: numpy array with shape (N,M),  alpha_bg: numpy array with shape(K),
    alpha_mw: numpy array with shape(K), num_iter: int
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
