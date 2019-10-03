from train_generator import *
import math
import numpy as np
from collections import Counter
import pdb
import random
from train_helper_functions import make_deepcopy, extract_start_pos, next_state


def metropolis_hastings_gibbs_sampler(observations, train_track, no_iterations, error_prob=0.1):
    """
        A Metropolis-Hastings sampler (for the swith states) within a Gibbs sampler (for s1)
    """
    sampled_startpositions = []  # Used to store samples for the start positions.
    sampled_switch_settings = []  # Used to store sampled switch states.

    # Used to calcule the acceptence ratio of proposed switch states.
    no_accepted_proposals = 0
    no_proposals = 0

    # Sample and store intial switch settings and startpositions.
    sampled_switch_settings.append(sample_switch_states(
        train_track.lattice_size))  # generate initial switch state
    # set the initial start position as the one at G[0][0]
    sampled_startpositions.append(sample_start_pos(train_track))

    """ Metropolis-Hastings within gibbs sampling. """
    for it in range(no_iterations):

        """ Gibbs sampling of the startposition for the current iteration conditioned
        on the graph, observations and current switch states. """

        sampled_startposition = gibbs_sampler(
            train_track, sampled_switch_settings, observations, error_prob)

        # Store the new sampled start position.
        sampled_startpositions.append(sampled_startposition)

        """ Metropolis hastings sampling for the switch settings conditioned on OBS"""

        sampled_switch_setting, no_proposals, no_accepted_proposals = metropolis_hastings_sampler(
            train_track, sampled_startpositions, sampled_switch_settings, observations, error_prob, no_proposals, no_accepted_proposals)

        # Store the new sampled switch settings for the train tracks.
        sampled_switch_settings.append(sampled_switch_setting)

    # Display the proposal acceptence ratio of the Metropolis Hastings sampler for the
    # switch settings.
    acceptence_ratio = no_accepted_proposals / no_proposals
    print('Acceptance rate ', acceptence_ratio)

    return sampled_startpositions, sampled_switch_settings


def gibbs_sampler(train_track, sampled_switch_settings, observations, error_prob):
    """ Gibbs sampling of the trains start position conditioned on the track switch settings,
        the track and the observations.
    """
    log_likelihoods = []

    for row in range(train_track.lattice_size):
        for col in range(train_track.lattice_size):

            switches_prev_it = sampled_switch_settings[-1]

            no_start_positions = train_track.lattice_size * train_track.lattice_size
            log_prior = math.log(1 / no_start_positions)
            log_likelihood = compute_conditional_likelihood(
                observations, train_track, train_track.get_node(row, col), switches_prev_it, error_prob)

            # We multiply the likelihood with the prior. However, we have an
            # uniformative prior so we might as well exclude it but we include
            # it to showcase our understanding.
            log_likelihoods.append(log_likelihood + log_prior)

    # 'Unlogging' the likelihoods
    likelihoods = unlog_likelihoods(log_likelihoods)

    # Normalization of the likelihoods into a valid distribution.
    distribution = normalize_distribution(likelihoods)

    # Categorical sampling of new start position.
    sampled_startposition = categorical_sampling(distribution)
    sampled_startposition = extract_start_pos(len(sampled_switch_settings[0]), sampled_startposition, train_track)

    return sampled_startposition


def metropolis_hastings_sampler(train_track, sampled_startpositions, sampled_switch_settings, observations, error_prob, no_proposals, no_accepted_proposals):
    """ Perform metropolis hastings sampling for the switch settings of the train track.
    """

    # First we make a deepcopy of the switchsettinsg sampled from the previous iteration
    last_sampled_switch_setting = sampled_switch_settings[-1]
    previous_switch_setting = make_deepcopy(
        last_sampled_switch_setting, train_track.lattice_size)

    # Initilize a matrix our new no_accepted_proposals switches
    new_switch_setting = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # We make a copy of the last no_accepted_proposals swittchsettings so that we can condition
    # on the rest of the nodes
    proposal_switch_setting = make_deepcopy(
        last_sampled_switch_setting, train_track.lattice_size)

    # We want to sample a new switch setting for every switch setting.
    for row in range(train_track.lattice_size):
        for col in range(train_track.lattice_size):

            # Extract the sampled startposition form the previous iteration.
            # we want to condition on s1 from the last iteration
            previous_startposition = sampled_startpositions[-1]

            # For every node we sample a proposed new switch setting once from uniform distribution.
            proposal_switch_setting[row][col] = np.random.randint(
                1, train_track.lattice_size + 1)

            # Compute the log_likelihoodikelihoods for the new proposed switch setting and the switch setting from the previous iteration.
            log_likelihood_previous = compute_conditional_likelihood(
                observations, train_track, previous_startposition, previous_switch_setting, error_prob)
            log_likelihood_proposal = compute_conditional_likelihood(
                observations, train_track, previous_startposition, proposal_switch_setting, error_prob)

            # We draw a random sample from a uniform distribution over [0, 1).
            u = np.random.rand()

            # We compute the acceptence probability.
            acceptance_prob = min(
                math.exp(log_likelihood_proposal - log_likelihood_previous), 1)

            # If the acceptence probability is larger than an uniform sample we accpt the proposal
            if u < acceptance_prob:

                # We update the parameters used to calcule the acceptence ratio.
                no_accepted_proposals += 1
                no_proposals += 1

                # Store the proposed switch setting.
                previous_switch_setting = make_deepcopy(
                    proposal_switch_setting, train_track.lattice_size)
                new_switch_setting[row][col] = proposal_switch_setting[row][col]

            # Otherwise we reject the the new proposed switch setting
            else:

                # We update a parameter used to calcule the acceptence ratio.
                no_proposals += 1

                # Store the proposed switch setting as the one from the last lap since we rejected the current proposal.
                proposal_switch_setting = make_deepcopy(
                    previous_switch_setting, train_track.lattice_size)
                new_switch_setting[row][col] = previous_switch_setting[row][col]

    return new_switch_setting, no_proposals, no_accepted_proposals


def unlog_likelihoods(log_likelihoods):

    likelihoods = np.exp(log_likelihoods - np.max(log_likelihoods))

    return likelihoods


def normalize_distribution(likelihoods):

    distribution = likelihoods / np.sum(likelihoods)

    return distribution


def categorical_sampling(distribution):
    """ Performs a sampling from a categorical distribution.
    """
    no_drawings = 1
    sample = np.argmax(np.random.multinomial(no_drawings, distribution))

    return sample


def compute_conditional_likelihood(observations, train_track, startposition, switch_settings, error_prob):
    """
        Computes the conditional likelihood of the observations given the train track,
        startposition and train track switch settings.
    """

    no_time_steps = len(observations)

    # Initialisation: We start by getting the next node/state from the starting node
    new_startposition = train_track.get_next_node(
        startposition, 0, switch_settings)[0]
    current_node = train_track.get_node(startposition.row, startposition.col)
    previous_direction = train_track.get_entry_direction(
        current_node, new_startposition)

    log_likelihood = 0

    """ The conditional probability can be calculed as a sum of transition and emission density functions.
        See page 5 in the assignment description for more information. """
    for t in range(1, no_time_steps):

        # If the previous observation was 1, 2 or 3
        if previous_direction != 0:

            # If the previous direction was not 0, then we know that the next correct
            # observation should be zero.

            # Therefore we know that if the next observation is 0 this observation
            # is accurate.
            if observations[t] == 0:

                # Therefore we add log(1-p) i.e 0.9 if the next observation is 0. Since
                # we did not enter through zero and now have to exit through another switch.
                log_likelihood += math.log(1 - error_prob)

            # If the observations instead is not zero it is incorrect.
            else:

                # Therefore we add log(p) to the conditional likelihood.
                log_likelihood += math.log(error_prob)

        # If the previous observation was 0.
        else:

            # We know that the previous observation was zero. Therefore if the switch setting is correct
            # we should exit through the switch setting.
            true_switch_setting = switch_settings[new_startposition.row][new_startposition.col]

            if observations[t] == true_switch_setting:

                # Therefore we add log(1-p) i.e log(0.9) if the next observation was accurate.
                log_likelihood += math.log(1 - error_prob)

            else:
                # If the next observation was inaccurate we add log(p) i.e log(0.1).
                log_likelihood += math.log(error_prob)

        startposition, new_startposition, previous_direction = next_state(
            new_startposition, train_track, startposition, switch_settings)

    return log_likelihood
