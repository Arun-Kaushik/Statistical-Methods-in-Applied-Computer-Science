from train_generator import *
import math
import numpy as np
from collections import Counter
import pdb
import random
from train_helper_functions import next_state, make_deepcopy, extract_start_pos


def gibbs_sampler(observations, train_track, num_iter, prob_err=0.1):
    """
        A gibbs sampler of the switch settings and start positions.
    """

    start_positions = []  # Start position samples
    switch_settings = []  # Switch setting samples

    # The size of the train track in terms of: track_size x track_size
    track_size = len(train_track.G)

    # Generate initial switch state
    switch_settings.append(sample_switch_states(train_track.lattice_size))

    # Set the initial start position as the one at G[ 0 , 0 ]
    start_positions.append(sample_start_pos(train_track))

    for n in range(num_iter):

        """ Gibbs sampling of the start position """
        start_position_sample = gibbs_sampler_start_position(track_size, train_track, switch_settings, observations, prob_err)

        # Storing the current sample so that we can plot the samples in a histogram.
        start_positions.append(start_position_sample)

        """ Gibbs sampling of the switch settings """
        # Making a deepcopy of the last sampled switch setting so that we can make a new proposal
        # based on this setting by updating its elements.
        proposed_switch_setting = make_deepcopy(switch_settings[-1], train_track.lattice_size)

        switch_setting_sample = gibbs_sampler_switch_settings(track_size, proposed_switch_setting, start_position_sample, observations, train_track, prob_err, start_positions)

        # Storing the current sample so that we can plot the samples in a histogram.
        switch_settings.append(switch_setting_sample)

    return start_positions, switch_settings


def gibbs_sampler_start_position(track_size, train_track, switch_settings, observations, prob_err):
    """
        Gibbs sampling of the start position conditioned on the train tracks,
        obseravtions and the last sampled switch settings.
    """

    log_likelihoods = []


    # For the gibbs sampler we want to go thorugh every possible start position
    # in the graph for the train track.
    for row in range(track_size):
        for col in range(track_size):

            # We compute the loged likelihood for every startposition
            start_position = train_track.get_node(row, col)
            last_sampled_switch_setting = switch_settings[-1]

            log_likelihoods.append(compute_conditional_likelihood(observations, train_track, start_position, last_sampled_switch_setting, prob_err))

    # "unlogging" the logged likelihoods
    likelihoods = unlog_likelihoods(log_likelihoods)
    distribution = normalize_distribution(likelihoods)

    # Sampling from a Categorical
    sampled_startposition =  categorical_sampling(distribution)

    # Extracting the node for the sampled start position
    sampled_startposition = extract_start_pos(len(switch_settings[0]), sampled_startposition, train_track)

    return sampled_startposition


def gibbs_sampler_switch_settings( track_size, proposed_switch_setting, start_position, observations, train_track, prob_err, start_positions):
    """
        Gibbs samping of switch settings conditioned on the current start position, train track graph and observations.
    """

    for row in range(track_size):
        for col in range(track_size):

            log_likelihoods = []

            switch_settings_upper = 4
            switch_settings_lower = 1

            # For every node in the current Graph we want to go through every
            # possible switch setting, i.e. 1, 2 and 3
            for switch_setting in range(switch_settings_lower, switch_settings_upper):

                proposed_switch_setting[row][col] = switch_setting
                start_position = start_positions[-1]

                # The prior is given in the assignment as 1/3
                prior = math.log(1 / 3)

                log_likelihood = compute_conditional_likelihood(observations, train_track, start_position, proposed_switch_setting, prob_err)
                log_likelihoods.append(log_likelihood + prior)

            likelihoods = unlog_likelihoods(log_likelihoods)
            distribution = normalize_distribution(likelihoods)

            # Categorical sampling of new switch setting.
            sampled_switch_setting = categorical_sampling(distribution)

            # We store the sampled switch setting amongst the other switch settings for the train track.
            proposed_switch_setting[row][col] = 1 + sampled_switch_setting

    return proposed_switch_setting


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


def normalize_distribution(likelihoods):

    distribution = likelihoods / np.sum(likelihoods)

    return distribution


def unlog_likelihoods(log_likelihoods):

    likelihoods = np.exp(log_likelihoods - np.max(log_likelihoods))

    return likelihoods


def categorical_sampling(distribution):
    """ Performs a sampling from a categorical distribution.
    """
    no_drawings = 1
    sample = np.argmax(np.random.multinomial(no_drawings, distribution))

    return sample
