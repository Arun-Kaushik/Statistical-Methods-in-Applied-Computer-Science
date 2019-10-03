from train_generator import *
import math
import numpy as np

import random
from train_helper_functions import next_state, make_deepcopy, extract_start_pos, convert_array_to_matrix, convert_matrix_to_array


def blocked_gibbs_sampler(observations, train_track, num_iter, error_prob=0.1):
    """
        A blocked gibbs sampler for the switch settings and start position of the train.
    """

    start_positions = []  # store samples for the start positions
    switch_settings = []  # store switch states

    # generate initial switch state
    switch_settings.append(sample_switch_states(train_track.lattice_size))
    # set the initial start position as the one at G[0][0]
    start_positions.append(sample_start_pos(train_track))

    for n in range(num_iter):

        """ Gibbs sampling of the start position of the train """
        start_position_sampled = gibbs_sampler_start_position(
            train_track, observations, switch_settings, error_prob)

        # we extract a node for the index for the sampled start position and add it to our list of smaples
        start_positions.append(extract_start_pos(
            len(switch_settings[0]), start_position_sampled, train_track))


        switch_setting_sample = blocked_gibbs_sampler_switch_settings(switch_settings, train_track, observations, error_prob, start_positions)
        switch_settings.append(convert_array_to_matrix(switch_setting_sample))

    return start_positions, switch_settings


def extractor(x, switch_settings_block2, switch_settings_block3, X, b):

    X[b[2]] = np.mod(x, switch_settings_block3) + 1
    X[b[1]] = np.mod(np.floor_divide(x, switch_settings_block3),
                     switch_settings_block2) + 1
    X[b[0]] = np.floor_divide(
        x, (switch_settings_block2 * switch_settings_block3)) + 1
    return X


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


def gibbs_sampler_start_position(train_track, observations, switch_settings, error_prob):
    """
        Gibbs sampling of the start positions.
    """


    log_likelihoods = []

    for row in range(train_track.lattice_size):
        for col in range(train_track.lattice_size):

            last_switch_setting = switch_settings[-1]

            # We go thorugh every possible value the start pos
            start_position = train_track.get_node(row, col)

            log_likelihood = compute_conditional_likelihood(
                observations, train_track, start_position, last_switch_setting, error_prob)
            log_likelihoods.append(log_likelihood)

    likelihoods = unlog_likelihoods(log_likelihoods)

    # Normalize the likelihoods
    distribution = normalize_distribution(likelihoods)

    # Categorical sampling of new start position
    start_position_sampled = categorical_sampling(distribution)

    return start_position_sampled


def blocked_gibbs_sampler_switch_settings(switch_settings, train_track, observations, error_prob, start_positions):
    """
        A blocked sampler for the switch settings.
    """

	block_1 = [0, 2, 4]
	block_2 = [3, 5, 7]
	block_3 = [1, 6, 8]

	block_indicies = [block_1, block_2, block_3]

	# We create an array of the switches in the last samplex switch setting(s) X
	switch_settings_last = switch_settings[-1]
	switch_settings_last = make_deepcopy(
		switch_settings_last, train_track.lattice_size)
	switch_settings_last = convert_matrix_to_array(switch_settings_last)

	for block in block_indicies:
		log_likelihoods = []
		# So for the three blocks we want to go though all possible swithc setting for each position
		# Also note that we need three nested loops since we want to compute all possible cobintaions of switch settiings
		# for the three blocks

		switch_settings_upper = 4
		switch_settings_lower = 1

		for switch_settings_block1 in range(switch_settings_lower, switch_settings_upper):
			for switch_settings_block2 in range(switch_settings_lower, switch_settings_upper):
				for switch_settings_block3 in range(switch_settings_lower, switch_settings_upper):

					ind_b1 = block[0]
					switch_settings_last[ind_b1] = switch_settings_block1

					ind_b2 = block[1]
					switch_settings_last[ind_b2] = switch_settings_block2

					ind_b3 = block[2]
					switch_settings_last[ind_b3] = switch_settings_block3

					# We convert the array into a matrix again so taht we can send it into compute_conditional_likelihood()
					switch_settings_last = convert_array_to_matrix(
						switch_settings_last)

					start_position = start_positions[-1]

					prior = math.log(1 / 3)
					log_likelihood = compute_conditional_likelihood(
						observations, train_track, start_position, switch_settings_last, error_prob) + prior
					log_likelihoods.append(log_likelihood)
					switch_settings_last = convert_matrix_to_array(
						switch_settings_last)

		likelihoods = unlog_likelihoods(log_likelihoods)
		distribution = normalize_distribution(likelihoods)

		# Categorical resmapling
		switch_setting_sample = categorical_sampling(distribution)

		# We extract the new indicies for the new sampled switch setting.
		switch_setting_sample = extractor(switch_setting_sample, switch_settings_block2,
						  switch_settings_block3, switch_settings_last, block)

	return switch_setting_sample


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
