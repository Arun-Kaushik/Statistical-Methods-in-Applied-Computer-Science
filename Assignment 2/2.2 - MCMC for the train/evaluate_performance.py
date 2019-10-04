from train_helper_functions import convert_node_to_string
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pdb

def apply_burn_in(samples, burn_in):

    samples = samples[burn_in:-1]
    return samples


def apply_lag(samples, lag):

    samples = samples[0::lag]
    return samples



def calculate_accuracy(burn_in, lag, s, s_truth):
    """Calculates the accuracy of a start position sequence
    """
    s_b = s[burn_in:-1]
    s_lag = s_b[0::lag]
    s_str = convert_node_to_string(s_lag)

    cnt = Counter(s_str)
    tmp1 = cnt.most_common(9)
    occurances_most_common = tmp1[0][0][1]

    s1_truth_str_rep = str(convert_node_to_string([s_truth])[0])

    print('The most common sample was correct! It was: ', s1_truth_str_rep)
    print('Accuracy: ', int(occurances_most_common) / len(s_lag))
    print()


def histogram_plotter( sampled_start_positions, start_position_truth, sampling_method):
    """"Computes histograms for 3 chains for all algorithms for s1 """

    # Convert the true start position into its string representation.
    start_position_truth = str(
        convert_node_to_string([start_position_truth])[0])

    # Convert the sampled start positions into its string representations.
    sampled_start_positions = convert_node_to_string(sampled_start_positions)

    # Counting the occurances of the different possible start positions.
    height = [sampled_start_positions.count('0 0'), sampled_start_positions.count('0 1'), sampled_start_positions.count('0 2'), sampled_start_positions.count('1 0'), sampled_start_positions.count(
        '1 1'), sampled_start_positions.count('1 2'), sampled_start_positions.count('2 0'), sampled_start_positions.count('2 1'), sampled_start_positions.count('2 2')]

    # Possible start positions.
    bars = ['(0,0)', '(0,1)', '(0,2)', '(1,0)', '(1,1)',
            '(1,2)', '(2,0)', '(2,1)', '(2,2)']

    plt.bar(np.arange(len(bars)), height, color='b')

    title_str = 'Sampled Start Positions - ' + sampling_method + ' - True Start Position=' + \
        str(start_position_truth)
    plt.title(title_str)

    plt.xlabel('Start positions')
    plt.ylabel('Occurances')

    plt.xticks(np.arange(len(bars)), bars)

    plt.show()
