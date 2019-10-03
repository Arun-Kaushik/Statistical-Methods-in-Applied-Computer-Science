from train_helper_functions import convert_node_to_string
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pdb


def calc_acc(burn_in,lag,s,s_truth):
	"""Calculates the accuracy of a start position sequence
	"""
	s_b=s[burn_in:-1]
	s_lag=s_b[0::lag]
	s_str=convert_node_to_string(s_lag)

	cnt = Counter(s_str)
	tmp1=cnt.most_common(9)
	occurances_most_common=tmp1[0][0][1]

	s1_truth_str_rep=str(convert_node_to_string([s_truth])[0])


	print('The most common sample was correct! It was: ',s1_truth_str_rep)
	print('Accuracy: ',int(occurances_most_common)/len(s_lag))
	print()


def convergence_histogram_plotter(burn_in,lag,s,s_truth):
	""""Computes histograms for 3 chains for all algorithms for s1 """
	print()
	true_s1_str=str(convert_node_to_string([s_truth])[0])

	s_mhg=s[0]
	sg=s[1]
	sbg=s[2]

	s_mhg=s_mhg[burn_in:-1]
	s_mhg=s_mhg[0::lag]

	sg=sg[burn_in:-1]
	sg=sg[0::lag]

	sbg=sbg[burn_in:-1]
	sbg=sbg[0::lag]
	#
	sbg_str_rep=convert_node_to_string(sbg)
	sg_str_rep=convert_node_to_string(sg)
	s_mhg_str_rep=convert_node_to_string(s_mhg)


	height0 = [sbg_str_rep.count('0 0'), sbg_str_rep.count('0 1'), sbg_str_rep.count('0 2'), sbg_str_rep.count('1 0'),sbg_str_rep.count('1 1'),sbg_str_rep.count('1 2'), sbg_str_rep.count('2 0'), sbg_str_rep.count('2 1'), sbg_str_rep.count('2 2')]
	bars0 = ['(0,0)','(0,1)', '(0,2)', '(1,0)', '(1,1)', '(1,2)','(2,0)','(2,1)','(2,2)']
	y0_pos = np.arange(len(bars0))
	plt.bar(y0_pos, height0, color = 'r')
	title_str='Start Positions - Blocked Gibbs - last half of samples - True s1='+str(true_s1_str)
	plt.title(title_str)
	plt.xlabel('Start positions')
	plt.ylabel('Occurances')
	plt.xticks(y0_pos, bars0)
	plt.show()

	height0 = [sg_str_rep.count('0 0'), sg_str_rep.count('0 1'), sg_str_rep.count('0 2'), sg_str_rep.count('1 0'),sg_str_rep.count('1 1'),sg_str_rep.count('1 2'), sg_str_rep.count('2 0'), sg_str_rep.count('2 1'), sg_str_rep.count('2 2')]
	bars0 = ['(0,0)','(0,1)', '(0,2)', '(1,0)', '(1,1)', '(1,2)','(2,0)','(2,1)','(2,2)']
	y0_pos = np.arange(len(bars0))
	plt.bar(y0_pos, height0, color = 'r')
	title_str='Start Positions - Gibbs - last half of samples - True s1='+str(true_s1_str)
	plt.title(title_str)
	plt.xlabel('Start positions')
	plt.ylabel('Occurances')
	plt.xticks(y0_pos, bars0)
	plt.show()

	height0 = [s_mhg_str_rep.count('0 0'), s_mhg_str_rep.count('0 1'), s_mhg_str_rep.count('0 2'), s_mhg_str_rep.count('1 0'),s_mhg_str_rep.count('1 1'),s_mhg_str_rep.count('1 2'), s_mhg_str_rep.count('2 0'), s_mhg_str_rep.count('2 1'), s_mhg_str_rep.count('2 2')]
	bars0 = ['(0,0)','(0,1)', '(0,2)', '(1,0)', '(1,1)', '(1,2)','(2,0)','(2,1)','(2,2)']
	y0_pos = np.arange(len(bars0))
	plt.bar(y0_pos, height0, color = 'r')
	title_str='Start Positions - MH within Gibbs - last half of samples - True s1='+str(true_s1_str)
	plt.title(title_str)
	plt.xlabel('Start positions')
	plt.ylabel('Occurances')
	plt.xticks(y0_pos, bars0)
	plt.show()
