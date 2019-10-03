import numpy as np

def make_deepcopy( Switches, Graph_sz ):
    """ Creates a deep copy of all switch settiings within a graph"""
    switch_settings_deep_copy = [[Switches[y][x] for x in range(Graph_sz)] for y in range(Graph_sz)]
    return switch_settings_deep_copy


def extract_start_pos(sz,new_s1,G):
	""" Extraxt a new sampled startposition  node from the graph"""
	c=np.mod(new_s1,sz)
	r=(int)(new_s1/sz)
	return G.get_node(r,c)


def next_state(st_n, G, start, X):
	st_hold=st_n
	entry_dir=G.get_entry_direction(start,st_n)
	st_n=G.get_next_node(st_n,entry_dir,X)
	start=st_hold
	st_n=st_n[0]
	node_prev=G.get_node(start.row,start.col)
	node_new=G.get_node(st_n.row,st_n.col)
	prev_dir=G.get_entry_direction(node_prev,node_new)
	return start, st_n, prev_dir


def convert_node_to_string(sequence):
	""" Converts a node to astring representation"""
	str_list=[]
	for s in sequence:
		str_list.append(str(s.row)+' '+str(s.col))
	return str_list


def convert_array_to_matrix(array_list):
	"""Converts an array of size 9 into a 3x3 matrix
	"""
	return [array_list[0:3], array_list[3:6], array_list[6:9]]


def convert_matrix_to_array( matrix ):
	"""converts a matrix into an array
	"""
	return [val for row in matrix for val in row]
