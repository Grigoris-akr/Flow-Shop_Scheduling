import numpy as np
import os
import multiprocessing as mp
import time
import csv
import atexit
from queue import Empty
from pso import pso, pso2

# import logging
# logger = logging.getLogger(__name__)

@atexit.register
def save_output():
	""" saves results in a .csv format upon exiting even in case of an exception """
	with open('output.csv', 'w') as output:
		writer = csv.writer(output)
		for key, value in results.items():
			writer.writerow([key, value])


def create_setups_dict():
	""" creates a dictionary with information for all taillard tables available """

	setup_names = ['20x5',  '20x10',  '20x20',  '50x5',   '50x10',  '50x20', 
				   '100x5', '100x10', '100x20', '200x10', '200x20', '500x20']
	list_of_files = sorted(os.listdir('taillard_tables'))
	table_paths = ['taillard_tables' + "/{0}".format(x) for x in list_of_files]

	setups = {}
	for setup_name, table in zip(setup_names, table_paths):
		with open(table) as f:
			lines = f.readlines()
		n_jobs, n_machines = np.fromstring(lines[1], dtype=int, sep=' ')[:2]

		setups[setup_name] = {'table_path': table,
							  'n_jobs': n_jobs,
							  'n_machines': n_machines,
							  'initial_arrays': None}
	return setups

def create_initial_arrays(setups):
	""" Reads Initial arrays from all taillard tables and adds them to the corresponding key in setups dictionary """

	for curr_setup in list(setups.keys()):
		n_machines 	= setups[curr_setup]['n_machines']
		n_jobs 		= setups[curr_setup]['n_jobs']
		table_path 	= setups[curr_setup]['table_path']

		with open(table_path) as f:
			lines = f.readlines()

		# iterate though all data on a specific taillard table.
		initial_arrays = np.zeros([10, n_machines, n_jobs]) # <-- hardcoded values here
		for i, j in enumerate(range(1, (n_machines + 3)*10, n_machines+3)):
			# upper_bound, lower_bound = np.fromstring(lines[j], dtype=int, sep=' ')[3:5]
			for k in range(n_machines):
				initial_arrays[i, k] = np.fromstring(lines[j+k+2], dtype=int, sep=' ')

		setups[curr_setup]['initial_arrays'] =  initial_arrays

	return setups # no need for a return though, setups dict is passed by reference


def pso_worker(pending_jobs_queue, output_queue, max_iter, n_particles, reloc_iter, exchange_iter):
	""" Worker function. Reads from pending_jobs_queue performs PSO on the given array of a setup and writes to output_queue """

	current = mp.current_process()
	while True:
		try:
			(initial_array, array_id, seed) = pending_jobs_queue.get(timeout = 1)
		except Empty:
			# signaling parent that there are no more jobs available and exiting loop
			print(write_line.format(line = int(current.name)+3, column = 0, val = f"worker:{current.name} finished {' '*40}"))
			output_queue.put(None)
			break

		startTime = time.perf_counter()
		n_machines = initial_array.shape[0]
		n_jobs = initial_array.shape[1]
		print(write_line.format(line = int(current.name)+3, column = 0, val = f"worker:{current.name} at {n_jobs}x{n_machines} array id: {array_id} seed: {seed}      "))

		cost = pso2(initial_array, seed, max_iter, n_particles, reloc_iter, exchange_iter)

		exec_time = time.perf_counter() - startTime
		output = {'setup': f"{n_jobs}x{n_machines}",
				  'cost': cost,
				  'seed': seed,
				  'array_id': array_id,
				  'exec_time': exec_time
				 }

		output_queue.put(output)

if __name__ == '__main__':
	max_iter = 800 # 800 
	reloc_iter = 10 # 10
	exchange_iter = 10 # 10
	n_particles = 500 # 500
	seeds = [42, 1337]

	## -- probably unix-only -- ##
	os.system('clear')
	write_line = "\033[{line};{column}H{val}"
	print(write_line.format(line = 1, column = 0, val = "One little foo and we are out!"))
	## ------------------------ ##

	setups = create_setups_dict()
	setups = create_initial_arrays(setups)

	setups_to_run = ['20x5',  '20x10',  '20x20',  '50x5',   '50x10',  '50x20', 
					 '100x5', '100x10', '100x20', '200x10', '200x20', '500x20']

	# create a global variable to store results
	global results 
	results = {}
	for name in setups_to_run:
		results[name] =  np.zeros([10,len(seeds)*2]) # < -- hard-coded array size - idgaf

	# counting available cores and instantiating data transfering queues
	# nb_cores = min(10, mp.cpu_count()) # i think default affinity maxes out at 10 for windows. i can change it but i dont care for it.
	# nb_cores = mp.cpu_count()
	nb_cores = len(os.sched_getaffinity(0))
	pending_jobs_queue = mp.Queue()
	output_queue = mp.Queue()
	process_list = []
	print(write_line.format(line = 2, column = 0, val = f"Number of cores: {nb_cores}"))

	# create and start processes
	for worker in range(nb_cores):
		process_list.append(mp.Process(target = pso_worker, name = str(worker), args = (pending_jobs_queue, output_queue, max_iter, 
																						 n_particles, reloc_iter, exchange_iter)))

	for process in process_list:
		process.start()

	# filling jobs queue
	for curr_setup in setups_to_run:
		for array_id, init_array in enumerate(setups[curr_setup]['initial_arrays']):
			for seed in seeds:
				pending_jobs_queue.put((init_array, array_id, seed))

	# while the are workers get results
	finished_workers_counter = 0
	while finished_workers_counter < nb_cores:
		x = output_queue.get()
		if x is None:
			finished_workers_counter += 1
		else:
			results[x['setup']][x['array_id'], seeds.index(x['seed'])] = x['cost']
			results[x['setup']][x['array_id'], seeds.index(x['seed'])+2] = x['exec_time']

	# terminating processes
	for process in process_list:
		process.join()

	print('All workers have finished: normal termination')



# -- obsolete --
# def runner(pending_jobs_queue, max_iter, n_particles, reloc_iter, exchange_iter):
# 	while True:
# 		# (initial_array, array_id, seed) = pending_jobs_queue.get()
		# if initial_array is None:
		# 	# signaling parent that there are no more jobs available and exiting loop
		# 	print(write_line.format(line = int(current.name)+2, column = 0, val = f"worker:{current.name} finished {' '*40}"))
		# 	output_queue.put(None)
		# 	break

# 		current = mp.current_process()

# 		n_machines = initial_arrays.shape[1]
# 		n_jobs = initial_arrays.shape[2]

# 		path = os.path.join('multWorker_output', f"{n_machines}x{n_jobs}.txt")
# 		file = open(path, "w")
# 		file.write("iter\tseed\tcost\tseed\tcost\n")
# 		for i, initial_array in enumerate(initial_arrays):
			
# 			output = []
# 			file.write(f"{i}")
# 			for j, seed in enumerate([42,1337]):
# 				print(write_line.format(line = int(current.name)+2, column = 0, val = f"worker:{current.name} at {n_jobs}x{n_machines}\t-progress: {i+1}/10"))
# 				np.random.seed(seed)
# 				output = pso(initial_array, max_iter, n_particles, n_jobs, n_machines, reloc_iter, exchange_iter, int(current.name))
# 				file.write(f"\t{seed}\t{output}")
# 			file.write('\n')
