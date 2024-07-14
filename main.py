import numpy as np
import os
import multiprocessing as mp
import time
import csv
from queue import Empty
from pso import pso


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
		initial_arrays = np.zeros([10, n_machines, n_jobs])
		for i, j in enumerate(range(1, (n_machines + 3)*10, n_machines+3)):
			for k in range(n_machines):
				initial_arrays[i, k] = np.fromstring(lines[j+k+2], dtype=int, sep=' ')
		setups[curr_setup]['initial_arrays'] =  initial_arrays

	return setups


def pso_worker(pending_jobs_queue, output_queue, max_iter, n_particles, reloc_iter, exchange_iter):
	""" Worker function. Reads from pending_jobs_queue performs PSO on the given array and writes to output_queue """

	current = mp.current_process()
	while True:
		try:
			(initial_array, array_id, seed) = pending_jobs_queue.get(timeout = 1)
		except Empty:
			# signaling parent that there are no more jobs available and exiting loop
			print(f"worker:{current.name} finished")
			output_queue.put(None)
			break

		startTime = time.perf_counter()
		n_machines = initial_array.shape[0]
		n_jobs = initial_array.shape[1]

		print(f"worker:{current.name} at {n_jobs}x{n_machines} array id: {array_id} seed: {seed}")
		cost = pso(initial_array, seed, max_iter, n_particles, reloc_iter, exchange_iter)

		exec_time = time.perf_counter() - startTime
		output = {'setup': f"{n_jobs}x{n_machines}",
				  'cost': cost,
				  'seed': seed,
				  'array_id': array_id,
				  'exec_time': exec_time
				 }

		output_queue.put(output)

if __name__ == '__main__':
	startTime = time.perf_counter()
	max_iter = 800
	reloc_iter = 10
	exchange_iter = 10
	n_particles = 500
	seeds = [42, 1337]

	setups = create_setups_dict()
	setups = create_initial_arrays(setups)

	setups_to_run = ['20x5',  '20x10',  '20x20',  '50x5',   '50x10',  '50x20', 
					 '100x5', '100x10', '100x20', '200x10', '200x20', '500x20']

	# create a variable to store results 
	results = {}
	for name in setups_to_run:
		results[name] =  np.zeros([10,len(seeds)*2])

	# counting available cores and instantiating data transfering queues
	nb_cores = mp.cpu_count()
	pending_jobs_queue = mp.Queue()
	output_queue = mp.Queue()
	print(f"Number of cores: {nb_cores}")

	# create and start processes
	process_list = []
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

	# wait for results while there are unfinished jobs
	finished_workers_counter = 0
	while finished_workers_counter < nb_cores:
		res = output_queue.get()
		if res is None:
			finished_workers_counter += 1
		else:
			results[res['setup']][res['array_id'], seeds.index(res['seed'])] = res['cost']
			results[res['setup']][res['array_id'], seeds.index(res['seed'])+2] = res['exec_time']

			with open('output.csv', 'w') as output:
				writer = csv.writer(output)
				for key, value in results.items():
					writer.writerow([key, value])

	# terminating processes
	for process in process_list:
		process.join()

	# saving total execution time in .txt
	exec_time = time.perf_counter() - startTime
	with open('total_exec_time', 'w+') as f:
		f.write(str(exec_time))

	print('All workers have finished: Normal termination')