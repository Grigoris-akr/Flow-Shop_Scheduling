import numpy as np

def cost_fun(array_zero, particle):
	particle = [number - 1 for number in particle]
	array_zero = array_zero[:, particle]

	x = np.shape(array_zero)[0] # jobs maybe
	y = np.shape(array_zero)[1] # machines maybe

	array_main = np.zeros((x,y))

	for i in range(0,x):
		for j in range(0,y):
			if(j==0):
				if i == 0:
					array_main[i,j] =  array_zero[i,j]
				else:
					array_main[i,j] = array_main[i-1,j] + array_zero[i,j]
				
			elif(i==0):
				array_main[i,j] = array_main[i,j-1] + array_zero[i,j]
			else:
				array_main[i,j] = max(array_main[i,j-1], array_main[i-1,j]) + array_zero[i,j]			
	
	return array_main[-1,-1]

def exchange(seq):
	new_seq = seq.copy()
	a = np.random.randint(0, len(seq)-1)
	b = np.random.randint(0, len(seq)-1)
	temp = new_seq[a]
	new_seq[a] = new_seq[b]
	new_seq[b] = temp
	return new_seq

def relocate(seq):
	a = np.random.randint(0, len(seq)-1)
	b = np.random.randint(0, len(seq)-2)
	node = seq[a]
	seq = np.delete(seq, a)
	seq = np.insert(seq, b, node)
	return seq


def pso2(initial_array, seed, max_iter, n_particles, reloc_iter, exchange_iter):
	np.random.seed(seed)
	n_jobs = initial_array.shape[1]

	# create random combination for each particle
	X_list = [None] * n_particles
	lr = list(range(1, n_jobs + 1))
	
	for i in range(n_particles):
		# X_list [i] = np.random.sample(lr, n_jobs) 
		X_list [i] = np.random.choice(lr, n_jobs, replace=False)
	
	# create particles
	particle = X_list[:n_particles]
	
	speed = np.array(particle) * np.random.uniform(0.1,0.3)
	
	cost_function = np.zeros(len(particle))
	cost_function_best = np.zeros(len(particle))
	pbest = np.zeros(len(particle))
	
	c1 = 2.05 
	c2 = 2.05
	w = 0.5
	
	pbest = particle
	max_particle = np.max(particle[0])
	
	for i in range(0, len(particle)):
		cost_function_best[i] = cost_fun(initial_array, particle[i])
	
	cost_function_best_global = np.min(cost_function_best)
	p_best_global = particle[np.argmin(cost_function_best)]
	
	k = 0
	while k < max_iter:
		# compute makespan for each particle
		for j in range(0, len(particle)):
			cost_function[j] = cost_fun(initial_array,particle[j])
			if (cost_function[j] <= cost_function_best[j]):
				cost_function_best[j] = cost_function[j]
				pbest[j] = particle[j]
				
			if (cost_function[j] <= cost_function_best_global):
				cost_function_best_global = cost_function[j]
				p_best_global = particle[j]

		# 1-0 relocate
		for _ in range(reloc_iter):
			new_pbest_global = relocate(p_best_global)
			
			if(cost_fun(initial_array, new_pbest_global) < cost_fun(initial_array, p_best_global)):    
				p_best_global = new_pbest_global

		# 1-1 exchange        
		for _ in range(exchange_iter):
			new_pbest_global = exchange(p_best_global)
			
			if(cost_fun(initial_array, new_pbest_global) < cost_fun(initial_array, p_best_global)):    
				p_best_global = new_pbest_global

		# SVP and new speed calculation
		for j in range(len(particle)):
			particle_mirror = particle[j]
			particle[j]= [x / max_particle for x in particle[j]]
			pbest_temp = [x / max_particle for x in pbest[j]]
			p_best_global_temp = [x / max_particle for x in p_best_global]
	
			cc1 = w * speed[j] 
			cc2 = c1 * np.random.uniform(0,1) * (np.array(pbest_temp) - np.array(particle[j])) 
			cc3 = c2 * np.random.uniform(0,1) * (np.array(p_best_global_temp) - np.array(particle[j]))
			speed[j] = cc1 + cc2 + cc3
			particle[j] = particle[j] + speed[j]

			# unSVP 
			binder = dict(zip(particle_mirror, particle[j]))
			binder_sorted = {k: v for k, v in sorted(binder.items(), key=lambda item: item[1])}
			particle[j] = list(binder_sorted.keys())
			
		k += 1

	return cost_function_best_global


## --------------------------------------------- cut here --------------------------------------------- ##

# -- obsolete --
def pso(initial_array, max_iter, n_particles, n_jobs, n_machines, reloc_iter, exchange_iter, worker_name):
	import random
	from random import randint
	# create random combination for each particle

	X_list = [None] * n_particles
	lr = list(range(1, n_jobs + 1))
	
	for i in range(n_particles):
		
		X_list [i] = random.sample(lr, n_jobs)
		# X_list [i] = np.random.choice(lr, n_jobs, replace=False)
	
	# FTIAXNW PARTICLES
	particle = X_list[:n_particles]
	
	speed = np.array(particle) * random.uniform(0.1,0.3)
	
	cost_function = np.zeros(len(particle))
	cost_function_best = np.zeros(len(particle))
	pbest = np.zeros(len(particle))
	
	c1 = 2.05 
	c2 = 2.05
	w = 0.5
	print_res = False
	# anneal w during training - pso paper
	
	k = 0
	
	pbest = particle
	max_particle = max(particle[0])
	
	for i in range(0, len(particle)):
		cost_function_best[i] = cost_fun(initial_array, particle[i])
	
	cost_function_best_global = min(cost_function_best)
	p_best_global = particle[np.argmin(cost_function_best)]
	
	xx = 0
	# KSEKINAW ITERATIONS
	while k < max_iter:
		if not print_res:
			pass
			# printProgressBar(k + 1, max_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
			# printOffset = '\t' * ((worker_name-1)*2+2)
			# print(f"{printOffset}Worker{worker_name}:{((k + 1)/ (max_iter)):.1f}% || ",  end = '\r')

		# c1 = 1 - (k / max_iter)
		# c2 = 1 - ((max_iter - k) / max_iter)
		# w = (1 - (k / max_iter)) * 0.5
		
		xx += 1
		# compute makespan for each particle
		for j in range(0, len(particle)):
			cost_function[j] = cost_fun(initial_array,particle[j])
			if (cost_function[j] <= cost_function_best[j]):
				cost_function_best[j] = cost_function[j]
				pbest[j] = particle[j]
				
			if (cost_function[j] <= cost_function_best_global):
				cost_function_best_global = cost_function[j]
				p_best_global = particle[j]
				xx = 0

		# print(cost_function_best_global)

		# #check gia 1-0 relocate
		# for _ in range(reloc_iter):
		# 	new_pbest_global = relocate(p_best_global)
			
		# 	if(cost_fun(initial_array, new_pbest_global) < cost_fun(initial_array, p_best_global)):    
		# 		p_best_global = new_pbest_global
		# # check gia 1-1 exchange        
		# for _ in range(exchange_iter):
		# 	new_pbest_global = exchange(p_best_global)
			
		# 	if(cost_fun(initial_array, new_pbest_global) < cost_fun(initial_array, p_best_global)):    
		# 		p_best_global = new_pbest_global

		#SVP KAI IPOLOGISMOS NEAS TAXITITAS 
		
		for j in range(0, len(particle)):
			
			particle_mirror = particle[j]
			if j ==0 and print_res:
				print('pre_speed')
				print('particle')
				print(particle[j])
				print('best')
				print(p_best_global)

	
			particle[j]= [x / max_particle for x in particle[j]]
			pbest_temp = [x / max_particle for x in pbest[j]]
			p_best_global_temp = [x / max_particle for x in p_best_global]
			if j ==0 and print_res:
				print('pre_speed_svp')
				print(particle[j])
	
			cc1 = w * speed[j] 
			cc2 = c1 * random.uniform(0,1) * (np.array(pbest_temp) - np.array(particle[j])) 
			cc3 = c2 * random.uniform(0,1) * (np.array(p_best_global_temp) - np.array(particle[j]))
			# cc2 = c1 * random.uniform(0,1) * (np.array(pbest[j]) - np.array(particle[j])) 
			# cc3 = c2 * random.uniform(0,1) * (np.array(p_best_global) - np.array(particle[j]))
#                 speed[j] = (w*speed[j] 
#                            + c1 * random.uniform(0,1) * (np.array(pbest[j]) - np.array(particle[j])) 
#                            + c2 * random.uniform(0,1) * (np.array(p_best_global) - np.array(particle[j])))
			speed[j] = cc1 + cc2 + cc3
			# particle[j] = particle[j] + list(speed[j])
			particle[j] = particle[j] + speed[j]
			if j == 0 and print_res:
				# print('speed',speed[0])
				print('pre svp', particle[0])

			#unSVP 
			binder = dict(zip(particle_mirror, particle[j]))
			
			binder_sorted = {k: v for k, v in sorted(binder.items(), key=lambda item: item[1])}
			
			particle[j] = list(binder_sorted.keys())
			
			if j == 0 and print_res:
				print('after svp', particle[0])
		k += 1
		
	# value = cost_function_best_global
	# pos =  p_best_global
	# resulting_values.append(value)

	# print('iters without impovement', xx)
	# # print(sorted(resulting_values))
	# # find mean
	# results = sorted(resulting_values)[:2]
	# mean = np.mean(results)

	# np.savetxt(file_name + "_results.txt", mean)
	return cost_function_best_global
		


if __name__ == '__main__':
	# data = np.load('taillard_tables/020_05.txt')
	import os
	import time
	startTime = time.time()
	
	list_of_files = os.listdir('taillard_tables')
	list_of_files = sorted(list_of_files)

	# print(list_of_files)
	list_of_files = list_of_files[:1]
	print(list_of_files)

	table_paths = ['taillard_tables' + "/{0}".format(x) for x in list_of_files]
	
	# table_paths = table_paths[0:1]
	# list_of_files = list_of_files[0:1]
	with open(table_paths[0]) as f:
		lines = f.readlines()
	print(table_paths[0])
	n_jobs, n_machines = np.fromstring(lines[1], dtype=int, sep=' ')[:2]
	
	reloc_iter = 2
	max_iter = 1000
	n_particles = 5
	# n_machines = 5
	# n_jobs = 20
	outer_iter = 1
	c = 0

	# for j in range(1, (n_machines + 3) * 10, n_machines + 3):
	j = 1
	print(c)
	upper_bound, lower_bound = np.fromstring(lines[j], dtype=int, sep=' ')[3:5]
	
	initial_array = np.zeros([n_machines, n_jobs])
	for k in range(n_machines):
		initial_array[k] = np.fromstring(lines[j+k+2], dtype=int, sep=' ')

	minim = []
	# for seed in [0,42,1337]:
	for seed in [0]:
		np.random.seed(seed)
		minim.append(pso(initial_array, max_iter, n_particles, n_jobs, n_machines, outer_iter, reloc_iter, exchange_iter))
	# print(minim)
	# minim = np.min(minim)

	# results[i,c] = pso(initial_array, max_iter, n_particles, n_jobs, n_machines, outer_iter)
	results = minim
	best_results = lower_bound
	c += 1

	executionTime = (time.time() - startTime)
	np.save('time.npy', executionTime)
	np.save('res.npy', results)
	np.save('b_res.npy', best_results)

	print(executionTime)
	print(results)
	print(best_results)
	
	
	
	