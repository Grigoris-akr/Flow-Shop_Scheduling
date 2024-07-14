import numpy as np

def cost_fun(array_zero, particle):
	""" makespan calculation function """

	particle = [number - 1 for number in particle]
	array_zero = array_zero[:, particle]

	jobs = np.shape(array_zero)[0]
	machines = np.shape(array_zero)[1]

	array_main = np.zeros((jobs,machines))

	for i in range(jobs):
		for j in range(machines):
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
	""" 1-1 exchange function"""

	new_seq = seq.copy()
	a = np.random.randint(0, len(seq)-1)
	b = np.random.randint(0, len(seq)-1)
	temp = new_seq[a]
	new_seq[a] = new_seq[b]
	new_seq[b] = temp
	return new_seq

def relocate(seq):
	""" 1-0 relocate function"""

	a = np.random.randint(0, len(seq)-1)
	b = np.random.randint(0, len(seq)-2)
	node = seq[a]
	seq = np.delete(seq, a)
	seq = np.insert(seq, b, node)
	return seq


def pso(initial_array, seed, max_iter, n_particles, reloc_iter, exchange_iter):
	""" Particle Swarm Optimization """

	np.random.seed(seed)
	n_jobs = initial_array.shape[1]

	# create random combination for each particle
	X_list = [None] * n_particles
	lr = list(range(1, n_jobs + 1))
	
	for i in range(n_particles):
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
	
			temp1 = w * speed[j] 
			temp2 = c1 * np.random.uniform(0,1) * (np.array(pbest_temp) - np.array(particle[j])) 
			temp3 = c2 * np.random.uniform(0,1) * (np.array(p_best_global_temp) - np.array(particle[j]))
			speed[j] = temp1 + temp2 + temp3
			particle[j] = particle[j] + speed[j]

			# unSVP 
			binder = dict(zip(particle_mirror, particle[j]))
			binder_sorted = {k: v for k, v in sorted(binder.items(), key=lambda item: item[1])}
			particle[j] = list(binder_sorted.keys())
			
		k += 1

	return cost_function_best_global