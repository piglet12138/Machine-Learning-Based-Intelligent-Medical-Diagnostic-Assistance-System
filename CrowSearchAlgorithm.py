# -------------------------------------------------


import random	#random Function
import numpy 	#numpy operations
import math		#ceil function


def init(n, pd, l, u): #init the matrix problem
	x = []
	for i in range (n):
		x.append([])
		for j in range (pd):
			x[i].append(l-(l-u)*(random.random()))
	return x

def fitness(xn, n ,pd):	#function for fitness calculation
    fitness = []
    for i in range(n):
        fitness.append(0)
        for j in range(pd):
            fitness[i] = fitness[i]+pow(xn[(i, j)], 2)
    return fitness

# variables initialization #
pd = 10		#Problem dimension (number of decision variables)
n = 20		#Flock (population) size
ap = 0.1	#Awareness probability
fl = 2		#Flight length (fl)
l = -100	#Lower
u = 100		#Uper
x = numpy.matrix(init(n, pd, l, u))
xn = x.copy()
ft = numpy.array(fitness(xn, n, pd))
mem = x.copy()			#Memory initialization
fit_mem = ft.copy()		#Fitness of memory positions
tmax = 2500				#Max numuber of iterations (itermax)
ffit = numpy.array([])	# Best fit of each iteration

#Iteration begin

for t in range(tmax):

	num = numpy.array([random.randint(0, n-1) for _ in range(n)]) # Generation of random candidate crows for following (chasing)
	xnew = numpy.empty((n,pd))
	for i in range (n):
		if(random.random() > ap):
			for j in range (pd):
				xnew[(i,j)] = x[(i,j)]+fl*((random.random())*(mem[(num[i],j)]-x[(i,j)]))
		else:
			for j in range (pd):
				xnew[(i, j)] = l-(l-u)*random.random()
	xn = xnew.copy()
	ft = numpy.array(fitness(xn, n, pd)) #Function for fitness evaluation of new solutions

	#Update position and memory#
	for i in range(n):
		if(xnew[i].all() > l and xnew[i].all() < u):
			x[i] = xnew[i].copy()		#Update position
			if(ft[i] < fit_mem[i]):
				mem[i] = xnew[i].copy()	#Update memory
				fit_mem[i] = ft[i]
	ffit = numpy.append(ffit, numpy.amin(fit_mem)) 	#Best found value until iteration t

ngbest, = numpy.where(numpy.isclose(fit_mem, min(fit_mem)))
print(mem[ngbest[0]])

import random
import numpy as np


class CrowSearchAlgorithm:
	def __init__(self, fitness_func, problem_dim=10, population_size=20, awareness_prob=0.1, flight_length=2,
				 lower_bound=[], upper_bound=[], max_iter=100):
		self.fitness_func = fitness_func
		self.pd = problem_dim
		self.n = population_size
		self.ap = awareness_prob
		self.fl = flight_length
		self.l = lower_bound
		self.u = upper_bound
		self.tmax = max_iter

		# Population and Memory initialization
		self.x = np.array(init(self.n, self.pd, self.l, self.u))
		self.mem = self.x.copy()
		self.fit_mem = np.array(self._evaluate_fitness(self.x))
		self.ffit = np.array([])

	def _init_population(self):
		x = []
		for i in range(n):
			x.append([])
			for j in range(pd):
				x[i].append(l[j] - (l[j] - u[j]) * (random.random()))
		return x

	def _evaluate_fitness(self, positions):
		return np.array([self.fitness_func(pos) for pos in positions])

	def optimize(self):
		for t in range(self.tmax):
			num = np.random.randint(0, self.n, self.n)  # Random candidate crows for following
			xnew = np.empty_like(self.x)
			for i in range(self.n):
				if (random.random() > self.ap):
					for j in range(self.pd):
						xnew[i][j] = self.x[i][j] + self.fl * (random.random() * (self.mem[num[i]][j] - self.x[i][j]))
				else:
					xnew[i] = [self.l - (self.l - self.u) * random.random() for _ in range(self.pd)]

			ft = self._evaluate_fitness(xnew)

			# Update position and memory
			for i in range(self.n):
				if (np.all(xnew[i] > self.l) and np.all(xnew[i] < self.u)):
					self.x[i] = xnew[i].copy()
					if ft[i] < self.fit_mem[i]:
						self.mem[i] = xnew[i].copy()
						self.fit_mem[i] = ft[i]

			self.ffit = np.append(self.ffit, np.min(self.fit_mem))

		global_best_index = np.argmin(self.fit_mem)
		self.best_position = self.mem[global_best_index]
		self.best_fitness = self.fit_mem[global_best_index]
		return self.best_position, self.best_fitness


# Example usage:
def sample_fitness_function(params):
	return np.sum(np.square(params))


csa = CrowSearchAlgorithm(fitness_func=sample_fitness_function)
best_position, best_fitness = csa.optimize()
print(f"Best Position: {best_position}")
print(f"Best Fitness: {best_fitness}")
