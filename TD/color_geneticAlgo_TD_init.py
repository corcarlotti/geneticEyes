# me - this DAT
# scriptOp - the OP which is cooking

# press 'Setup Parameters' in the OP to call this function to re-create the parameters.

def onSetupParameters(scriptOp):
	print()
	pM = op("constant1")["pM"]
	pC = op("constant1")["pC"]
	dna = int(op("constant1")["dna"])
	pop_size = int(op("constant1")["pop_size"])
	dna_size = int(op("constant1")["dna_size"])
	goal = int(op("constant1")["goal"])
	last_population = np.transpose(setup_algorithm(dna, pop_size, dna_size))

	best = run_algorithm_step(pM, pC, pop_size, dna_size, goal, last_population)
	out_array = np.ones((dna_size,pop_size,3), dtype = np.uint8)

	out_array[:,:,0] = last_population
	out_array[:,:,1] = last_population
	out_array[:,:,2] = last_population

	print(last_population)

	scriptOp.copyNumpyArray(out_array)
	return

### Genetic Algorithm with Python Round 1.0 ###

from fileinput import filename
from os import lseek
from turtle import color
import numpy as np
import random
import math
import colorsys
import sys

## create an individual
## dna is seed
##class Individual:
##    
##    def _init_(self, dna):
##        if self.dna == dna:
##            dna.slice()
##        else:
##            return []
##
##    ##def mutate(self, probMutate):
##        mutatedRoute = self.dna.slice()
##
##    ##def getDistance(self):
##        
##    
##    ##def getFitness(self):
##        return (1/getDistance(self))
##

## probability of mutation = pM
## probability of crossover = pC

## create a new population of size x
##def population (size, seed, pC, pM):
	
def generate_individual(dna, size):
	individual = np.random.randint(1, size,size)
	return (individual)
			  
def get_fitness(individual, size, goal):
	distance = np.count_nonzero(individual==goal)
	fitness = distance/size
	return fitness

def generate_population(dna,dna_size, pop_size):
	np.random.seed(dna)
	start_population = []
	for i in range(pop_size):
		start_population.append(generate_individual(dna,dna_size))
	return (start_population)

def current_fitnesses(population, size, goal):
	current_fitnesses = []
	for x in population:
		current_fitnesses.append(get_fitness(x, size, goal))
	return current_fitnesses

def select_parent(population, size, goal):
	fitnessArray = current_fitnesses(population, size, goal)
	fitnessSum = sum(map(float,fitnessArray))
	roll = random.uniform(0,1) * fitnessSum

	for i in range(len(population)):
		if roll <= fitnessArray[i]:
			return population[i]
		else:
			roll -= fitnessArray[i]

def orderedCross(startInd, endInd, p1, p2):
	dna_length = len(p1)
	choose = random.choice([1,2])
	if choose == 1:
		childDNA = list(p1[0:(startInd)]) + list(p2[startInd:endInd]) + list(p1[(endInd):dna_length])
	else:
		childDNA = list(p2[0:(startInd)]) + list(p1[startInd:endInd]) + list(p2[(endInd):dna_length])
	return childDNA   

def crossover(parent1, parent2):
	num1 = math.floor((len(parent1)-1)*random.uniform(0,1))
	num2 = math.floor((len(parent2)-1)*random.uniform(0,1))

	segmentStart = min(num1,num2)
	segmentEnd = max(num1,num2)

	offspring = orderedCross(segmentStart, segmentEnd, parent1, parent2)
	return (offspring)
 
def mutate(individual):
	index = len(individual) - 1 
	mutation_position = random.randint(0,index)
	individual[mutation_position] = random.randint(1,(index+1))
	return(individual)

def two_crossover(parent1, parent2):
	num1 = math.floor((len(parent1)-1)*random.uniform(0,1))
	num2 = math.floor((len(parent2)-1)*random.uniform(0,1))
 
	segmentStart = min(num1,num2)
	segmentEnd = max(num1,num2)

	offspring1 = orderedCross(segmentStart, segmentEnd, parent1, parent2)
	offspring2 = orderedCross(segmentStart, segmentEnd, parent2, parent1)
	return [offspring1, offspring2]

def make_two_children(population, pC, pM, size, goal):
	p1 = select_parent(population, size, goal)
	p2 = select_parent(population, size, goal)
	if pC > random.uniform(0,1):
		two_children = two_crossover(p1, p2)
	else:
		two_children = [p1,p2]
		if pM > random.uniform(0,1):
			two_children = map(mutate,two_children)
		else:
			two_children = two_children
	return two_children        

def next_hopefully_diverser_gen(population, posCr, posMut, dna_size, goal):
	diverser_pop = []
	while len(diverser_pop) < len(population):
		new_kids = make_two_children(population, posCr, posMut, dna_size, goal)
		for child in new_kids:
			diverser_pop.append(np.array(child))
	if len(diverser_pop) > len(population):
		diverser_pop = diverser_pop[:-1]
	return diverser_pop

def get_best(population, size, goal):
	(m,i) = max((fitness,index) for index,fitness in enumerate(current_fitnesses(population, size, goal)))
	return population[i]



def setup_algorithm(dna, pop_size, dna_size):
	last_population = generate_population(dna, dna_size, pop_size)
	return last_population

def run_algorithm_step(pM, pC, pop_size, dna_size, goal, last_population):
	#population_zero = next_generation(population_zero, pC, pM)
	last_population = next_hopefully_diverser_gen(last_population, pC, pM, dna_size, goal)
	best = get_best(last_population, pop_size, goal)
	print(best)
	return best

def run_algorithm(pM, pC, dna, pop_size, dna_size, goal):

	while np.count_nonzero(get_best(last_population, pop_size, goal)==goal) != dna_size:
		run_algorithm_step(pM, pC, pop_size, dna_size, goal, last_population)

	return 0



# called whenever custom pulse parameter is pushed
def onPulse(par):
	return

def onCook(scriptOp):


	return
