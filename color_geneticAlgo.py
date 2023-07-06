### Genetic Algorithm with Python Round 1.0 ###

import numpy as np
import random
import math

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
    
#create a individual with rgb value
def generate_individual(dna):
    individual = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
    return (individual)
 
#get the fitness of individual = distance to color value, 0 is the best fitness
def get_individuals_fitness(individual, color):
    fitness=0
    for v in range(3):
        distance = color[v] - individual[v]
        fitness += abs(distance)
    return (fitness/3)

#generate a "size"-size population of individuals 
def generate_population(dna,size):
    np.random.seed(dna)
    start_population = []
    for i in range(size):
        start_population.append(generate_individual(dna))
    return (start_population)


#get the fitnesses of the population
def get_populations_fitnesses(population, color):
    current_fitnesses = []
    for x in population:
        current_fitnesses.append(get_individuals_fitness(x, color))
        
        
    return current_fitnesses



#choose individual with best fitness
def select_parent(population, color):
    popFitnessArray = get_populations_fitnesses(population, color)
    fitnessSum = sum(map(float, popFitnessArray))/len(population)
    #print(fitnessSum)
    roll = random.uniform(0,0.1) * fitnessSum
    #print (roll)

    for i in range(len(population)):
        if roll >= popFitnessArray[i]:
            #print(population[i])
            return population[i]
        else:
            roll += popFitnessArray[i] * random.uniform(0,0.1)
            #print(roll)
            #roll -= random.uniform(0,1) * fitnessSum

y=generate_population(20, 23)
#print(y)
#print(get_populations_fitnesses(y, (23,210,128)))
#print(select_parent(y, (23,210,128)))

#exchange dna
def orderedCross(startInd, endInd, p1, p2):
    dna_length = len(p1)
    choose = random.choice([1,2])
    if choose == 1:
        childDNA = list(p1[0:(startInd)]) + list(p2[startInd:endInd]) + list(p1[(endInd):dna_length])
    else:
        childDNA = list(p2[0:(startInd)]) + list(p1[startInd:endInd]) + list(p2[(endInd):dna_length])
    return childDNA   

#exchange dna for one child of two parents
def one_crossover(parent1, parent2):
    num1 = math.floor((len(parent1)-1)*random.uniform(0,1))
    num2 = math.floor((len(parent2)-1)*random.uniform(0,1))

    segmentStart = min(num1,num2)
    segmentEnd = max(num1,num2)

    offspring = orderedCross(segmentStart, segmentEnd, parent1, parent2)
    #print(offspring)
    return (offspring)

    
def mutate(individual):
    index = len(individual) - 1 
    mutation_position = random.randint(0,index)
    l_individual = list(individual)
    l_individual[mutation_position] = random.randint(0,255)
    t_individual = tuple(l_individual)
    return(t_individual)
#print(mutate(x))

##def make_a_child(population, pC, pM):
##    p1 = select_parent(population)
##    p2 = select_parent(population)
##    #print(p1,p2)
##    if pC > random.uniform(0,1):
##        child = crossover(p1,p2)
##    else:
##        child = random.choice([p1,p2])
##    
##    if pM > random.uniform(0,1):
##        possiblyMutated = mutate(child)
##    else:
##        possiblyMutated = child
##    return (possiblyMutated) 
###print(make_a_child(y, 0.9, 0.3))

#exchange dna for two children of two parents
def two_crossover(parent1, parent2):
    num1 = math.floor((len(parent1)-1)*random.uniform(0,1))
    num2 = math.floor((len(parent2)-1)*random.uniform(0,1))

    segmentStart = min(num1,num2)
    segmentEnd = max(num1,num2)

    offspring1 = orderedCross(segmentStart, segmentEnd, parent1, parent2)
    offspring2 = orderedCross(segmentStart, segmentEnd, parent2, parent1)
    return [offspring1, offspring2]

#make two children
def make_two_children(population, pC, pM, color):
    p1 = select_parent(population, color)
    p2 = select_parent(population, color)
    if pC > random.uniform(0,1):
        two_children = two_crossover(p1, p2)
    else:
        two_children = [p1,p2]
    if pM > random.uniform(0,1):
        two_possibly_mutated = map(mutate,two_children)
    else:
        two_possibly_mutated = two_children
    return two_possibly_mutated        

def next_hopefully_diverser_gen(population, posCr, posMut, color):
    diverser_pop = []
    while len(diverser_pop) < len(population):
        new_kids = make_two_children(population, posCr, posMut, color)
        for child in new_kids:
            diverser_pop.append(np.array(child))
    if len(diverser_pop) > len(population):
        diverser_pop = diverser_pop[:-1]
    return diverser_pop
        

##def next_generation(population, posCr, posMut):
##     evolved_pop = []
##     while len(evolved_pop) < len(population):
##         evolved_pop.append(make_a_child(population, posCr, posMut))
##     #if len(evolved_pop) == len(population)
##     return evolved_pop

#print(current_fitnesses(y))  
#print(sum(current_fitnesses(y)))
#new_gen=next_hopefully_diverser_gen(y, 0.5, 0.1)

#print(new_gen)
#print(current_fitnesses(new_gen))
#print(sum(current_fitnesses(new_gen)))

def get_best(population, color):
    #fitnesses = get_populations_fitnesses(population)
    (m,i) = max((fitness,index) for index,fitness in enumerate(get_populations_fitnesses(population, color)))
    return population[i]

#print(get_best(y))
#print(get_best(new_gen))


def run_algorithm(pM, pC, dna, size, color):
    population_zero = generate_population(dna, size)
    #print(get_best(population_zero))
    while sum(get_best(population_zero, color)) != 0:
        #population_zero = next_generation(population_zero, pC, pM)
        population_zero = next_hopefully_diverser_gen(population_zero, pC, pM, color)
        print(population_zero)
        #print(get_best(population_zero, color))
    return population_zero

run_algorithm(0.32 ,0.12 ,420, 40, (233,34, 23))
        
