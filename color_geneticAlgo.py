### Genetic Algorithm with Python Round 1.0 ###

from fileinput import filename
from os import lseek
from turtle import color
import numpy as np
import random
import math
import colorsys
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plotly.colors as pc
from matplotlib.animation import FuncAnimation
from PIL import Image

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
    np.random.seed(dna)
    individual = np.random.randint(1, size,size)
    return (individual)
#print(generate_individual(101))

#x = generate_individual(13)
#print (x)
#print(get_distance(x))
              
def get_fitness(individual, size, goal):
    distance = np.count_nonzero(individual==goal)
    fitness = distance/size
    #print(fitness)
    return fitness

def generate_population(dna,dna_size, pop_size):
    np.random.seed(dna)
    start_population = []
    for i in range(pop_size):
        start_population.append(generate_individual(dna,dna_size))
    return (start_population)

#y= generate_population(10,10)
#print(y)

def current_fitnesses(population, size, goal):
    current_fitnesses = []
    for x in population:
        current_fitnesses.append(get_fitness(x, size, goal))
    return current_fitnesses
#print(current_fitnesses(y))

def select_parent(population, size, goal):
    fitnessArray = current_fitnesses(population, size, goal)
    fitnessSum = sum(map(float,fitnessArray))
    roll = random.uniform(0,1) * fitnessSum
    #print (roll)

    for i in range(len(population)):
        if roll <= fitnessArray[i]:
            #print(population[i])
            return population[i]
        else:
            roll -= fitnessArray[i]
            #roll -= random.uniform(0,1) * fitnessSum
#print(y)
#print(select_parent(y))

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
    #print(offspring)
    return (offspring)
 
def mutate(individual):
    index = len(individual) - 1 
    mutation_position = random.randint(0,index)
    individual[mutation_position] = random.randint(1,(index+1))
    return(individual)
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

def get_best(population, size, goal):
    (m,i) = max((fitness,index) for index,fitness in enumerate(current_fitnesses(population, size, goal)))
    #print(len(population[i]))
    return population[i]

#print(get_best(y))
#print(get_best(new_gen))




from matplotlib.widgets import Slider, Button, RadioButtons

def color_dict(size):
    HSV_tuples = [(x*1/size, 0.35, 0.35) for x in range(size)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)


    ordered_scale = pc.make_colorscale(list(HSV_tuples))
    ordered_tuples = pc.colorscale_to_colors(ordered_scale)

    keys = [i for i in range(0,size)]
    color_dict = {k: v for k, v in zip(keys, ordered_tuples)}
    return color_dict

def get_colors(the_dict, indi):
    colors = [the_dict.get(i) for i in indi]
    return colors

# def plot_a_pie(individual):
#
#     s = len(individual)
#     values = np.empty(s)
#     values.fill(100/s)
#
#     return values
#
# def update_pie_colors(values, colors, ax):
#     ax.clear()
#     ax.axis('equal')
#
#     #colors = get_colors(dictionary, individual)
#     ax.pie(values, colors=colors, wedgeprops=dict(width=0.8), startangle=-20)
def plot_a_pie(individual, dictionary, pM, pC, dna_size, pop_size):
    
    s = len(individual)
    values = np.empty(s)
    values.fill(100/s)

    colors = get_colors(dictionary, individual)
    #fig = plt.figure(figsize=(8, 8))
    plt.pie(values, colors=colors, startangle=-20, wedgeprops={"edgecolor":'w',
       "linewidth":1, "width":0.7})
    plt.pie([1], colors='k', radius=0.295);
    
    txt="Mutation Prop: " + str(pM) + "  Recombination Prop: " + str(pC) + "\nDNA Size: " + str(dna_size) + "  Population Size: " + str(pop_size)
    plt.figtext(0.5, 0.05, txt, wrap=True, horizontalalignment='center', fontsize=10, color="white")
    # show plot
    #plt.show()

    #return plt, 
    return 0

# def update(pM, pC, last_population, pop_size, dna_size, goal, color_dict, ax):
#     ax.cla() #Clear ax
#     ax.grid()
#     #col_dict = color_dict(dna_size)
#     if np.count_nonzero(get_best(last_population, pop_size, goal)==goal) != dna_size:
#         individual = get_best(last_population, pop_size, goal)
#         last_population = next_hopefully_diverser_gen(last_population, pC, pM, dna_size, goal)
#         values, colors = plot_a_pie(individual, color_dict)
#         ax.pie(values, colors=colors, wedgeprops=dict(width=0.8), startangle=-20)
#         return plt, last_population
#     else:   
#         individual = get_best(last_population, pop_size, goal)
#         values, colors = plot_a_pie(individual, color_dict)
#         ax.pie(values, colors=colors, wedgeprops=dict(width=0.8), startangle=-20)
#         exit()


def run_algorithm(pM, pC, dna, pop_size, dna_size, goal):
    
    fig = plt.gcf()
    fig.patch.set_facecolor('#262626')
    #fig.show()
    fig.canvas.draw()
    i = 0
    col_dict = color_dict(dna_size)
    last_population = generate_population(dna, dna_size, pop_size)
    #print(get_best(population_zero))
    #while sum(get_best(population_zero, size, goal)) != goal*size:


    while np.count_nonzero(get_best(last_population, pop_size, goal)==goal) != dna_size:
        #population_zero = next_generation(population_zero, pC, pM)
        last_population = next_hopefully_diverser_gen(last_population, pC, pM, dna_size, goal)
        best = get_best(last_population, pop_size, goal)
        print(best)
        plot_a_pie(best, col_dict, pM, pC, dna_size, pop_size)
        #plt.pause(0.0001)
        
        #fig.canvas.draw()
        f_name = r"/Users/roschkach/Projekte/Python Projekte/geneticAlgorithm/plot_3_1_1/plot_3_1_1_" + str(i) + ".png"
        i+=1
        #pil_img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb()).resize((50, 50),Image.NEAREST)
        #plt.savefig(f_name)
        plt.clf()
    return 0 #last_population

    #anim = FuncAnimation(fig, update(pM, pC, population, pop_size, dna_size, goal, col_dict, ax), repeat= False)
    #anim.save('geneticEyegorithm.gif', writer='imagemagick', fps=30, dpi=40)
    #plt.show()
    #plt.close()


run_algorithm(0.1 ,0.1, 65, 120, 15, 2)