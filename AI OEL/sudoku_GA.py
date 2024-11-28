import time
import random
from collections import Counter


def read_puzzle(address):
    puzzle = []
    f = open(address, 'r')
    for row in f:
        temp = row.split()   #['2','4','8']
        puzzle.append([int(c) for c in temp])   #[2,4,8] converting it into integer
    print(f"Input Problem Statement{puzzle}")
    return puzzle


def make_population(count, initial=None):  #count=1000
    if initial is None:   #create a 9x9 grid of values 0
        initial = [[0] * 9 for r in range(9)]
    population = []
    for r in range(count):  #we will call chromosomes to solve our 9x9 grid 1000 times
        population.append(make_chromosome(initial))  #we will append 9x9 grids in population
    return population


def make_chromosome(initial=None):   #filled input sudoku with random values, for empty spaces
    if initial is None:
        initial = [[0] * 9 for _ in range(9)]
    chromosome = []
    for i in range(9):
        chromosome.append(make_gene(initial[i]))  #after gene it will get 9 rows of distinct values with respect to our initial
    return chromosome


def make_gene(initial=None):
    if initial is None:
        initial = [0] * 9
    mapp = {}  #dictionary
    gene = list(range(1, 10))
    random.shuffle(gene)
    
    #initial = [5, 0, 0, 3, 0, 8, 0, 0, 7]
    #gene = [2, 9, 7, 1, 4, 3, 8, 5, 6]
    
    for i in range(9):
        mapp[gene[i]] = i    #map= {2: 0, 9: 1, 7: 2, 1: 3, 4: 4, 3: 5, 8: 6, 5: 7, 6: 8}
        
    #i=0: initial[0] is 5, which is at gene[7]. Swap gene[0] and gene[7].
    #i=3: initial[3] is 3, which is at gene[5]. Swap gene[3] and gene[5].
    
    for i in range(9):
        if initial[i] != 0 and gene[i] != initial[i]:
            #temp = tuple of two values
            temp = gene[i], gene[mapp[initial[i]]]  #gene[7]    #(1,2)
            gene[mapp[initial[i]]], gene[i] = temp    #(2,1)
            mapp[initial[i]], mapp[temp[0]] = i, mapp[initial[i]]   
    return gene


def get_fitness(chromosome):
    """Calculate the fitness of a chromosome (puzzle)."""
    fitness = 0
    # Check columns and rows
    for i in range(9):
        #the counter in the built in python func which counts the occurances of numbers instead of zero
        column_counts = Counter(chromosome[j][i] for j in range(9) if chromosome[j][i] != 0)
        row_counts = Counter(chromosome[i])
        fitness += sum(count - 1 for count in column_counts.values())  # Penalize duplicates in columns
        fitness += sum(count - 1 for count in row_counts.values())  # Penalize duplicates in rows

    # Check 3x3 grids
    for row in range(3):
        for col in range(3):
            square_counts = Counter(
                chromosome[i][j]
                for i in range(row * 3, (row + 1) * 3)
                for j in range(col * 3, (col + 1) * 3)
                if chromosome[i][j] != 0
            )
            fitness += sum(count - 1 for count in square_counts.values())  # Penalize duplicates in squares
    
    # Return negative since higher duplicates mean lower fitness like -30, if the fitness is in -1 to -10 then our algorithm is very close to the solution        
    return -fitness  


def crossover(ch1, ch2):
    new_child_1 = []
    new_child_2 = []
    for i in range(9):
        x = random.randint(0, 1)
        #on every iteration ek ek kar keh cross karenge overall nahi hoga half half
        if x == 1:
            #simply placing the values
            new_child_1.append(ch1[i]) 
            new_child_2.append(ch2[i])
        elif x == 0:
            #crossing the values
            new_child_2.append(ch1[i])
            new_child_1.append(ch2[i])
    return new_child_1, new_child_2

def mutation(ch, pm, initial):
    for i in range(9):
        x = random.randint(0, 100)
        if x < pm * 100:
            ch[i] = make_gene(initial[i])  # this is not done on every row, only for which the prob is greater than the random value
    return ch


def elitism_selection(population, elite_size=10):
    #here we will find the fitness value of our each chromosome that we have geenrated randomly
    fitness_list = [(get_fitness(chromosome), chromosome) for chromosome in population]  
    # Sort based on fitness, best first
    fitness_list.sort(reverse=True, key=lambda x: x[0])
    # Select the top 10 chromosomes
    return [chromosome for _, chromosome in fitness_list[:elite_size]]


def get_offsprings(population, initial, pm, pc):  #pm = 0.1, pc= 0.98
    new_pool = []
    i = 0
    while i < len(population):
        ch1 = population[i]
        ch2 = population[(i + 1) % len(population)]
        x = random.randint(0, 100)
        if x < pc * 100: 
            ch1, ch2 = crossover(ch1, ch2)   #create the childs and those will be passed on for mutation
        new_pool.append(mutation(ch1, pm, initial))  
        new_pool.append(mutation(ch2, pm, initial))
        i += 2
    return new_pool

def w_get_mating_pool(population):
    fitness_list = [] #contains tuple(fitness, chromosome)
    pool = []
    
    # Calculate fitness for each chromosome
    for chromosome in population:
        fitness = get_fitness(chromosome)
        fitness_list.append((fitness, chromosome))
    
    # Sort fitness list to get the minimum fitness
    fitness_list.sort()
    min_fitness = fitness_list[0][0]   #minimum fitness at index 0
    #fitness = [-30, -20, -10] , min_fitness = -30
    
    # Calculate weights by normalizing against the minimum fitness
    weight = [fit[0] - min_fitness + 1 for fit in fitness_list]  # Add 1 to ensure positive weights
    #weight = [-30 - (-30) + 1, -20 - (-30) + 1, -10 - (-30) + 1]
       #    = [1, 11, 21]
    #weights refers to probablity to be selected randomly

    # Check if all weights are zero
    if sum(weight) == 0:
        # Assign equal weight if all values are zero
        weight = [1] * len(fitness_list)
    
    # Generate mating pool using weights
    for _ in range(len(population)):
        ch = random.choices(fitness_list, weights=weight)[0] #we will select the chromosomes based on the weights probability
        pool.append(ch[1])  #ch[1] refers to chromosome
    
    #the core difference between elitism_selection and this is that in elitism we select the bestest fitness chromosome but here we will
    #going to select lowest fitness value, as we are selecting randomly and also for diversity
    return pool




# Population size
POPULATION = 500

# Number of generations
REPETITION = 500

# Probability of mutation
PM = 0.1

# Probability of crossover
PC = 0.95

# Main genetic algorithm function
def genetic_algorithm(initial_file):
    initial = read_puzzle(initial_file)   #input puzzle read
    population = make_population(POPULATION, initial)
    
    print("The calculation is in the progesss......")
    
    for i in range(REPETITION):
        # Step 1: Select elite chromosomes
        elite_chromosomes = elitism_selection(population, elite_size=10) 

        # Step 2: Generate the rest of the population through the genetic algorithm
        mating_pool = w_get_mating_pool(population)
        random.shuffle(mating_pool)
        offspring_population = get_offsprings(mating_pool, initial, PM, PC)

        # Step 3: Combine elite chromosomes and offspring
        population = elite_chromosomes + offspring_population[:POPULATION - len(elite_chromosomes)]
        #my ec are 10, and offspring population woulf be 1000 , so we will choose 990 , so that 10+990=1000 and donot exceed the actual population

        # Check for solution (fitness = 0)
        fit = [get_fitness(c) for c in population]
        m = max(fit)
        if m == 0:   #the perfect suduko solution population will be passed if m==0, else added population will be passed
            return population 
    
    return population

def pch(ch):
    for i in range(9):
        for j in range(9):
            print(ch[i][j], end=" ")
        print("")
        
        
tic = time.time()
r = genetic_algorithm("test3.txt")
toc = time.time()
print("time_taken: ", toc - tic)

fit = [get_fitness(c) for c in r]
m = max(fit)
print("Max Fitness:", m)

# Print the chromosome with the highest fitness
for c in r:
    if get_fitness(c) == m:
        pch(c)
        break
    
    
print("End!!!")