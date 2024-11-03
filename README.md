# CI2024_lab2

## Summary
- Gready approach and explaination
    - Greedy code
- Evolutionary Algorithm
- Conclusions
- Code

## Gready approach and explaination
Instead of using prof. Squillero Gready as a starting point I tried to evoid using the np.inf.   
The main idea is to use the visited vector as a mask for the current city distance array (distance_matrix[city]).  
Let's call a = distance_matrix[city] and b = visited (after marking visited[city] = True). In a first moment I was using as mask m = a AND not(b):  
 
|a|b|not(b)|mask|  
|-|-|-|-| 
|T|T|F|F|
|T|F|T|T|
|F|T|F|F|
|F|F|T|F|

After i realised the entry a = F b = F was not possible, due to the fact that if a = F it means I'm using distance_matrix[city], so visited[city] is True (b = T), I remade the table and the mask simply become not(b) - which is the same of mask:

|a   |b   |not(b)|mask|
|----|----|----|---|
|T|T|F|F|
|T|F|T|T|
|F|T|F|F|

Using a mask i can avoid picking the city itself (argmin[a] = city because it is 0) and also cities already in the trip (if a city is already visited, b = T and mask = not(b) = F).  
The return value of a[mask] is k, and it is the index of the "True" value you have to pick in the vector: we need an iterative function that search for the "k+1" True value inside the mask. Let's use an example to better understand:
- a = np.array([0, 2, 4, 1, 5]), mask = not(b) = [False, True, False,True, True]  
- a[mask] = [2, 1, 5]  
- k = np.argmin(a[mask]) = 1, the index of 1 in the array a[mask]. Since arrays starts from 0, we add 1 to k to find the cardinality of the True value we are looking for (the second in this case)
- finally, with a function find_kth_element, we found the index of the second True value inside the mask: this is the closest city

### Code

Function to find the k-th occurrence:

    def find_kth_occurrence(lst, element, k):
        occurrences = (i for i, x in enumerate(lst) if x == element)
        return next(islice(occurrences, k-1, k), -1)  

Initialization of visited, trip, starting point (it can be random) and cost

    visited = np.full((cities.shape[0]), False)

    city = 0
    visited[city] = True

    trip = []
    trip.append(city)
    cost = 0

Body of the greedy

    while not np.all(visited):

        support = np.logical_not(visited)
        k = np.argmin(distance_matrix[city][support])
        closest_city = find_kth_occurrence(support, 1, k+1)

        print(f"{names[city]} -> {names[closest_city]} {distance_matrix[city][closest_city]:.2f}")

        city = closest_city
        visited[city] = True
        trip.append(city)
        cost += distance_matrix[trip[-2], trip[-1]]

    cost += distance_matrix[trip[-1], trip[0]]
    trip.append(0)


## Evolutionary Algorithm

In the first instance my starting point was the greedy explained above, but starting from a local optima was not good.  
At the end of the lesson on November 31, thanks to the professor, i implemented a new version of greedy using some sort of 'roulette wheel' approch: for every city there is a probability p to choose the best path or discard the best value and repeat the process, again with a new probability p.  
I defined a threshold (strength) and everytime p is less than this value the searching is repeated.  
  
This approach helps in adding some randomicity creating the population for the evolutionary algorithm

### Code

    def find_path(closest_city, list, support, strength = 0.5, p = 1 ):
        while p > strength:
            support[closest_city] = False
            try:
                k = np.argmin(list[support])
            except:
                break
            p = np.random.rand()
            closest_city = find_kth_occurrence(support, 1, k+1)
        return closest_city

### Parent selection, selective pressure and fitness

Parent selection is random, selective pressure is very low and the approach is Steady State. This because i want to keep solution far from local optima, to help exploration.  
Fitness is simply the cost of the path: i want to have the lowest possible

### Genetic operators

I wanted to implement both mutation, crossover and mutation + crossover. I defined a function to randomly select the operator:  

    def genetic_operator(population):
    p = np.random.rand()
    if p < 0.3:
        p1 = parent_selection(population)
        p2 = parent_selection(population)
        return crossover(p1, p2)
    elif p < 0.6:
        p = parent_selection(population)
        return mutation(p)
    else:
        p1 = parent_selection(population)
        p2 = parent_selection(population)
        return crossover(mutation(p1), mutation(p2))

Crossover is Inver-Over crossover, Mutation is Inversion mutation

## Conclusions

The algorithm returns a better value then the greedy for every instance. While I'm writing this report, China is still running improving the result.  
The main idea is to keep the takeover time very high, to explorate the fitness landscape. The main problem is the time required to compute large instances. Last China run was about 2h, and the current one is 5h estimated.  

The evolution of this algorithm was to start from good solution, but not locally optimal, and also to set a long takeover time ( using a large population and offspring, with a "small" number of generation - for China with small i mean 10k )  

### Some results with China

About China, I tried with these parameter values:  

- POPULATION_SIZE = 400, OFFSPRING_SIZE = 250, GENERATIONS = 5_000  
    - cost: 68906.63... (very bad)
- POPULATION_SIZE = 400, OFFSPRING_SIZE = 250, GENERATIONS = 10_000
    - cost: 67645.07...
- POPULATION_SIZE = 500, OFFSPRING_SIZE = 300, GENERATIONS = 30_000
    - cost: 59535.56... (best found)

(using a small population size and a very large number of generation i get a value very close to the greedy one)
- POPULATION_SIZE = 100, OFFSPRING_SIZE = 50, GENERATIONS = 100_000
    - cost: 64975.95... (similar to greedy)

I assume this is a good path to follow to find a good value. Unfortunately, since the last run was 313 min, I can't put more results in the repo at the moment (don't have time). I will upload it after the commit check with some additional info, if I think them are useful.

## Code ( clear version, without comments )

    import numpy as np
    import pandas as pd
    from icecream import ic
    from geopy.distance import geodesic
    import networkx as nx
    from itertools import combinations, islice

    from matplotlib import pyplot as plt
    from tqdm.auto import tqdm
    from dataclasses import dataclass

Read cities

    vanuatu = pd.read_csv("cities\\vanuatu.csv", names = ['city', 'lat', 'lon'])
    china = pd.read_csv("cities\\china.csv", names = ['city', 'lat', 'lon'])
    usa = pd.read_csv("cities\\us.csv", names = ['city', 'lat', 'lon'])
    russia = pd.read_csv("cities\\russia.csv", names = ['city', 'lat', 'lon'])
    italy = pd.read_csv("cities\\italy.csv", names = ['city', 'lat', 'lon'])

    cities = china
    names = cities['city'].values

Distance and trip visualization functions

    def geo_distance(x1, y1, x2, y2):
        return geodesic((x1, y1), (x2, y2)).km

    def trip_visualization(trip):
        if trip[-1] != trip[0]:
            trip = np.concatenate((trip, [trip[0]]))
        latitudes = cities.iloc[trip]['lat']
        longitudes = cities.iloc[trip]['lon']
        
        plt.figure(figsize=(10, 16))
        plt.plot(longitudes, latitudes, marker='o') 
        for idx in range(len(cities)):
            plt.text(cities.iloc[idx]['lon'], cities.iloc[idx]['lat'], cities.iloc[idx]['city'], fontsize=12)
        plt.title("Trip Route")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        return plt.show()

Distance matrix generation

    distance_matrix = np.zeros((cities.shape[0], cities.shape[0]))
    for i in tqdm(range(cities.shape[0])):
        for j in range(cities.shape[0]):
            distance_matrix[i, j] = geo_distance(cities.iloc[i, 1], cities.iloc[i, 2], cities.iloc[j, 1], cities.iloc[j, 2])

Encapsulating Individual

    @dataclass
    class Individual:
        genome: np.ndarray = None
        cost: float = None  

### Greedy generator

    def find_kth_occurrence(lst, element, k):
        occurrences = (i for i, x in enumerate(lst) if x == element)

        return next(islice(occurrences, k-1, k), -1)

    def find_path(closest_city, list, support, strength = 0.5, p = 1 ):
        while p > strength:
            support[closest_city] = False
            try:
                k = np.argmin(list[support])
            except:
                break
            p = np.random.rand()
            closest_city = find_kth_occurrence(support, 1, k+1)

        return closest_city

    def individual_generator(cities, distance_matrix, strength = 0.5):

        visited = np.full((cities.shape[0]), False)
        city = np.random.randint(0, cities.shape[0])
        visited[city] = True

        trip = []
        trip.append(city)
        distance = 0

        while not np.all(visited):
            support = np.logical_not(visited)
            closest_city = find_path(city, distance_matrix[city], support, strength)
            city = closest_city
            visited[city] = True
            trip.append(city)
            distance += distance_matrix[trip[-2], trip[-1]]
        distance+= distance_matrix[trip[-1], trip[0]]

        return Individual(trip, distance)

### Utility functions

Parent selection  

    def parent_selection(population):
        parents = sorted(np.random.choice(population, 2), key=lambda x: x.cost, reverse=True)

        return parents[0]

    def distance_computation(trip):
        distance = 0
        for i in range(len(trip) - 1):
            distance += distance_matrix[trip[i], trip[i + 1]]
        distance += distance_matrix[trip[-1], trip[0]]
        
        return distance

Crossover ( Inver-Over)  

    def rearrange(arr, start):
        new_arr = np.zeros_like(arr)
        len = arr.shape[0]

        for i in range(len):
            new_arr[(start + i) % len] = arr[i]

        return Individual(new_arr, distance_computation(new_arr))

    def crossover(p1: Individual, p2: Individual):
        gene1 = np.concatenate((p1.genome, p1.genome))
        gene2 = p2.genome

        i = np.random.randint(0, len(p1.genome))
        j = np.where(gene2 == gene1[i])[0][0]

        gene1 = gene1[i:i + len(gene2)]
        index = (j + 1) % len(gene2)
        z = np.where(gene1 == gene2[index])[0][0]
        slice_ = gene1[1:z][::-1]
        new_gene = np.concatenate(([gene1[0], gene1[z]], slice_, gene1[z + 1:]))

        return rearrange(new_gene, i)

Mutation ( inversion )  

    def mutation(p: Individual):
        gene = p.genome.copy()
        i, j = np.random.choice(range(len(gene)), 2, replace=False)
        gene[i], gene[j] = gene[j], gene[i]
        return Individual(gene, distance_computation(gene))

Genetic operator  

    def genetic_operator(population):
        p = np.random.rand()
        if p < 0.3:
            p1 = parent_selection(population)
            p2 = parent_selection(population)
            return crossover(p1, p2)
        elif p < 0.6:
            p = parent_selection(population)
            return mutation(p)
        else:
            p1 = parent_selection(population)
            p2 = parent_selection(population)
            return crossover(mutation(p1), mutation(p2))

Fitness  

    def fitness(solution):
        return (solution.cost)

Population generator  

    def population_generator(cities, distance_matrix, population_size, strength = 0.5):
        population = []
        for _ in range(population_size):
            population.append(individual_generator(cities, distance_matrix, strength))
        return population

### Body of the program

Values are from the last computation    

Population generator

    POPULATION_SIZE = 500
    STRENGTH = 0.5
    
    population = population_generator(cities, distance_matrix, POPULATION_SIZE, STRENGTH)

Evolutionary body

    OFFSPRING_SIZE = 300
    GENERATIONS = 30_000

    for _ in tqdm(range(GENERATIONS)):
        offspring = []
        for _ in range(OFFSPRING_SIZE):
            offspring.append(genetic_operator(population))
        # steady state
        population.extend(offspring)
        # simple survival selection
        population = sorted(population, key=lambda x: x.cost)
        population = population[:POPULATION_SIZE]

    ic(population[0].cost)