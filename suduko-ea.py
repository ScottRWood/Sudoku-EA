from matplotlib import pyplot as plt
from random import shuffle, choice, random
from time import time


def analysis():
    """
    Runs the analysis of all experiments on all grids
    """
    generations = []
    fits = []
    times = []

    for i in range(1, 4):
        grid = read_grid_file("Grid" + str(i) + ".ss")
        sizes = [10, 100, 1000, 10000]

        avg_generation = []
        avg_fit = []
        avg_time = []

        for population_size in sizes:
            generation_found = []
            fit_found = []
            time_found = []

            for j in range(5):
                fail_count = 0
                start = time()
                while True:
                    generation, fit = evolve(grid, population_size)

                    if fit == 0:
                        break
                    if fail_count < 3:
                        fail_count += 1
                        print("Local minimum reached - recreating seed. (Grid: ", i, " Population: ", population_size,
                              " Test: ", j, ")")
                    else:
                        print("Local minimum reached - population size ineffective.")
                        break
                        
                time_elapsed = time() - start

                generation_found.append(generation)
                fit_found.append(fit)
                time_found.append(time_elapsed)

                print("Grid: ", i, " Population: ", population_size, " Test ", j, " Time: ", time_elapsed, " Fitness: ",
                      fit, " generation: ", generation)

            avg_generation.append(sum(generation_found) / len(generation_found))
            avg_fit.append(sum(fit_found) / len(fit_found))
            avg_time.append(sum(time_found) / len(time_found))

        generations.append(avg_generation)
        fits.append(avg_fit)
        times.append(avg_time)

    # Plot the data.
    xs = ['10', '100', '1000', '10000']

    plt.figure(num=None, figsize=(8, 15))
    plt.subplot(3, 1, 1)
    plt.title("Experiments")
    for g in generations:
        plt.plot(xs, g)
    plt.legend(['Grid1.ss', 'Grid2.ss', 'Grid3.ss'], loc='upper right')
    plt.ylabel('Generation')

    plt.subplot(3, 1, 2)
    for f in fits:
        plt.plot(xs, f)
    plt.ylabel('Fitness')

    plt.subplot(3, 1, 3)
    for t in times:
        plt.plot(xs, t)
    plt.ylabel('Time')

    plt.xlabel("Population Size")
    plt.show()


def read_grid_file(file_name):
    """
    Reads a grid file
    :param file_name: Path of file to be opened
    :type file_name: str
    """
    with open(file_name) as f:
        grid = []
        for line in f.readlines():
            if line[0] == '-':
                continue
            row = []
            for c in line:
                if c == '.':
                    row.append(0)
                elif c in [str(i) for i in range(1, 10)]:
                    row.append(int(c))
            grid.append(row)
    f.close()
    return grid


def evolve(goal, population_size, mutation_rate=0.075, truncation_rate=0.5):
    """
    Runs the EA on given parameters
    :param goal: The grid for the EA to be run on
    :type goal: list(list(int))
    :param population_size: The size of the population
    :type population_size: int
    :param mutation_rate: Likelihood of mutation occurring
    :type mutation_rate: float
    :param truncation_rate: The proportion of candidates that are selected
    :type truncation_rate: float
    :return: the generation number and its fitness value
    """
    population = create_seed(goal, population_size)
    fit = check_fitness(population)

    overall_best_fit = 145
    generation = 0
    fail_count = 0

    if population_size == 10:
        fail_limit = 10000
    elif population_size == 100:
        fail_limit = 7500
    elif population_size == 1000:
        fail_limit = 500
    elif population_size == 10000:
        fail_limit = 250
    else:
        fail_limit = 1000

    while overall_best_fit > 0 and fail_count < fail_limit:
        generation += 1
        population = select_population(population, fit, population_size, truncation_rate)
        population = crossover_population(population, population_size)
        population = mutate_population(goal, population, mutation_rate)
        fit = check_fitness(population)
        best_child, best_fit = best_population(population, fit)
        if best_fit < overall_best_fit:
            overall_best_fit = best_fit
            fail_count = 0
        else:
            fail_count += 1

    del population, fit
    return generation, overall_best_fit


def create_seed(host, population_size):
    """
    Creates initial population from problem grid an population size
    :param host: The problem grid
    :type host: list(list(int))
    :param population_size: The size of the population
    :type population_size: int
    :return: A list of possible solutions
    """
    population = []
    for i in range(population_size):
        solution = clone(host)
        for line in solution:
            missing = list({1, 2, 3, 4, 5, 6, 7, 8, 9} - set(line))
            shuffle(missing)
            while missing:
                for j in range(len(line)):
                    if line[j] == 0:
                        line[j] = missing.pop()
                        break
        population.append(solution)
    return population


def best_population(population, fitness_population):
    """
    Finds the best member of the population
    :param population: The population to be selected from
    :type population: list(list(list(int)))
    :param fitness_population: The fitness score of each member
    :type fitness_population: list(int)
    :return: The best member of the population and its fitness
    """
    return sorted(zip(population, fitness_population), key=lambda ind_fit: ind_fit[1])[0]


def select_population(population, fitness_population, population_size, truncation_rate):
    """
    Select the best n individuals based on population size and truncation rate
    :param population: The population to be selected from
    :type population: list(list(list(int)))
    :param fitness_population: The fitness score of each member
    :type fitness_population: list(int)
    :param population_size: The size of the population
    :type population_size: int
    :param truncation_rate: The proportion that is to be selected
    :type truncation_rate: float
    :return: a list of selected members
    """
    sorted_population = sorted(zip(population, fitness_population), key=lambda ind_fit: ind_fit[1])
    return [individual for individual, fitness in sorted_population[:int(population_size * truncation_rate)]]


def crossover_population(population, population_size):
    """
    Perform the crossover operator on the population
    :param population: Population to be used
    :type population: list(list(list(int)))
    :param population_size: Size of the population
    :type population_size: int
    :return: The new members generated by the crossover operator
    """
    cross = [crossover_individual(choice(population), choice(population)) for _ in range(population_size - 1)]
    cross.append(population[0])
    del population
    return cross


def crossover_individual(indiviudal_1, indiviudal_2):
    """
    Perform the crossover operator using 2 given parents
    :param indiviudal_1: The first parent
    :type indiviudal_1: list(list(int))
    :param indiviudal_2: The second parent
    :type indiviudal_2: list(list(int))
    :return: The child created via crossover
    """
    child = []
    for i in range(0, 9):
        if random() > 0.5:
            child.append(indiviudal_1[i][:])
        else:
            child.append(indiviudal_2[i][:])
    return child


def check_changeable(grid, x, y):
    """
    Check if a cell can be changed
    :param grid: The problem grid
    :type grid: list(list(int))
    :param x: The x-coordinate of the cell
    :type x: int
    :param y: The y-coordinate of the cell
    :type y: int
    :return: Boolean specifying whether the cell can be changed
    """
    return grid[x][y] == 0


def mutate_population(grid, population, mutation_rate):
    """
    Perform the mutation operator on the population
    :param grid: The problem grid
    :type grid: list(list(int))
    :param population: The population to be operated on
    :type population: list(list(list(int)))
    :param mutation_rate: The proportion to be mutated
    :type mutation_rate: float
    :return: The mutated population
    """
    return [mutate_individual(grid, clone(solution), mutation_rate) for solution in population]


def mutate_individual(grid, solution, mutation_rate):
    """
    Mutate a candidate
    :param grid: The problem grid
    :type grid: list(list(int))
    :param solution: The current solution
    :type solution: list(list(int))
    :param mutation_rate: The proportion to be mutated
    :type mutation_rate: float
    :return: The mutated candidate (if mutated)
    """
    for x in range(9):
        for y in range(9):
            if check_changeable(grid, x, y):
                if random() < mutation_rate:
                    switch = get_pivot(grid, solution, x)
                    if switch != -1:
                        solution[x][y], solution[x][switch] = solution[x][switch], solution[x][y]
                    continue
    return solution


def get_pivot(grid, solution, x, check=1):
    """
    Generate a pivot for the mutation
    :param grid: The problem grid
    :type grid: list(list(int))
    :param solution: The current solution
    :type solution: list(list(int))
    :param x: The row being checked
    :type x: int
    :param check: A number to limit the number of pivot checks that can be made
    :type check: int
    :return: The column index to be swapped with or -1 if no pivot found
    """
    if check < 8:
        y = choice(range(9))
        return y if check_changeable(grid, x, y) else get_pivot(grid, solution, x, check+1)
    else:
        return -1


def clone(grid):
    """
    An auxillary method to clone a grid
    :param grid: The grid to be cloned
    :type grid: list(list(int))
    :return: A clone of the grid
    """
    return [x[:] for x in grid]


def check_fitness(population):
    """
    Check the fitness of the population
    :param population: The population to be checked
    :type population: list(list(list(int)))
    :return: A list of fitness values
    """
    return [fitness(solution) for solution in population]


def fitness(solution):
    """
    Calculate the fitness of a solution
    :param solution: Solution to be checked
    :type solution: list(list(int))
    :return: fitness score
    """
    return check_vertical(solution) + check_squares(solution)


def check_vertical(solution):
    """
    Check for duplicates in columns
    :param solution: Solution to be checked
    :type solution: list(list(int))
    :return: number of errors
    """
    errors = 0
    for line in [list(i) for i in zip(*solution)]:
        errors += check_line(line)
    return errors


def check_squares(solution):
    """
    Check for duplicates in 3x3 squares
    :param solution: Solution to be checked
    :type solution: list(list(int))
    :return: number of errors
    """
    errors = 0
    for x in range(0, 9, 3):
        for y in range(0, 9, 3):
            errors += check_line(sum([row[y:y+3] for row in solution[x:x+3]], []))
    return errors


def check_line(line):
    """
    Checks duplicates in a line
    :param line: The line to be checked
    :type line: list(int)
    :return: number of duplicates
    """
    return 9 - len(set(line))


if __name__ == "__main__":
    analysis()
