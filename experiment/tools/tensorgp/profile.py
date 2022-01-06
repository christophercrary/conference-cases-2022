import math
import os
import pickle
import random
import sys
import timeit

import numpy as np
from scipy.stats import iqr
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join('./', 'tensorgp')))
from tensorgp.engine import *

# Useful path directory.
root_dir = (f'{os.path.dirname(os.path.abspath(__file__))}/../../results/'
            f'programs')

# Random seed, for reproducibility.
seed = 37
random.seed(seed)

########################################################################
# Some helper function(s).
########################################################################

def get_max_size(m, d):
    """Return the maximum possible size for a program.
    
    The program is considered to be `m`-ary, of depth `d`.
    """
    if (m==1):
        return d+1
    else:
        return int((1-m**(d+1))/(1-m))

######################################################################## Profiling of program evaluation mechanism given by TensorGP.
########################################################################

def initialize_terminals(dimensions, n):
    # Read fitness case data from file. The data is stored as 
    # a tuple of tuples, effectively a two-dimensional array,
    # with the columns specifying variables, and the rows 
    # specifying particular fitness cases.
    with open(f'{root_dir}/fitness_cases.pkl', 'rb') as f:
        fitness_cases = pickle.load(f)

    #print('n:', n)
    
    # Infer two-dimensional `NumPy` array from tuple of tuples.
    fitness_cases = np.array(fitness_cases)

    #print('Fitness cases:', fitness_cases)

    # Extract relevant variable data from the overall
    # array of fitness cases, casting this data to `float32`.
    res = tf.cast(fitness_cases[:dimensions[0], n], tf.float32)
    #print('Res:', res)

    return res

def r2(**kwargs):
    """R-squared fitness function."""
    population = kwargs.get('population')
    tensors = kwargs.get('tensors')
    target = kwargs.get('target')

    # print('Population size:', len(population))
    # print('Shape of tensors:', tf.shape(tensors))
    # print('Shape of target:', tf.shape(target))

    fitness = []
    best_ind = 0

    max_fit = float('0')

    for i in range(len(tensors)):

        # print(f'Tensors[{i}]:', tensors[i])
        # print(f'Target:', target)

        fit = tf_r2(target, tensors[i]).numpy()
        # print('Fitness:', fit)

        if fit > max_fit:
            max_fit = fit
            best_ind = i

        fitness.append(fit)
        population[i]['fitness'] = fit

        # print(f'Tensor[{i}]: {tensors[i]}')
        # print(f'Target: {target}')

    # print(f'`Length of `fitness: {len(fitness)}')

    return population, population[best_ind]


# Parameter for debug logging within TensorGP.
debug = 0

# Computing devices to utilize.
devices = ('/cpu:0', '/gpu:0')

# Overall set of functions.
functions = {'add', 'aq', 'exp', 'log', 'mul', 
             'sin', 'sqrt', 'sub', 'tanh'}

# Dictionary for particular function set criteria.
function_sets = {
    'nicolau_a': (4, 2, 7, 2),
    'nicolau_b': (6, 2, 5, 1),
    'nicolau_c': (9, 2, 5, 1)
}

# Number of programs per size bin.
num_programs_per_size_bin = 1

# Numbers of fitness cases.
num_fitness_cases = (10, 100, 1000, 10000)

# Overall target.
with open(f'{root_dir}/target.pkl', 'rb') as f:
    target_ = pickle.load(f)

# Value for the `repeat` argument of the `timeit.repeat` method.
repeat = 3

# Value for the `number` argument of the `timeit.repeat` method.
number = 2

# Number of times in which the `timeit.repeat` function is
# called, in order to generate a list of minimum average
# runtimes.
num_epochs = 3

# Median average runtimes for programs within each size bin,
# for each number of fitness cases, for each function set.
med_avg_runtimes = []

# # Average of *minimum average runtimes* for each size bin,
# # for each number of fitness cases, for each function set.
# avg_min_avg_runtimes = []

# # Median of *minimum average runtimes* for each size bin,
# # for each number of fitness cases, for each function set.
# med_min_avg_runtimes = []

# # Minimum of *minimum average runtimes* for each size bin,
# # for each number of fitness cases, for each function set.
# min_min_avg_runtimes = []

# # Maximum of *minimum average runtimes* for each size bin,
# # for each number of fitness cases, for each function set.
# max_min_avg_runtimes = []

# # Standard deviation of *minimum average runtimes* for each size 
# # bin, for each number of fitness cases, for each function set.
# std_dev_min_avg_runtimes = []

# # Interquartile range of *minimum average runtimes* for each size 
# # bin, for each number of fitness cases, for each function set.
# iqr_min_avg_runtimes = []

# # Median node evaluations per second (NEPS) for each size bin,
# # for each number of fitness cases, for each function set.
# med_neps = []

for device in devices:
    # For each device...

    # Prepare for statistics relevant to function set.
    med_avg_runtimes.append([])

    for name, (num_functions, max_arity, 
        max_depth, bin_size) in function_sets.items():
        # For each function set...
        print(f'Function set `{name}`:')

        # Number of variables for given function set.
        num_variables = num_functions-1

        # Maximum program size for function set.
        max_possible_size = get_max_size(max_arity, max_depth)

        # Number of size bins.
        num_size_bins = int(math.ceil(max_possible_size/bin_size))

        # Prepare for statistics relevant to function set.
        # sizes.append([])
        med_avg_runtimes[-1].append([])
        # avg_min_avg_runtimes.append([])
        # med_min_avg_runtimes.append([])
        # min_min_avg_runtimes.append([])
        # max_min_avg_runtimes.append([])
        # std_dev_min_avg_runtimes.append([])
        # iqr_min_avg_runtimes.append([])
        # med_neps.append([])

        # Read in the programs relevant to the function set from file.
        # This file contains `population_size * num_size_bins` programs,
        # representing the `population_size` programs for each of the
        # `num_size_bins` size bins.
        with open(
            f'{root_dir}/{name}/programs_tensorgp.txt', 'r') as f:
            programs = f.readlines()

        for nfc in num_fitness_cases:
            # For each number of fitness cases...
            print(f'Number of fitness cases: `{nfc}`')

            # Tensor dimensions for this test case.
            #
            # There are `num_variables` dimensions, each
            # consisting of `nfc` fitness cases.
            # target_dims = (nfc, nfc)
            target_dims = (nfc,)
            #target_dims = tuple(nfc for _ in range(num_variables))

            # Target for given number of fitness cases.
            target = tf.cast(tf.convert_to_tensor(target_[:nfc]), tf.float32)

            #print(f'Shape of `target`: {tf.shape(target)}')

            # Prepare for statistics relevant to the 
            # numbers of fitness cases and size bins.
            med_avg_runtimes[-1][-1].append([[] for _ in range(num_size_bins)])

            # Create an appropriate GP engine.
            engine = Engine(debug=debug,
                            seed=seed,
                            device=device,
                            operators=functions,
                            target_dims=target_dims,
                            target=target,
                            fitness_func=r2,
                            population_size=num_programs_per_size_bin,
                            var_func=initialize_terminals,
                            num_variables = num_variables)

            for i in range(num_size_bins):
                # For each size bin, calculate the relevant statistics.

                # Population relevant to the current size bin.
                population, *_ = engine.generate_pop_from_expr(
                    programs[(i) * num_programs_per_size_bin:
                            (i+1) * num_programs_per_size_bin])

                for _ in range(num_epochs):
                    # For each epoch...

                    # Raw runtimes after running the `fitness_func_wrap` 
                    # function a total of `repeat * number` times. 
                    # The resulting object is a list of `repeat` values,
                    # where each represents a raw runtime after running
                    # the relevant code `number` times.
                    runtimes = timeit.Timer(
                        'engine.fitness_func_wrap(population=population)',
                        globals=globals()).repeat(repeat=repeat, number=number)

                    # Average runtimes, taking into account `number`.
                    avg_runtimes = [runtime/number for runtime in runtimes]

                    # Calculate and append median average runtime.
                    med_avg_runtimes[-1][-1][-1][i].append(
                        np.median(avg_runtimes))

# Preserve results.
with open(f'{root_dir}/../results_tensorgp.pkl', 'wb') as f:
    pickle.dump(med_avg_runtimes, f)

        # # Average of *minimum average runtimes* for each size bin.
        # avg_min_avg_runtimes[-1].append(
        #     [np.mean(min_avg_runtimes[-1][-1][i]) 
        #     for i in range(num_size_bins)])
        # print('Averages of minimum average runtimes:', 
        #         avg_min_avg_runtimes[-1][-1])
        # print('\n')

        # # Median of *minimum average runtimes* for each size bin.
        # med_min_avg_runtimes[-1].append(
        #     [np.median(min_avg_runtimes[-1][-1][i]) 
        #     for i in range(num_size_bins)])
        # print('Medians of minimum average runtimes:', 
        #         med_min_avg_runtimes[-1][-1])
        # print('\n')

        # # Minimum of *minimum average runtimes* for each size bin.
        # min_min_avg_runtimes[-1].append(
        #     [min(min_avg_runtimes[-1][-1][i]) 
        #     for i in range(num_size_bins)])
        # print('Minimums of minimum average runtimes:', 
        #         min_min_avg_runtimes[-1][-1])
        # print('\n')

        # # Maximum of *minimum average runtimes* for each size bin.
        # max_min_avg_runtimes[-1].append(
        #     [max(min_avg_runtimes[-1][-1][i]) 
        #     for i in range(num_size_bins)])
        # print('Maximums of minimum average runtimes:', 
        #         max_min_avg_runtimes[-1][-1])
        # print('\n')

        # # Standard deviation of minimum average runtimes for each size bin.
        # std_dev_min_avg_runtimes[-1].append(
        #     [np.std(min_avg_runtimes[-1][-1][i]) 
        #     for i in range(num_size_bins)])
        # print('Standard deviations of minimum average runtimes:', 
        #         std_dev_min_avg_runtimes[-1][-1])
        # print('\n')

        # # Interquartile range of minimum average runtimes for each size bin.
        # iqr_min_avg_runtimes[-1].append(
        #     [iqr(min_avg_runtimes[-1][-1][i]) 
        #     for i in range(num_size_bins)])
        # print('Interquartile range of minimum average runtimes:', 
        #         iqr_min_avg_runtimes[-1][-1])
        # print('\n\n')

        #print('Sizes:', sizes[-1])

        # Median node evaluations per second relevant to function set.
        # med_neps.append([(size*nfc)/med 
        #     for size, med in zip(sizes[-1], med_min_avg_runtimes[-1])])
        # print('Median node evaluations per second:', med_neps[-1])
        # print('\n')


        # Plot graph of median node evaluations per second, 
        # for each function set.
        # for i, (name, 
        #     (num_functions, max_arity, max_depth, bin_size)) in enumerate(
        #         function_sets.items()):

        #     # Maximum program size for function set.
        #     max_possible_size = get_max_size(max_arity, max_depth)

        #     # Number of size bins.
        #     num_size_bins = int(math.ceil(max_possible_size/bin_size))

        #     # Index range for plot.
        #     index = range(1, num_size_bins+1)

            # Plot for function set.
            # plt.plot(index, [size*nfc for size in sizes[i]])
            # plt.plot(index, med_neps[i], label=f'Function set {name}')

        # plt.xlabel('Size bin number')
        # plt.ylabel('Median of node evaluations per second')
        # plt.title('Median of node evaluations per second vs. size bin number')
        # plt.legend(loc='upper left')

        # plt.show()
    