from os import getcwd
import numpy as np
from tudatpy.util import result2array
from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt

###################################################################

design_space_method = 'factorial_design'
run_to_show = 1

###################################################################

def str2vec(string: str, separator: str) -> np.ndarray:
    return np.array([float(element) for element in string.split(separator)])

def read_vector_history_from_file(file_name: str) -> dict:

    with open(file_name, 'r') as file: lines = file.readlines()
    keys = [float(line.split('\t')[0]) for line in lines]
    solution = dict.fromkeys(keys)
    for idx in range(len(keys)): solution[keys[idx]] = str2vec(lines[idx], '\t')[1:]

    return solution

def extract_elements_from_history(history: dict, index) -> dict:

    if type(index) is int: index = [index]
    elif type(index) is list: pass
    else: raise TypeError('(extract_element_from_history): Illegal index type.')


    n = len(index)
    new_history = dict.fromkeys(list(history.keys()))
    for key in list(new_history.keys()):
        new_history[key] = np.zeros(n)
        k = 0
        for current_index in index:
            new_history[key][k] = history[key][current_index]
            k = k + 1

    return new_history

filename = getcwd() + '/DesignSpace_' + design_space_method + '/Run_' + str(run_to_show) + '/dependent_variable_history.dat'
force_coefficients = result2array(extract_elements_from_history(read_vector_history_from_file(filename), [-3, -2, -1]))

plt.figure()
plt.plot(force_coefficients[:,0], force_coefficients[:,1], label = r'$C_x$')
plt.plot(force_coefficients[:,0], force_coefficients[:,2], label = r'$C_y$')
plt.plot(force_coefficients[:,0], force_coefficients[:,3], label = r'$C_z$')
plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('C [-]')
plt.title('Aerodynamic force coefficients')