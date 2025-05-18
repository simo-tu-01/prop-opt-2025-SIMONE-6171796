"""
Copyright (c) 2010-2021, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Shape Optimization
First name: Simone
Last name: Guccione
Student number: 6171796

This module computes the dynamics of a capsule re-entering the atmosphere of the Earth, using a variety of integrator
and propagator settings.  For each run, the differences w.r.t. a benchmark propagation are computed, providing a proxy
for setting quality. The benchmark settings are currently defined semi-randomly, and are to be analyzed/modified.

The trajectory of the capsule is heavily dependent on the shape and orientation of the vehicle. Here, the shape is
determined here by the five parameters, which are used to compute the aerodynamic accelerations on the vehicle using a
modified Newtonian flow (see Dirkx and Mooij, "Conceptual Shape Optimization of Entry Vehicles" 2018). The bank angle
and sideslip angles are set to zero. The vehicle shape and angle of attack are defined by values in the vector shape_parameters.

The vehicle starts 120 km above the surface of the planet, with a speed of 7.83 km/s in an Earth-fixed frame (see
getInitialState function).

The propagation is terminated as soon as one of the following conditions is met (see 
get_propagation_termination_settings() function):
- Altitude < 25 km
- Propagation time > 24 hr

This propagation assumes only point mass gravity by the Earth and aerodynamic accelerations.

The entries of the vector 'shape_parameters' contains the following:
- Entry 0:  Nose radius
- Entry 1:  Middle radius
- Entry 2:  Rear length
- Entry 3:  Rear angle
- Entry 4:  Side radius
- Entry 5:  Constant Angle of Attack

Details on the outputs written by this file can be found:
- benchmark data: comments for 'generateBenchmarks' function
- results for integrator/propagator variations: comments under "RUN SIMULATION FOR VARIOUS SETTINGS"
- files defining the points and surface normals of the mesh used for the aerodynamic analysis (save_vehicle_mesh_to_file)

Frequent warnings and/or errors that might pop up:
* One frequent warning could be the following (mock values):
    "Warning in interpolator, requesting data point outside of boundaries, requested data at 7008 but limit values are
    0 and 7002, applying extrapolation instead."
It can happen that the benchmark ends earlier than the regular simulation, due to the smaller step size. Therefore,
the code will be forced to extrapolate the benchmark states (or dependent variables) to compare them to the
simulation output, producing a warning. This warning can be deactivated by forcing the interpolator to use the boundary
value instead of extrapolating (extrapolation is the default behavior). This can be done by setting:

    interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation = interpolators.extrapolate_at_boundary)

* One frequent error could be the following:
    "Error, propagation terminated at t=4454.723896, returning propagation data up to current time."
    This means that an error occurred with the given settings. Typically, this implies that the integrator/propagator
    combination is not feasible. It is part of the assignment to figure out why this happens.

* One frequent error can be one of:
    "Error in RKF integrator, step size is NaN"
    "Error in ABM integrator, step size is NaN"
    "Error in BS integrator, step size is NaN"

This means that a variable time-step integrator wanting to take a NaN time step. In such cases, the selected
integrator settings are unsuitable for the problem you are considering.

NOTE: When any of the above errors occur, the propagation results up to the point of the crash can still be extracted
as normal. It can be checked whether any issues have occured by using the function

dynamics_simulator.integration_completed_successfully

which returns a boolean (false if any issues have occured)

* A frequent issue can be that a simulation with certain settings runs for too long (for instance if the time steo
becomes excessively small). To prevent this, you can add an additional termination setting (on top of the existing ones!)

    cpu_tim_termination_settings = propagation_setup.propagator.cpu_time_termination(
        maximum_cpu_time )

where maximum_cpu_time is a varaiable (float) denoting the maximum time in seconds that your simulation is allowed to
run. If the simulation runs longer, it will terminate, and return the propagation results up to that point.

* Finally, if the following error occurs, you can NOT extract the results up to the point of the crash. Instead,
the program will immediately terminate

    SPICE(DAFNEGADDR) --

    Negative value for BEGIN address: -214731446

This means that a state is extracted from Spice at a time equal to NaN. Typically, this is indicative of a
variable time-step integrator wanting to take a NaN time step, and the issue not being caught by Tudat.
In such cases, the selected integrator settings are unsuitable for the problem you are considering.
"""

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import numpy as np
import matplotlib.pyplot as plt

# Tudatpy imports
from tudatpy.data import save2txt
from tudatpy import constants
from tudatpy.interface import spice as spice_interface
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import environment
from tudatpy import numerical_simulation
from tudatpy.math import interpolators
from tudatpy import astro

# Problem-specific imports
import CapsuleEntryUtilities as Util
import pandas as pd
import random

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()
# NOTE TO STUDENTS: INPUT YOUR PARAMETER SET HERE, FROM THE INPUT FILES
# ON BRIGHTSPACE, FOR YOUR SPECIFIC STUDENT NUMBER
shape_parameters = [6.6048232258,
                    2.8463109182,
                    3.2004148801,
                    -0.4142178644,
                    0.3201519188,
                    0.4232153013]
# Choose whether benchmark is run
use_benchmark = True
additional_benchmark_analysis = False
benchmark_plot = False

run_integrator_analysis = True
variable_analysis = True
mc_analysis = False 
mc_analysis_2 = False

# Choose which question(s) to run
question_1 = False
question_2 = False
question_3 = True


# Choose whether output of the propagation is written to files
write_results_to_file = True
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Set simulation start epoch
simulation_start_epoch = 0.0  # s
# Set termination conditions
maximum_duration = constants.JULIAN_DAY  # s
termination_altitude = 25.0E3  # m
# Set vehicle properties
capsule_density = 250.0  # kg m-3

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Define settings for celestial bodies
bodies_to_create = ['Earth']
# Define coordinate system
global_frame_origin = 'Earth'
global_frame_orientation = 'J2000'

# Create body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)
# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create and add capsule to body system
# NOTE TO STUDENTS: When making any modifications to the capsule vehicle, do NOT make them in this code, but in the
# add_capsule_to_body_system function
Util.add_capsule_to_body_system(bodies,
                                shape_parameters,
                                capsule_density)


###########################################################################
# CREATE (CONSTANT) PROPAGATION SETTINGS ##################################
###########################################################################

# Retrieve termination settings
termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                     maximum_duration,
                                                     termination_altitude)
# Retrieve dependent variables to save
dependent_variables_to_save = Util.get_dependent_variable_save_settings()
# Check whether there is any
are_dependent_variables_to_save = False # if not dependent_variables_to_save else True


###########################################################################
# IF DESIRED, GENERATE BENCHMARK ##########################################
###########################################################################

# NOTE TO STUDENTS: MODIFY THE CODE INSIDE THIS "IF STATEMENT" (AND CALLED FUNCTIONS, IF NEEDED)
# TO ASSESS THE QUALITY OF VARIOUS BENCHMARK SETTINGS
if question_1:

    # Define benchmark interpolator settings to make a comparison between the two benchmarks
    benchmark_interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation = interpolators.extrapolate_at_boundary_with_warning)

    # Create propagator settings for benchmark (Cowell)
    benchmark_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                 bodies,
                                                                 simulation_start_epoch,
                                                                 termination_settings,
                                                                 dependent_variables_to_save )
    # Set output path for the benchmarks
    benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/' if write_results_to_file else None

    # Generate benchmarks
    benchmark_time_steps = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24] # [1/16, 1/8, 1/4, 1/2, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]  # s
    max_errors = np.empty(len(benchmark_time_steps))
    max_interpol_errors = np.empty(len(benchmark_time_steps))
    interpolation_relative_error_1_dict = dict()
    interpolation_relative_error_2_dict = dict()

    for i, benchmark_time_step in enumerate(benchmark_time_steps):

        benchmark_list = Util.generate_benchmarks(benchmark_time_step,
                                                simulation_start_epoch,
                                                bodies,
                                                benchmark_propagator_settings,
                                                are_dependent_variables_to_save,
                                                benchmark_output_path)

        # Extract benchmark states (first one is run with benchmark_time_step; second with 2.0*benchmark_time_step)
        first_benchmark_state_history = benchmark_list[0]
        second_benchmark_state_history = benchmark_list[1]
        time = np.array(list(second_benchmark_state_history.keys()))

        # Compare benchmark states, returning interpolator of the first benchmark, and writing difference to file if
        # write_results_to_file is set to True
        benchmark_state_difference = Util.compare_benchmarks(first_benchmark_state_history,
                                                            second_benchmark_state_history,
                                                            benchmark_output_path,
                                                            'benchmarks_state_difference.dat')
        
        position_difference = np.array([np.linalg.norm(benchmark_state_difference [epoch][:3]) for epoch in benchmark_state_difference.keys()])
        position_difference = position_difference[:-1]  # Remove last element, as it is corrupted (due to extrapolation)
        time = time[:-1]
        max_errors[i] = np.max(position_difference)

        
        benchmark_interpolation_difference = Util.evaluate_interpolation(first_benchmark_state_history, second_benchmark_state_history)
        benchmark_interpolation_difference = {epoch: value for epoch, value in list(benchmark_interpolation_difference.items())[6:-6]}
        interpolation_position_difference = np.array([np.linalg.norm(benchmark_interpolation_difference[epoch][:3]) for epoch in benchmark_interpolation_difference.keys()])
        
        max_interpol_errors[i] = np.max(interpolation_position_difference)

        # interpolation_relative_error_1, interpolation_relative_error_2 = Util.evaluate_interpolation_error(benchmark_interpolation_difference)
        # interpolation_relative_error_1_dict[i] = interpolation_relative_error_1
        # interpolation_relative_error_2_dict[i] = interpolation_relative_error_2

        # Extract benchmark dependent variables, if present
        if are_dependent_variables_to_save:
            first_benchmark_dependent_variable_history = benchmark_list[2]
            second_benchmark_dependent_variable_history = benchmark_list[3]

            # Create dependent variable interpolator for first benchmark
            benchmark_dependent_variable_interpolator = interpolators.create_one_dimensional_vector_interpolator(
                first_benchmark_dependent_variable_history,
                benchmark_interpolator_settings)

            # Compare benchmark dependent variables, returning interpolator of the first benchmark, and writing difference
            # to file if write_results_to_file is set to True
            benchmark_dependent_difference = Util.compare_benchmarks(first_benchmark_dependent_variable_history,
                                                                    second_benchmark_dependent_variable_history,
                                                                    benchmark_output_path,
                                                                    'benchmarks_dependent_variable_difference.dat')
            
        # plt.plot(list(benchmark_interpolation_difference.keys())[::2], interpolation_position_difference[::2], linewidth=2, label=f"{benchmark_time_step}")
        ls = '-' if i < 10 else '--'
        plt.plot(time, position_difference, linewidth=2.5, label=f"{2*benchmark_time_step}", linestyle=ls)

    plt.xlabel('Time [s]', fontsize=20)
    plt.ylabel(r'$||\epsilon_r (t,\Delta t)||$ [m]', fontsize=20)
    plt.yscale('log')
    plt.axhline(y=1, color='black', linestyle='--', linewidth=2, label='1 m threshold')
    plt.legend(title='Step Size [s]', title_fontsize=18, fontsize=18, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    plt.show()
            
    plt.figure(figsize=(10, 6))
    plt.plot([2 * step for step in benchmark_time_steps], max_errors, linewidth=2, marker='o', label='Max position error')
    plt.plot([2 * step for step in benchmark_time_steps], max_interpol_errors, linewidth=2, marker='d', linestyle='--', label='Max position error at the interpolated points')
    plt.xlabel('Step Size [s]', fontsize=20)
    plt.ylabel(r'$\epsilon_{max} (\Delta t)$ [m]', fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([2 * step for step in benchmark_time_steps], labels=[str(step) for step in [2 * step for step in benchmark_time_steps]], fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=2, label='1 m threshold')
    plt.legend(fontsize=15)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot the position error on the first y-axis
    ax1.plot([2 * step for step in benchmark_time_steps], max_errors, linewidth=3, marker='o', markersize=10, label='Max position error', markeredgecolor='black')
    ax1.plot([2 * step for step in benchmark_time_steps], max_interpol_errors, linewidth=3, marker='^', markersize=10, linestyle='--', label='Max position error at the interpolated points', markeredgecolor='black')
    ax1.set_xlabel('Step Size [s]', fontsize=20)
    ax1.set_ylabel(r'$\epsilon_{max} (\Delta t)$ [m]', fontsize=20)
    ax1.tick_params(axis='y')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks([2 * step for step in benchmark_time_steps])
    ax1.set_xticklabels([str(step) for step in [2 * step for step in benchmark_time_steps]], fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.grid(which='both', linestyle='--', linewidth=0.5)
    ax1.axhline(y=1, color='black', linestyle='--', linewidth=2, label='1 m threshold')
    # Add legend
    ax1.legend(fontsize=18, loc='upper left')

    # Calculate the relative error between the two maximum errors
    relative_error = np.abs(max_errors - max_interpol_errors) / max_errors

    # Create a second y-axis for the relative error
    ax2 = ax1.twinx()
    ax2.plot([2 * step for step in benchmark_time_steps], relative_error * 1e3, 'r-d', markersize=8, markeredgecolor='black', linewidth=2, label='Relative Error')
    ax2.set_ylabel('Relative Error [%]', fontsize=20, color='red')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='red', labelsize=20)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize=18, loc='upper left')

    plt.show()

    """
    for i, time_step in enumerate(benchmark_time_steps):

        interpolation_relative_error_1 = interpolation_relative_error_1_dict[i]
        interpolation_relative_error_2 = interpolation_relative_error_2_dict[i]

        time = interpolation_relative_error_1.keys()

        plt.plot(time, interpolation_relative_error_1.values(), linewidth=2.5, linestyle='-', label={time_step})
        plt.plot(time, interpolation_relative_error_2.values(), linewidth=2.5, linestyle='--')

    plt.xlabel('Time [s]', fontsize=20)
    plt.ylabel('Interpolation Error Ratio [-]', fontsize=20)
    plt.yscale('log')
    # Add legend for linestyle
    handles, labels = plt.gca().get_legend_handles_labels()
    linestyle_legend = [
        plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Previous Neighbour'),
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Next Neighbour')
    ]
    handles.extend(linestyle_legend)
    labels.extend(['Previous Neighbour', 'Next Neighbour'])
    plt.legend(handles=handles, title='Step Size', title_fontsize=18, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    plt.show()
    """


if use_benchmark:
    # Generate benchmark solution with fixed step size previously found
    benchmark_step_size = 0.08 # s
    benchmark_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                 bodies,
                                                                 simulation_start_epoch,
                                                                 termination_settings,
                                                                 dependent_variables_to_save)
    
    benchmark_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        benchmark_step_size,
        propagation_setup.integrator.CoefficientSets.rkf_56)
    benchmark_propagator_settings.print_settings.print_dependent_variable_indices = True

    benchmark_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        benchmark_propagator_settings )
    
    benchmark_time = benchmark_dynamics_simulator.state_history.keys()
    benchmark_state_history = benchmark_dynamics_simulator.state_history
    benchmark_dependent_variables_history = benchmark_dynamics_simulator.dependent_variable_history

    # Extract and process dependent variables
    benchmark_dependent_variables_dict = Util.save_dependent_variables_to_dict(benchmark_dependent_variables_history)

    if additional_benchmark_analysis:

        # Compare different time steps with benchmark
        time_steps = [0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48] 
        max_errors = np.empty(len(time_steps))
        max_interpol_errors = np.empty(len(time_steps))

        # Create state interpolator for benchmark
        benchmark_interpolator_settings = interpolators.lagrange_interpolation(
            8, boundary_interpolation = interpolators.extrapolate_at_boundary_with_warning)
        benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            benchmark_state_history,
            benchmark_interpolator_settings)

        for i, time_step in enumerate(time_steps):

            propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                bodies,
                                                                simulation_start_epoch,
                                                                termination_settings,
                                                                dependent_variables_to_save)

            propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
                time_step,
                propagation_setup.integrator.CoefficientSets.rkf_56)
            propagator_settings.print_settings.print_dependent_variable_indices = True

            dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                bodies,
                propagator_settings)
            
            state_history = dynamics_simulator.state_history
            time = np.array(list(state_history.keys()))
            
            benchmark_state_difference = dict()
            for epoch in time:
                benchmark_state_difference[epoch] = benchmark_state_interpolator.interpolate(epoch) - state_history[epoch]
            
            position_difference = np.array([np.linalg.norm(benchmark_state_difference [epoch][:3]) for epoch in benchmark_state_difference.keys()])
            position_difference = position_difference[:-1]  # Remove last element, as it is corrupted (due to extrapolation)
            time = time[:-1]
            max_errors[i] = np.max(position_difference)
            
            benchmark_interpolation_difference = Util.evaluate_interpolation(benchmark_state_history, state_history)
            benchmark_interpolation_difference = {epoch: value for epoch, value in list(benchmark_interpolation_difference.items())[6:-6]}
            interpolation_position_difference = np.array([np.linalg.norm(benchmark_interpolation_difference[epoch][:3]) for epoch in benchmark_interpolation_difference.keys()])
            
            max_interpol_errors[i] = np.max(interpolation_position_difference)
                
            # plt.plot(list(benchmark_interpolation_difference.keys())[::2], interpolation_position_difference[::2], linewidth=2, label=f"{benchmark_time_step}")
            plt.plot(time, position_difference, linewidth=2, label=f"{time_step}")

        plt.xlabel('Time [s]', fontsize=20)
        plt.ylabel(r'$||\epsilon_r (t,\Delta t)||$ [m]', fontsize=20)
        plt.yscale('log')
        plt.legend(title='Step Size [s]', title_fontsize=15, fontsize=15, bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.grid(True)
        plt.show()
                
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, max_errors, linewidth=2, marker='o', label='Max position error')
        plt.plot(time_steps, max_interpol_errors, linewidth=2, marker='d', linestyle='--', label='Max position error at the interpolated points')
        plt.xlabel('Step Size [s]', fontsize=20)
        plt.ylabel(r'$\epsilon_{max} (\Delta t)$ [m]', fontsize=20)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(time_steps, labels=[str(step) for step in time_steps], fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.axhline(y=1, color='r', linestyle='--', linewidth=1.5, label='10 m threshold')
        plt.legend(fontsize=15)
        plt.show()

    if benchmark_plot:
        Util.plot_dependent_variables(benchmark_dependent_variables_dict)

    

###########################################################################
# RUN SIMULATION FOR VARIOUS SETTINGS #####################################
###########################################################################
"""
Code below propagates states using each propagator (propagator_index=0..6), four multi-stage variable step-size 
integrators (integrator_index j=0..3) and an RK4 integrator (j=4). For the variable-step integrators, 4 different
tolerances are used (step_size_index=0..3). For the RK4, 6 different step sizes are used (step_size_index=0..5),
see use of number_of_integrator_step_size_settings variable. See get_integrator_settings function for more details.

For each combination of i, j, and k, results are written to directory:
    ShapeOptimization/SimulationOutput/prop_i/int_j/setting_k/

Specifically:
     state_History.dat                                  Cartesian states as function of time
     dependent_variable_history.dat                     Dependent variables as function of time
     state_difference_wrt_benchmark.dat                 Difference of dependent variables w.r.t. benchmark
     dependent_variable_difference_wrt_benchmark.dat    Difference of states w.r.t. benchmark
     ancillary_simulation_info.dat                      Other information about the propagation (number of function
                                                        evaluations, etc...)

NOTE TO STUDENTS: THE NUMBER, TYPES, SETTINGS OF PROPAGATORS/INTEGRATORS/INTEGRATOR STEPS,TOLERANCES,ETC. SHOULD BE
MODIFIED FOR ASSIGNMENT 1.
"""
if run_integrator_analysis:

    # Define list of propagators
    available_propagators = [propagation_setup.propagator.cowell,
                             propagation_setup.propagator.encke,
                             propagation_setup.propagator.gauss_keplerian,
                             propagation_setup.propagator.gauss_modified_equinoctial,
                             propagation_setup.propagator.unified_state_model_quaternions,
                             propagation_setup.propagator.unified_state_model_modified_rodrigues_parameters,
                             propagation_setup.propagator.unified_state_model_exponential_map]
    number_of_propagators = len(available_propagators)
    names_of_propagators = ['Cowell', 'Encke', 'Kepler (Gauss)', 'MEE', 'USM6', 'USM7', 'USM-EM']
    dependent_variables_per_popagator = {
        'Cowell': [],
        'Encke': [],
        'Kepler (Gauss)': [],
        'MEE': [],
        'USM6': [],
        'USM7': [],
        'USM-EM': []
    }
    dependent_variables_per_popagator_dict = {
        'Cowell': [],
        'Encke': [],
        'Kepler (Gauss)': [],
        'MEE': [],
        'USM6': [],
        'USM7': [],
        'USM-EM': []
    }

    # Define list of integrators
    
    names_of_integrators = ['RKF45', 'RKF56', 'RKDP78', 'RKF1210', 'RK4', 'RK5','RK7', 'RK10']
    number_of_integrators = len(names_of_integrators)

    number_of_function_evaluations = np.empty_like(names_of_propagators)
    max_errors = np.empty_like(names_of_propagators)
    propagator_benchmark_differences = []
    propagator_unprocessed_elements = []
    propagator_state_history = []

    # Define benchmark interpolator settings 
    benchmark_interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation = interpolators.extrapolate_at_boundary_with_warning)

    if question_2:

        step_size = 2.56# s

        for propagator_index in range(number_of_propagators):

            # Get current propagator, and define propagation settings
            current_propagator = available_propagators[propagator_index]
            current_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                       bodies,
                                                                       simulation_start_epoch,
                                                                       termination_settings,
                                                                       dependent_variables_to_save,
                                                                       current_propagator)

            # Create integrator settings
            current_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
                step_size,
                propagation_setup.integrator.CoefficientSets.rkf_56)

            dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                bodies, current_propagator_settings)

            # Retrieve propagated state and dependent variables
            state_history = dynamics_simulator.state_history
            propagator_state_history.append(state_history)
            unprocessed_state_history = dynamics_simulator.unprocessed_state_history
            propagator_unprocessed_elements.append(unprocessed_state_history)

            # Save dependent variables for the current propagator
            dependent_variable_history = dynamics_simulator.dependent_variable_history
            dependent_variables_dict = Util.save_dependent_variables_to_dict(dependent_variable_history)
            dependent_variables_per_popagator[names_of_propagators[propagator_index]] = dependent_variable_history
            dependent_variables_per_popagator_dict[names_of_propagators[propagator_index]] = dependent_variables_dict


            # Get the number of function evaluations (for comparison of different integrators)
            function_evaluation_dict = dynamics_simulator.cumulative_number_of_function_evaluations
            number_of_function_evaluations[propagator_index] = list(function_evaluation_dict.values())[-1]



            # Compare the simulation to the benchmarks 
            benchmark_difference = Util.compare_benchmarks(benchmark_state_history,
                                                           state_history,                                                           
                                                           None,
                                                           None)

            benchmark_difference = {epoch: value for epoch, value in list(benchmark_difference.items())[:-1]}
            propagator_benchmark_differences.append(benchmark_difference)
            benchmark_position_difference = np.array([np.linalg.norm(benchmark_difference[epoch][:3]) for epoch in benchmark_difference.keys()])

            benchmark_difference_no_extrapolation = Util.compare_benchmarks_no_extrapolation(benchmark_state_history,
                                                                                    state_history,                                                           
                                                                                    None,
                                                                                    None)
            benchmark_position_difference_no_extrapolation = np.array([np.linalg.norm(benchmark_difference_no_extrapolation[epoch][:3]) for epoch in benchmark_difference_no_extrapolation.keys()])
            max_errors[propagator_index] = np.max(benchmark_position_difference_no_extrapolation)

            ls = ['-', '-', '--', ':', ':']
            color = ['royalblue', 'royalblue',  'darkorange', 'deeppink', 'cyan']
            lw = ['4', '4', '3.5', '3.5', '3.5']
            labels = ['Cowell-Encke', 'Cowell-Encke', 'Kepler', 'MEE', 'USM7-USM6-USMEM']
            if propagator_index == 1 or propagator_index == 2 or propagator_index == 3 or propagator_index == 4:
                plt.plot(benchmark_difference_no_extrapolation.keys(), benchmark_position_difference_no_extrapolation,
                        label=f"{labels[propagator_index]}", 
                        linestyle=ls[propagator_index],
                        color=color[propagator_index],
                        linewidth=lw[propagator_index])  
                # Add a circular marker at the end of each line
                plt.plot(list(benchmark_difference_no_extrapolation.keys())[-1], benchmark_position_difference_no_extrapolation[-1], 
                         marker='o', color=color[propagator_index], markersize=8, label=None)

        extrapolation_threshold = list(benchmark_state_history.keys())[-1]

        plt.xlabel('Time [s]', fontsize=20)
        plt.ylabel('Position Error [m]', fontsize=20)
        plt.yscale('log')
        plt.axhline(y=1, color='black', linestyle='-', linewidth=2, label='1 m requirement')
        plt.legend(fontsize=15, loc='upper left')
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.grid(True)
        plt.show()

        # Print a table with max error and number of function evaluations for each propagator

        # Create a dictionary with propagator names as columns
        data = {
            "Propagator": names_of_propagators,
            "Max Error [m]": max_errors,
            "Function Evaluations": number_of_function_evaluations
        }

        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(data)

        # Print the table
        print(df.to_string(index=False))

        elements_cowell = propagator_unprocessed_elements[0]
        elements_enke = propagator_unprocessed_elements[1]
        elements_gauss = propagator_unprocessed_elements[2]
        elements_gme = propagator_unprocessed_elements[3]
        elements_usmq = propagator_unprocessed_elements[4]
        elements_usmmrp = propagator_unprocessed_elements[5]
        elements_usmem = propagator_unprocessed_elements[6]

        state_cowell = propagator_state_history[0]
        state_enke = propagator_state_history[1]
        state_gauss = propagator_state_history[2]
        state_gme = propagator_state_history[3]
        state_usmq = propagator_state_history[4]
        state_usmmrp = propagator_state_history[5]
        state_usmem = propagator_state_history[6]

        # they dont work anymore
        dependent_cowell = dependent_variables_per_popagator['Cowell']
        dependent_enke = dependent_variables_per_popagator['Encke']
        dependent_gauss = dependent_variables_per_popagator['Kepler (Gauss)']
        dependent_gme = dependent_variables_per_popagator['MEE']
        dependent_usmq = dependent_variables_per_popagator['USM6']
        dependent_usmmrp = dependent_variables_per_popagator['USM7']
        dependent_usmem = dependent_variables_per_popagator['USM-EM']

        for propagator_index, prop in enumerate(names_of_propagators):

            dependent = dependent_variables_per_popagator[prop]
            
            # Directly use altitude and altitude_benchmark as they are already dictionaries
            dependent_benchmark_difference = Util.compare_benchmarks_no_extrapolation(benchmark_dependent_variables_history,
                                                                                     dependent, None, None)
            
            time = dependent_benchmark_difference.keys()
            altitude_difference = np.array([np.linalg.norm(dependent_benchmark_difference[epoch][1]) for epoch in time])

            ls = ['-', '-', '--', ':', ':']
            color = ['royalblue', 'royalblue',  'darkorange', 'deeppink', 'cyan']
            lw = ['4', '4', '3.5', '3.5', '3.5']
            labels = ['Cowell-Encke', 'Cowell-Encke', 'Kepler', 'MEE', 'USM7-USM6-USMEM']
            if propagator_index == 1 or propagator_index == 2 or propagator_index == 3 or propagator_index == 4:
                plt.plot(time, altitude_difference,
                        label=f"{labels[propagator_index]}", 
                        linestyle=ls[propagator_index],
                        color=color[propagator_index],
                        linewidth=lw[propagator_index])  
                plt.plot(list(time)[-1], altitude_difference[-1], 
                         marker='o', color=color[propagator_index], markersize=8, label=None)
            
        plt.xlabel('Time [s]', fontsize=20)
        plt.ylabel('Altitude Error [m]', fontsize=20)
        plt.yscale('log')
        plt.axhline(y=1, color='black', linestyle='-', linewidth=2, label='1 m requirement')
        # plt.legend(fontsize=15, loc='upper left')
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.grid(True)
        plt.show()
            


        # Util.plot_cowell_state_elements(elements_cowell)
        #Util.plot_enke_state_elements(elements_enke)
        #Util.plot_gauss_keplerian_elements(elements_gauss)

        Util.plot_gme_elements(elements_gme)
        #Util.plot_cowell_state_elements(state_gme)
        dependent_gme = dependent_variables_per_popagator_dict['MEE']
        # Util.plot_dependent_variables(dependent_gme)

        # Interpolate the benchmark at the time points of state_usmq
        benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            benchmark_state_history,
            interpolators.lagrange_interpolation(8, boundary_interpolation=interpolators.extrapolate_at_boundary_with_warning)
        )

        
        # Util.plot_usm_quaternions_elements(elements_usmq)
        dependent_usmq = dependent_variables_per_popagator_dict['USM6']
        # Util.plot_dependent_variables(dependent_usmq)
        # Util.plot_cowell_state_elements(state_usmq)

        interpolated_benchmark_states = {epoch: benchmark_interpolator.interpolate(epoch) for epoch in state_usmq.keys()}

        #Util.plot_usm_mrp_elements(elements_usmmrp)
        #Util.plot_dependent_variables(dependent_usmmrp)
        #Util.plot_cowell_state_elements(state_usmmrp)

        #Util.plot_usmem_elements(elements_usmem)
        #Util.plot_dependent_variables(dependent_usmem)
        #Util.plot_cowell_state_elements(state_usmem)
            
    if question_3:

        step_sizes = [1.28, 2.56, 5.12, 10.24, 20.48] #, 40.96] # s
        tolerances = [1.0E-14, 1.0E-12, 1.0E-10, 1.0E-8, 1.0E-6] # s

        markers = ['s', '^', 'P', 'D', 'v', '<', '>', 'o']  # Define a list of markers

        # Define list of propagators
        available_propagators = [propagation_setup.propagator.cowell,
                                propagation_setup.propagator.encke,
                                propagation_setup.propagator.gauss_modified_equinoctial]
        number_of_propagators = len(available_propagators)
        names_of_propagators = ['Cowell', 'Encke', 'MEE']

        for integrator_index in range(int(number_of_integrators/2)):

            for propagator_index in range(number_of_propagators):

                # Get current propagator, and define propagation settings
                current_propagator = available_propagators[propagator_index]
                current_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                        bodies,
                                                                        simulation_start_epoch,
                                                                        termination_settings,
                                                                        dependent_variables_to_save,
                                                                        current_propagator )
                    
                function_evaluations_variable_sorted = []
                max_position_differences_variable_sorted = []

                function_evaluations_fixed_sorted = []
                max_position_differences_fixed_sorted = []


                # Loop over all tolerances / step sizes
                for step_tolerance_index in range(len(step_sizes)):
                    """
                    # Print status
                    to_print = 'Current run: \n propagator_index = ' + str(propagator_index) + \
                            '\n integrator_index = ' + str(integrator_index) \
                            + '\n step_size_index = ' + str(step_tolerance_index)
                    print(to_print)
                    # Set output path
                    output_path = current_dir + '/SimulationOutput/prop_' + str(propagator_index) + \
                                '/int_' + str(integrator_index) + '/step_size_' + str(step_tolerance_index) + '/'
                                """
                    
                    
                    
                    # Create integrator settings for variable step size                    
                    tolerance = tolerances[step_tolerance_index]
                    current_integrator_settings_variable = Util.get_integrator_settings(current_propagator_settings,
                                                                            integrator_index,
                                                                            tolerance)
                    dynamics_simulator_variable = numerical_simulation.create_dynamics_simulator(
                        bodies, current_integrator_settings_variable )
                    
                    # Create integrator settings for fixed step size
                    step_size = step_sizes[step_tolerance_index]
                    current_integrator_settings_fixed = Util.get_integrator_settings(current_propagator_settings,
                                                                            integrator_index,
                                                                            step_size, 
                                                                            fixed = True)
                    dynamics_simulator_fixed = numerical_simulation.create_dynamics_simulator(
                        bodies, current_integrator_settings_fixed )

                    ### OUTPUT OF THE SIMULATION ###
                    # Retrieve propagated state and dependent variables
                    state_history_variable = dynamics_simulator_variable.state_history
                    unprocessed_state_history_variable = dynamics_simulator_variable.unprocessed_state_history
                    dependent_variable_history_variable = dynamics_simulator_variable.dependent_variable_history

                    state_history_fixed = dynamics_simulator_fixed.state_history
                    unprocessed_state_history_fixed = dynamics_simulator_fixed.unprocessed_state_history
                    dependent_variable_history_fixed = dynamics_simulator_fixed.dependent_variable_history


                    # Get the number of function evaluations (for comparison of different integrators)
                    function_evaluation_dict_variable = dynamics_simulator_variable.cumulative_number_of_function_evaluations
                    number_of_function_evaluations_variable = list(function_evaluation_dict_variable.values())[-1]
                    function_evaluations_variable_sorted.append(number_of_function_evaluations_variable)
                    propagation_outcome_variable = dynamics_simulator_variable.integration_completed_successfully

                    function_evaluation_dict_fixed = dynamics_simulator_fixed.cumulative_number_of_function_evaluations
                    number_of_function_evaluations_fixed = list(function_evaluation_dict_fixed.values())[-1]
                    function_evaluations_fixed_sorted.append(number_of_function_evaluations_fixed)
                    propagation_outcome_fixed = dynamics_simulator_fixed.integration_completed_successfully

                    print(propagation_outcome_variable)
                    print(propagation_outcome_fixed)


                    # Compare the simulation to the benchmarks and write differences to files
                    if use_benchmark:
                        # Initialize containers
                        state_difference_variable = dict()
                        state_difference_fixed = dict()

                        # Loop over the propagated states and use the benchmark interpolators
                        benchmark_difference_variable = Util.compare_benchmarks_no_extrapolation(benchmark_state_history,
                                                                                        state_history_variable,                                                           
                                                                                        None,
                                                                                        None)
                        benchmark_difference_fixed = Util.compare_benchmarks_no_extrapolation(benchmark_state_history,
                                                                                        state_history_fixed,                                                           
                                                                                        None,
                                                                                        None)

                        extrapolation_threshold = list(benchmark_state_history.keys())[-1]
                        
                        position_difference_variable = {epoch: np.linalg.norm(benchmark_difference_variable[epoch][:3]) for epoch in benchmark_difference_variable.keys() if epoch < extrapolation_threshold}
                        time_variable = np.array(list(position_difference_variable.keys()))
                        
                        position_difference_fixed = {epoch: np.linalg.norm(benchmark_difference_fixed[epoch][:3]) for epoch in benchmark_difference_fixed.keys() if epoch < extrapolation_threshold}
                        time_fixed = np.array(list(position_difference_fixed.keys()))

                        """
                        # Plot position difference vs time
                        plt.plot(time, position_difference.values(), linewidth=2, label=f"Integrator {integrator_index}, Step/Tolerance {step_tolerance_index}")
                        if integrator_index < 4:
                            plt.title(f'Propagator: {names_of_propagators[propagator_index]}, Integrator: {names_of_integrators[integrator_index]}, Tolerance: {tolerances[step_tolerance_index]:.0e}', fontsize=16)
                        else:
                            plt.title(f'Propagator: {names_of_propagators[propagator_index]}, Integrator: {names_of_integrators[integrator_index]}, Step Size: {step_sizes[step_tolerance_index]} s', fontsize=16)
                        plt.show()
                        plt.close()
                        """

                        max_error_variable = max(position_difference_variable.values())
                        max_position_differences_variable_sorted.append(max_error_variable)
                        
                        max_error_fixed = max(position_difference_fixed.values())
                        max_position_differences_fixed_sorted.append(max_error_fixed)


                    # Use consistent colormaps for step sizes and tolerances
                    color_map_variable = plt.cm.viridis  # Colormap for variable step size
                    color_variable = color_map_variable(step_tolerance_index / len(tolerances))
                    label_variable = f'{tolerances[step_tolerance_index]:.0e}'
                    
                    color_map_fixed = plt.cm.spring  # Colormap for fixed step size (yellow-orange-red scale)
                    color_fixed = color_map_fixed(step_tolerance_index / len(step_sizes))
                    label_fixed = f'{step_sizes[step_tolerance_index]:.2f}s'
                    
                    # Plot the scatter point with the corresponding color and label for variable step size
                    if propagation_outcome_variable:
                        plt.scatter(number_of_function_evaluations_variable, max_error_variable, 
                                    marker=markers[propagator_index], 
                                    color=color_variable, label=label_variable, s=200, zorder=3,
                                    edgecolor='black', linewidth=1.5)
                    else:
                        plt.scatter(number_of_function_evaluations_variable, max_error_variable, 
                                    marker='x', color='red', label='Failed Propagation', s=200, zorder=3, linewidths=2)

                    # Plot the scatter point with the corresponding color and label for fixed step size
                    if propagation_outcome_fixed:
                        plt.scatter(number_of_function_evaluations_fixed, max_error_fixed, 
                                    marker=markers[propagator_index], 
                                    color=color_fixed, label=label_fixed, s=200, zorder=3,
                                    edgecolor='black', linewidth=1.5)
                    else:
                        plt.scatter(number_of_function_evaluations_fixed, max_error_fixed, 
                                    marker='x', color='red', label='Failed Propagation', s=200, zorder=3, linewidths=2)
                

                # Plot the line connecting the points for the current propagator
                plt.plot(
                    function_evaluations_variable_sorted,
                    max_position_differences_variable_sorted,
                    linestyle='--',
                    color='black',
                    linewidth=1.5,
                    zorder=2,
                )

                plt.plot(
                    function_evaluations_fixed_sorted,
                    max_position_differences_fixed_sorted,
                    linestyle='-',
                    color='black',
                    linewidth=1.5,
                    zorder=2,
                )

                dict_to_write_fixed = dict()
                dict_to_write_variable = dict()

                dict_to_write_fixed['Number of function evaluation'] = function_evaluations_fixed_sorted
                dict_to_write_fixed['Maximum position difference'] = max_position_differences_fixed_sorted

                dict_to_write_variable['Number of function evaluation'] = function_evaluations_variable_sorted
                dict_to_write_variable['Maximum position difference'] = max_position_differences_variable_sorted


                output_path_variable = current_dir + '/SimulationOutput/int_' + str(integrator_index) + \
                                '/prop_' + str(propagator_index) + '/variable_step_size/'
                output_path_fixed = current_dir + '/SimulationOutput/int_' + str(integrator_index) + \
                                '/prop_' + str(propagator_index) + '/fixed_step_size/'

                save2txt(dict_to_write_fixed, 'max_pos_diff__func_eval.txt',   output_path_fixed)
                save2txt(dict_to_write_variable, 'amax_pos_diff__func_eval.txt',   output_path_variable)

            # Add legends for step size
            step_size_legend_handles = [
                plt.Line2D([0], [0], color=plt.cm.spring(i / len(step_sizes)), lw=4, label=f'{step_size} s')
                for i, step_size in enumerate(step_sizes)
            ]
            step_size_legend = plt.legend(handles=step_size_legend_handles, title='Step Size:', title_fontsize=15, fontsize=15, loc='upper left', bbox_to_anchor=(1.02, 1.035))
            plt.gca().add_artist(step_size_legend)

            # Add legends for tolerance
            tolerance_legend_handles = [
                plt.Line2D([0], [0], color=plt.cm.viridis(i / len(tolerances)), lw=4, label=f'{tolerance:.0e}')
                for i, tolerance in enumerate(tolerances)
            ]
            tolerance_legend = plt.legend(handles=tolerance_legend_handles, title='Tolerance:', title_fontsize=15, fontsize=15, loc='upper left', bbox_to_anchor=(1.02, 0.7))
            plt.gca().add_artist(tolerance_legend)

            # Add legends for propagator types with markers
            propagator_legend_handles = [
                plt.Line2D([0], [0], color='black', marker=markers[i % len(markers)], markersize=10, label=names_of_propagators[i])
                for i in range(number_of_propagators)
            ]
            propagator_legend = plt.legend(handles=propagator_legend_handles, title='Propagator Types:', title_fontsize=15, fontsize=15, loc='upper left', bbox_to_anchor=(1.02, 0.4))
            plt.gca().add_artist(propagator_legend)

            # Add legends for propagator types with linestyles
            linestyle_legend_handles = [
                plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label=f'{names_of_integrators[integrator_index]} - Variable Step Size'),
                plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label=f'{names_of_integrators[4 + integrator_index]} - Fixed Step Size'),
            ]
            linestyle_legend = plt.legend(handles=linestyle_legend_handles, title_fontsize=20, fontsize=20) #loc='upper right', bbox_to_anchor=(0.98, 0.98))
            plt.gca().add_artist(linestyle_legend)

            # Add legend for failed propagation
            failed_legend_handle = plt.Line2D([0], [0], color='red', marker='x', markersize=10, linestyle='None', label='Integration Failed')
            failed_legend = plt.legend(handles=[failed_legend_handle], loc='upper left', bbox_to_anchor=(1.02, 0.1), fontsize=15)
            plt.gca().add_artist(failed_legend)

            plt.xlabel('Cumulative Number of Function Evaluations [-]', fontsize=20)
            plt.ylabel('Maximum Position Difference [m]', fontsize=20)
            plt.yscale('log')
            plt.xscale('log')
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.axhline(y=1, color='black', linestyle='--', linewidth=1.5)
            plt.grid()
            plt.show()










    if variable_analysis:

        step_sizes = [1.0E-14, 1.0E-12, 1.0E-10, 1.0E-8, 1.0E-6] # s

        # Define list of propagators
        available_propagators = [propagation_setup.propagator.cowell,
                                propagation_setup.propagator.encke,
                                propagation_setup.propagator.gauss_modified_equinoctial]
        number_of_propagators = len(available_propagators)
        names_of_propagators = ['Cowell', 'Encke', 'MEE']

        time_step_size_dict = dict()

        # Loop over propagators
        for propagator_index in range(number_of_propagators):

            # Get current propagator, and define propagation settings
            current_propagator = available_propagators[propagator_index]
            current_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                    bodies,
                                                                    simulation_start_epoch,
                                                                    termination_settings,
                                                                    dependent_variables_to_save,
                                                                    current_propagator )
            
            time_step_size_per_propagator = []

            # Loop over different integrators
            for integrator_index in range(int(number_of_integrators/2)):

                time_step_size = []

                for step_tolerance_index in range(5):
                    
                    tolerance = step_sizes[step_tolerance_index]
                    # Create integrator settings
                    current_integrator_settings = Util.get_integrator_settings(current_propagator_settings,
                                                                            integrator_index,
                                                                            tolerance)
                    

                    # Create Shape Optimization Problem object
                    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                        bodies, current_propagator_settings )

                    ### OUTPUT OF THE SIMULATION ###
                    # Retrieve propagated state and dependent variables
                    state_history = dynamics_simulator.state_history
                    unprocessed_state_history = dynamics_simulator.unprocessed_state_history
                    dependent_variable_history = dynamics_simulator.dependent_variable_history

                    time = np.array(list(state_history.keys()))
                    time_step_size.append([time[:-1], np.diff(time)])


                    # Compare the simulation to the benchmarks and write differences to files
                    if use_benchmark:
                        # Initialize containers
                        state_difference = dict()

                        # Loop over the propagated states and use the benchmark interpolators
                        # NOTE TO STUDENTS: it can happen that the benchmark ends earlier than the regular simulation, due to
                        # the shorter step size. Therefore, the following lines of code will be forced to extrapolate the
                        # benchmark states (or dependent variables), producing a warning. Be aware of it!
                        benchmark_difference = Util.compare_benchmarks_no_extrapolation(benchmark_state_history,
                                                                                        state_history,                                                           
                                                                                        None,
                                                                                        None)

                        extrapolation_threshold = list(benchmark_state_history.keys())[-1]
                        
                        position_difference = {epoch: np.linalg.norm(benchmark_difference[epoch][:3]) for epoch in benchmark_difference.keys() if epoch < extrapolation_threshold}
                        time = np.array(list(position_difference.keys()))

                        """
                        # Plot position difference vs time
                        plt.plot(time, position_difference.values(), linewidth=2, label=f"Integrator {integrator_index}, Step/Tolerance {step_tolerance_index}")
                        if integrator_index < 4:
                            plt.title(f'Propagator: {names_of_propagators[propagator_index]}, Integrator: {names_of_integrators[integrator_index]}, Tolerance: {tolerances[step_tolerance_index]:.0e}', fontsize=16)
                        else:
                            plt.title(f'Propagator: {names_of_propagators[propagator_index]}, Integrator: {names_of_integrators[integrator_index]}, Step Size: {step_sizes[step_tolerance_index]} s', fontsize=16)
                        plt.show()
                        plt.close()
                        """

                time_step_size_per_propagator.append(time_step_size)

            time_step_size_dict[propagator_index] = time_step_size_per_propagator

        # Plot time step size as a function of time for each propagator
        fig, axes = plt.subplots(1, len(names_of_propagators), figsize=(18, 6), sharey=True)
        linestyles = ['-', '--', '-', ':', (0, (3, 1, 1, 1))]  # Define linestyles for tolerances
        colors = plt.cm.tab10.colors  # Use a colormap for integrators

        for propagator_index, propagator_name in enumerate(names_of_propagators):
            ax = axes[propagator_index]
            time_step_sizes = time_step_size_dict[propagator_index]

            for integrator_index, step_sizes_per_integrator in enumerate(time_step_sizes):
                for tolerance_index, step_sizes in enumerate(step_sizes_per_integrator):
                    time = step_sizes[0]  # Exclude the last time point
                    step_sizes = step_sizes[1]
                    if tolerance_index == 2:
                        ax.plot(
                        time,
                        step_sizes,
                        label=f"Int {names_of_integrators[integrator_index]}, Tol {step_sizes[tolerance_index]:.0e}",
                        color=colors[integrator_index % len(colors)],
                        linestyle=linestyles[tolerance_index % len(linestyles)],
                        linewidth=3,
                        )

            ax.set_title(propagator_name, fontsize=20)
            ax.set_xlabel('Time [s]', fontsize=20)
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=18)
            if propagator_index == 0:
                ax.set_ylabel('Time Step Size [s]', fontsize=20)
            # Only add legend to the last subplot
            if propagator_index == len(names_of_propagators) - 1:
                integrator_legend_handles = [
                    plt.Line2D([0], [0],
                            color=colors[i % len(colors)],
                            lw=3,
                            label=names_of_integrators[i])
                    for i in range(len(names_of_integrators))
                ]
                ax.legend(
                    handles=integrator_legend_handles,
                    title='Integrators:',
                    title_fontsize=12,
                    fontsize=10,
                    loc='upper right',
                    bbox_to_anchor=(1.3, 1.0)
                )

        plt.tight_layout()
        plt.show()




    if mc_analysis:

        step_sizes = [1.28, 2.56, 5.12] # s

        # Define list of propagators
        available_propagators = [propagation_setup.propagator.cowell,
                                propagation_setup.propagator.encke,]
        number_of_propagators = len(available_propagators)
        names_of_propagators = ['Cowell', 'Encke']

        names_of_integrators = ['RK4', 'RK5']
        number_of_integrators = len(names_of_integrators)

        number_of_function_evaluations = np.empty_like(names_of_propagators)
        max_errors = np.empty_like(names_of_propagators)
        propagator_benchmark_differences = []
        propagator_state_history = []

        # Define benchmark interpolator settings 
        benchmark_interpolator_settings = interpolators.lagrange_interpolation(
            8, boundary_interpolation = interpolators.extrapolate_at_boundary_with_warning)
        
        ###################################################################################

        # Number of Monte Carlo samples
        num_samples = 100

        # Define the range for each shape parameter (30% variation)
        shape_parameter_ranges = [
            (0.8 * shape_parameters[0], 1.2 * shape_parameters[0]),  # Nose radius (±20%)
            (0.8 * shape_parameters[1], 1.2 * shape_parameters[1]),  # Middle radius (±20%)
            (0.8 * shape_parameters[2], 1.2 * shape_parameters[2]),  # Rear length (±20%)
            (0.8 * shape_parameters[3], 1.2 * shape_parameters[3]),  # Rear angle (±20%)
            (0.8 * shape_parameters[4], 1.2 * shape_parameters[4]),  # Side radius (±20%)
            (shape_parameters[5], shape_parameters[5])               # Constant Angle of Attack (unchanged)
        ]

        # Generate Monte Carlo samples
        monte_carlo_samples = []
        for _ in range(num_samples):
            sample = [
                random.uniform(*shape_parameter_ranges[i]) for i in range(len(shape_parameters))
            ]
            monte_carlo_samples.append(sample)

        #######################################################################################

        mc_results_per_propopagator = dict()

        # Loop over propagators
        for propagator_index in range(number_of_propagators):

            # Get current propagator, and define propagation settings
            current_propagator = available_propagators[propagator_index]

            mc_results_per_integrator = dict()

            # Loop over different integrators
            for integrator_index in range(int(number_of_integrators)):

                mc_results_per_step_size = dict()

                for step_tolerance_index in range(len(step_sizes)):
                    
                    tolerance = step_sizes[step_tolerance_index]

                    func_eval_list = []
                    max_error_list = []

                    for sample in monte_carlo_samples:

                        shape_parameters = sample
                        bodies = environment_setup.create_system_of_bodies(body_settings)
                        Util.add_capsule_to_body_system(bodies,
                                                        shape_parameters,
                                                        capsule_density)


                        current_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                        bodies,
                                                                        simulation_start_epoch,
                                                                        termination_settings,
                                                                        dependent_variables_to_save,
                                                                        current_propagator )
                        current_integrator_settings = Util.get_integrator_settings(current_propagator_settings,
                                                                                integrator_index + 4,
                                                                                tolerance,
                                                                                fixed= True)
                    
                    
                        # Create Shape Optimization Problem object
                        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                            bodies, current_integrator_settings )

                        ### OUTPUT OF THE SIMULATION ###
                        # Retrieve propagated state and dependent variables
                        state_history = dynamics_simulator.state_history
                        dependent_variable_history = dynamics_simulator.dependent_variable_history
                        function_evaluation_dict = dynamics_simulator.cumulative_number_of_function_evaluations
                        number_of_function_evaluations = list(function_evaluation_dict.values())[-1]
                        func_eval_list.append(number_of_function_evaluations)
                        

                        # Compare the simulation to the benchmarks and write differences to files
                        if use_benchmark:

                            # Generate benchmark solution with fixed step size previously found
                            benchmark_step_size = 0.16 # s
                            benchmark_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                                        bodies,
                                                                                        simulation_start_epoch,
                                                                                        termination_settings,
                                                                                        dependent_variables_to_save)
                            
                            benchmark_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
                                benchmark_step_size,
                                propagation_setup.integrator.CoefficientSets.rkf_56)
                            benchmark_propagator_settings.print_settings.print_dependent_variable_indices = True

                            benchmark_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                                bodies,
                                benchmark_propagator_settings )
                            
                            benchmark_state_history = benchmark_dynamics_simulator.state_history

                            # Initialize containers
                            state_difference = dict()

                            # Loop over the propagated states and use the benchmark interpolators
                            # NOTE TO STUDENTS: it can happen that the benchmark ends earlier than the regular simulation, due to
                            # the shorter step size. Therefore, the following lines of code will be forced to extrapolate the
                            # benchmark states (or dependent variables), producing a warning. Be aware of it!
                            benchmark_difference = Util.compare_benchmarks_no_extrapolation(benchmark_state_history,
                                                                                            state_history,                                                           
                                                                                            None,
                                                                                            None)

                            extrapolation_threshold = list(benchmark_state_history.keys())[-1]
                            
                            position_difference = {epoch: np.linalg.norm(benchmark_difference[epoch][:3]) for epoch in benchmark_difference.keys() if epoch < extrapolation_threshold}
                            time = np.array(list(position_difference.keys()))
                            max_error = max(position_difference.values())
                            max_error_list.append(max_error)

                            """
                            # Plot position difference vs time
                            plt.plot(time, position_difference.values(), linewidth=2, label=f"Integrator {integrator_index}, Step/Tolerance {step_tolerance_index}")
                            if integrator_index < 4:
                                plt.title(f'Propagator: {names_of_propagators[propagator_index]}, Integrator: {names_of_integrators[integrator_index]}, Tolerance: {tolerances[step_tolerance_index]:.0e}', fontsize=16)
                            else:
                                plt.title(f'Propagator: {names_of_propagators[propagator_index]}, Integrator: {names_of_integrators[integrator_index]}, Step Size: {step_sizes[step_tolerance_index]} s', fontsize=16)
                            plt.show()
                            plt.close()
                            """


                    mc_results_per_step_size[step_tolerance_index] = [func_eval_list, max_error_list]
                    
                mc_results_per_integrator[integrator_index] = mc_results_per_step_size
                
            mc_results_per_propopagator[propagator_index] = mc_results_per_integrator

        # Create a figure with two subplots for Cowell and Encke
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

        # Define colors for integrators and markers for step sizes
        colors = ['blue', 'red']  # Use blue and red for the two integrators
        markers = ['o', 's', '^']  # Define a list of markers for step sizes

        # Loop through propagators (Cowell and Encke)
        for propagator_index, propagator_name in enumerate(['Cowell', 'Encke']):
            ax = axes[propagator_index]
            mc_results_per_integrator = mc_results_per_propopagator[propagator_index]

            # Loop through integrators
            for integrator_index in range(number_of_integrators):

                mc_results_per_step_size = mc_results_per_integrator[integrator_index]
                # Loop through step sizes
                for step_size_index in range(len(step_sizes)):

                    mc_results = mc_results_per_step_size[step_size_index]
                    func_eval_list = mc_results[0]
                    max_error_list = mc_results[1]

                    # Plot the data
                    ax.scatter(
                        func_eval_list,
                        max_error_list,
                        color=colors[integrator_index],
                        marker=markers[step_size_index],
                        s=50,
                        label=f"Int: {names_of_integrators[integrator_index]}, Step: {step_sizes[step_size_index]} [s]",
                        edgecolor='black'
                    )

            # Set plot title and labels
            ax.set_title(f"{propagator_name} Propagator", fontsize=20)
            ax.set_xlabel("Number of Function Evaluations [-]", fontsize=20)
            ax.set_xscale("log")
            ax.grid(True)
            ax.tick_params(axis="both", which="major", labelsize=18)
            ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, label='1 m requirement')

        # Set shared y-axis label
        axes[0].set_ylabel("Maximum Position Error [m]", fontsize=20)
        axes[0].set_yscale("log")

        # Add a single legend to the first subplot (Cowell) in the upper right
        axes[0].legend(fontsize=16, loc='upper right')



        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
        # Save the plot in the current directory
        output_plot_path = os.path.join(current_dir, "monte_carlo_analysis_plot.png")
        plt.savefig(output_plot_path, bbox_inches="tight", dpi=300)








    if mc_analysis_2:

            tolerances = [1.0E-12, 1.0E-10, 1.0E-8, 1.0E-6] # s


            # Define list of propagators
            available_propagators = [propagation_setup.propagator.encke]
            number_of_propagators = len(available_propagators)
            names_of_propagators = ['Encke']

            names_of_integrators = ['RKF4(5)', 'RK5(6)']
            number_of_integrators = len(names_of_integrators)

            number_of_function_evaluations = np.empty_like(names_of_propagators)
            max_errors = np.empty_like(names_of_propagators)
            propagator_benchmark_differences = []
            propagator_state_history = []

            # Define benchmark interpolator settings 
            benchmark_interpolator_settings = interpolators.lagrange_interpolation(
                8, boundary_interpolation = interpolators.extrapolate_at_boundary_with_warning)
            
            ###################################################################################

            # Number of Monte Carlo samples
            num_samples = 100

            # Define the range for each shape parameter (30% variation)
            shape_parameter_ranges = [
                (0.8 * shape_parameters[0], 1.2 * shape_parameters[0]),  # Nose radius (±20%)
                (0.8 * shape_parameters[1], 1.2 * shape_parameters[1]),  # Middle radius (±20%)
                (0.8 * shape_parameters[2], 1.2 * shape_parameters[2]),  # Rear length (±20%)
                (0.8 * shape_parameters[3], 1.2 * shape_parameters[3]),  # Rear angle (±20%)
                (0.8 * shape_parameters[4], 1.2 * shape_parameters[4]),  # Side radius (±20%)
                (shape_parameters[5], shape_parameters[5])               # Constant Angle of Attack (unchanged)
            ]

            # Generate Monte Carlo samples
            monte_carlo_samples = []
            for _ in range(num_samples):
                sample = [
                    random.uniform(*shape_parameter_ranges[i]) for i in range(len(shape_parameters))
                ]
                monte_carlo_samples.append(sample)

            #######################################################################################

            mc_results_per_propopagator = dict()

            # Loop over propagators
            for propagator_index in range(number_of_propagators):

                # Get current propagator, and define propagation settings
                current_propagator = available_propagators[propagator_index]

                mc_results_per_integrator = dict()

                # Loop over different integrators
                for integrator_index in range(int(number_of_integrators)):

                    mc_results_per_step_size = dict()

                    for step_tolerance_index in range(3):
                        
                        if propagator_index == 0:
                           tol_idx = step_tolerance_index + 1
                        
                        tolerance = tolerances[tol_idx]

                        func_eval_list = []
                        max_error_list = []

                        for sample in monte_carlo_samples:

                            shape_parameters = sample
                            bodies = environment_setup.create_system_of_bodies(body_settings)
                            Util.add_capsule_to_body_system(bodies,
                                                            shape_parameters,
                                                            capsule_density)


                            current_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                            bodies,
                                                                            simulation_start_epoch,
                                                                            termination_settings,
                                                                            dependent_variables_to_save,
                                                                            current_propagator )
                            current_integrator_settings = Util.get_integrator_settings(current_propagator_settings,
                                                                                    integrator_index,
                                                                                    tolerance)
                        
                        
                            # Create Shape Optimization Problem object
                            dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                                bodies, current_integrator_settings )

                            ### OUTPUT OF THE SIMULATION ###
                            # Retrieve propagated state and dependent variables
                            state_history = dynamics_simulator.state_history
                            dependent_variable_history = dynamics_simulator.dependent_variable_history
                            function_evaluation_dict = dynamics_simulator.cumulative_number_of_function_evaluations
                            number_of_function_evaluations = list(function_evaluation_dict.values())[-1]
                            func_eval_list.append(number_of_function_evaluations)
                            

                            # Compare the simulation to the benchmarks and write differences to files
                            if use_benchmark:

                                # Generate benchmark solution with fixed step size previously found
                                benchmark_step_size = 0.16 # s
                                benchmark_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                                            bodies,
                                                                                            simulation_start_epoch,
                                                                                            termination_settings,
                                                                                            dependent_variables_to_save)
                                
                                benchmark_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
                                    benchmark_step_size,
                                    propagation_setup.integrator.CoefficientSets.rkf_56)
                                benchmark_propagator_settings.print_settings.print_dependent_variable_indices = True

                                benchmark_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                                    bodies,
                                    benchmark_propagator_settings )
                                
                                benchmark_state_history = benchmark_dynamics_simulator.state_history

                                # Initialize containers
                                state_difference = dict()

                                # Loop over the propagated states and use the benchmark interpolators
                                # NOTE TO STUDENTS: it can happen that the benchmark ends earlier than the regular simulation, due to
                                # the shorter step size. Therefore, the following lines of code will be forced to extrapolate the
                                # benchmark states (or dependent variables), producing a warning. Be aware of it!
                                benchmark_difference = Util.compare_benchmarks_no_extrapolation(benchmark_state_history,
                                                                                                state_history,                                                           
                                                                                                None,
                                                                                                None)

                                extrapolation_threshold = list(benchmark_state_history.keys())[-1]
                                
                                position_difference = {epoch: np.linalg.norm(benchmark_difference[epoch][:3]) for epoch in benchmark_difference.keys() if epoch < extrapolation_threshold}
                                time = np.array(list(position_difference.keys()))
                                max_error = max(position_difference.values())
                                max_error_list.append(max_error)

                                """
                                # Plot position difference vs time
                                plt.plot(time, position_difference.values(), linewidth=2, label=f"Integrator {integrator_index}, Step/Tolerance {step_tolerance_index}")
                                if integrator_index < 4:
                                    plt.title(f'Propagator: {names_of_propagators[propagator_index]}, Integrator: {names_of_integrators[integrator_index]}, Tolerance: {tolerances[step_tolerance_index]:.0e}', fontsize=16)
                                else:
                                    plt.title(f'Propagator: {names_of_propagators[propagator_index]}, Integrator: {names_of_integrators[integrator_index]}, Step Size: {step_sizes[step_tolerance_index]} s', fontsize=16)
                                plt.show()
                                plt.close()
                                """


                        mc_results_per_step_size[step_tolerance_index] = [func_eval_list, max_error_list]
                        
                    mc_results_per_integrator[integrator_index] = mc_results_per_step_size
                    
                mc_results_per_propopagator[propagator_index] = mc_results_per_integrator




            # Create a single plot for Encke
            fig, ax = plt.subplots(figsize=(10, 7))

            # Only one propagator: Encke
            propagator_index = 0
            propagator_name = 'Encke'
            mc_results_per_integrator = mc_results_per_propopagator[propagator_index]

            colors = ['blue', 'red']
            markers = ['o', 's', '^']

            # Loop through integrators
            for integrator_index in range(number_of_integrators):
                mc_results_per_step_size = mc_results_per_integrator[integrator_index]
                # Loop through step sizes (3 for Encke)
                for step_size_index in range(3):
                    mc_results = mc_results_per_step_size[step_size_index]
                    func_eval_list = mc_results[0]
                    max_error_list = mc_results[1]

                    if integrator_index == 0:
                        tol_idx = step_size_index + 1
                    else:
                        tol_idx = step_size_index 

                    ax.scatter(
                        func_eval_list,
                        max_error_list,
                        marker=markers[step_size_index % len(markers)],
                        color=colors[integrator_index % len(colors)],
                        s=60,
                        label=f"{names_of_integrators[integrator_index]}, Tol: {tolerances[tol_idx]:.0e}",
                        edgecolor='black'
                    )

            ax.set_title(f"{propagator_name}", fontsize=20)
            ax.set_xlabel("Number of Function Evaluations [-]", fontsize=20)
            ax.set_ylabel("Maximum Position Error [m]", fontsize=20)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True)
            ax.tick_params(axis="both", which="major", labelsize=18)
            ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, label='1 m requirement')
            ax.legend(fontsize=14, loc='best')




            # Adjust layout and show the plot
            plt.tight_layout()
            plt.show()
            # Save the plot in the current directory
            output_plot_path = os.path.join(current_dir, "monte_carlo_analysis_plot.png")
            plt.savefig(output_plot_path, bbox_inches="tight", dpi=300)



        