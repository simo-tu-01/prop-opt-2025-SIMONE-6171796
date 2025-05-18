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

# Choose which question(s) to run
question_1 = False
question_2 = False
question_3 = True
question_4 = False

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
    benchmark_time_steps = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48] # [1/16, 1/8, 1/4, 1/2, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]  # s
    max_errors = np.empty(len(benchmark_time_steps))
    max_interpol_errors = np.empty(len(benchmark_time_steps))

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
        ls = '--' if i < 4 else '-.' if i > 9 else '-'
        plt.plot(time, position_difference, linewidth=2.5, label=f"{2*benchmark_time_step}", linestyle=ls)

    plt.xlabel('Time [s]', fontsize=20)
    plt.ylabel(r'$||\epsilon_r (t,\Delta t)||$ [m]', fontsize=20)
    plt.yscale('log')
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
    plt.axhline(y=10, color='r', linestyle='--', linewidth=1.5, label='10 m threshold')
    plt.legend(fontsize=15)
    plt.show()


    relative_interpolation_error = abs(max_interpol_errors - max_errors) # / max_errors

    plt.figure(figsize=(10, 6))
    plt.plot([2 * step for step in benchmark_time_steps], relative_interpolation_error, linewidth=2, marker='o', label='Relative Interpolation Error')
    plt.xlabel('Step Size [s]', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([2 * step for step in benchmark_time_steps], labels=[str(step) for step in [2 * step for step in benchmark_time_steps]], fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=15)
    plt.show()


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
        plt.axhline(y=10, color='r', linestyle='--', linewidth=1.5, label='10 m threshold')
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

        step_size = 5.12# s

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
            dependent_variables_per_popagator[names_of_propagators[propagator_index]] = dependent_variables_dict


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
            max_errors[propagator_index] = np.max(benchmark_position_difference)

            markers = ['o', 's', 'D', '^', 'v', '<', '>']  # Define a list of markers
            plt.plot(benchmark_difference.keys(), benchmark_position_difference, linewidth=2, 
                     label=f"{names_of_propagators[propagator_index]}", 
                     marker='o',             # markers[propagator_index % len(markers)],
                     markersize=6, 
                     markevery=(propagator_index * 2, 14))  # Offset markers for each curve

        extrapolation_threshold = list(benchmark_state_history.keys())[-1]

        plt.xlabel('Time [s]', fontsize=20)
        plt.ylabel(r'$||\epsilon_r (t,\Delta t)||$ [m]', fontsize=20)
        plt.yscale('log')
        plt.axvline(x=extrapolation_threshold, color='k', linestyle='--', linewidth=1.5, label='Extrapolation threshold')
        plt.legend(fontsize=15, loc='upper left')
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.grid(True)
        plt.show()

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

        dependent_cowell = dependent_variables_per_popagator['Cowell']
        dependent_enke = dependent_variables_per_popagator['Encke']
        dependent_gauss = dependent_variables_per_popagator['Kepler (Gauss)']
        dependent_gme = dependent_variables_per_popagator['MEE']
        dependent_usmq = dependent_variables_per_popagator['USM6']
        dependent_usmmrp = dependent_variables_per_popagator['USM7']
        dependent_usmem = dependent_variables_per_popagator['USM-EM']

        # Util.plot_cowell_state_elements(elements_cowell)
        # Util.plot_enke_state_elements(elements_enke)
        # Util.plot_gauss_keplerian_elements(elements_gauss)

        Util.plot_gme_elements(elements_gme)
        Util.plot_cowell_state_elements(state_gme)
        Util.plot_dependent_variables(dependent_gme)

        # Interpolate the benchmark at the time points of state_usmq
        benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            benchmark_state_history,
            interpolators.lagrange_interpolation(8, boundary_interpolation=interpolators.extrapolate_at_boundary_with_warning)
        )

        kepler_gme = dependent_gme['kepler_elements']
        kepler_benchmark = benchmark_dependent_variables_dict['kepler_elements']

        # Interpolate the benchmark Kepler elements at the time points of kepler_gme
        benchmark_interpolator_kepler = interpolators.create_one_dimensional_vector_interpolator(
            kepler_benchmark,
            interpolators.lagrange_interpolation(8, boundary_interpolation=interpolators.extrapolate_at_boundary_with_warning)
        )

        interpolated_benchmark_kepler = {epoch: benchmark_interpolator_kepler.interpolate(epoch) for epoch in kepler_gme.keys() if epoch < extrapolation_threshold}

        # Compute the error between the GME propagator's Kepler elements and the interpolated benchmark
        kepler_error = {epoch: (kepler_gme[epoch] - interpolated_benchmark_kepler[epoch]) / interpolated_benchmark_kepler[epoch] for epoch in interpolated_benchmark_kepler.keys()}

        Util.plot_kepler_elements(kepler_error)

        gme_benchmark = dict()

        for epoch in (epoch for epoch in elements_gme.keys() if epoch < extrapolation_threshold):
            gme_benchmark[epoch] = astro.element_conversion.cartesian_to_mee(benchmark_interpolator.interpolate(epoch), bodies.get_body('Earth').gravitational_parameter)

        gme_error = {epoch: (elements_gme[epoch] - gme_benchmark[epoch]) / gme_benchmark[epoch] for epoch in gme_benchmark.keys()}

        Util.plot_gme_elements(gme_error)

        
        Util.plot_usm_quaternions_elements(elements_usmq)
        Util.plot_dependent_variables(dependent_usmq)
        # Util.plot_cowell_state_elements(state_usmq)

        interpolated_benchmark_states = {epoch: benchmark_interpolator.interpolate(epoch) for epoch in state_usmq.keys()}

        # Plot the velocity norm with time for the USM6 propagator and interpolated benchmark
        velocity_norm_usmq = [np.linalg.norm(state[3:]) for state in state_usmq.values()]
        velocity_norm_benchmark = [np.linalg.norm(state[3:]) for state in interpolated_benchmark_states.values()]
        time_usmq = list(state_usmq.keys())

        plt.figure(figsize=(10, 6))
        plt.plot(time_usmq, velocity_norm_usmq, linewidth=2, label='USM6 Velocity Norm')
        plt.plot(time_usmq, velocity_norm_benchmark, linewidth=2, linestyle='--', label='Interpolated Benchmark Velocity Norm')
        plt.xlabel('Time [s]', fontsize=20)
        plt.ylabel('Velocity Norm [m/s]', fontsize=20)
        plt.yscale('log')
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.title('Velocity Norm vs Time', fontsize=20)
        plt.show()

        # Plot the position norm with time for the USM6 propagator and interpolated benchmark
        position_norm_usmq = [np.linalg.norm(state[:3]) for state in state_usmq.values()]
        position_norm_benchmark = [np.linalg.norm(state[:3]) for state in interpolated_benchmark_states.values()]

        plt.figure(figsize=(10, 6))
        plt.plot(time_usmq, position_norm_usmq, linewidth=2, label='USM6 Position Norm')
        plt.plot(time_usmq, position_norm_benchmark, linewidth=2, linestyle='--', label='Interpolated Benchmark Position Norm')
        plt.xlabel('Time [s]', fontsize=20)
        plt.ylabel('Position Norm [m]', fontsize=20)
        plt.yscale('log')
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.title('Position Norm vs Time', fontsize=20)
        plt.show()

    
        #Util.plot_usm_mrp_elements(elements_usmmrp)
        #Util.plot_dependent_variables(dependent_usmmrp)
        #Util.plot_cowell_state_elements(state_usmmrp)

        #Util.plot_usmem_elements(elements_usmem)
        #Util.plot_dependent_variables(dependent_usmem)
        #Util.plot_cowell_state_elements(state_usmem)
            
    if question_3:

        step_sizes = [2.56, 5.12, 10.24, 20.48, 40.96] # s
        tolerances = [1.0E-14, 1.0E-12, 1.0E-10, 1.0E-8, 1.0E-6] # s

        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P']  # Define a list of markers
        linestyle_options = ['-', '--', '-.', ':']  # Define line styles 
        

        # Define list of propagators
        available_propagators = [propagation_setup.propagator.cowell,
                                propagation_setup.propagator.encke,
                                propagation_setup.propagator.gauss_modified_equinoctial]
        number_of_propagators = len(available_propagators)
        names_of_propagators = ['Cowell', 'Encke', 'MEE']

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

            # Loop over different integrators
            for integrator_index in range(number_of_integrators):

                function_evaluations_sorted = []
                max_position_differences_sorted = []

                # Loop over all tolerances / step sizes
                for step_tolerance_index in range(5):
                    # Print status
                    to_print = 'Current run: \n propagator_index = ' + str(propagator_index) + \
                            '\n integrator_index = ' + str(integrator_index) \
                            + '\n step_size_index = ' + str(step_tolerance_index)
                    print(to_print)
                    # Set output path
                    output_path = current_dir + '/SimulationOutput/prop_' + str(propagator_index) + \
                                '/int_' + str(integrator_index) + '/step_size_' + str(step_tolerance_index) + '/'
                    

                    if integrator_index < 4:                        
                        tolerance = tolerances[step_tolerance_index]
                        # Create integrator settings
                        current_integrator_settings = Util.get_integrator_settings(current_propagator_settings,
                                                                                integrator_index,
                                                                                tolerance)
                    elif 4 <= integrator_index <= 7:     
                        step_size = step_sizes[step_tolerance_index]
                        # Create integrator settings
                        current_integrator_settings = Util.get_integrator_settings(current_propagator_settings,
                                                                                integrator_index,
                                                                                step_size)

                    
                    # Create Shape Optimization Problem object
                    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                        bodies, current_propagator_settings )

                    ### OUTPUT OF THE SIMULATION ###
                    # Retrieve propagated state and dependent variables
                    state_history = dynamics_simulator.state_history
                    unprocessed_state_history = dynamics_simulator.unprocessed_state_history
                    dependent_variable_history = dynamics_simulator.dependent_variable_history

                    # Get the number of function evaluations (for comparison of different integrators)
                    function_evaluation_dict = dynamics_simulator.cumulative_number_of_function_evaluations
                    number_of_function_evaluations = list(function_evaluation_dict.values())[-1]
                    function_evaluations_sorted.append(number_of_function_evaluations)

                    # Add it to a dictionary
                    dict_to_write = {'Number of function evaluations (ignore the line above)': number_of_function_evaluations}
                    # Check if the propagation was run successfully
                    propagation_outcome = dynamics_simulator.integration_completed_successfully
                    dict_to_write['Propagation run successfully'] = propagation_outcome
                    # Note if results were written to files
                    dict_to_write['Results written to file'] = write_results_to_file
                    # Note if benchmark was run
                    dict_to_write['Benchmark run'] = use_benchmark
                    # Note if dependent variables were present
                    dict_to_write['Dependent variables present'] = are_dependent_variables_to_save

                    # Save results to a file
                    if write_results_to_file:
                        save2txt(state_history, 'state_history.dat', output_path)
                        save2txt(unprocessed_state_history, 'unprocessed_state_history.dat', output_path)
                        save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)
                        save2txt(dict_to_write, 'ancillary_simulation_info.txt',   output_path)
                        environment.save_vehicle_mesh_to_file(
                            bodies.get_body('Capsule').aerodynamic_coefficient_interface, output_path)

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

                        max_error = max(position_difference.values())
                        max_position_differences_sorted.append(max_error)

                        # Write differences with respect to the benchmarks to files
                        if write_results_to_file:
                            save2txt(benchmark_difference, 'state_difference_wrt_benchmark.dat', output_path)

                        # Do the same for dependent variables, if present
                        if are_dependent_variables_to_save:
                            # Initialize containers
                            dependent_difference = dict()
                            # Loop over the propagated dependent variables and use the benchmark interpolators
                            for epoch in dependent_variable_history.keys():
                                dependent_difference[epoch] = dependent_variable_history[epoch] - benchmark_dependent_variable_interpolator.interpolate(epoch)
                            # Write differences with respect to the benchmarks to files
                            if write_results_to_file:
                                save2txt(dependent_difference, 'dependent_variable_difference_wrt_benchmark.dat',   output_path)


                    
                    # Use different colormaps for step sizes and tolerances
                    if integrator_index < 4:
                        color_map = plt.cm.tab10  # Colormap for tolerances
                        color = color_map(step_tolerance_index / len(tolerances))
                        label = f'Tolerance: {tolerances[step_tolerance_index]:.0e}' if integrator_index == 0 else None
                    else:
                        color_map = plt.cm.Dark2  # Colormap for step sizes
                        color = color_map(step_tolerance_index / len(step_sizes))
                        label = f'Step Size: {step_sizes[step_tolerance_index]:.2f}s' if integrator_index == 4 else None
                    
                    # Plot the scatter point with the corresponding color and label
                    plt.scatter(number_of_function_evaluations, max_error, 
                                marker=markers[integrator_index % len(markers)], 
                                color=color, label=label,  s=100, zorder=3)
                    
                linestyle = linestyle_options[integrator_index % len(linestyle_options)]
                # Plot the line connecting the points for the current integrator
                plt.plot(
                    function_evaluations_sorted,
                    max_position_differences_sorted,
                    linestyle=linestyle,
                    color='black',
                    linewidth=1.5,
                    zorder=2,
                )

            # Create a new plot for each propagator
            plt.title(f'Performance Analysis for {names_of_propagators[propagator_index]} Propagator', fontsize=20)

            # Add legends for step size
            step_size_legend_handles = [
                plt.Line2D([0], [0], color=plt.cm.Dark2(i / len(step_sizes)), lw=4, label=f'{step_size} s')
                for i, step_size in enumerate(step_sizes)
            ]
            step_size_legend = plt.legend(handles=step_size_legend_handles, title='Step Size:', title_fontsize=15, fontsize=15, loc='upper left', bbox_to_anchor=(1.02, 1.035))
            plt.gca().add_artist(step_size_legend)

            # Add legends for tolerance
            tolerance_legend_handles = [
                plt.Line2D([0], [0], color=plt.cm.tab10(i / len(tolerances)), lw=4, label=f'{tolerance:.0e}')
                for i, tolerance in enumerate(tolerances)
            ]
            tolerance_legend = plt.legend(handles=tolerance_legend_handles, title='Tolerance:', title_fontsize=15, fontsize=15, loc='upper left', bbox_to_anchor=(1.02, 0.7))
            plt.gca().add_artist(tolerance_legend)

            # Add legends for integrator types with markers and linestyles
            integrator_legend_handles = [
                plt.Line2D([0], [0], color='black', marker=markers[i % len(markers)], linestyle=linestyle_options[i % len(linestyle_options)], markersize=10, label=names_of_integrators[i])
                for i in range(number_of_integrators)
            ]
            integrator_legend = plt.legend(handles=integrator_legend_handles, title='Integrator Types:', title_fontsize=15, fontsize=15, loc='upper left', bbox_to_anchor=(1.02, 0.4))
            plt.gca().add_artist(integrator_legend)

            plt.xlabel('Cumulative Number of Function Evaluations [-]', fontsize=20)
            plt.ylabel('Maximum Position Difference [m]', fontsize=20)
            plt.yscale('log')
            plt.xscale('log')
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.axhline(y=10, color='black', linestyle='--', linewidth=1.5)
            plt.grid()
            plt.show()



        # Print the ancillary information
        print('\n### ANCILLARY SIMULATION INFORMATION ###')
        for (elem, (info, result)) in enumerate(dict_to_write.items()):
            if elem > 1:
                print(info + ': ' + str(result))
