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
First name: ***COMPLETE HERE***
Last name: ***COMPLETE HERE***
Student number: ***COMPLETE HERE***

This module is the main script that executes the propagation and optimization. It relies on two other modules, defined
for a more practical organization of functions and classes, which are imported below.

This function computes the dynamics of a capsule re-entering the atmosphere of the Earth, using a variety of integrator
and propagator settings (see comments under "RUN SIMULATION FOR VARIOUS SETTINGS"). For each run, the differences w.r.t. 
a benchmark propagation are computed, providing a proxy for setting quality.

The vehicle starts 120 km above the surface of the planet, with a speed of 7.83 km/s in an Earth-fixed frame (see
getInitialState function).

The propagation is terminated as soon as one of the following conditions is met (see 
get_propagation_termination_settings() function):
- Altitude < 25 km
- Propagation time > 24 hr

This propagation assumes only point mass gravity by the Earth and aerodynamic accelerations.

The trajectory of the capsule is heavily dependent on the shape and orientation of the vehicle. Here, the shape is
determined here by the five parameters, which are used to compute the aerodynamic accelerations on the vehicle using a 
modified Newtonian flow (see also Dirkx and Mooij, 2018). The bank angle and sideslip angles are set to zero.
The vehicle shape and angle of attack are defined by values in the vector shapeParameters.

The entries of the vector 'shapeParameters' contains the following:
- Entry 0:  Nose radius
- Entry 1:  Middle radius
- Entry 2:  Rear length
- Entry 3:  Rear angle
- Entry 4:  Side radius
- Entry 5:  Constant Angle of Attack

The benchmark is run if the variable use_benchmark is True.
The output is written if the variable write_results_to_file is true.

Details on the outputs written by this file can be found:
- benchmark data: comments for 'generateBenchmarks' function
- results for integrator/propagator variations: comments under "RUN SIMULATION FOR VARIOUS SETTINGS"

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

import sys
sys.path.insert(0, '/home/dominic/Software/tudat-bundle/build-tudat-bundle-Desktop-Default/tudatpy/')

# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators

# Problem-specific imports
import ShapeOptimizationUtilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()
# NOTE TO STUDENTS: INPUT YOUR PARAMETER SET HERE, FROM THE INPUT FILES
# ON BRIGHTSPACE, FOR YOUR SPECIFIC STUDENT NUMBER
shape_parameters = [8.148730872315355,
                    2.720324489288032,
                    0.2270385167794302,
                    -0.4037530896422072,
                    0.2781438040896319,
                    0.4559143679738996]
# Choose whether benchmark is run
use_benchmark = True
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
# N.B.: all the bodies added after this function is called will automatically
# be placed in the same reference frame, which is the same for the full
# system of bodies
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
are_dependent_variables_to_save = False if not dependent_variables_to_save else True


###########################################################################
# IF DESIRED, GENERATE BENCHMARK ##########################################
###########################################################################

# NOTE TO STUDENTS: MODIFY THE CODE INSIDE THIS "IF STATEMENT" (AND CALLED FUNCTIONS, IF NEEDED)
# TO ASSESS THE QUALITY OF VARIOUS BENCHMARK SETTINGS
if use_benchmark:
    # Define benchmark interpolator settings to make a comparison between the two benchmarks
    benchmark_interpolator_settings = interpolators.lagrange_interpolation(
        8,boundary_interpolation = interpolators.extrapolate_at_boundary)

    # Create propagator settings for benchmark (Cowell)
    benchmark_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                 bodies,
                                                                 simulation_start_epoch,
                                                                 termination_settings,
                                                                 dependent_variables_to_save )
    # Set output path for the benchmarks
    benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/' if write_results_to_file else None

    # Generate benchmarks
    benchmark_time_step = 4.0
    benchmark_list = Util.generate_benchmarks(benchmark_time_step,
                                              simulation_start_epoch,
                                              bodies,
                                              benchmark_propagator_settings,
                                              are_dependent_variables_to_save,
                                              benchmark_output_path)

    # Extract benchmark states (first one is run with benchmark_time_step; second with 2.0*benchmark_time_step)
    first_benchmark_state_history = benchmark_list[0]
    second_benchmark_state_history = benchmark_list[1]
    # Create state interpolator for first benchmark
    benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_benchmark_state_history,
        benchmark_interpolator_settings)

    # Compare benchmark states, returning interpolator of the first benchmark, and writing difference to file if
    # write_results_to_file is set to True
    benchmark_state_difference = Util.compare_benchmarks(first_benchmark_state_history,
                                                         second_benchmark_state_history,
                                                         benchmark_output_path,
                                                         'benchmarks_state_difference.dat')

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
# Define list of propagators
available_propagators = [propagation_setup.propagator.cowell,
                         propagation_setup.propagator.encke,
                         propagation_setup.propagator.gauss_keplerian,
                         propagation_setup.propagator.gauss_modified_equinoctial,
                         propagation_setup.propagator.unified_state_model_quaternions,
                         propagation_setup.propagator.unified_state_model_modified_rodrigues_parameters,
                         propagation_setup.propagator.unified_state_model_exponential_map]

# Define settings to loop over
number_of_propagators = len(available_propagators)
number_of_integrators = 5

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
        # For RK4, more step sizes are used. NOTE TO STUDENTS, MODIFY THESE AS YOU SEE FIT!
        if integrator_index > 3:
            number_of_integrator_step_size_settings = 6
        else:
            number_of_integrator_step_size_settings = 4

        # Loop over all tolerances / step sizes
        for step_size_index in range(number_of_integrator_step_size_settings):
            # Print status
            to_print = 'Current run: \n propagator_index = ' + str(propagator_index) + \
                       '\n integrator_index = ' + str(integrator_index) \
                       + '\n step_size_index = ' + str(step_size_index)
            print(to_print)
            # Set output path
            output_path = current_dir + '/SimulationOutput/prop_' + str(propagator_index) + \
                          '/int_' + str(integrator_index) + '/step_size_' + str(step_size_index) + '/'

            # Create integrator settings
            current_integrator_settings = Util.get_integrator_settings(propagator_index,
                                                                       integrator_index,
                                                                       step_size_index,
                                                                       simulation_start_epoch)
            # Create Shape Optimization Problem object
            dynamics_simulator = numerical_simulation.SingleArcSimulator(
                bodies, current_integrator_settings, current_propagator_settings, print_dependent_variable_data=False )


            ### OUTPUT OF THE SIMULATION ###
            # Retrieve propagated state and dependent variables
            state_history = dynamics_simulator.state_history
            unprocessed_state_history = dynamics_simulator.unprocessed_state_history
            dependent_variable_history = dynamics_simulator.dependent_variable_history

            # Get the number of function evaluations (for comparison of different integrators)
            function_evaluation_dict = dynamics_simulator.cumulative_number_of_function_evaluations
            number_of_function_evaluations = list(function_evaluation_dict.values())[-1]
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

            # Compare the simulation to the benchmarks and write differences to files
            if use_benchmark:
                # Initialize containers
                state_difference = dict()

                # Loop over the propagated states and use the benchmark interpolators
                # NOTE TO STUDENTS: it can happen that the benchmark ends earlier than the regular simulation, due to
                # the shorter step size. Therefore, the following lines of code will be forced to extrapolate the
                # benchmark states (or dependent variables), producing a warning. Be aware of it!
                for epoch in state_history.keys():
                    state_difference[epoch] = state_history[epoch] - benchmark_state_interpolator.interpolate(epoch)

                # Write differences with respect to the benchmarks to files
                if write_results_to_file:
                    save2txt(state_difference, 'state_difference_wrt_benchmark.dat', output_path)

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

# Print the ancillary information
print('\n### ANCILLARY SIMULATION INFORMATION ###')
for (elem, (info, result)) in enumerate(dict_to_write.items()):
    if elem > 1:
        print(info + ': ' + str(result))
