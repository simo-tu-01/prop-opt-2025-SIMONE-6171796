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
- files defining the points and surface normals of the mesg used for the aerodynamic analysis (save_vehicle_mesh_to_file)

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

import sys
sys.path.insert(0, '/home/dominic/Tudat/tudat-bundle/tudat-bundle/cmake-build-default/tudatpy')

# General imports
import os

# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators

# Problem-specific imports
import CapsuleEntryUtilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()
parameter_vector = [[7.067287765, 2.4170219984, 2.2363750229, -0.5273354261, 0.4838446208, 0.1162353527],
[8.1487308723, 2.7203244893, 0.2270385168, -0.4037530896, 0.2781438041, 0.455914368],
[7.041740653, 2.3023325677, 2.2330797267, -0.558728029, 0.3602598373, 0.4809835448],
[9.0071362922, 2.9990405154, 2.4752705714, -0.512795019, 0.3084274431, 0.5128362015],
[6.2537561791, 2.1467558926, 2.1598022277, -0.2586121879, 0.3518871239, 0.2557315011],
[7.6983117481, 2.0923385955, 1.7186406196, -0.255984141, 0.1158838553, 0.3203083369],
[3.8686343422, 2.6697460404, 0.6877576649, -0.7652400717, 0.3522259173, 0.2548030601],
[5.2722659144, 2.9355390726, 2.4773619827, -0.8864559158, 0.4525574779, 0.5187926269],
[6.6048232258, 2.8463109182, 3.2004148801, -0.4142178644, 0.3201519188, 0.4232153013],
[6.6198516141, 2.5245481627, 3.4468273757, -0.6040103968, 0.290454681, 0.4290749526],
[4.0663404303, 2.6704675076, 3.9481422229, -0.495868522, 0.302693632, 0.1435113375],
[7.7131171976, 2.9139620224, 1.3666405992, -0.242625805, 0.4078543976, 0.1068951771],
[6.1495506149, 2.918920762, 1.3073884546, -0.453146517, 0.4924978005, 0.3022936508],
[5.2048962004, 2.130641664, 0.9266435643, -0.803848226, 0.2672311423, 0.4606871979]
]
for iteration in range(len(parameter_vector)):

# NOTE TO STUDENTS: INPUT YOUR PARAMETER SET HERE, FROM THE INPUT FILES
# ON BRIGHTSPACE, FOR YOUR SPECIFIC STUDENT NUMBER
shape_parameters = parameter_vector[ iteration ]
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


# Propagate dynamics
current_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                 bodies,
                                                                 simulation_start_epoch,
                                                                 termination_settings,
                                                                 dependent_variables_to_save )
current_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
    4.0,
    propagation_setup.integrator.CoefficientSets.rkdp_87)

dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, current_propagator_settings)

### OUTPUT OF THE SIMULATION ###
# Retrieve propagated state and dependent variables
state_history = dynamics_simulator.state_history
unprocessed_state_history = dynamics_simulator.unprocessed_state_history
dependent_variable_history = dynamics_simulator.dependent_variable_history

output_path = current_dir + '/SimulationOutput/MonteCarlo/'
save2txt(dependent_variable_history, 'dependent_variable_history'+str(iteration)+'.dat', output_path)

#
# ###########################################################################
# # IF DESIRED, GENERATE BENCHMARK ##########################################
# ###########################################################################
#
# # NOTE TO STUDENTS: MODIFY THE CODE INSIDE THIS "IF STATEMENT" (AND CALLED FUNCTIONS, IF NEEDED)
# # TO ASSESS THE QUALITY OF VARIOUS BENCHMARK SETTINGS
# if use_benchmark:
#     # Define benchmark interpolator settings to make a comparison between the two benchmarks
#     benchmark_interpolator_settings = interpolators.lagrange_interpolation(
#         8,boundary_interpolation = interpolators.extrapolate_at_boundary)
#
#     # Create propagator settings for benchmark (Cowell)
#     benchmark_propagator_settings = Util.get_propagator_settings(shape_parameters,
#                                                                  bodies,
#                                                                  simulation_start_epoch,
#                                                                  termination_settings,
#                                                                  dependent_variables_to_save )
#     # Set output path for the benchmarks
#     benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/' if write_results_to_file else None
#
#     # Generate benchmarks
#     benchmark_time_step = 4.0
#     benchmark_list = Util.generate_benchmarks(benchmark_time_step,
#                                               simulation_start_epoch,
#                                               bodies,
#                                               benchmark_propagator_settings,
#                                               are_dependent_variables_to_save,
#                                               benchmark_output_path)
#
#     # Extract benchmark states (first one is run with benchmark_time_step; second with 2.0*benchmark_time_step)
#     first_benchmark_state_history = benchmark_list[0]
#     second_benchmark_state_history = benchmark_list[1]
#     # Create state interpolator for first benchmark
#     benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
#         first_benchmark_state_history,
#         benchmark_interpolator_settings)
#
#     # Compare benchmark states, returning interpolator of the first benchmark, and writing difference to file if
#     # write_results_to_file is set to True
#     benchmark_state_difference = Util.compare_benchmarks(first_benchmark_state_history,
#                                                          second_benchmark_state_history,
#                                                          benchmark_output_path,
#                                                          'benchmarks_state_difference.dat')
#
#     # Extract benchmark dependent variables, if present
#     if are_dependent_variables_to_save:
#         first_benchmark_dependent_variable_history = benchmark_list[2]
#         second_benchmark_dependent_variable_history = benchmark_list[3]
#
#         # Create dependent variable interpolator for first benchmark
#         benchmark_dependent_variable_interpolator = interpolators.create_one_dimensional_vector_interpolator(
#             first_benchmark_dependent_variable_history,
#             benchmark_interpolator_settings)
#
#         # Compare benchmark dependent variables, returning interpolator of the first benchmark, and writing difference
#         # to file if write_results_to_file is set to True
#         benchmark_dependent_difference = Util.compare_benchmarks(first_benchmark_dependent_variable_history,
#                                                                  second_benchmark_dependent_variable_history,
#                                                                  benchmark_output_path,
#                                                                  'benchmarks_dependent_variable_difference.dat')
#
# ###########################################################################
# # RUN SIMULATION FOR VARIOUS SETTINGS #####################################
# ###########################################################################
# """
# Code below propagates states using each propagator (propagator_index=0..6), four multi-stage variable step-size
# integrators (integrator_index j=0..3) and an RK4 integrator (j=4). For the variable-step integrators, 4 different
# tolerances are used (step_size_index=0..3). For the RK4, 6 different step sizes are used (step_size_index=0..5),
# see use of number_of_integrator_step_size_settings variable. See get_integrator_settings function for more details.
#
# For each combination of i, j, and k, results are written to directory:
#     ShapeOptimization/SimulationOutput/prop_i/int_j/setting_k/
#
# Specifically:
#      state_History.dat                                  Cartesian states as function of time
#      dependent_variable_history.dat                     Dependent variables as function of time
#      state_difference_wrt_benchmark.dat                 Difference of dependent variables w.r.t. benchmark
#      dependent_variable_difference_wrt_benchmark.dat    Difference of states w.r.t. benchmark
#      ancillary_simulation_info.dat                      Other information about the propagation (number of function
#                                                         evaluations, etc...)
#
# NOTE TO STUDENTS: THE NUMBER, TYPES, SETTINGS OF PROPAGATORS/INTEGRATORS/INTEGRATOR STEPS,TOLERANCES,ETC. SHOULD BE
# MODIFIED FOR ASSIGNMENT 1.
# """
# # Define list of propagators
# available_propagators = [propagation_setup.propagator.cowell,
#                          propagation_setup.propagator.encke,
#                          propagation_setup.propagator.gauss_keplerian,
#                          propagation_setup.propagator.gauss_modified_equinoctial,
#                          propagation_setup.propagator.unified_state_model_quaternions,
#                          propagation_setup.propagator.unified_state_model_modified_rodrigues_parameters,
#                          propagation_setup.propagator.unified_state_model_exponential_map]
#
# # Define settings to loop over
# number_of_propagators = len(available_propagators)
# number_of_integrators = 5
#
# # Loop over propagators
# for propagator_index in range(number_of_propagators):
#
#     # Get current propagator, and define propagation settings
#     current_propagator = available_propagators[propagator_index]
#     current_propagator_settings = Util.get_propagator_settings(shape_parameters,
#                                                                bodies,
#                                                                simulation_start_epoch,
#                                                                termination_settings,
#                                                                dependent_variables_to_save,
#                                                                current_propagator )
#
#     # Loop over different integrators
#     for integrator_index in range(number_of_integrators):
#         # For RK4, more step sizes are used. NOTE TO STUDENTS, MODIFY THESE AS YOU SEE FIT!
#         if integrator_index > 3:
#             number_of_integrator_step_size_settings = 6
#         else:
#             number_of_integrator_step_size_settings = 4
#
#         # Loop over all tolerances / step sizes
#         for step_size_index in range(number_of_integrator_step_size_settings):
#             # Print status
#             to_print = 'Current run: \n propagator_index = ' + str(propagator_index) + \
#                        '\n integrator_index = ' + str(integrator_index) \
#                        + '\n step_size_index = ' + str(step_size_index)
#             print(to_print)
#             # Set output path
#             output_path = current_dir + '/SimulationOutput/prop_' + str(propagator_index) + \
#                           '/int_' + str(integrator_index) + '/step_size_' + str(step_size_index) + '/'
#
#             # Create integrator settings
#             current_integrator_settings = Util.get_integrator_settings(propagator_index,
#                                                                        integrator_index,
#                                                                        step_size_index,
#                                                                        simulation_start_epoch)
#             current_propagator_settings.integrator_settings = current_integrator_settings
#             # Create Shape Optimization Problem object
#             dynamics_simulator = numerical_simulation.create_dynamics_simulator(
#                 bodies, current_propagator_settings )
#
#
#             ### OUTPUT OF THE SIMULATION ###
#             # Retrieve propagated state and dependent variables
#             state_history = dynamics_simulator.state_history
#             unprocessed_state_history = dynamics_simulator.unprocessed_state_history
#             dependent_variable_history = dynamics_simulator.dependent_variable_history
#
#             # Get the number of function evaluations (for comparison of different integrators)
#             function_evaluation_dict = dynamics_simulator.cumulative_number_of_function_evaluations
#             number_of_function_evaluations = list(function_evaluation_dict.values())[-1]
#             # Add it to a dictionary
#             dict_to_write = {'Number of function evaluations (ignore the line above)': number_of_function_evaluations}
#             # Check if the propagation was run successfully
#             propagation_outcome = dynamics_simulator.integration_completed_successfully
#             dict_to_write['Propagation run successfully'] = propagation_outcome
#             # Note if results were written to files
#             dict_to_write['Results written to file'] = write_results_to_file
#             # Note if benchmark was run
#             dict_to_write['Benchmark run'] = use_benchmark
#             # Note if dependent variables were present
#             dict_to_write['Dependent variables present'] = are_dependent_variables_to_save
#
#             # Save results to a file
#             if write_results_to_file:
#                 save2txt(state_history, 'state_history.dat', output_path)
#                 save2txt(unprocessed_state_history, 'unprocessed_state_history.dat', output_path)
#                 save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)
#                 save2txt(dict_to_write, 'ancillary_simulation_info.txt',   output_path)
#                 environment.save_vehicle_mesh_to_file(
#                     bodies.get_body('Capsule').aerodynamic_coefficient_interface, output_path)
#
#             # Compare the simulation to the benchmarks and write differences to files
#             if use_benchmark:
#                 # Initialize containers
#                 state_difference = dict()
#
#                 # Loop over the propagated states and use the benchmark interpolators
#                 # NOTE TO STUDENTS: it can happen that the benchmark ends earlier than the regular simulation, due to
#                 # the shorter step size. Therefore, the following lines of code will be forced to extrapolate the
#                 # benchmark states (or dependent variables), producing a warning. Be aware of it!
#                 for epoch in state_history.keys():
#                     state_difference[epoch] = state_history[epoch] - benchmark_state_interpolator.interpolate(epoch)
#
#                 # Write differences with respect to the benchmarks to files
#                 if write_results_to_file:
#                     save2txt(state_difference, 'state_difference_wrt_benchmark.dat', output_path)
#
#                 # Do the same for dependent variables, if present
#                 if are_dependent_variables_to_save:
#                     # Initialize containers
#                     dependent_difference = dict()
#                     # Loop over the propagated dependent variables and use the benchmark interpolators
#                     for epoch in dependent_variable_history.keys():
#                         dependent_difference[epoch] = dependent_variable_history[epoch] - benchmark_dependent_variable_interpolator.interpolate(epoch)
#                     # Write differences with respect to the benchmarks to files
#                     if write_results_to_file:
#                         save2txt(dependent_difference, 'dependent_variable_difference_wrt_benchmark.dat',   output_path)
#
# # Print the ancillary information
# print('\n### ANCILLARY SIMULATION INFORMATION ###')
# for (elem, (info, result)) in enumerate(dict_to_write.items()):
#     if elem > 1:
#         print(info + ': ' + str(result))
