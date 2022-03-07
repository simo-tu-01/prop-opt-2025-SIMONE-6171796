"""
Copyright (c) 2010-2022, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Low Thrust
First name: ***COMPLETE HERE***
Last name: ***COMPLETE HERE***
Student number: ***COMPLETE HERE***

This module computes the dynamics of an interplanetary low-thrust trajectory, using a thrust profile determined from
a semi-analytical Hodographic shaping method (see Gondelach and Noomen, 2015). This file propagates the dynamics
using a variety of  integrator and propagator settings. For each run, the differences w.r.t. a benchmark propagation are
computed, providing a proxy for setting quality. The benchmark settings are currently defined semi-randomly, and are to be
analyzed/modified.

The semi-analytical trajectory of the vehicle is determined by its departure and arrival time (which define the initial and final states)
as well as the free parameters of the shaping method. The free parameters of the shaping method defined here are the same
as for the 'higher-order solution' in Section V.A of Gondelach and Noomen (2015). The free parameters define the amplitude
of specific types of velocity shaping functions. The low-thrust hodographic trajectory is parameterized by the values of
the variable trajectory_parameters (see below). The low-thrust trajectory computed by the shape-based method starts
at the Earth's center of mass, and terminates at Mars's center of mass.

The semi-analytical model is used to compute the thrust as a function of time (along the ideal semi-analytical trajectory).
This function is then used to define a thrust model in the numerical propagation

In the propagation, the vehicle starts on the Hodographic low-thrust trajectory, 30 days
(defined by the time_buffer variable) after it 'departs' the Earth's center of mass.

The propagation is terminated as soon as one of the following conditions is met (see
get_propagation_termination_settings() function):

* Distance to Mars < 50000 km
* Propagation time > Time-of-flight of hodographic trajectory
 
This propagation as provided assumes only point mass gravity by the Sun and thrust acceleration of the vehicle.
Both the translational dynamics and mass of the vehicle are propagated, using a fixed specific impulse.

The entries of the vector 'trajectory_parameters' contains the following:
* Entry 0: Departure time (from Earth's center-of-mass) in Julian days since J2000
* Entry 1: Time-of-flight from Earth's center-of-mass to Mars' center-of-mass, in Julian days
* Entry 2: Number of revolutions around the Sun
* Entry 3,4: Free parameters for radial shaping functions
* Entry 5,6: Free parameters for normal shaping functions
* Entry 7,8: Free parameters for axial shaping functions
  
Details on the outputs written by this file can be found:
* Benchmark data: comments for 'generate_benchmarks()' and 'compare_benchmarks()' function
* Results for integrator/propagator variations: comments under "RUN SIMULATION FOR VARIOUS SETTINGS"
* Trajectory for semi-analytical hodographic shape-based solution: comments with, and call to
    get_hodographic_trajectory() function

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

    cpu_time_termination_settings = propagation_setup.propagator.cpu_time_termination(
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
import numpy as np
import os

# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.math import interpolators

# Problem-specific imports
import LowThrustUtilities as Util
from LowThrustProblem import LowThrustProblem

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()
# NOTE TO STUDENTS: INPUT YOUR PARAMETER SET HERE, FROM THE INPUT FILES
# ON BRIGHTSPACE, FOR YOUR SPECIFIC STUDENT NUMBER.
trajectory_parameters = [570727221.2273525 / constants.JULIAN_DAY,
                         37073942.58665284 / constants.JULIAN_DAY,
                         0,
                         2471.19649906354,
                         4207.587982407276,
                         -5594.040587888714,
                         8748.139268525232,
                         -3449.838496679572]

# Choose whether output of the propagation is written to files
write_results_to_file = False
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Vehicle settings
vehicle_mass = 4.0E3
specific_impulse = 3000.0
# Fixed parameters
minimum_mars_distance = 5.0E7
# Time since 'departure from Earth CoM' at which propagation starts (and similar
# for arrival time)
time_buffer = 30.0 * constants.JULIAN_DAY

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Define settings for celestial bodies
bodies_to_create = ['Earth',
                    'Mars',
                    'Sun',
                    'Jupiter']
# Define coordinate system
global_frame_origin = 'SSB'
global_frame_orientation = 'ECLIPJ2000'
# Create body settings
body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                            global_frame_origin,
                                                            global_frame_orientation)

# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create vehicle object and add it to the existing system of bodies
bodies.create_empty_body('Vehicle')
bodies.get_body('Vehicle').mass = vehicle_mass

###########################################################################
# CREATE PROPAGATOR SETTINGS ##############################################
###########################################################################

# Retrieve dependent variables to save
dependent_variables_to_save = Util.get_dependent_variable_save_settings()
# Check whether there is any
are_dependent_variables_to_save = False if not dependent_variables_to_save else True


################################
### Design Space Exploration ###
################################

# Create Low Thrust Problem object
decision_variable_range = \
    [[0.0, 100.0, 0, -10000, -10000, -10000, -10000, -10000], #Made list of lists instead of tuple, because the ffd requires a third level
     [6000.0, 800.0, 2.9999,10000,10000,10000,10000,10000]]

design_space_method = 'monte_carlo'

number_of_parameters = len(decision_variable_range[0])

if design_space_method == 'monte_carlo':
    number_of_simulations = 100
    random_seed = 42
    np.random.seed(random_seed)
    print('\n Random Seed :', random_seed, '\n')

elif design_space_method == 'fractional_factorial_design': 
    no_of_factors = 4 
    no_of_levels = 2
    if no_of_levels == 3:
        mid_range_list = [(decision_variable_range[1][i] + decision_variable_range[0][i])/2 for i in range(number_of_parameters)]
        decision_variable_range.insert(1, mid_range_list)
    FFD_array, ierror = Util.orth_arrays(no_of_factors, no_of_levels)
    number_of_simulations = len(FFD_array)

elif design_space_method == 'factorial_design':
    no_of_levels = 3
    no_of_factors = number_of_parameters
    yates_array = Util.yates_array(no_of_levels, no_of_factors)
    design_variable_arr = np.zeros((no_of_levels, no_of_factors))
    for par in range(no_of_factors):
        design_variable_arr[:, par] = np.linspace(decision_variable_range[0][par], decision_variable_range[1][par], no_of_levels, endpoint=True)
        number_of_simulations = len(yates_array)

parameters = np.zeros((number_of_simulations, number_of_parameters))

for simulation_index in range(number_of_simulations):
    print(simulation_index)

    if design_space_method == 'factorial_design':
        level_combination = yates_array[simulation_index, :]
        for it, j in enumerate(level_combination): #Run through the row of levels from 0 to no_of_levels
            trajectory_parameters[it] = design_variable_arr[j, it]
    else:
        for parameter_index in range(number_of_parameters):
            if design_space_method == 'monte_carlo':
                trajectory_parameters[parameter_index] = np.random.uniform(decision_variable_range[0][parameter_index], decision_variable_range[1][parameter_index])
            elif design_space_method == 'fractional_factorial_design':
                if FFD_array[simulation_index,parameter_index] == -1:
                    trajectory_parameters[parameter_index] = decision_variable_range[0][parameter_index]
                elif no_of_levels == 2 and FFD_array[simulation_index,parameter_index] == 1:
                    trajectory_parameters[parameter_index] = decision_variable_range[1][parameter_index]
                elif no_of_levels == 3 and FFD_array[simulation_index,parameter_index] == 0:
                    trajectory_parameters[parameter_index] = decision_variable_range[1][parameter_index]
                elif no_of_levels == 3 and FFD_array[simulation_index,parameter_index] == 1:
                    trajectory_parameters[parameter_index] = decision_variable_range[2][parameter_index]
                else:
                    print('Error something went wrong with assigning parameters')

    parameters[simulation_index, :] = trajectory_parameters.copy()

    # Create integrator settings
    integrator_settings = Util.get_integrator_settings(0, 0, 0, initial_propagation_time)

    current_low_thrust_problem = LowThrustProblem(bodies,
                                                     integrator_settings,
                                                     specific_impulse,
                                                      minimum_mars_distance,
                                                      time_buffer,
                                                      vehicle_mass,
                                                      decision_variable_range,
                                                      True)

    fitness = current_low_thrust_problem.fitness(trajectory_parameters)

    ### OUTPUT OF THE SIMULATION ###
    # Retrieve propagated state and dependent variables
    state_history = current_low_thrust_problem.get_last_run_dynamics_simulator().state_history
    dependent_variable_history = current_low_thrust_problem.get_last_run_dynamics_simulator().dependent_variable_history

    # Set time limits to avoid numerical issues at the boundaries due to the interpolation
    propagation_times = list(state_history.keys())
    limit_times = {propagation_times[3]: propagation_times[-3]}

    # Get output path
    subdirectory = '/DesignSpace_%s/Run_%s'%(design_space_method, simulation_index)

    # Decide if output writing is required
    if write_results_to_file:
        output_path = current_dir + subdirectory
    else:
        output_path = None

    # If desired, write output to a file
    if write_results_to_file:
        save2txt(state_history, 'state_history.dat', output_path)
        save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)
        save2txt(limit_times, 'limit_times.dat', output_path)


