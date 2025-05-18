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
#
# import sys
# sys.path.insert(0, "/home/dominic/Tudat/tudat-bundle/tudat-bundle/cmake-build-default/tudatpy")

# General imports
import numpy as np
import os

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
import time
import matplotlib.pyplot as plt

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
# Choose whether output of the propagation is written to files
write_results_to_file = True
# Get path of current directory
current_dir = os.path.dirname(__file__) 

acceleration_environment_analysis = True
spherical_harmonics_analysis = True


# Love numbers
h2_love_number = 0.6
l2_shida_number = 0.08
# Shape model Constants
body_radius = 6378.0E3
body_flattening = 1.0 / 300.0

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
# CREATE (CONSTANT) PROPAGATION SETTINGS ##################################
###########################################################################

fixed_step_size = 2.56  # s
current_coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_45


if acceleration_environment_analysis:
    # Initialize dictionary to save simulation output
    simulation_results = dict()

    # Set number of models to loop over
    
    models_names = ['Nominal', 'Exponential Atmosphere', 'No Sun 3rd body', 'No Moon 3rd body-tides', 
                    'No SRP', 'No solid tides', 'No schwarzschild', 'Spherical Earth', 'Simple Rotation Model']
    number_of_models = len(models_names)

    # Loop over different model settings
    for model_test in range(number_of_models):

        # Define settings for celestial bodies
        bodies_to_create = ['Earth','Moon','Sun']
        # Define coordinate system
        global_frame_origin = 'Earth'
        global_frame_orientation = 'J2000'

        # Create body settings
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create,
            global_frame_origin,
            global_frame_orientation)
        
        if model_test == 1:
            Util.get_environment_settings(body_settings, global_frame_orientation, simulation_start_epoch,
                                atmosphere= 'exponential',
                                shape= 'oblate_spheroid',
                                rotation_model= 'IAU',
                                tides = True)
        elif model_test == 3:
            Util.get_environment_settings(body_settings, global_frame_orientation, simulation_start_epoch,
                                atmosphere= 'US76',
                                shape= 'oblate_spheroid',
                                rotation_model= 'IAU',
                                tides = False)
        elif model_test == 5:
            Util.get_environment_settings(body_settings, global_frame_orientation, simulation_start_epoch,
                                atmosphere= 'US76',
                                shape= 'oblate_spheroid',
                                rotation_model= 'IAU',
                                tides = False)
        elif model_test == 7:
            Util.get_environment_settings(body_settings, global_frame_orientation, simulation_start_epoch,
                                atmosphere= 'US76',
                                shape= 'sphere',
                                rotation_model= 'IAU',
                                tides = True)
        elif model_test == 8:
            Util.get_environment_settings(body_settings, global_frame_orientation, simulation_start_epoch,
                                atmosphere= 'US76',
                                shape= 'oblate_spheroid',
                                rotation_model= 'simple',
                                tides = True)       
        else:
            Util.get_environment_settings(body_settings, global_frame_orientation, simulation_start_epoch,
                                atmosphere= 'US76',
                                shape= 'oblate_spheroid',
                                rotation_model= 'IAU',
                                tides = True)
            

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

        # Create propagator settings for benchmark (Cowell)
        propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                        bodies,
                                                        simulation_start_epoch,
                                                        termination_settings,
                                                        dependent_variables_to_save,
                                                        current_propagator=propagation_setup.propagator.cowell,
                                                        model_choice = model_test )

        # Create integrator settings
        propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
            fixed_step_size,
            coefficient_set=current_coefficient_set,
            order_to_use=propagation_setup.integrator.lower
        )
        propagator_settings.print_settings.print_dependent_variable_indices = True

        # Create Shape Optimization Problem object
        print(f'{models_names[model_test]}')
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, propagator_settings)
        

        ### OUTPUT OF THE SIMULATION ###
        # Retrieve propagated state and dependent variables
        state_history = dynamics_simulator.state_history
        dependent_variable_history = dynamics_simulator.dependent_variable_history

        # Save results to a dictionary
        simulation_results[model_test] = [state_history, dependent_variable_history]

        # Get output path
        if model_test == 0:
            subdirectory = '/NominalCase/'
        else:
            subdirectory = '/Model_' + str(model_test) + '/'

        # Decide if output writing is required
        if write_results_to_file:
            output_path = current_dir + subdirectory
        else:
            output_path = None

        # If desired, write output to a file
        if write_results_to_file:
            save2txt(state_history, 'state_history.dat', output_path)
            save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)



    """
    The first index of the dictionary simulation_results refers to the model case, while the second index can be 0 (states)
    or 1 (dependent variables).
    """
    # Compare all the model settings with the nominal case - Position
    for model_test in range(1, number_of_models):
        # Get output path
        output_path = current_dir + '/Model_' + str(model_test) + '/'

        # Set time limits to avoid numerical issues at the boundaries due to the interpolation
        nominal_state_history = simulation_results[0][0]
        nominal_dependent_variable_history = simulation_results[0][1]
        nominal_times = list(nominal_state_history.keys())

        # Retrieve current state and dependent variable history
        current_state_history = simulation_results[model_test][0]
        current_dependent_variable_history = simulation_results[model_test][1]
        current_times = list(current_state_history.keys())

        # Get limit times at which both histories can be validly interpolated
        interpolation_lower_limit = max(nominal_times[3],current_times[3])
        interpolation_upper_limit = min(nominal_times[-3],current_times[-3])

        # Create vector of epochs to be compared (boundaries are referred to the first case)
        
        unfiltered_interpolation_epochs = [n for n in current_times if n <= interpolation_upper_limit]
        interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n >= interpolation_lower_limit]

        # Compare state history
        state_difference_wrt_nominal = Util.compare_models(current_state_history,
                                                        simulation_results[0][0],
                                                        interpolation_epochs,
                                                        output_path,
                                                        'state_difference_wrt_nominal_case.dat')
        # Compare dependent variable history
        dependent_variable_difference_wrt_nominal = Util.compare_models(current_dependent_variable_history,
                                                                        simulation_results[0][1],
                                                                        interpolation_epochs,
                                                                        output_path,
                                                                        'dependent_variable_difference_wrt_nominal_case.dat')
        
        position_difference = np.array([np.linalg.norm(state_difference_wrt_nominal[epoch][:3]) for epoch in state_difference_wrt_nominal.keys()])
        time_values = state_difference_wrt_nominal.keys()
        altitude_difference = np.array([np.linalg.norm(dependent_variable_difference_wrt_nominal[epoch][1]) for epoch in dependent_variable_difference_wrt_nominal.keys()])

        ls = '-'
        mrk = None
        if model_test == 1: 
            ls = '--'

        plt.plot(time_values, position_difference, linewidth=2.5, label=f"{models_names[model_test]}", linestyle = ls, marker=mrk)
    plt.xlabel('Time [s]', fontsize=20)
    plt.ylabel('Position Error [m]', fontsize=20)
    plt.yscale('log')
    plt.axhline(y=10, color='black', linestyle='--', linewidth=2, label='10 m threshold')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    plt.show()


    # Compare all the model settings with the nominal case - Altitude
    for model_test in range(1, number_of_models):
        # Get output path
        output_path = current_dir + '/Model_' + str(model_test) + '/'

        # Set time limits to avoid numerical issues at the boundaries due to the interpolation
        nominal_state_history = simulation_results[0][0]
        nominal_dependent_variable_history = simulation_results[0][1]
        nominal_times = list(nominal_state_history.keys())

        # Retrieve current state and dependent variable history
        current_state_history = simulation_results[model_test][0]
        current_dependent_variable_history = simulation_results[model_test][1]
        current_times = list(current_state_history.keys())

        # Get limit times at which both histories can be validly interpolated
        interpolation_lower_limit = max(nominal_times[3],current_times[3])
        interpolation_upper_limit = min(nominal_times[-3],current_times[-3])

        # Create vector of epochs to be compared (boundaries are referred to the first case)
        
        unfiltered_interpolation_epochs = [n for n in current_times if n <= interpolation_upper_limit]
        interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n >= interpolation_lower_limit]

        # Compare state history
        state_difference_wrt_nominal = Util.compare_models(current_state_history,
                                                        simulation_results[0][0],
                                                        interpolation_epochs,
                                                        output_path,
                                                        'state_difference_wrt_nominal_case.dat')
        # Compare dependent variable history
        dependent_variable_difference_wrt_nominal = Util.compare_models(current_dependent_variable_history,
                                                                        simulation_results[0][1],
                                                                        interpolation_epochs,
                                                                        output_path,
                                                                        'dependent_variable_difference_wrt_nominal_case.dat')
        
        position_difference = np.array([np.linalg.norm(state_difference_wrt_nominal[epoch][:3]) for epoch in state_difference_wrt_nominal.keys()])
        time_values = state_difference_wrt_nominal.keys()
        altitude_difference = np.array([np.linalg.norm(dependent_variable_difference_wrt_nominal[epoch][1]) for epoch in dependent_variable_difference_wrt_nominal.keys()])

        ls = '-'
        mrk = None
        if model_test == 1: 
            ls = '--'

        plt.plot(time_values, altitude_difference, linewidth=2.5, label=f"{models_names[model_test]}", linestyle = ls, marker=mrk)
    plt.xlabel('Time [s]', fontsize=20)
    plt.ylabel('Altitude Error[m]', fontsize=20)
    plt.yscale('log')
    plt.axhline(y=10, color='black', linestyle='--', linewidth=2, label='10 m threshold')
    plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    plt.show()


###########################################################################
# SPHERICAL HARMONICS ANALYSIS           ##################################
###########################################################################

# No SRP
# No Tides
# No Shwartzshild
# Rotation Model: Simple
# Atmosphere Model: US76
# Shape Model: Oblate Spheroid

if spherical_harmonics_analysis:
    # Initialize dictionary to save simulation output
    simulation_results = dict()
    cpu_time_list = []

    # Set number of models to loop over
    
    models_names = ['200/200', '128/128', '64/64', '32/32', '16/16', 
                    '8/8', '4/4', '2/2', '2/0']
    sh_values = [(200,200), (128, 128), (64, 64), (32, 32), (16, 16), 
                 (8, 8), (4, 4), (2, 2), (2, 0)]
    number_of_models = len(models_names)

    # Loop over different model settings
    for model_test in range(number_of_models):

        sh_degree, sh_order = sh_values[model_test]

        # Define settings for celestial bodies
        bodies_to_create = ['Earth', 'Moon', 'Sun']
        # Define coordinate system
        global_frame_origin = 'Earth'
        global_frame_orientation = 'J2000'

        # Create body settings
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create,
            global_frame_origin,
            global_frame_orientation)
        
        Util.get_environment_settings(body_settings, global_frame_orientation, simulation_start_epoch,
                                atmosphere= 'US76',
                                shape= 'oblate_spheroid',
                                rotation_model= 'simple',
                                tides = False)
              
        # Create bodies
        bodies = environment_setup.create_system_of_bodies(body_settings)

        # Create and add capsule to body system
        Util.add_capsule_to_body_system(bodies,
                                        shape_parameters,
                                        capsule_density)


        ###########################################################################
        # CREATE PROPAGATION SETTINGS            ##################################
        ###########################################################################

        # Retrieve termination settings
        termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                            maximum_duration,
                                                            termination_altitude)
        # Retrieve dependent variables to save
        dependent_variables_to_save = Util.get_dependent_variable_save_settings()

        # Create propagator settings for benchmark (Cowell)
        if model_test == 0:
            propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                            bodies,
                                                            simulation_start_epoch,
                                                            termination_settings,
                                                            dependent_variables_to_save,
                                                            current_propagator=propagation_setup.propagator.cowell,
                                                            model_choice = 'sh',
                                                            sh_degree= sh_degree,
                                                            sh_order= sh_order)
        else:
            propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                            bodies,
                                                            simulation_start_epoch,
                                                            termination_settings,
                                                            dependent_variables_to_save,
                                                            current_propagator=propagation_setup.propagator.cowell,
                                                            model_choice = 'sh_no',
                                                            sh_degree= sh_degree,
                                                            sh_order= sh_order)

        # Create integrator settings
        propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
            fixed_step_size,
            coefficient_set=current_coefficient_set,
            order_to_use=propagation_setup.integrator.lower
        )
        propagator_settings.print_settings.print_dependent_variable_indices = True

        # Create Shape Optimization Problem object
        print(f'{models_names[model_test]}')
        start_time = time.perf_counter()
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, propagator_settings)
        cpu_time = time.perf_counter() - start_time

        ### OUTPUT OF THE SIMULATION ###
        # Retrieve propagated state and dependent variables
        state_history = dynamics_simulator.state_history
        dependent_variable_history = dynamics_simulator.dependent_variable_history



        # Save results to a dictionary
        simulation_results[model_test] = [state_history, dependent_variable_history]
        cpu_time_list.append(cpu_time)

        # Get output path
        if model_test == 0:
            subdirectory = 'SH/NominalCase/'
        else:
            subdirectory = 'SH/Model_' + str(models_names[model_test]) + '/'

        # Decide if output writing is required
        if write_results_to_file:
            output_path = current_dir + subdirectory
        else:
            output_path = None

        # If desired, write output to a file
        if write_results_to_file:
            save2txt(state_history, 'state_history.dat', output_path)
            save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)



    """
    The first index of the dictionary simulation_results refers to the model case, while the second index can be 0 (states)
    or 1 (dependent variables).
    """
    # Compare all the model settings with the nominal case - Position
    for model_test in range(1, number_of_models):
        # Get output path
        output_path = current_dir + '/Model_' + str(model_test) + '/'

        # Set time limits to avoid numerical issues at the boundaries due to the interpolation
        nominal_state_history = simulation_results[0][0]
        nominal_dependent_variable_history = simulation_results[0][1]
        nominal_times = list(nominal_state_history.keys())

        # Retrieve current state and dependent variable history
        current_state_history = simulation_results[model_test][0]
        current_dependent_variable_history = simulation_results[model_test][1]
        current_times = list(current_state_history.keys())

        # Get limit times at which both histories can be validly interpolated
        interpolation_lower_limit = max(nominal_times[3],current_times[3])
        interpolation_upper_limit = min(nominal_times[-3],current_times[-3])

        # Create vector of epochs to be compared (boundaries are referred to the first case)
        
        unfiltered_interpolation_epochs = [n for n in current_times if n <= interpolation_upper_limit]
        interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n >= interpolation_lower_limit]

        # Compare state history
        state_difference_wrt_nominal = Util.compare_models(current_state_history,
                                                        simulation_results[0][0],
                                                        interpolation_epochs,
                                                        output_path,
                                                        'state_difference_wrt_nominal_case.dat')
        # Compare dependent variable history
        dependent_variable_difference_wrt_nominal = Util.compare_models(current_dependent_variable_history,
                                                                        simulation_results[0][1],
                                                                        interpolation_epochs,
                                                                        output_path,
                                                                        'dependent_variable_difference_wrt_nominal_case.dat')
        
        position_difference = np.array([np.linalg.norm(state_difference_wrt_nominal[epoch][:3]) for epoch in state_difference_wrt_nominal.keys()])
        time_values = state_difference_wrt_nominal.keys()
        altitude_difference = np.array([np.linalg.norm(dependent_variable_difference_wrt_nominal[epoch][1]) for epoch in dependent_variable_difference_wrt_nominal.keys()])

        ls = '-'
        mrk = None
        plt.plot(time_values, position_difference, linewidth=2.5, label=f"{models_names[model_test]}", linestyle = ls, marker=mrk)

    plt.xlabel('Time [s]', fontsize=20)
    plt.ylabel('Position Error [m]', fontsize=20)
    plt.yscale('log')
    plt.axhline(y=10, color='black', linestyle='--', linewidth=2, label='10 m threshold')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    plt.show()



    # Compare all the model settings with the nominal case - Altitude
    for model_test in range(1, number_of_models):
        # Get output path
        output_path = current_dir + '/Model_' + str(model_test) + '/'

        # Set time limits to avoid numerical issues at the boundaries due to the interpolation
        nominal_state_history = simulation_results[0][0]
        nominal_dependent_variable_history = simulation_results[0][1]
        nominal_times = list(nominal_state_history.keys())

        # Retrieve current state and dependent variable history
        current_state_history = simulation_results[model_test][0]
        current_dependent_variable_history = simulation_results[model_test][1]
        current_times = list(current_state_history.keys())

        # Get limit times at which both histories can be validly interpolated
        interpolation_lower_limit = max(nominal_times[3],current_times[3])
        interpolation_upper_limit = min(nominal_times[-3],current_times[-3])

        # Create vector of epochs to be compared (boundaries are referred to the first case)
        
        unfiltered_interpolation_epochs = [n for n in current_times if n <= interpolation_upper_limit]
        interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n >= interpolation_lower_limit]

        # Compare state history
        state_difference_wrt_nominal = Util.compare_models(current_state_history,
                                                        simulation_results[0][0],
                                                        interpolation_epochs,
                                                        output_path,
                                                        'state_difference_wrt_nominal_case.dat')
        # Compare dependent variable history
        dependent_variable_difference_wrt_nominal = Util.compare_models(current_dependent_variable_history,
                                                                        simulation_results[0][1],
                                                                        interpolation_epochs,
                                                                        output_path,
                                                                        'dependent_variable_difference_wrt_nominal_case.dat')
        
        position_difference = np.array([np.linalg.norm(state_difference_wrt_nominal[epoch][:3]) for epoch in state_difference_wrt_nominal.keys()])
        time = state_difference_wrt_nominal.keys()
        altitude_difference = np.array([np.linalg.norm(dependent_variable_difference_wrt_nominal[epoch][1]) for epoch in dependent_variable_difference_wrt_nominal.keys()])

        ls = '-'
        mrk = None
        plt.plot(time, altitude_difference, linewidth=2.5, label=f"{models_names[model_test]}", linestyle = ls, marker=mrk)

    plt.xlabel('Time [s]', fontsize=20)
    plt.ylabel('Altitude Error [m]', fontsize=20)
    plt.yscale('log')
    plt.axhline(y=10, color='black', linestyle='--', linewidth=2, label='10 m threshold')
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    plt.show()




    # Initialize containers
    labels = []
    delta_qc_max = []
    delta_qc_alt = []
    delta_q_max = []
    delta_q_alt = []
    delta_ntot_max = []
    delta_ntot_alt = []
    cpu_time_vals = []

    # Optional: if you have the labels like '2/0', '4/4', etc.
    # degree_order_labels = ['2/0', '2/2', '4/4', ..., '128/128']

    # Main loop
    for model_test in range(1, number_of_models):
        output_path = current_dir + '/Model_' + str(model_test) + '/'
        
        nominal_dependent_variable_history = simulation_results[0][1]
        current_dependent_variable_history = simulation_results[model_test][1]
        
        current_metrics = Util.compute_aerodynamic_metrics(current_dependent_variable_history)
        nominal_metrics = Util.compute_aerodynamic_metrics(nominal_dependent_variable_history)
        
        percentage_errors = {}
        for key in nominal_metrics:
            if 'altitude' in key:
                percentage_errors[key] = abs(current_metrics[key] - nominal_metrics[key])
            elif isinstance(nominal_metrics[key], (int, float)) and nominal_metrics[key] != 0:
                percentage_errors[key] = 100.0 * abs(current_metrics[key] - nominal_metrics[key]) / abs(nominal_metrics[key])
            else:
                percentage_errors[key] = np.nan

        # Append label (you can use custom D/O labels here)
        labels.append(str(models_names[model_test]))  # or degree_order_labels[model_test - 1]

        # Extract metrics
        delta_qc_max.append(percentage_errors['max_heat_flux'])
        delta_qc_alt.append(percentage_errors['altitude_at_max_heat_flux'])

        delta_q_max.append(percentage_errors['max_heat_load'])
        delta_q_alt.append(percentage_errors['altitude_at_max_heat_load'])

        delta_ntot_max.append(percentage_errors['max_load_factor'])
        delta_ntot_alt.append(percentage_errors['altitude_at_max_load_factor'])

        cpu_time_vals.append(cpu_time_list[model_test])

    # === Plot ===
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # % Errors (log scale)
    marker_sizes = [8, 6, 7]
    ax1.plot(labels, delta_qc_max, 's-', label=r'$\Delta q_{c,\max}$ [%]', markersize=8, markeredgecolor='black')
    ax1.plot(labels, delta_q_max, 'o-', label=r'$\Delta Q_{\max}$ [%]', markersize=6, markeredgecolor='black')
    ax1.plot(labels, delta_ntot_max, '^-', label=r'$\Delta n_{\mathrm{tot},\max}$ [%]', markersize=7, markeredgecolor='black')
    ax1.set_ylabel('Percentage Error [%]', fontsize=20)
    ax1.set_xlabel('Degree/Order)', fontsize=20)
    ax1.set_yscale('log')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend(fontsize=16,  bbox_to_anchor=(1.1, 1.02), loc='upper left')
    ax1.tick_params(axis='both', which='major', labelsize=18)

    # CPU time (log scale)
    ax2 = ax1.twinx()
    ax2.plot(labels, cpu_time_vals, 'd--', color='gray', label='CPU Time [s]', markersize=7)
    # Move the CPU time y-label closer to the edge of the figure
    ax2.set_ylabel('CPU Time [s]', fontsize=20, labelpad=-25)
    ax2.set_yscale('log')
    ax2.legend(fontsize=16, bbox_to_anchor=(1.1, 0.75), loc='lower left')
    ax2.tick_params(axis='both', which='major', labelsize=18)

    # Plot altitude errors on a third y-axis, using same color/marker as % error, but different linestyle
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Offset third y-axis

    # plt.title('Trajectory Sensitivity and CPU Time vs. Spherical Harmonics D/O', fontsize=20)
    plt.tight_layout()
    plt.show()






    # Compare all the model settings with the nominal case - Position
    for model_test in range(1, number_of_models):
        # Get output path
        output_path = current_dir + '/Model_' + str(model_test) + '/'

        nominal_dependent_variable_history = simulation_results[0][1]

        current_dependent_variable_history = simulation_results[model_test][1]

        cpu_time = cpu_time_list

        current_metrics = Util.compute_aerodynamic_metrics(current_dependent_variable_history)

        nominal_metrics = Util.compute_aerodynamic_metrics(nominal_dependent_variable_history)

        # Compute percentage errors for each metric
        percentage_errors = {}
        for key in nominal_metrics:
            if 'altitude' in key:
                # For altitude metrics, provide absolute error (not percentage)
                percentage_errors[key] = abs(current_metrics[key] - nominal_metrics[key])
            elif isinstance(nominal_metrics[key], (int, float)) and nominal_metrics[key] != 0:
                percentage_errors[key] = 100.0 * abs(current_metrics[key] - nominal_metrics[key]) / abs(nominal_metrics[key])
            else:
                percentage_errors[key] = np.nan

        

        # Prepare table row
        sh_deg, sh_ord = sh_values[model_test]
        if model_test == 1:
            table = []
        table.append([
            f"{sh_deg}/{sh_ord}",
            f"{percentage_errors['max_heat_flux']:.3f}",
            f"{percentage_errors['altitude_at_max_heat_flux']:.3f}",
            f"{percentage_errors['max_heat_load']:.3f}",
            f"{percentage_errors['altitude_at_max_heat_load']:.3f}",
            f"{percentage_errors['max_load_factor']:.3f}",
            f"{percentage_errors['altitude_at_max_load_factor']:.3f}",
            f"{cpu_time[model_test]:.3f}"
        ])

        # Print table header
        headers = [
        "SH Degree/Order",
        "%Err Max Heat Flux",
        "%Err Alt@Max Heat Flux",
        "%Err Max Heat Load",
        "%Err Alt@Max Heat Load",
        "%Err Max Load Factor",
        "%Err Alt@Max Load Factor",
        "CPU Time [s]"
        ]
        print("\nSpherical Harmonics Metrics Comparison Table:")
        print("{:<15} {:>20} {:>22} {:>20} {:>22} {:>20} {:>25} {:>15}".format(*headers))
        for row in table:
            print("{:<15} {:>20} {:>22} {:>20} {:>22} {:>20} {:>25} {:>15}".format(*row))











