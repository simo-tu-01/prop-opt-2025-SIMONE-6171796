'''
Copyright (c) 2010-2021, Delft University of Technology
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

This module defines useful functions that will be called by the main script, where the optimization is executed.
'''

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
from itertools import combinations as comb

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory


###########################################################################
# USEFUL FUNCTIONS ########################################################
###########################################################################


def get_termination_settings(trajectory_parameters,
                             minimum_mars_distance: float,
                             time_buffer: float) \
        -> tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings:
    """
    Get the termination settings for the simulation.

    Termination settings currently include:
    - simulation time (propagation stops if it is greater than the one provided by the hodographic trajectory)
    - distance to Mars (propagation stops if the relative distance is lower than the target distance)

    Parameters
    ----------
    trajectory_parameters : list[floats]
        List of trajectory parameters.
    minimum_mars_distance : float
        Minimum distance from Mars at which the propagation stops.
    time_buffer : float
        Time interval between the simulation start epoch and the beginning of the hodographic trajectory.

    Returns
    -------
    hybrid_termination_settings : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object.
    """
    # Create single PropagationTerminationSettings objects
    # Time
    final_time = get_trajectory_final_time(trajectory_parameters,
                                           time_buffer)
    time_termination_settings = propagation_setup.propagator.time_termination(
        final_time,
        terminate_exactly_on_final_condition=False)
    # Altitude
    relative_distance_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.relative_distance('Vehicle', 'Mars'),
        limit_value=minimum_mars_distance,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False)
    # Define list of termination settings
    termination_settings_list = [time_termination_settings,
                                 relative_distance_termination_settings]
    # Create termination settings object
    hybrid_termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                                  fulfill_single_condition=True)
    return hybrid_termination_settings


# NOTE TO STUDENTS: this function can be modified to save more/less dependent variables.
def get_dependent_variable_save_settings() -> list:
    """
    Retrieves the dependent variables to save.

    Currently, the dependent variables saved include the relative distance between the spacecraft and:
    - Earth
    - the Sun
    - Mars

    Parameters
    ----------
    none

    Returns
    -------
    dependent_variables_to_save : list[tudatpy.kernel.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    """
    dependent_variables_to_save = [propagation_setup.dependent_variable.relative_distance('Vehicle', 'Earth'),
                                   propagation_setup.dependent_variable.relative_distance('Vehicle', 'Sun'),
                                   propagation_setup.dependent_variable.relative_distance('Vehicle', 'Mars')]
    return dependent_variables_to_save


# NOTE TO STUDENTS: THIS FUNCTION SHOULD BE EXTENDED TO USE MORE INTEGRATORS FOR ASSIGNMENT 1.
def get_integrator_settings(propagator_index: int,
                            integrator_index: int,
                            settings_index: int,
                            simulation_start_epoch: float) \
        -> tudatpy.kernel.numerical_simulation.propagation_setup.integrator.IntegratorSettings:
    """

    Retrieves the integrator settings.

    It selects a combination of integrator to be used (first argument) and
    the related setting (tolerance for variable step size integrators
    or step size for fixed step size integrators). The code, as provided, runs the following:
    - if j=0,1,2,3: a variable-step-size, multi-stage integrator is used (see multiStageTypes list for specific type),
                     with tolerances 10^(-10+*k)
    - if j=4      : a fixed-step-size RK4 integrator is used, with step-size 7200*2^(k)

    Parameters
    ----------
    propagator_index : int
        Index that selects the propagator type (currently not used).
        NOTE TO STUDENTS: this argument can be used to select specific combinations of propagator and integrators
        (provided that the code is expanded).
    integrator_index : int
        Index that selects the integrator type as follows:
            0 -> RK4(5)
            1 -> RK5(6)
            2 -> RK7(8)
            3 -> RKDP7(8)
            4 -> RK4
    settings_index : int
        Index that selects the tolerance or the step size
        (depending on the integrator type).
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.

    Returns
    -------
    integrator_settings : tudatpy.kernel.numerical_simulation.propagation_setup.integrator.IntegratorSettings
        Integrator settings to be provided to the dynamics simulator.

    """
    # Define list of multi-stage integrators
    multi_stage_integrators = [propagation_setup.integrator.RKCoefficientSets.rkf_45,
                               propagation_setup.integrator.RKCoefficientSets.rkf_56,
                               propagation_setup.integrator.RKCoefficientSets.rkf_78,
                               propagation_setup.integrator.RKCoefficientSets.rkdp_87]

    # Use variable step-size integrator
    if integrator_index < 4:
        # Select variable-step integrator
        current_coefficient_set = multi_stage_integrators[integrator_index]
        # Compute current tolerance
        current_tolerance = 10.0 ** (-10.0 + settings_index)
        # Create integrator settings
        integrator = propagation_setup.integrator
        # Here (epsilon, inf) are set as respectively min and max step sizes
        # also note that the relative and absolute tolerances are the same value
        integrator_settings = integrator.runge_kutta_variable_step_size(simulation_start_epoch,
                                                                        1.0,
                                                                        current_coefficient_set,
                                                                        np.finfo(float).eps,
                                                                        np.inf,
                                                                        current_tolerance,
                                                                        current_tolerance)
    # Use fixed step-size integrator
    else:
        # Compute time step
        fixed_step_size = 7200.0 * 2.0 ** settings_index
        # Create integrator settings
        integrator = propagation_setup.integrator
        integrator_settings = integrator.runge_kutta_4(simulation_start_epoch,
                                                       fixed_step_size)
    return integrator_settings

def get_propagator_settings( trajectory_parameters,
                             bodies,
                             initial_propagation_time,
                             constant_specific_impulse,
                             vehicle_initial_mass,
                             termination_settings,
                             dependent_variables_to_save,
                             current_propagator = propagation_setup.propagator.cowell,
                             model_choice = 0 ):
    # Define bodies that are propagated and their central bodies of propagation
    bodies_to_propagate = ['Vehicle']
    central_bodies = ['Sun']
    # Retrieve thrust acceleration
    thrust_settings = get_hodograph_thrust_acceleration_settings(trajectory_parameters,
                                                                 bodies,
                                                                 constant_specific_impulse)
    # Define accelerations acting
    # Define accelerations acting on vehicle in nominal case
    acceleration_settings_on_vehicle = {
        'Sun': [propagation_setup.acceleration.point_mass_gravity()],
        'Vehicle': [thrust_settings],
        'Mars': [propagation_setup.acceleration.point_mass_gravity()],
        'Earth': [propagation_setup.acceleration.point_mass_gravity()]
    }
    # Here model settings are modified
    if model_choice == 1:
        del acceleration_settings_on_vehicle['Mars']
    elif model_choice == 2:
        del acceleration_settings_on_vehicle['Earth']
    elif model_choice > 2:
        acceleration_settings_on_vehicle['Jupiter'] = [propagation_setup.acceleration.point_mass_gravity()]
    # Create global accelerations dictionary
    acceleration_settings = {'Vehicle': acceleration_settings_on_vehicle}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    # Retrieve initial state
    initial_state = get_hodograph_state_at_epoch(trajectory_parameters,
                                                 bodies,
                                                 initial_propagation_time)

    # Create propagation settings for the benchmark
    translational_propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        termination_settings,
        current_propagator,
        output_variables=dependent_variables_to_save)

    # Create mass rate model
    mass_rate_settings_on_vehicle = {'Vehicle': [propagation_setup.mass_rate.from_thrust()]}
    mass_rate_models = propagation_setup.create_mass_rate_models(bodies,
                                                                 mass_rate_settings_on_vehicle,
                                                                 acceleration_models)
    # Create mass propagator settings (same for all propagations)
    mass_propagator_settings = propagation_setup.propagator.mass(bodies_to_propagate,
                                                                 mass_rate_models,
                                                                 np.array([vehicle_initial_mass]),
                                                                 termination_settings)

    # Create multi-type propagation settings list
    propagator_settings_list = [translational_propagator_settings,
                                mass_propagator_settings]

    # Create multi-type propagation settings object
    propagator_settings = propagation_setup.propagator.multitype(propagator_settings_list,
                                                                 termination_settings,
                                                                 dependent_variables_to_save)

    return propagator_settings



###########################################################################
# HODOGRAPH-SPECIFIC FUNCTIONS ############################################
###########################################################################


def get_trajectory_time_of_flight(trajectory_parameters: list) -> float:
    """
    Returns the time of flight in seconds.

    Parameters
    ----------
    trajectory_parameters : list of floats
        List of trajectory parameters to optimize.

    Returns
    -------
    float
        Time of flight [s].
    """
    return trajectory_parameters[1] * constants.JULIAN_DAY


def get_trajectory_initial_time(trajectory_parameters: list,
                                buffer_time: float = 0.0) -> float:
    """
    Returns the time of flight in seconds.

    Parameters
    ----------
    trajectory_parameters : list of floats
        List of trajectory parameters to optimize.
    buffer_time : float (default: 0.0)
        Delay between start of the hodographic trajectory and the start of the propagation.

    Returns
    -------
    float
        Initial time of the hodographic trajectory [s].
    """
    return trajectory_parameters[0] * constants.JULIAN_DAY + buffer_time


def get_trajectory_final_time(trajectory_parameters: list,
                              buffer_time: float = 0.0) -> float:
    """
    Returns the time of flight in seconds.

    Parameters
    ----------
    trajectory_parameters : list of floats
        List of trajectory parameters to optimize.
    buffer_time : float (default: 0.0)
        Delay between start of the hodographic trajectory and the start of the propagation.

    Returns
    -------
    float
        Final time of the hodographic trajectory [s].
    """
    # Get initial time
    initial_time = get_trajectory_initial_time(trajectory_parameters)
    return initial_time + get_trajectory_time_of_flight(trajectory_parameters) - buffer_time


def get_hodographic_trajectory(shaping_object: tudatpy.kernel.trajectory_design.shape_based_thrust.HodographicShaping,
                               trajectory_parameters: list,
                               specific_impulse: float,
                               output_path: str = None):
    """
    It computes the analytical hodographic trajectory and saves the results to a file, if desired.

    This function analytically calculates the hodographic trajectory from the Hodographic Shaping object. It
    retrieves both the trajectory and the acceleration profile; if desired, both are saved to files as follows:

    * hodographic_trajectory.dat: Cartesian states of semi-analytical trajectory;
    * hodographic_thrust_acceleration.dat: Thrust acceleration in inertial, Cartesian, coordinates, along the
    semi-analytical trajectory.

    NOTE: The independent variable (first column) does not represent the usual time (seconds since J2000), but instead
    denotes the time since departure.

    Parameters
    ----------
    shaping_object: tudatpy.kernel.trajectory_design.shape_based_thrust.HodographicShaping
        Hodographic shaping object.
    trajectory_parameters : list of floats
        List of trajectory parameters to be optimized.
    specific_impulse : float
        Constant specific impulse of the spacecraft.
    output_path : str (default: None)
        If and where to save the benchmark results (if None, results are NOT written).

    Returns
    -------
    none
    """
    # Set time parameters
    start_time = 0.0
    final_time = get_trajectory_time_of_flight(trajectory_parameters)
    # Set number of data points
    number_of_data_points = 10000
    # Compute step size
    step_size = (final_time - start_time) / (number_of_data_points - 1)
    # Create epochs vector
    epochs = np.linspace(start_time,
                         final_time,
                         number_of_data_points)
    # Create specific impulse lambda function
    specific_impulse_function = lambda t: specific_impulse
    # Retrieve thrust acceleration profile from shaping object
    # NOTE TO THE STUDENTS: do not uncomment
    # thrust_acceleration_profile = shaping_object.get_thrust_acceleration_profile(
    #     epochs,
    #     specific_impulse_function)
    # Retrieve trajectory from shaping object
    trajectory_shape = shaping_object.get_trajectory(epochs)
    # If desired, save results to files
    if output_path is not None:
        # NOTE TO THE STUDENTS: do not uncomment
        # save2txt(thrust_acceleration_profile,
        #          'hodographic_thrust_acceleration.dat',
        #          output_path)
        save2txt(trajectory_shape,
                 'hodographic_trajectory.dat',
                 output_path)


def get_radial_velocity_shaping_functions(trajectory_parameters: list,
                                          frequency: float,
                                          scale_factor: float,
                                          time_of_flight: float,
                                          number_of_revolutions: int) -> tuple:
    """
    Retrieves the radial velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to optimize.
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    time_of_flight: float
        Time of flight of the trajectory.
    number_of_revolutions: int
        Number of revolutions around the Sun (currently unused).

    Returns
    -------
    tuple
        A tuple composed by two lists: the radial velocity shaping functions and their free coefficients.
    """
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    radial_velocity_shaping_functions = shape_based_thrust.recommended_radial_hodograph_functions(time_of_flight)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    # Set free parameters
    free_coefficients = trajectory_parameters[3:5]
    return (radial_velocity_shaping_functions,
            free_coefficients)


def get_normal_velocity_shaping_functions(trajectory_parameters: list,
                                          frequency: float,
                                          scale_factor: float,
                                          time_of_flight: float,
                                          number_of_revolutions: int) -> tuple:
    """
    Retrieves the normal velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to optimize.
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    time_of_flight: float
        Time of flight of the trajectory.
    number_of_revolutions: int
        Number of revolutions around the Sun (currently unused).

    Returns
    -------
    tuple
        A tuple composed by two lists: the normal velocity shaping functions and their free coefficients.
    """
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    normal_velocity_shaping_functions = shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    # Set free parameters
    free_coefficients = trajectory_parameters[5:7]
    return (normal_velocity_shaping_functions,
            free_coefficients)


def get_axial_velocity_shaping_functions(trajectory_parameters: list,
                                         frequency: float,
                                         scale_factor: float,
                                         time_of_flight: float,
                                         number_of_revolutions: int) -> tuple:
    """
    Retrieves the axial velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to optimize.
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    time_of_flight: float
        Time of flight of the trajectory.
    number_of_revolutions: int
        Number of revolutions around the Sun.

    Returns
    -------
    tuple
        A tuple composed by two lists: the axial velocity shaping functions and their free coefficients.
    """
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    axial_velocity_shaping_functions = shape_based_thrust.recommended_axial_hodograph_functions(
        time_of_flight,
        number_of_revolutions)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    exponent = 4.0
    axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
        exponent=exponent,
        frequency=(number_of_revolutions + 0.5) * frequency,
        scale_factor=scale_factor ** exponent))
    axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
        exponent=exponent,
        frequency=(number_of_revolutions + 0.5) * frequency,
        scale_factor=scale_factor ** exponent))
    # Set free parameters
    free_coefficients = trajectory_parameters[7:9]
    return (axial_velocity_shaping_functions,
            free_coefficients)


def create_hodographic_shaping_object(trajectory_parameters: list,
                                      bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies) \
        -> tudatpy.kernel.trajectory_design.shape_based_thrust.HodographicShaping:
    """
    It creates and returns the hodographic shaping object, based on the trajectory parameters.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to be optimized.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.

    Returns
    -------
    hodographic_shaping_object : tudatpy.kernel.trajectory_design.shape_based_thrust.HodographicShaping
        Hodographic shaping object.
    """
    # Time settings
    initial_time = get_trajectory_initial_time(trajectory_parameters)
    time_of_flight = get_trajectory_time_of_flight(trajectory_parameters)
    final_time = get_trajectory_final_time(trajectory_parameters)
    # Number of revolutions
    number_of_revolutions = int(trajectory_parameters[2])
    # Compute relevant frequency and scale factor for shaping functions
    frequency = 2.0 * np.pi / time_of_flight
    scale_factor = 1.0 / time_of_flight
    # Retrieve shaping functions and free parameters
    radial_velocity_shaping_functions, radial_free_coefficients = get_radial_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)
    normal_velocity_shaping_functions, normal_free_coefficients = get_normal_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)
    axial_velocity_shaping_functions, axial_free_coefficients = get_axial_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)
    # Retrieve boundary conditions and central body gravitational parameter
    initial_state = bodies.get_body('Earth').state_in_base_frame_from_ephemeris(initial_time)
    final_state = bodies.get_body('Mars').state_in_base_frame_from_ephemeris(final_time)
    gravitational_parameter = bodies.get_body('Sun').gravitational_parameter
    # Create and return shape-based method
    hodographic_shaping_object = shape_based_thrust.HodographicShaping(initial_state,
                                                                       final_state,
                                                                       time_of_flight,
                                                                       gravitational_parameter,
                                                                       number_of_revolutions,
                                                                       radial_velocity_shaping_functions,
                                                                       normal_velocity_shaping_functions,
                                                                       axial_velocity_shaping_functions,
                                                                       radial_free_coefficients,
                                                                       normal_free_coefficients,
                                                                       axial_free_coefficients)
    return hodographic_shaping_object


def get_hodograph_thrust_acceleration_settings(trajectory_parameters: list,
                                               bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                                               specific_impulse: float) \
        -> tudatpy.kernel.trajectory_design.shape_based_thrust.HodographicShaping:
    """
    It extracts the acceleration settings resulting from the hodographic trajectory and returns the equivalent thrust
    acceleration settings object.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to be optimized.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    specific_impulse : float
        Constant specific impulse of the spacecraft.

    Returns
    -------
    tudatpy.kernel.numerical_simulation.propagation_setup.acceleration.ThrustAccelerationSettings
        Thrust acceleration settings object.
    """
    # Create shaping object
    shaping_object = create_hodographic_shaping_object(trajectory_parameters,
                                                       bodies)
    # Compute offset, which is the time since J2000 (when t=0 for tudat) at which the simulation starts
    # N.B.: this is different from time_buffer, which is the delay between the start of the hodographic
    # trajectory and the beginning of the simulation
    time_offset = get_trajectory_initial_time(trajectory_parameters)
    # Create specific impulse lambda function
    specific_impulse_function = lambda t: specific_impulse
    # Return acceleration settings
    return transfer_trajectory.get_low_thrust_acceleration_settings(shaping_object,
                                                                    bodies,
                                                                    'Vehicle',
                                                                    specific_impulse_function,
                                                                    time_offset)


def get_hodograph_state_at_epoch(trajectory_parameters: list,
                                 bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                                 epoch: float) -> np.ndarray:
    """
    It retrieves the Cartesian state, expressed in the inertial frame, at a given epoch of the analytical trajectory.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to be optimized.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.

    Returns
    -------
    np.ndarray
        Cartesian state in the inertial frame of the spacecraft at the given epoch.
    """
    # Create shaping object
    shaping_object = create_hodographic_shaping_object(trajectory_parameters,
                                                       bodies)
    # Define current hodograph time
    hodograph_time = epoch - get_trajectory_initial_time(trajectory_parameters)
    return shaping_object.get_state(hodograph_time)

###########################################################################
# BENCHMARK UTILITIES #####################################################
###########################################################################


# NOTE TO STUDENTS: THIS FUNCTION CAN BE EXTENDED TO GENERATE A MORE ROBUST BENCHMARK (USING MORE THAN 2 RUNS)
def generate_benchmarks(benchmark_step_size: float,
                        simulation_start_epoch: float,
                        bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                        benchmark_propagator_settings:
                        tudatpy.kernel.numerical_simulation.propagation_setup.propagator.MultiTypePropagatorSettings,
                        are_dependent_variables_present: bool,
                        output_path: str = None):
    """
    Function to generate to accurate benchmarks.

    This function runs two propagations with two different integrator settings that serve as benchmarks for
    the nominal runs. The state and dependent variable history for both benchmarks are returned and, if desired, 
    they are also written to files (to the directory ./SimulationOutput/benchmarks/) in the following way:
    * benchmark_1_states.dat, benchmark_2_states.dat
        The numerically propagated states from the two benchmarks.
    * benchmark_1_dependent_variables.dat, benchmark_2_dependent_variables.dat
        The dependent variables from the two benchmarks.

    Parameters
    ----------
    simulation_start_epoch : float
        The start time of the simulation in seconds.
    constant_specific_impulse : float
        Constant specific impulse of the vehicle.  
    minimum_mars_distance : float
        Minimum distance from Mars at which the propagation stops.
    time_buffer : float
        Time interval between the simulation start epoch and the beginning of the hodographic trajectory.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
        System of bodies present in the simulation.
    benchmark_propagator_settings
        Propagator settings object which is used to run the benchmark propagations.
    trajectory_parameters
        List that represents the trajectory parameters for the spacecraft.
    are_dependent_variables_present : bool
        If there are dependent variables to save.
    output_path : str (default: None)
        If and where to save the benchmark results (if None, results are NOT written).

    Returns
    -------
    return_list : list
        List of state and dependent variable history in this order: state_1, state_2, dependent_1_ dependent_2.
    """
    ### CREATION OF THE TWO BENCHMARKS ###
    # Define benchmarks' step sizes
    first_benchmark_step_size = benchmark_step_size
    second_benchmark_step_size = 2.0 * first_benchmark_step_size

    # Create integrator settings for the first benchmark, using a fixed step size RKDP8(7) integrator
    # (the minimum and maximum step sizes are set equal, while both tolerances are set to inf)
    benchmark_integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
        simulation_start_epoch,
        first_benchmark_step_size,
        propagation_setup.integrator.RKCoefficientSets.rkdp_87,
        first_benchmark_step_size,
        first_benchmark_step_size,
        np.inf,
        np.inf)

    print('Running first benchmark...')
    first_dynamics_simulator = numerical_simulation.SingleArcSimulator(
        bodies,
        benchmark_integrator_settings,
        benchmark_propagator_settings, print_dependent_variable_data=True)

    # Create integrator settings for the second benchmark in the same way
    benchmark_integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
        simulation_start_epoch,
        second_benchmark_step_size,
        propagation_setup.integrator.RKCoefficientSets.rkdp_87,
        second_benchmark_step_size,
        second_benchmark_step_size,
        np.inf,
        np.inf)

    print('Running second benchmark...')
    second_dynamics_simulator = numerical_simulation.SingleArcSimulator(
        bodies,
        benchmark_integrator_settings,
        benchmark_propagator_settings, print_dependent_variable_data=False)

    ### WRITE BENCHMARK RESULTS TO FILE ###
    # Retrieve state history
    first_benchmark_states = first_dynamics_simulator.state_history
    second_benchmark_states = second_dynamics_simulator.state_history
    # Write results to files
    if output_path is not None:
        save2txt(first_benchmark_states, 'benchmark_1_states.dat', output_path)
        save2txt(second_benchmark_states, 'benchmark_2_states.dat', output_path)
    # Add items to be returned
    return_list = [first_benchmark_states,
                   second_benchmark_states]

    ### DO THE SAME FOR DEPENDENT VARIABLES ###
    if are_dependent_variables_present:
        # Retrieve dependent variable history
        first_benchmark_dependent_variable = first_dynamics_simulator.dependent_variable_history
        second_benchmark_dependent_variable = second_dynamics_simulator.dependent_variable_history
        # Write results to file
        if output_path is not None:
            save2txt(first_benchmark_dependent_variable, 'benchmark_1_dependent_variables.dat', output_path)
            save2txt(second_benchmark_dependent_variable, 'benchmark_2_dependent_variables.dat', output_path)
        # Add items to be returned
        return_list.append(first_benchmark_dependent_variable)
        return_list.append(second_benchmark_dependent_variable)

    return return_list


def compare_benchmarks(first_benchmark: dict,
                       second_benchmark: dict,
                       output_path: str,
                       filename: str) -> dict:
    """
    It compares the results of two benchmark runs.

    It uses an 8th-order Lagrange interpolator to compare the state (or the dependent variable, depending on what is
    given as input) history. The difference is returned in form of a dictionary and, if desired, written to a file named
    filename and placed in the directory output_path.

    Parameters
    ----------
    first_benchmark : dict
        State (or dependent variable history) from the first benchmark.
    second_benchmark : dict
        State (or dependent variable history) from the second benchmark.
    output_path : str
        If and where to save the benchmark results (if None, results are NOT written).
    filename : str
        Name of the output file.

    Returns
    -------
    benchmark_difference : dict
        Interpolated difference between the two benchmarks' state (or dependent variable) history.
    """
    # Create 8th-order Lagrange interpolator for first benchmark
    benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(first_benchmark,
                                                                                      interpolators.lagrange_interpolation(8))
    # Calculate the difference between the benchmarks
    print('Calculating benchmark differences...')
    # Initialize difference dictionaries
    benchmark_difference = dict()
    # Calculate the difference between the states and dependent variables in an iterative manner
    for second_epoch in second_benchmark.keys():
        benchmark_difference[second_epoch] = benchmark_interpolator.interpolate(second_epoch) - \
                                             second_benchmark[second_epoch]
    # Write results to files
    if output_path is not None:
        save2txt(benchmark_difference, filename, output_path)
    # Return the interpolator
    return benchmark_difference


def compare_models(first_model: dict,
                   second_model: dict,
                   interpolation_epochs: np.ndarray,
                   output_path: str,
                   filename: str) -> dict:
    """
    It compares the results of two runs with different model settings.
    It uses an 8th-order Lagrange interpolator to compare the state (or the dependent variable, depending on what is
    given as input) history. The difference is returned in form of a dictionary and, if desired, written to a file named
    filename and placed in the directory output_path.
    Parameters
    ----------
    first_model : dict
        State (or dependent variable history) from the first run.
    second_model : dict
        State (or dependent variable history) from the second run.
    interpolation_epochs : np.ndarray
        Vector of epochs at which the two runs are compared.
    output_path : str
        If and where to save the benchmark results (if None, results are NOT written).
    filename : str
        Name of the output file.
    Returns
    -------
    model_difference : dict
        Interpolated difference between the two simulations' state (or dependent variable) history.
    """
    # Create interpolator settings
    interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation=interpolators.use_boundary_value)
    # Create 8th-order Lagrange interpolator for both cases
    first_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_model, interpolator_settings)
    second_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        second_model, interpolator_settings)
    # Calculate the difference between the first and second model at specific epochs
    model_difference = {epoch: second_interpolator.interpolate(epoch) - first_interpolator.interpolate(epoch)
                        for epoch in interpolation_epochs}
    # Write results to files
    if output_path is not None:
        save2txt(model_difference,
                 filename,
                 output_path)
    # Return the model difference
    return model_difference

##########################################
### Design Space Exploration Functions ###
##########################################

def orth_arrays(nfact : int, nlevels : int) -> tuple((np.array, int)):
    """ 
    Erwin's Matlab Steps:
    Create ortogonal arrays from Latin Square in 4 successive steps:
    
    0) Take the column from the smaller array to create 2 new
       columns and 2x new rows,
    1) block 1 (1/2 rows): take old values 2x for new columns,
    2) block 2 (1/2 rows): take old values, use Latin-Square for new
       columns,
    3) column 1: divide experiments into groups of 1,2.
    """

    ierror = 0
    icount = 0
    # Simple lambda functions to create size of orthogonal array
    row_number = lambda icount, nlevels : nlevels**(icount+1)
    col_number = lambda row_number : row_number-1

    ###################################
    ### If 2 Level orthogonal array ###
    ###################################

    #Determining the number of rows
    if nlevels == 2:
        if nfact >= 2 and nfact <= 3:
                icount = 1
        elif nfact >= 4 and nfact <= 7:
                icount = 2
        elif nfact >= 8 and nfact <= 15:
                icount = 3
        elif nfact >= 16 and nfact <= 31:
                icount = 4
        elif nfact >= 32 and nfact <= 63:
                icount = 5
        elif nfact >= 64 and nfact <= 127:
                icount = 6
        elif nfact >= 128 and nfact <= 255:
                icount = 7
        else:
                ierror = 1
                Lx = np.zeros(1)
                return Lx, ierror

        Lxrow = row_number(icount, nlevels)
        Lxcol = col_number(Lxrow)
        Lx = np.zeros((Lxrow,Lxcol))
        iaux = Lx.copy()
        
        ### Define the 2-level Latin Square ###
        LS = np.zeros((2,2))
        LS[0,0] = -1
        LS[0,1] =  1
        LS[1,0] =  1
        LS[1,1] = -1
        # Other relevant lists for filling in the 2-level array
        index_list = [0, 1]
        two_level = [-1, 1]
        
        # In case of only one factor, copy the first Latin Square and leave the subroutine.
        if icount == 0:
                Lx[0,0] = LS[0,1]
                Lx[1,0] = LS[0,1]
                return Lx, ierror
        
        iaux[0,0] = -1
        iaux[1,0] =  1
        irow = 2
        icol = 1

        # Some weirdness is required here because the original algorithm in Matlab starts from index 1
        Lx = np.hstack((np.zeros((len(Lx), 1)), Lx))
        Lx = np.vstack((np.zeros((1, len(Lx[0,:]))), Lx))
        iaux = np.hstack((np.zeros((len(iaux), 1)), iaux))
        iaux = np.vstack((np.zeros((1, len(iaux[0,:]))), iaux))
        
        ### Fill in orthogonal array ###
        for i1 in range(1, icount + 1):
                for i2 in range(1, irow + 1):
                        for i3 in range(1, icol + 1):
                                for p in range(2):
                                        for q in range(2):
                                                for r in range(2):
                                                        #Block 1.
                                                        if iaux[i2,i3] == two_level[q] and p == 0:
                                                                Lx[i2,i3*2 + index_list[r]] = two_level[q] 
                                                        #Block 2
                                                        if iaux[i2,i3] == two_level[q] and p == 1:
                                                                Lx[i2 + irow,i3*2 + index_list[r]] = LS[index_list[q], index_list[r]]
                                        Lx[i2 + irow*p,1] = two_level[p]

                if i1 == icount:
                        # Deleting extra row from Matlab artifact
                        Lx = np.delete(Lx, 0, 0)
                        Lx = np.delete(Lx, 0, 1)
                        return Lx, ierror
                irow = 2*irow
                icol = 2*icol+1
                for i2 in range(1, irow + 1):
                        for i3 in range(1, icol + 1):
                                iaux[i2,i3] = Lx[i2,i3]

    ###################################
    ### If 3 Level orthogonal array ###
    ###################################

    #Determining the number of rows
    elif nlevels == 3:
        if nfact >= 2 and nfact <= 4:
                icount = 1
        elif nfact >= 5 and nfact <= 13:
                icount = 2
        elif nfact >= 14 and nfact <= 40:
                icount = 3
        elif nfact >= 41 and nfact <= 121:
                icount = 4
        else:
                ierror = 1
                Lx = np.zeros(1)
                return Lx, ierror

        Lxrow = row_number(icount, nlevels)
        Lxcol = col_number(Lxrow) // 2
        Lx = np.zeros((Lxrow,Lxcol))
        iaux = Lx.copy()
        
        # Relevant lists for filling in the 3-level array
        index_list = [0, 1, 2]
        three_level = [-1, 0, 1]
        ### Define the two three-level Latin Squares. Latin Square 1 ###
        LS1 = np.zeros((3,3))
        for i in range(3):
                for j in range(3):
                                LS1[i,index_list[j]] = three_level[(j+i)%3];
        ### ... and Latin Square 2. ###
        LS2 = np.zeros((3,3))
        three_level_2 = [-1, 1, 0]
        for i in range(3):
                for j in range(3):
                        LS2[i, index_list[j]] = three_level_2[j-i]
         
        ### In case of only one factor, copy the first Latin Square and leave the subroutine. ###
        if icount == 0:
           Lx[0,0] = LS1[0,0];
           Lx[1,0] = LS1[0,1];
           Lx[2,0] = LS1[0,2];
           return Lx, ierror

        ### Define iaux for loops ###
        iaux[0,0] = -1
        iaux[1,0] = 0
        iaux[2,0] =  1
        irow = 3
        icol = 1

        # Some weirdness is required here because the original algorithm in Matlab starts from index 1
        Lx = np.hstack((np.zeros((len(Lx), 1)), Lx))
        Lx = np.vstack((np.zeros((1, len(Lx[0,:]))), Lx))
        iaux = np.hstack((np.zeros((len(iaux), 1)), iaux))
        iaux = np.vstack((np.zeros((1, len(iaux[0,:]))), iaux))
        
        ### Filling in orthogonal array ###
        for i1 in range(1, icount + 1):
                for i2 in range(1, irow + 1):
                        for i3 in range(1, icol + 1):
                                for p in range(3):
                                        for q in range(3):
                                                for r in range(3):
                                                        #Block 1.
                                                        if iaux[i2,i3] == three_level[q] and p == 0:
                                                                Lx[i2 + irow*p,i3*3 + three_level[r]] = three_level[q] 
                                                        #Block 2.
                                                        if iaux[i2,i3] == three_level[q] and p == 1:
                                                                Lx[i2 + irow*p,i3*3 + three_level[r]] = LS1[index_list[q], index_list[r]]
                                                        #Block 3.
                                                        if iaux[i2,i3] == three_level[q] and p == 2:
                                                                Lx[i2 + irow*p,i3*3 + three_level[r]] = LS2[index_list[q], index_list[r]]
                                        Lx[i2 + irow*p,1] = three_level[p]

                if i1 == icount:
                        # Deleting extra row from Matlab artifact
                        Lx = np.delete(Lx, 0, 0)
                        Lx = np.delete(Lx, 0, 1)
                        return Lx, ierror
                irow = 3*irow
                icol = 3*icol+1
                for i2 in range(1, irow + 1):
                        for i3 in range(1, icol + 1):
                                iaux[i2,i3] = Lx[i2,i3]
    else:
        print('These levels are not implemented yet. (You may wonder whether you need them)')



def yates_array(no_of_levels : int, no_of_factors : int) -> np.array:
    """
    Function that creates a yates array according to yates algorithm

    Sources: 
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda35i.htm 
    https://en.wikipedia.org/wiki/Yates_analysis

    no_of_levels : The number of levels a factor can attain

    no_of_factors : The number of design variables in the problem

    """

    # The values that can be entered into yates array, depends on the no_of_levels
    levels = []
    for i in range(no_of_levels+1):
        levels.append(i)

    n_rows = no_of_levels**no_of_factors
    n_cols = no_of_factors
    yates_array = np.zeros((n_rows, n_cols), dtype='int')

    row_seg = n_rows
    for col in range(n_cols):
        repetition_amount = no_of_levels**col # Number of times to repeat the row segment to fill the array
        row_seg = row_seg // no_of_levels # Get row segment divided by number of levels
        for j in range(repetition_amount):
            for i in range(no_of_levels): 
                # The values are entered from position i to position i + no_of_levels
                yates_array[(i*row_seg + j*row_seg*no_of_levels):((i+1)*row_seg + j*row_seg*no_of_levels), col] = levels[i] 
    return yates_array


def yates_array_v2(no_of_levels: int, no_of_factors: int) -> np.array:
    """
    Function that creates a yates array according to yates algorithm

    no_of_levels : The number of levels a factor can attain

    no_of_factors : The number of design variables in the problem

    Return : np.array (no_of_levels**no_of_factors, no_of_factors)

    """
    if no_of_levels == 2:
        levels = [-1, 1]
    #    elif no_of_levels == 3:
    #        levels = [-1, 0, 1]
    #    elif no_of_levels == 4:
    #        levels = [-2, -1, 1, 2]
    #    elif no_of_levels == 5:
    #        levels = [-2, -1, 0, 1, 2]
    #    elif no_of_levels == 6:
    #        levels = [-3, -2, -1, 1, 2, 3]
    #    elif no_of_levels == 7:
    #        levels = [-3, -2, -1, 0, 1, 2, 3]
    #    elif no_of_levels == 8:
    #        levels = [-4, -3, -2, -1, 1, 2, 3, 4]
    # levels = []
    # for i in range(no_of_levels+1):
    #    levels.append(i)

    n_rows = no_of_levels ** no_of_factors
    n_cols = no_of_factors
    yates_array = np.zeros((n_rows, n_cols), dtype='int')

    row_seg = n_rows
    for col in range(n_cols):
        repetition_amount = no_of_levels ** col  # Number of times to repeat the row segment to fill the array
        row_seg = row_seg // no_of_levels  # Get row segment divided by number of levels
        for j in range(repetition_amount):
            for it, level in enumerate(levels):
                # fill in from position i to position i + no_of_levels
                yates_array[(it * row_seg + j * row_seg * no_of_levels):((it + 1) * row_seg +
                                                                         j * row_seg * no_of_levels), col] = levels[it]
    return yates_array


def anova_analysis(objective_values,
                   factorial_design_array: np.array,
                   no_of_factors=4,
                   no_of_levels=2,
                   level_of_interactions=2):
    """
    Function that performs an ANOVA for 2 levels and n factors. After some definitions, we iterate
    through yates array, depending on the value being -1 or 1, we add the occurrence to a certain
    container. Later the amount of items in the container determines the contribution of that
    specific variable to an objective value. This iteration is done for individual, linear, and
    quadratic effects, thereby determing the contribution of all interactions.

    objective_values : list of objective function values following from the factorial design array

    factorial_design_array : np.array from yates_array_v2 function

    level_of_interactions : Integer either 2 or 3, depending on what interactions you want to
    include

    Return : This analysis returns the contributions of all parameters/interactions to the
    specified objective. Specifically, it returns 4 elements: Pi, Pij, Pijk, and Pe. These are lists
    that contain percentage contributions to the objective. Pi are the individual parameters, Pij
    are linear interactions, and Pijk are the quadratic interactions. Pe are errors.

    """

    assert len(objective_values) == len(factorial_design_array)  # Quick check for validity of data

    number_of_simulations = no_of_levels ** no_of_factors

    #########################
    ### Array definitions ###
    #########################

    # Lambda function to determine number of interaction columns
    interactions = lambda iterable, k: [i for i in comb(range(iterable), k)]
    no_of_interactions_2 = interactions(no_of_factors, 2)  # number of 2-level interactions
    no_of_interactions_3 = interactions(no_of_factors, 3)  # number of 3-level interactions

    # Arrays for individual, 2, and 3 level interactions - sum of squares
    sum_of_squares_i = np.zeros(no_of_factors)
    sum_of_squares_ij = np.zeros(len(no_of_interactions_2))
    sum_of_squares_ijk = np.zeros(len(no_of_interactions_3))

    # Arrays for individual, 2, and 3 level interactions - percentage contribution
    percentage_contribution_i = np.zeros(no_of_factors)
    percentage_contribution_ij = np.zeros(len(no_of_interactions_2))
    percentage_contribution_ijk = np.zeros(len(no_of_interactions_3))

    # Sum of objective values and mean of parameters
    sum_of_objective = np.sum(objective_values)
    mean_of_param = sum_of_objective / number_of_simulations

    # Variance and standard deviation of data
    variance_per_run = np.zeros(number_of_simulations)
    objective_values_squared = np.zeros(number_of_simulations)
    for i in range(len(factorial_design_array)):
        variance_per_run[i] = (objective_values[i] - mean_of_param) ** 2 / (number_of_simulations - 1)
        objective_values_squared[i] = objective_values[i] ** 2
    variance = np.sum(variance_per_run)
    standard_deviation = np.sqrt(variance)

    # Square of sums
    CF = sum_of_objective * mean_of_param
    # Difference square of sums and sum of squares
    sum_of_deviation = np.sum(objective_values_squared) - CF

    #####################################
    ### Iterations through yates array ###
    #####################################

    ### Linear effects ###
    # Container for appearance of minimum/maximum value
    Saux = np.zeros((no_of_factors, no_of_levels))
    number_of_appearanes = number_of_simulations / no_of_levels
    for j in range(number_of_simulations):
        for i in range(no_of_factors):
            if factorial_design_array[j, i] == -1:
                Saux[i, 0] += objective_values[j]
            else:
                Saux[i, 1] += objective_values[j]

    if sum_of_deviation > 1e-6:  # If there is a deviation, then there is a contribution.
        for i in range(no_of_factors):
            sum_of_squares_i[i] = (1 / 2) * (Saux[i, 1] - Saux[i, 0]) ** 2 / number_of_appearanes
            percentage_contribution_i[i] = 100 * sum_of_squares_i[i] / sum_of_deviation

    ### 2-level interaction ###
    # Container for appearance of minimum/maximum value
    Saux = np.zeros((len(no_of_interactions_2), no_of_levels))
    for j in range(number_of_simulations):
        for i in range(len(no_of_interactions_2)):
            # Interaction sequence of all possible 2-level interactions is created by multiplying the
            # two respective elements from yates array
            interaction = factorial_design_array[j, no_of_interactions_2[i][0]] * \
                          factorial_design_array[j, no_of_interactions_2[i][1]]
            if interaction == -1:
                Saux[i, 0] += objective_values[j]
            else:
                Saux[i, 1] += objective_values[j]

    if sum_of_deviation > 1e-6:  # If there is a deviation, then there is a contribution.
        for i in range(len(no_of_interactions_2)):
            sum_of_squares_ij[i] = (1 / 2) * (Saux[i, 1] - Saux[i, 0]) ** 2 / number_of_appearanes
            percentage_contribution_ij[i] = 100 * sum_of_squares_ij[i] / sum_of_deviation

    ### 3-level interaction ###
    # Container for appearance of minimum/maximum value
    Saux = np.zeros((len(no_of_interactions_3), no_of_levels))
    for j in range(number_of_simulations):
        for i in range(len(no_of_interactions_3)):
            # Interaction sequence of all possible 3-level interactions is created by multiplying the
            # three respective elements from yates array
            interaction = factorial_design_array[j, no_of_interactions_3[i][0]] * \
                          factorial_design_array[j, no_of_interactions_3[i][1]] * \
                          factorial_design_array[j, no_of_interactions_3[i][2]]
            if interaction == -1:
                Saux[i, 0] += objective_values[j]
            else:
                Saux[i, 1] += objective_values[j]

    if sum_of_deviation > 1e-6:  # If there is a deviation, then there is a contribution.
        for i in range(len(no_of_interactions_3)):
            sum_of_squares_ijk[i] = (1 / 2) * (Saux[i, 1] - Saux[i, 0]) ** 2 / number_of_appearanes
            percentage_contribution_ijk[i] = 100 * sum_of_squares_ijk[i] / sum_of_deviation

    ### Error contribution ###
    sum_of_squares_error = sum_of_deviation - np.sum(sum_of_squares_i) - \
                           np.sum(sum_of_squares_ij) - np.sum(sum_of_squares_ijk)

    percentage_contribution_error = 0
    if sum_of_deviation > 1e-6:  # If there is a deviation, then there is a contribution.
        percentage_contribution_error = 100 * sum_of_squares_error / sum_of_deviation

    """
    Because the function returns 4 separate variables, this is how data can be saved from this
    function:

    percentage_contribution_i, percentage_contribution_ij, percentage_contribution_ijk,
    percentage_contribution_e = Util.anova_analysis(<objectives>, <yates_array>)
    """

    return percentage_contribution_i, percentage_contribution_ij, \
           percentage_contribution_ijk, percentage_contribution_error