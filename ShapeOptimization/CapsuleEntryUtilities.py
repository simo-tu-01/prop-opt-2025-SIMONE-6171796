'''
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

This module defines useful functions that will be called by the main script, where the optimization is executed.
'''

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
from math import pi

# Tudatpy imports
import tudatpy
from tudatpy.data import save2txt
from tudatpy import constants
from tudatpy.interface import spice as spice_interface
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import environment
from tudatpy import numerical_simulation
from tudatpy.astro import element_conversion
from tudatpy.math import interpolators
from tudatpy.math import geometry

###########################################################################
# PROPAGATION SETTING UTILITIES ###########################################
###########################################################################

def get_initial_state(simulation_start_epoch: float,
                      bodies: tudatpy.numerical_simulation.environment.SystemOfBodies) -> np.ndarray:
    """
    Converts the initial state to inertial coordinates.

    The initial state is expressed in Earth-centered spherical coordinates.
    These are first converted into Earth-centered cartesian coordinates,
    then they are finally converted in the global (inertial) coordinate
    system.

    Parameters
    ----------
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    bodies : tudatpy.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.

    Returns
    -------
    initial_state_inertial_coordinates : np.ndarray
        The initial state of the vehicle expressed in inertial coordinates.
    """
    # Set initial spherical elements
    radial_distance = spice_interface.get_average_radius('Earth') + 120.0E3
    latitude = np.deg2rad(0.0)
    longitude = np.deg2rad(68.75)
    speed = 7.63E3
    flight_path_angle = np.deg2rad(-0.8)
    heading_angle = np.deg2rad(34.37)

    # Convert spherical elements to body-fixed cartesian coordinates
    initial_cartesian_state_body_fixed = element_conversion.spherical_to_cartesian_elementwise(
        radial_distance, latitude, longitude, speed, flight_path_angle, heading_angle)
    # Get rotational ephemerides of the Earth
    earth_rotational_model = bodies.get_body('Earth').rotation_model
    # Transform the state to the global (inertial) frame
    initial_cartesian_state_inertial = environment.transform_to_inertial_orientation(
        initial_cartesian_state_body_fixed,
        simulation_start_epoch,
        earth_rotational_model)

    return initial_cartesian_state_inertial

def get_termination_settings(simulation_start_epoch: float,
                             maximum_duration: float,
                             termination_altitude: float) \
        -> tudatpy.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings:
    """
    Get the termination settings for the simulation.

    Termination settings currently include:
    - simulation time (one day)
    - lower and upper altitude boundaries (0-100 km)
    - fuel run-out

    Parameters
    ----------
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    maximum_duration : float
        Maximum duration of the simulation [s].
    termination_altitude : float
        Minimum altitude [m].

    Returns
    -------
    hybrid_termination_settings : tudatpy.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object.
    """
    # Create single PropagationTerminationSettings objects
    # Time
    time_termination_settings = propagation_setup.propagator.time_termination(
        simulation_start_epoch + maximum_duration,
        terminate_exactly_on_final_condition=False
    )
    # Altitude
    lower_altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.altitude('Capsule', 'Earth'),
        limit_value=termination_altitude,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False
    )
    # Define list of termination settings
    termination_settings_list = [time_termination_settings,
                                 lower_altitude_termination_settings]

    # Create termination settings object (when either the time of altitude condition is reached: propaation terminates)
    hybrid_termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                                  fulfill_single_condition=True)
    return hybrid_termination_settings


# NOTE TO STUDENTS: this function can be modified to save more/less dependent variables.
def get_dependent_variable_save_settings() -> list:
    """
    Retrieves the dependent variables to save.

    Currently, the dependent variables saved include:
    - the Mach number
    - the altitude wrt the Earth

    Parameters
    ----------
    none

    Returns
    -------
    dependent_variables_to_save : list[tudatpy.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    """
    dependent_variables_to_save = [propagation_setup.dependent_variable.mach_number('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.altitude('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.keplerian_state('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.dynamic_pressure('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.latitude('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.longitude('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.heading_angle('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.flight_path_angle('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.total_acceleration_norm('Capsule'),
                                   propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.aerodynamic_type, 'Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.intermediate_aerodynamic_rotation_matrix_variable('Capsule', 
                                                                                                                          numerical_simulation.environment.inertial_frame, 
                                                                                                                          numerical_simulation.environment.aerodynamic_frame,
                                                                                                                          'Earth'),                                                                                    
                                   propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.point_mass_gravity_type, 'Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.tnw_to_inertial_rotation_matrix('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.airspeed('Capsule', 'Earth')] 
    
    return dependent_variables_to_save

# NOTE TO STUDENTS: THIS FUNCTION SHOULD BE EXTENDED TO USE MORE INTEGRATORS FOR ASSIGNMENT 1.
def get_integrator_settings_old(propagator_index: int,
                            integrator_index: int,
                            settings_index: int,
                            simulation_start_epoch: float) \
        -> tudatpy.numerical_simulation.propagation_setup.integrator.IntegratorSettings:
    """

    Retrieves the integrator settings.

    It selects a combination of integrator to be used (first argument) and
    the related setting (tolerance for variable step size integrators
    or step size for fixed step size integrators). The code, as provided, runs the following:
    - if j=0,1,2,3: a variable-step-size, multi-stage integrator is used (see multiStageTypes list for specific type),
                     with tolerances 10^(-10+*k)
    - if j=4      : a fixed-step-size RK4 integrator is used, with step-size 2^(k)

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
        Index that selects the tolerance or the step size (depending on the integrator type).
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.

    Returns
    -------
    integrator_settings : tudatpy.numerical_simulation.propagation_setup.integrator.IntegratorSettings
        Integrator settings to be provided to the dynamics simulator.
    """
    # Define list of multi-stage integrators
    multi_stage_integrators = [propagation_setup.integrator.CoefficientSets.rkf_45,
                               propagation_setup.integrator.CoefficientSets.rkf_56,
                               propagation_setup.integrator.CoefficientSets.rkf_78,
                               propagation_setup.integrator.CoefficientSets.rkdp_87]

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
        integrator_settings = integrator.runge_kutta_variable_step_size(
            simulation_start_epoch,
            1.0,
            current_coefficient_set,
            np.finfo(float).eps,
            np.inf,
            current_tolerance,
            current_tolerance)
    # Use fixed step-size integrator
    else:
        # Compute time step
        fixed_step_size = 2 ** settings_index
        # Create integrator settings
        integrator = propagation_setup.integrator
        integrator_settings = integrator.runge_kutta_fixed_step_size(
            fixed_step_size, propagation_setup.integrator.CoefficientSets.rk_4)

    return integrator_settings

def get_integrator_settings(propagator_settings: tudatpy.numerical_simulation.propagation_setup.integrator.IntegratorSettings,
                            integrator_index: int,
                            settings: float,
                            fixed = False) \
        -> tudatpy.numerical_simulation.propagation_setup.integrator.IntegratorSettings:
    """
    Retrieves the integrator settings.

    Supports 8 integrators:
    - integrator_index 0 to 3: Variable-step methods:
        0 -> RKF4(5)
        1 -> RKF5(6)
        2 -> RKDP8(7)
        3 -> RKF12(10)
    - integrator_index 4 to 7: Fixed-step methods using lower order of the above:
        4 -> RK4
        5 -> RK5
        6 -> RK7
        7 -> RK10

    settings_index defines:
        - the tolerance for variable-step methods: 10^(-10 + settings_index)
        - the fixed step size for fixed-step methods: 2^settings_index [s]

    Parameters
    ----------
    propagator_index : int
        (Unused, reserved for future use)
    integrator_index : int
        Index selecting one of the 8 integrators
    settings: float
        Tolerance (var-step) or step size (fixed-step)
    simulation_start_epoch : float
        Start time of the simulation 

    Returns
    -------
    integrator_settings : tudatpy.numerical_simulation.propagation_setup.integrator.IntegratorSettings
        The integrator settings object.
    """

    # Define integrator coefficient sets
    integrator_sets = [
        propagation_setup.integrator.rkf_45,
        propagation_setup.integrator.rkf_56,
        propagation_setup.integrator.rkdp_87,
        propagation_setup.integrator.rkf_1210
    ]

    if fixed:
        # Fixed-step integrator
        current_coefficient_set = integrator_sets[integrator_index - 4]
        fixed_step_size = settings

        propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
            fixed_step_size,
            coefficient_set=current_coefficient_set,
            order_to_use=propagation_setup.integrator.lower
        )
    else:
        # Variable-step integrator
        current_coefficient_set = integrator_sets[integrator_index]
        current_tolerance = settings

        minimum_step_size = 1.0e-12
        maximum_step_size = np.inf

        propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size( 
            1.0,  
            current_coefficient_set,
            minimum_step_size,
            maximum_step_size,
            current_tolerance,
            current_tolerance
        )

    return propagator_settings


def get_propagator_settings(shape_parameters,
                            bodies,
                            simulation_start_epoch,
                            termination_settings,
                            dependent_variables_to_save,
                            current_propagator = propagation_setup.propagator.cowell ):
    """
    Creates the propagator settings.

    This function creates the propagator settings for translational motion and mass, for the given simulation settings
    Note that, in this function, the entry of the shape_parameters representing the vehicle attitude (angle of attack)
    is processed to redefine the vehice attitude. The propagator settings that are returned as output of this function
    are not yet usable: they do not contain any integrator settings, which should be set at a later point by the user

    Parameters
    ----------
    shape_parameters : list[ float ]
        List of free parameters for the low-thrust model, which will be used to update the vehicle properties such that
        the new thrust/magnitude direction are used. The meaning of the parameters in this list is stated at the
        start of the *Propagation.py file
    bodies : tudatpy.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    termination_settings : tudatpy.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object to be used
    dependent_variables_to_save : list[tudatpy.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    current_propagator : tudatpy.numerical_simulation.propagation_setup.propagator.TranslationalPropagatorType
        Type of propagator to be used for translational dynamics

    Returns
    -------
    propagator_settings : tudatpy.numerical_simulation.propagation_setup.integrator.MultiTypePropagatorSettings
        Propagator settings to be provided to the dynamics simulator.
    """

    # Define bodies that are propagated and their central bodies of propagation
    bodies_to_propagate = ['Capsule']
    central_bodies = ['Earth']

    # Define accelerations acting on capsule
    acceleration_settings_on_vehicle = {
        'Earth': [propagation_setup.acceleration.point_mass_gravity(),
                  propagation_setup.acceleration.aerodynamic()]
    }
    # Create acceleration models.
    acceleration_settings = {'Capsule': acceleration_settings_on_vehicle}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    new_angles = np.array([shape_parameters[5], 0.0, 0.0])
    new_angle_function = lambda time : new_angles
    bodies.get_body('Capsule').rotation_model.reset_aerodynamic_angle_function( new_angle_function )


    # Retrieve initial state
    initial_state = get_initial_state(simulation_start_epoch, bodies)

    # Create propagation settings for the translational dynamics. NOTE: these are not yet 'valid', as no
    # integrator settings are defined yet
    propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                     acceleration_models,
                                                                     bodies_to_propagate,
                                                                     initial_state,
                                                                     simulation_start_epoch,
                                                                     None,
                                                                     termination_settings,
                                                                     current_propagator,
                                                                     output_variables=dependent_variables_to_save)
    
    return propagator_settings


###########################################################################
# CAPSULE SHAPE/AERODYNAMICS UTILITIES ####################################
###########################################################################


def get_capsule_coefficient_interface(capsule_shape: tudatpy.math.geometry.Capsule) \
        -> tudatpy.numerical_simulation.environment.HypersonicLocalInclinationAnalysis:
    """
    Function that creates an aerodynamic database for a capsule, based on a set of shape parameters.

    The Capsule shape consists of four separate geometrical components: a sphere segment for the nose, a torus segment
    for the shoulder/edge, a conical frustum for the rear body, and a sphere segment for the rear cap (see Dirkx and
    Mooij, 2016). The code used in this function discretizes these surfaces into a structured mesh of quadrilateral
    panels. The parameters number_of_points and number_of_lines define the number of discretization points (for each
    part) in both independent directions (lengthwise and circumferential). The list selectedMethods defines the type of
    aerodynamic analysis method that is used.

    Parameters
    ----------
    capsule_shape : tudatpy.math.geometry.Capsule
        Object that defines the shape of the vehicle.

    Returns
    -------
    hypersonic_local_inclination_analysis : tudatpy.environment.HypersonicLocalInclinationAnalysis
        Database created through the local inclination analysis method.
    """

    # Define settings for surface discretization of the capsule
    number_of_lines = [31, 31, 31, 11]
    number_of_points = [31, 31, 31, 11]
    # Set side of the vehicle (DO NOT CHANGE THESE: setting to true will turn parts of the vehicle 'inside out')
    invert_order = [0, 0, 0, 0]

    # Define moment reference point. NOTE: This value is chosen somewhat arbitrarily, and will only impact the
    # results when you consider any aspects of moment coefficients
    moment_reference = np.array([-0.6624, 0.0, 0.1369])

    # Define independent variable values
    independent_variable_data_points = []
    # Mach
    mach_points = environment.get_default_local_inclination_mach_points()
    independent_variable_data_points.append(mach_points)
    # Angle of attack
    angle_of_attack_points = np.linspace(np.deg2rad(-40),np.deg2rad(40),17)
    independent_variable_data_points.append(angle_of_attack_points)
    # Angle of sideslip
    angle_of_sideslip_points = environment.get_default_local_inclination_sideslip_angle_points()
    independent_variable_data_points.append(angle_of_sideslip_points)

    # Define local inclination method to use (index 0=Newtonian flow)
    selected_methods = [[0, 0, 0, 0], [0, 0, 0, 0]]

    # Get the capsule middle radius
    capsule_middle_radius = capsule_shape.middle_radius
    # Calculate reference area
    reference_area = np.pi * capsule_middle_radius ** 2

    # Create aerodynamic database
    hypersonic_local_inclination_analysis = environment.HypersonicLocalInclinationAnalysis(
        independent_variable_data_points,
        capsule_shape,
        number_of_lines,
        number_of_points,
        invert_order,
        selected_methods,
        reference_area,
        capsule_middle_radius,
        moment_reference)
    return hypersonic_local_inclination_analysis


def set_capsule_shape_parameters(shape_parameters: list,
                                 bodies: tudatpy.numerical_simulation.environment.SystemOfBodies,
                                 capsule_density: float):
    """
    It computes and creates the properties of the capsule (shape, mass, aerodynamic coefficient interface...).

    Parameters
    ----------
    shape_parameters : list of floats
        List of shape parameters to be optimized.
    bodies : tudatpy.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    capsule_density : float
        Constant density of the vehicle.

    Returns
    -------
    none
    """
    # Compute shape constraint
    length_limit = shape_parameters[1] - shape_parameters[4] * (1 - np.cos(shape_parameters[3]))
    length_limit /= np.tan(- shape_parameters[3])
    # Add safety factor
    length_limit -= 0.01
    # Apply constraint
    if shape_parameters[2] >= length_limit:
        shape_parameters[2] = length_limit

    # Create capsule
    new_capsule = geometry.Capsule(*shape_parameters[0:5])
    # Compute new body mass
    new_capsule_mass = capsule_density * new_capsule.volume
    # Set capsule mass
    bodies.get_body('Capsule').mass = new_capsule_mass
    # Create aerodynamic interface from shape parameters (this calls the local inclination analysis)
    new_aerodynamic_coefficient_interface = get_capsule_coefficient_interface(new_capsule)
    # Update the Capsule's aerodynamic coefficient interface
    bodies.get_body('Capsule').aerodynamic_coefficient_interface = new_aerodynamic_coefficient_interface


# NOTE TO STUDENTS: if and when making modifications to the capsule shape, do include them in this function and not in
# the main code.
def add_capsule_to_body_system(bodies: tudatpy.numerical_simulation.environment.SystemOfBodies,
                               shape_parameters: list,
                               capsule_density: float):
    """
    It creates the capsule body object and adds it to the body system, setting its shape based on the shape parameters
    provided.

    Parameters
    ----------
    bodies : tudatpy.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    shape_parameters : list of floats
        List of shape parameters to be optimized.
    capsule_density : float
        Constant density of the vehicle.

    Returns
    -------
    none
    """
    # Create new vehicle object and add it to the existing system of bodies
    bodies.create_empty_body('Capsule')
    constant_angles = np.zeros([3,1])
    constant_angles[ 0 ] = shape_parameters[ 5 ]
    angle_function = lambda time : constant_angles
    environment_setup.add_rotation_model( bodies, 'Capsule',
                                          environment_setup.rotation_model.aerodynamic_angle_based(
                                              'Earth', 'J2000', 'CapsuleFixed', angle_function ))
    # Update the capsule shape parameters
    set_capsule_shape_parameters(shape_parameters,
                                 bodies,
                                 capsule_density)



###########################################################################
# BENCHMARK UTILITIES #####################################################
###########################################################################


# NOTE TO STUDENTS: THIS FUNCTION CAN BE EXTENDED TO GENERATE A MORE ROBUST BENCHMARK (USING MORE THAN 2 RUNS)
def generate_benchmarks(benchmark_step_size,
                        simulation_start_epoch: float,
                        bodies: tudatpy.numerical_simulation.environment.SystemOfBodies,
                        benchmark_propagator_settings: tudatpy.numerical_simulation.propagation_setup.propagator.TranslationalStatePropagatorSettings,
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
    benchmark_step_size : float
        Time step of the benchmark that will be used. Two benchmark simulations will be run, both fixed-step 8th order
         (first benchmark uses benchmark_step_size, second benchmark uses 2.0 * benchmark_step_size)
    bodies : tudatpy.numerical_simulation.environment.SystemOfBodies,
        System of bodies present in the simulation.
    benchmark_propagator_settings
        Propagator settings object which is used to run the benchmark propagations.
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
    first_benchmark_step_size = benchmark_step_size  # s
    second_benchmark_step_size = 2.0 * first_benchmark_step_size

    # Create integrator settings for the first benchmark, using a fixed step size RKDP8(7) integrator
    # (the minimum and maximum step sizes are set equal, while both tolerances are set to inf)
    benchmark_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        first_benchmark_step_size,
        propagation_setup.integrator.CoefficientSets.rkf_56)
    benchmark_propagator_settings.print_settings.print_dependent_variable_indices = True

    first_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        benchmark_propagator_settings )

    # Create integrator settings for the second benchmark in the same way
    benchmark_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        second_benchmark_step_size,
        propagation_setup.integrator.CoefficientSets.rkf_56)
    benchmark_propagator_settings.print_settings.print_dependent_variable_indices = False

    second_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        benchmark_propagator_settings )


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
            save2txt(first_benchmark_dependent_variable, 'benchmark_1_dependent_variables.dat',  output_path)
            save2txt(second_benchmark_dependent_variable,  'benchmark_2_dependent_variables.dat',  output_path)
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
    benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_benchmark, interpolators.lagrange_interpolation(8, boundary_interpolation = interpolators.extrapolate_at_boundary_with_warning) )
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


def compare_benchmarks_no_extrapolation(first_benchmark: dict,
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
    benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_benchmark, interpolators.lagrange_interpolation(8, boundary_interpolation = interpolators.extrapolate_at_boundary_with_warning) )
    # Initialize difference dictionaries
    benchmark_difference = dict()


    # Calculate the difference between the states and dependent variables in an iterative manner
    for second_epoch in second_benchmark.keys():
        if second_epoch <= list(first_benchmark.keys())[-1]:
            benchmark_difference[second_epoch] = benchmark_interpolator.interpolate(second_epoch) - \
                                                second_benchmark[second_epoch]
    # Write results to files
    if output_path is not None:
        save2txt(benchmark_difference, filename, output_path)
    # Return the interpolator
    return benchmark_difference


def evaluate_interpolation(first_benchmark: dict,
                           second_benchmark: dict) -> dict:
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
    benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        second_benchmark, interpolators.lagrange_interpolation(8, boundary_interpolation = interpolators.extrapolate_at_boundary_with_warning) )
    # Initialize difference dictionaries
    benchmark_difference = dict()
    # Calculate the difference between the states and dependent variables in an iterative manner
    for epoch in first_benchmark.keys():
        benchmark_difference[epoch] = benchmark_interpolator.interpolate(epoch) - \
                                             first_benchmark[epoch]

    # Return the interpolator max error
    return benchmark_difference


def evaluate_interpolation_error(state_history_difference: dict):

    interpolation_relative_error_1 = dict()
    interpolation_relative_error_2 = dict()

    # state_history_difference = {t: state for i, (t, state) in enumerate(state_history_difference.items()) if i > 5}

    for i, epoch in enumerate(list(state_history_difference.keys())[1::2]):
        prev_epoch = list(state_history_difference.keys())[2 * i]
        next_epoch = list(state_history_difference.keys())[2 * i + 2]

        interpolation_relative_error_1[epoch] = np.linalg.norm(state_history_difference[epoch][:3]) / np.linalg.norm(state_history_difference[prev_epoch][:3])
        interpolation_relative_error_2[epoch] = - np.linalg.norm(state_history_difference[epoch][:3]) / np.linalg.norm(state_history_difference[next_epoch][:3])

    return (interpolation_relative_error_1, interpolation_relative_error_2)






################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################


import matplotlib.pyplot as plt

def plot_cowell_state_elements(state_history):
    """
    Plots the position components (x, y, z) and velocity components (vx, vy, vz) over time in a 1x2 subplot layout.

    Parameters
    ----------
    state_history : dict
        Dictionary containing the state elements over time. Keys are time, values are [x, y, z, vx, vy, vz].

    Returns
    -------
    None
    """
    # Extract time and state elements
    time = np.array(list(state_history.keys()))
    states = np.array(list(state_history.values()))
    x, y, z, vx, vy, vz = states[:, 0] / 1e3, states[:, 1] / 1e3, states[:, 2] / 1e3, states[:, 3] / 1e3, states[:, 4] / 1e3, states[:, 5] / 1e3

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot position components
    axes[0].plot(time, x, label='x', linewidth=3)
    axes[0].plot(time, y, label='y', linewidth=3)
    axes[0].plot(time, z, label='z', linewidth=3)
    axes[0].set_xlabel('Time (s)', fontsize=20)
    axes[0].set_ylabel('Position (km)', fontsize=20)
    axes[0].tick_params(axis='both', labelsize=18)
    axes[0].legend(fontsize=18)
    axes[0].grid()

    # Plot velocity components
    axes[1].plot(time, vx, label=r'$v_x$', linewidth=3)
    axes[1].plot(time, vy, label=r'$v_y$', linewidth=3)
    axes[1].plot(time, vz, label=r'$v_z$', linewidth=3)
    axes[1].set_xlabel('Time (s)', fontsize=20)
    axes[1].set_ylabel('Velocity (km/s)', fontsize=20)
    axes[1].tick_params(axis='both', labelsize=18)
    axes[1].legend(fontsize=18)
    axes[1].grid()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_enke_state_elements(state_difference):
    """
    Plots the state differences (dx, dy, dz, dvx, dvy, dvz) over time in a 1x2 subplot layout.

    Parameters
    ----------
    state_difference : dict
        Dictionary containing the state differences over time. Keys are time, values are [dx, dy, dz, dvx, dvy, dvz].

    Returns
    -------
    None
    """
    # Extract time and state differences
    time = np.array(list(state_difference.keys()))
    differences = np.array(list(state_difference.values()))
    dx, dy, dz, dvx, dvy, dvz = differences[:, 0] / 1e3, differences[:, 1] / 1e3, differences[:, 2] / 1e3, differences[:, 3] / 1e3, differences[:, 4] / 1e3, differences[:, 5] / 1e3

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot position differences
    axes[0].plot(time, dx, label=r'$\delta x$', linewidth=3)
    axes[0].plot(time, dy, label=r'$\delta y$', linewidth=3)
    axes[0].plot(time, dz, label=r'$\delta z$', linewidth=3)
    axes[0].set_xlabel('Time (s)', fontsize=20)
    axes[0].set_ylabel('Position Difference (km)', fontsize=20)
    axes[0].tick_params(axis='both', labelsize=18)
    axes[0].legend(fontsize=18)
    axes[0].grid()

    # Plot velocity differences
    axes[1].plot(time, dvx, label=r'$\delta v_x$', linewidth=3)
    axes[1].plot(time, dvy, label=r'$\delta v_y$', linewidth=3)
    axes[1].plot(time, dvz, label=r'$\delta v_z$', linewidth=3)
    axes[1].set_xlabel('Time (s)', fontsize=20)
    axes[1].set_ylabel('Velocity Difference (km/s)', fontsize=20)
    axes[1].tick_params(axis='both', labelsize=18)
    axes[1].legend(fontsize=18)
    axes[1].grid()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()




def plot_kepler_elements(state_history):
    """
    Plots all 6 Kepler elements (a, e, i, RAAN, omega, theta) over time in a 2x3 subplot layout.

    Parameters
    ----------
    state_history : dict
        Dictionary containing the Keplerian elements over time. Keys are time, values are [a, e, i, RAAN, omega, theta].

    Returns
    -------
    None
    """
    # Extract time and Keplerian elements
    # Filter the state history for time > 1500
    filtered_state_history = {t: state for t, state in state_history.items() if t > 1550}

    # Extract time and Keplerian elements for the filtered data
    time = np.array(list(filtered_state_history.keys()))
    elements = np.array(list(filtered_state_history.values()))
    a, e, i, RAAN, omega, theta = elements[:, 0], elements[:, 1], np.rad2deg(elements[:, 2]), \
                                    np.rad2deg(elements[:, 3]), np.rad2deg(elements[:, 4]), np.rad2deg(elements[:, 5])


    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Plot each Keplerian element
    axes[0].plot(time, a, label='a')
    axes[0].set_title('Semi-major Axis (a) vs Time')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('a (m)')
    axes[0].grid()

    axes[1].plot(time, e, label='e')
    axes[1].set_title('Eccentricity (e) vs Time')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('e (-)')
    axes[1].grid()

    axes[2].plot(time, i, label='i')
    axes[2].set_title('Inclination (i) vs Time')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('i (deg)')
    axes[2].grid()

    axes[3].plot(time, RAAN, label='RAAN')
    axes[3].set_title('RAAN vs Time')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('RAAN (deg)')
    axes[3].grid()

    axes[4].plot(time, omega, label='ω')
    axes[4].set_title('Argument of Periapsis (ω) vs Time')
    axes[4].set_xlabel('Time (s)')
    axes[4].set_ylabel('ω (deg)')
    axes[4].grid()

    axes[5].plot(time, theta, label='θ')
    axes[5].set_title('True Anomaly (θ) vs Time')
    axes[5].set_xlabel('Time (s)')
    axes[5].set_ylabel('θ (deg)')
    axes[5].grid()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_gauss_keplerian_elements(state_history):
    """
    Plots the Gauss Keplerian elements (e, omega, theta) over time in a single plot with two y-axes.

    Parameters
    ----------
    state_history : dict
        Dictionary containing the Keplerian elements over time. Keys are time, values are [a, e, i, RAAN, omega, theta].

    Returns
    -------
    None
    """
    # Extract time and Keplerian elements
    time = np.array(list(state_history.keys()))
    elements = np.array(list(state_history.values()))
    e, omega, theta = elements[:, 1], np.rad2deg(elements[:, 3]), np.rad2deg(elements[:, 5])

    # Find the time where eccentricity is minimum
    min_eccentricity_index = np.argmin(e)
    min_eccentricity_time = time[min_eccentricity_index]

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot eccentricity on the first y-axis
    ax1.plot(time, e, 'b-', label='Eccentricity (e)', linewidth=3)
    ax1.set_xlabel('Time [s]', fontsize=25)
    ax1.set_ylabel('Eccentricity [e]', fontsize=25)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.grid()

    # Add a vertical dashed line at the minimum eccentricity
    ax1.axvline(x=min_eccentricity_time, color='k', linestyle='--', label='Min Eccentricity')

    true_anomaly_derivative = np.diff(theta) / np.diff(time)  # dθ/dt using finite differences
    true_anomaly_derivative = np.append(true_anomaly_derivative, true_anomaly_derivative[-1])  # Append last value to match array length

    omega_dot = np.diff(omega) / np.diff(time)
    omega_dot = np.append(omega_dot, omega_dot[-1])

    # Create a second y-axis for omega and theta
    ax2 = ax1.twinx()
    ax2.plot(time, omega_dot, color='orange', label='dω/dt', linewidth=3)
    ax2.plot(time, true_anomaly_derivative, color='green', label='dθ/dt', linewidth=3)
    ax2.set_ylabel('[deg/s]',  fontsize=25)
    ax2.tick_params(axis='y', labelsize=20)

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=20)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_gme_elements(state_history):
    """
    Plots the Gauss Modified Equinoctial elements (p, f, g, h, k, L) over time in a 2x3 subplot layout.

    Parameters
    ----------
    state_history : dict
        Dictionary containing the Modified Equinoctial elements over time. Keys are time, values are [p, f, g, h, k, L].

    Returns
    -------
    None
    """

    # Extract time and Modified Equinoctial elements
    time = np.array(list(state_history.keys()))
    elements = np.array(list(state_history.values()))
    f, g, h, k, L = elements[:, 1], elements[:, 2], elements[:, 3], elements[:, 4], elements[:, 5]

    # Create subplots for f, g, h, k, L
    fig, axes = plt.subplots(1, 4, figsize=(25, 6))

    # Plot f
    axes[0].plot(time, f, label='f', linewidth=3, color='blue')
    axes[0].set_xlabel('Time (s)', fontsize=20)
    axes[0].legend(fontsize=18)
    axes[0].tick_params(axis='both', labelsize=18)
    axes[0].grid()

    # Plot g
    axes[1].plot(time, g, label='g', linewidth=3, color='orange')
    axes[1].set_xlabel('Time (s)', fontsize=20)
    axes[1].legend(fontsize=18)
    axes[1].tick_params(axis='both', labelsize=18)
    axes[1].grid()

    # Plot h
    axes[2].plot(time, h, label='h', linewidth=3, color='green')
    axes[2].set_xlabel('Time (s)', fontsize=20)
    axes[2].legend(fontsize=18)
    axes[2].tick_params(axis='both', labelsize=18)
    axes[2].grid()

    # Plot k
    axes[3].plot(time, k, label='k', linewidth=3, color='red')
    axes[3].set_xlabel('Time (s)', fontsize=20)
    axes[3].legend(fontsize=18)
    axes[3].tick_params(axis='both', labelsize=18)
    axes[3].grid()

    # Adjust layout and show the figure
    plt.show()

def plot_usm_quaternions_elements(state_history):
    """
    Plots the USM-Quaternions elements (C, Rf1, Rf2, ε1, ε2, ε3, η) over time in a 2x4 subplot layout.

    Parameters
    ----------
    state_history : dict
        Dictionary containing the USM-Quaternions elements over time. Keys are time, values are [C, Rf1, Rf2, ε1, ε2, ε3, η].

    Returns
    -------
    None
    """
    # Extract time and USM-Quaternions elements
    time = np.array(list(state_history.keys()))
    elements = np.array(list(state_history.values()))
    C, Rf1, Rf2, epsilon_1, epsilon_2, epsilon_3, eta = elements[:, 0], elements[:, 1], elements[:, 2], elements[:, 3], elements[:, 4], elements[:, 5], elements[:, 6]


    # Create a single plot for C, Rf1, and Rf2
    plt.figure(figsize=(10, 6))
    plt.plot(time, C, label='C', color='blue', linewidth=3)
    plt.plot(time, Rf1, label='Rf1', color='orange', linewidth=3, linestyle='--')
    plt.plot(time, Rf2, label='Rf2', color='green', linewidth=3, linestyle='-.')
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('(m/s)', fontsize=20)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_usm_mrp_elements(state_history):
    """
    Plots the USM-MRP elements (C, Rf1, Rf2, σ1, σ2, σ3, S) over time in a 2x4 subplot layout.

    Parameters
    ----------
    state_history : dict
        Dictionary containing the USM-MRP elements over time. Keys are time, values are [C, Rf1, Rf2, σ1, σ2, σ3, S].

    Returns
    -------
    None
    """
    # Extract time and USM-MRP elements
    time = np.array(list(state_history.keys()))
    elements = np.array(list(state_history.values()))
    C, Rf1, Rf2, sigma_1, sigma_2, sigma_3, S = elements[:, 0], elements[:, 1], elements[:, 2], elements[:, 3], elements[:, 4], elements[:, 5], elements[:, 6]

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Plot each USM-MRP element
    axes[0].plot(time, C, label='C')
    axes[0].set_title('C vs Time')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('C (m/s)')
    axes[0].grid()

    axes[1].plot(time, Rf1, label='Rf1')
    axes[1].set_title('Rf1 vs Time')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Rf1 (m/s)')
    axes[1].grid()

    axes[2].plot(time, Rf2, label='Rf2')
    axes[2].set_title('Rf2 vs Time')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Rf2 (m/s)')
    axes[2].grid()

    axes[3].plot(time, sigma_1, label='σ1')
    axes[3].set_title('σ1 vs Time')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('σ1 (-)')
    axes[3].grid()

    axes[4].plot(time, sigma_2, label='σ2')
    axes[4].set_title('σ2 vs Time')
    axes[4].set_xlabel('Time (s)')
    axes[4].set_ylabel('σ2 (-)')
    axes[4].grid()

    axes[5].plot(time, sigma_3, label='σ3')
    axes[5].set_title('σ3 vs Time')
    axes[5].set_xlabel('Time (s)')
    axes[5].set_ylabel('σ3 (-)')
    axes[5].grid()

    axes[6].plot(time, S, label='S')
    axes[6].set_title('S vs Time')
    axes[6].set_xlabel('Time (s)')
    axes[6].set_ylabel('S (-)')
    axes[6].grid()

    # Hide the last subplot (empty)
    axes[7].axis('off')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_usmem_elements(state_history):
    """
    Plots the USM-Exponential Map elements (C, Rf1, Rf2, a1, a2, a3, a^S) over time in a 2x4 subplot layout.

    Parameters
    ----------
    state_history : dict
        Dictionary containing the USM-Exponential Map elements over time. Keys are time, values are [C, Rf1, Rf2, a1, a2, a3, a^S].

    Returns
    -------
    None
    """
    # Extract time and USM-Exponential Map elements
    time = np.array(list(state_history.keys()))
    elements = np.array(list(state_history.values()))
    C, Rf1, Rf2, a1, a2, a3, a_shadow = elements[:, 0], elements[:, 1], elements[:, 2], elements[:, 3], elements[:, 4], elements[:, 5], elements[:, 6]

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Plot each USM-Exponential Map element
    axes[0].plot(time, C, label='C')
    axes[0].set_title('C vs Time')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('C (m/s)')
    axes[0].grid()

    axes[1].plot(time, Rf1, label='Rf1')
    axes[1].set_title('Rf1 vs Time')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Rf1 (m/s)')
    axes[1].grid()

    axes[2].plot(time, Rf2, label='Rf2')
    axes[2].set_title('Rf2 vs Time')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Rf2 (m/s)')
    axes[2].grid()

    axes[3].plot(time, a1, label='a1')
    axes[3].set_title('a1 vs Time')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('a1 (-)')
    axes[3].grid()

    axes[4].plot(time, a2, label='a2')
    axes[4].set_title('a2 vs Time')
    axes[4].set_xlabel('Time (s)')
    axes[4].set_ylabel('a2 (-)')
    axes[4].grid()

    axes[5].plot(time, a3, label='a3')
    axes[5].set_title('a3 vs Time')
    axes[5].set_xlabel('Time (s)')
    axes[5].set_ylabel('a3 (-)')
    axes[5].grid()

    axes[6].plot(time, a_shadow, label='a^S')
    axes[6].set_title('S vs Time')
    axes[6].set_xlabel('Time (s)')
    axes[6].set_ylabel('S (-)')
    axes[6].grid()

    # Hide the last subplot (empty)
    axes[7].axis('off')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()



#############################################################################################
#############################################################################################

def save_dependent_variables_to_dict(dependent_variables_history):

    

    time = list(dependent_variables_history.keys())

    dependent_variable_dict = {
        "mach_number": {},
        "altitude": {},
        "kepler_elements": {},
        "dynamic_pressure": {},
        "latitude_angle": {},
        "longitude_angle": {},
        "heading_angle": {},
        "flight_path_angle": {},
        "total_acceleration_norm": {},
        "aerodynamic_acceleration": {},
        "aero_rotation_matrix": {},
        "point_mass_acceleration": {},
        "rws_rotation_matrix": {},
        "airspeed": {}
    }

    for t, values in dependent_variables_history.items():

        dependent_variable_dict["mach_number"][t] = values[0]
        dependent_variable_dict["altitude"][t] = values[1]
        dependent_variable_dict["kepler_elements"][t] = values[2:8]
        dependent_variable_dict["dynamic_pressure"][t] = values[8]
        dependent_variable_dict["latitude_angle"][t] = values[9]
        dependent_variable_dict["longitude_angle"][t] = values[10]
        dependent_variable_dict["heading_angle"][t] = values[11]
        dependent_variable_dict["flight_path_angle"][t] = values[12]
        dependent_variable_dict["total_acceleration_norm"][t] = values[13]
        dependent_variable_dict["aerodynamic_acceleration"][t] = values[14:17]
        dependent_variable_dict["aero_rotation_matrix"][t] = values[17:26]
        dependent_variable_dict["point_mass_acceleration"][t] = values[26:29]
        dependent_variable_dict["rws_rotation_matrix"][t] = values[29:38]
        dependent_variable_dict["airspeed"][t] = values[38]

    return dependent_variable_dict


# Plot Kepler elements

def plot_dependent_variables(dependent_variable_data):

    plt.figure(figsize=(12, 8))
    time = list(dependent_variable_data['kepler_elements'].keys())
    # Check if all required keys exist in the dictionary
    required_keys = ["kepler_elements", "mach_number", "altitude", "latitude_angle", "longitude_angle", 
                     "heading_angle", "flight_path_angle", "total_acceleration_norm", 
                     "aerodynamic_acceleration", "aero_rotation_matrix","point_mass_acceleration", "rws_rotation_matrix"]
    for key in required_keys:
        print(key)
        if key not in dependent_variable_data:
            raise KeyError(f"Missing key in dependent_variable_data: {key}")

    kepler_elements = np.array([dependent_variable_data["kepler_elements"][t] for t in time])
    mach_numbers = np.array([dependent_variable_data["mach_number"][t] for t in time])
    altitudes = np.array([dependent_variable_data["altitude"][t] for t in time])
    latitude_angles = np.array([dependent_variable_data["latitude_angle"][t] for t in time])
    longitude_angles = np.array([dependent_variable_data["longitude_angle"][t] for t in time])
    heading_angles = np.array([dependent_variable_data["heading_angle"][t] for t in time])
    flight_path_angles = np.array([dependent_variable_data["flight_path_angle"][t] for t in time])
    total_acceleration_norms = np.array([dependent_variable_data["total_acceleration_norm"][t] for t in time])
    aerodynamic_accelerations = np.array([dependent_variable_data["aerodynamic_acceleration"][t] for t in time])
    rotation_matrices = np.array([dependent_variable_data["aero_rotation_matrix"][t] for t in time])
    point_mass_accelerations = np.array([dependent_variable_data["point_mass_acceleration"][t] for t in time])
    rws_rotation_matrices = np.array([dependent_variable_data["rws_rotation_matrix"][t] for t in time])
    airspeed = np.array([dependent_variable_data["airspeed"][t] for t in time])

    time = np.array(time)
    kepler_labels = ['Semi-major axis [m]', 'Eccentricity [-]', 'Inclination [rad]', 
                    'Argument of Periapsis [rad]', 'RAAN [rad]', 'True Anomaly [rad]']

    for i in range(6):
        plt.subplot(3, 2, i + 1)
        if i == 5:
            # Find the index of the minimum true anomaly
            true_anomaly = kepler_elements[:, 5]
            min_true_anomaly_index = np.argmin(true_anomaly)
            # Adjust true anomaly by adding 2π from the minimum index onwards
            true_anomaly[min_true_anomaly_index:] += 2 * np.pi
            plt.plot(time, true_anomaly, label=kepler_labels[i])
        else:
            plt.plot(time, kepler_elements[:, i], label=kepler_labels[i])
            plt.xlabel('Time [s]', fontsize=12)
            plt.ylabel(kepler_labels[i], fontsize=12)
            plt.grid()
            plt.tight_layout()

    plt.suptitle('Kepler Elements', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Create a single figure with 1 row and 3 columns for the plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Mach number vs Altitude
    ax1 = axes[0]
    ax2 = ax1.twiny()  # Create a second x-axis sharing the same y-axis
    ax1.plot(mach_numbers, altitudes * 1e-3, label='Mach', color='blue', linewidth=2.5)
    ax1.set_xlabel('Mach Number [-]', fontsize=20)
    ax1.set_ylabel('Altitude [km]', fontsize=20)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(3))
    ax1.grid()
    ax2.plot(airspeed * 1e-3, altitudes * 1e-3, label='Airspeed', color='red', linewidth=2.5)  # Invisible plot for scaling
    ax2.set_xlabel('Airspeed [km/s]', fontsize=20)
    ax2.tick_params(axis='x', labelsize=18)
    # Add legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left', fontsize=16)

    # Plot altitude vs flight path angle
    ax1 = axes[1]
    ax2 = ax1.twiny()  # Create a second x-axis sharing the same y-axis
    ax1.plot(flight_path_angles * 180/np.pi, altitudes * 1e-3, label='Flight Path Angle', color='green', linewidth=2.5)
    ax1.set_xlabel('Flight Path Angle [deg]', fontsize=20)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.grid()
    ax2.plot(kepler_elements[:, 1], altitudes * 1e-3, label='Eccentricity', color='orange', linewidth=2.5)  # Invisible plot for scaling
    ax2.set_xlabel('Eccentricity [-]', fontsize=20)
    ax2.tick_params(axis='x', labelsize=18)
    # Add legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=16, loc='upper center')

    # Plot aerodynamic acceleration norm as a function of altitude
    g = 9.81  # Gravitational acceleration [m/s²]
    axes[2].plot(total_acceleration_norms / g, altitudes * 1e-3, label='Total Acceleration [g]', linewidth=2.5)
    axes[2].set_xlabel('Acceleration [g]', fontsize=20)
    axes[2].tick_params(axis='both', labelsize=18)
    axes[2].grid()

    aero_acceleration_rsw_frame = []
    for i, (aero_acceleration, point_mass_acceleration) in enumerate(zip(aerodynamic_accelerations, point_mass_accelerations)):
        rsw_rotation_matrix = np.array(rws_rotation_matrices[i]).reshape(3, 3)
        aero_acceleration_rsw = np.dot(rsw_rotation_matrix.T, aero_acceleration)
        aero_acceleration_rsw_frame.append(aero_acceleration_rsw)

    aero_acceleration_rsw_frame = np.array(aero_acceleration_rsw_frame)

    # Plot aerodynamic acceleration components in the trajectory frame
    axes[2].plot(np.abs(aero_acceleration_rsw_frame[:, 0]) / g, altitudes * 1e-3, label='Along-track [g]', linestyle='-', linewidth=2)
    axes[2].plot(np.abs(aero_acceleration_rsw_frame[:, 1]) / g, altitudes * 1e-3, label='Radial [g]', linestyle='-', linewidth=2)
    axes[2].plot(np.abs(aero_acceleration_rsw_frame[:, 2]) / g, altitudes * 1e-3, label='Cross-track [g]', linestyle='-', linewidth=2)
    axes[2].legend(fontsize=16)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Compute angular momentum and derivative of true anomaly
    angular_momentum = np.array([np.sqrt(a * (1 - e**2)) for a, e in zip(kepler_elements[:, 0], kepler_elements[:, 1])])  # h = sqrt(a * (1 - e^2))
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot angular momentum on the first y-axis
    ax1.plot(time, angular_momentum, 'b-', label=r'Semilatus Rectus (h/$\mu^2$)', linewidth=2)
    ax1.set_xlabel('Time [s]', fontsize=20)
    ax1.set_ylabel('p [m]', color='b', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='b', labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)
    ax1.grid()

    # Create a second y-axis for the derivative of the true anomaly
    ax2 = ax1.twinx()
    ax2.plot(time, true_anomaly, 'r', label='θ', linewidth=2)
    ax2.set_ylabel('θ [rad]', color='r', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=18)

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center left', fontsize=18)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()

    # Plot altitude, latitude, longitude, Mach, heading, and flight path angle
    plt.figure(figsize=(12, 8))

    # Top 3 plots
    plt.subplot(2, 3, 1)
    plt.plot(time, altitudes, label='Altitude [m]')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Altitude [m]', fontsize=12)
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.plot(time, latitude_angles, label='Latitude [rad]')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Latitude [rad]', fontsize=12)
    plt.grid()

    plt.subplot(2, 3, 3)
    plt.plot(time, longitude_angles, label='Longitude [rad]')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Longitude [rad]', fontsize=12)
    plt.grid()

    # Bottom 3 plots
    plt.subplot(2, 3, 4)
    plt.plot(time, mach_numbers, label='Mach Number [-]')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Mach Number [-]', fontsize=12)
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.plot(time, heading_angles, label='Heading Angle [rad]')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Heading Angle [rad]', fontsize=12)
    plt.grid()

    plt.subplot(2, 3, 6)
    plt.plot(time, flight_path_angles, label='Flight Path Angle [rad]')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Flight Path Angle [rad]', fontsize=12)
    plt.grid()

    plt.suptitle('Altitude, Latitude, Longitude, Mach, Heading, Flight Path Angle', fontsize=16)
    plt.tight_layout()
    plt.show()

    

    # Compute aerodynamic acceleration components in the trajectory frame
    aero_acceleration_trajectory_frame = []

    for i, aero_acceleration in enumerate(aerodynamic_accelerations):
        rotation_matrix = np.array(rotation_matrices[i]).reshape(3, 3)
        trajectory_frame_acceleration = np.dot(rotation_matrix, aero_acceleration)
        aero_acceleration_trajectory_frame.append(trajectory_frame_acceleration)

    aero_acceleration_trajectory_frame = np.array(aero_acceleration_trajectory_frame)

    # Plot aerodynamic acceleration components in the trajectory frame
    plt.figure(figsize=(10, 6))
    plt.plot(time, np.abs(aero_acceleration_trajectory_frame[:, 0]), label='Drag [m/s²]')
    plt.plot(time, np.abs(aero_acceleration_trajectory_frame[:, 1]), label='Side [m/s²]')
    plt.plot(time, np.abs(aero_acceleration_trajectory_frame[:, 2]), label='Lift [m/s²]')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Aerodynamic Acceleration [m/s²]', fontsize=12)
    plt.yscale('log')
    plt.title('Aerodynamic Acceleration Components in Trajectory Frame', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()
    

    # Compute aerodynamic and point mass acceleration components in the RSW frame
    aero_acceleration_rsw_frame = []
    point_mass_acceleration_rsw_frame = []

    for i, (aero_acceleration, point_mass_acceleration) in enumerate(zip(aerodynamic_accelerations, point_mass_accelerations)):
        rsw_rotation_matrix = np.array(rws_rotation_matrices[i]).reshape(3, 3)
        aero_acceleration_rsw = np.dot(rsw_rotation_matrix.T, aero_acceleration)
        point_mass_acceleration_rsw = np.dot(rsw_rotation_matrix.T, point_mass_acceleration)
        aero_acceleration_rsw_frame.append(aero_acceleration_rsw)
        point_mass_acceleration_rsw_frame.append(point_mass_acceleration_rsw)

    aero_acceleration_rsw_frame = np.array(aero_acceleration_rsw_frame)
    point_mass_acceleration_rsw_frame = np.array(point_mass_acceleration_rsw_frame)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot aerodynamic acceleration components in the RSW frame
    axes[0].plot(time, aero_acceleration_rsw_frame[:, 1], label='Along-track [m/s²]', linewidth=2)
    axes[0].plot(time, aero_acceleration_rsw_frame[:, 0], label='Radial [m/s²]', linewidth=2)
    axes[0].plot(time, aero_acceleration_rsw_frame[:, 2], label='Cross-track [m/s²]', linewidth=2)
    axes[0].set_title('Aerodynamic Acceleration in RSW Frame', fontsize=16)
    axes[0].set_xlabel('Time [s]', fontsize=14)
    axes[0].set_ylabel('Acceleration [m/s²]', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid()

    # Plot point mass acceleration components in the RSW frame
    axes[1].plot(time, point_mass_acceleration_rsw_frame[:, 1], label='Along-track [m/s²]', linewidth=2)
    axes[1].plot(time, point_mass_acceleration_rsw_frame[:, 0], label='Radial [m/s²]', linewidth=2)
    axes[1].plot(time, point_mass_acceleration_rsw_frame[:, 2], label='Cross-track [m/s²]', linewidth=2)
    axes[1].set_title('Point Mass Acceleration in RSW Frame', fontsize=16)
    axes[1].set_xlabel('Time [s]', fontsize=14)
    axes[1].set_ylabel('Acceleration [m/s²]', fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid()

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()
