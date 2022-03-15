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

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.math import geometry

###########################################################################
# PROPAGATION SETTING UTILITIES ###########################################
###########################################################################

def get_initial_state(simulation_start_epoch: float,
                      bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies) -> np.ndarray:
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
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
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
        -> tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings:
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
    hybrid_termination_settings : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
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
    dependent_variables_to_save : list[tudatpy.kernel.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    """
    dependent_variables_to_save = [propagation_setup.dependent_variable.mach_number('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.altitude('Capsule', 'Earth')]
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
        integrator_settings = integrator.runge_kutta_4(simulation_start_epoch,
                                                       fixed_step_size)
    return integrator_settings


def get_propagator_settings(shape_parameters,
                            bodies,
                            simulation_start_epoch,
                            termination_settings,
                            dependent_variables_to_save,
                            current_propagator = propagation_setup.propagator.cowell,
                            model_choice = 0,
                            initial_state_perturbation = np.zeros( 6 ) ):

    # Define bodies that are propagated and their central bodies of propagation
    bodies_to_propagate = ['Capsule']
    central_bodies = ['Earth']

    # Define accelerations for the nominal case
    acceleration_settings_on_vehicle = {'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(2, 2),
                                                  propagation_setup.acceleration.aerodynamic()]}
    # Here different acceleration models are defined
    if model_choice == 1:
        acceleration_settings_on_vehicle['Earth'][0] = propagation_setup.acceleration.point_mass_gravity()
    elif model_choice == 2:
        acceleration_settings_on_vehicle['Earth'][0] = propagation_setup.acceleration.spherical_harmonic_gravity(4, 4)

    # Create global accelerations' dictionary
    acceleration_settings = {'Capsule': acceleration_settings_on_vehicle}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    # Set vehicle body orientation (constant angle of attack, zero sideslip and bank angle)
    environment_setup.set_constant_aerodynamic_orientation(
        bodies.get_body('Capsule'),shape_parameters[5], 0.0, 0.0,
        silence_warnings=True )

    # Retrieve initial state
    initial_state = get_initial_state(simulation_start_epoch,bodies)

    # Create propagation settings for the benchmark
    propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                     acceleration_models,
                                                                     bodies_to_propagate,
                                                                     initial_state,
                                                                     termination_settings,
                                                                     current_propagator,
                                                                     output_variables=dependent_variables_to_save)
    return propagator_settings


###########################################################################
# CAPSULE SHAPE/AERODYNAMICS UTILITIES ####################################
###########################################################################


def get_capsule_coefficient_interface(capsule_shape: tudatpy.kernel.math.geometry.Capsule) \
        -> tudatpy.kernel.numerical_simulation.environment.HypersonicLocalInclinationAnalysis:
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
    capsule_shape : tudatpy.kernel.math.geometry.Capsule
        Object that defines the shape of the vehicle.

    Returns
    -------
    hypersonic_local_inclination_analysis : tudatpy.kernel.environment.HypersonicLocalInclinationAnalysis
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
                                 bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                                 capsule_density: float):
    """
    It computes and creates the properties of the capsule (shape, mass, aerodynamic coefficient interface...).

    Parameters
    ----------
    shape_parameters : list of floats
        List of shape parameters to be optimized.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
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
    bodies.get_body('Capsule').set_constant_mass(new_capsule_mass)
    # Create aerodynamic interface from shape parameters (this calls the local inclination analysis)
    new_aerodynamic_coefficient_interface = get_capsule_coefficient_interface(new_capsule)
    # Update the Capsule's aerodynamic coefficient interface
    bodies.get_body('Capsule').aerodynamic_coefficient_interface = new_aerodynamic_coefficient_interface


# NOTE TO STUDENTS: if and when making modifications to the capsule shape, do include them in this function and not in
# the main code.
def add_capsule_to_body_system(bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                               shape_parameters: list,
                               capsule_density: float):
    """
    It creates the capsule body object and adds it to the body system, setting its shape based on the shape parameters
    provided.

    Parameters
    ----------
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
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
                        bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                        benchmark_propagator_settings: tudatpy.kernel.numerical_simulation.propagation_setup.propagator.TranslationalStatePropagatorSettings,
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
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
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
        first_benchmark, interpolators.lagrange_interpolation(8))
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
