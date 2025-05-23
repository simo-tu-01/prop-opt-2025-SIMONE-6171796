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
First name: Simone
Last name: Guccione
Student number: 6171796

This module defines useful functions that will be called by the main script, where the optimization is executed.
'''

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np

# Tudatpy imports
import tudatpy
from tudatpy.data import save2txt
from tudatpy import constants
from tudatpy.interface import spice
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
    radial_distance = spice.get_average_radius('Earth') + 120.0E3
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
                                   propagation_setup.dependent_variable.altitude('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.density('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.airspeed('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.aerodynamic_type, 'Capsule', 'Earth')]
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
                            sh_degree = 0,
                            sh_order = 0,
                            initial_state_perturbation = np.zeros(6)):
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
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    termination_settings : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object to be used
    dependent_variables_to_save : list[tudatpy.kernel.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    current_propagator : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.TranslationalPropagatorType
        Type of propagator to be used for translational dynamics

    Returns
    -------
    propagator_settings : tudatpy.kernel.numerical_simulation.propagation_setup.integrator.MultiTypePropagatorSettings
        Propagator settings to be provided to the dynamics simulator.
    """

    # Define bodies that are propagated and their central bodies of propagation
    bodies_to_propagate = ['Capsule']
    central_bodies = ['Earth']

    # Here different acceleration models are defined
    if model_choice == 0 or model_choice == 5 or model_choice == 7 or model_choice == 8 or model_choice == 9:
        acceleration_settings_on_vehicle = {
            'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(sh_degree, sh_order),
                    propagation_setup.acceleration.aerodynamic(),
                    propagation_setup.acceleration.relativistic_correction(use_schwarzschild= True)],
            'Sun':[propagation_setup.acceleration.point_mass_gravity(),
                    propagation_setup.acceleration.radiation_pressure()],
            'Moon': [propagation_setup.acceleration.point_mass_gravity()],
        }
    elif model_choice == 1:
        acceleration_settings_on_vehicle = {
            'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(sh_degree, sh_order),
                    propagation_setup.acceleration.aerodynamic()],
            'Sun':[propagation_setup.acceleration.point_mass_gravity()],
            'Moon': [propagation_setup.acceleration.point_mass_gravity()],
        }
    elif model_choice == 2:
        acceleration_settings_on_vehicle = {
            'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(sh_degree, sh_order),
                    propagation_setup.acceleration.aerodynamic(),
                    propagation_setup.acceleration.relativistic_correction(use_schwarzschild= True)],
            'Sun':[propagation_setup.acceleration.radiation_pressure()],
            'Moon': [propagation_setup.acceleration.point_mass_gravity()],
        }
    elif model_choice == 3:
        acceleration_settings_on_vehicle = {
            'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(sh_degree, sh_order),
                    propagation_setup.acceleration.aerodynamic(),
                    propagation_setup.acceleration.relativistic_correction(use_schwarzschild= True)],
            'Sun':[propagation_setup.acceleration.point_mass_gravity(),
                    propagation_setup.acceleration.radiation_pressure()],
        }
    elif model_choice == 4:
        acceleration_settings_on_vehicle = {
            'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(sh_degree, sh_order),
                    propagation_setup.acceleration.aerodynamic(),
                    propagation_setup.acceleration.relativistic_correction(use_schwarzschild= True)],
            'Sun':[propagation_setup.acceleration.point_mass_gravity()],
            'Moon': [propagation_setup.acceleration.point_mass_gravity()],
        }
    elif model_choice == 6:
        acceleration_settings_on_vehicle = {
            'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(sh_degree, sh_order),
                    propagation_setup.acceleration.aerodynamic()],
            'Sun':[propagation_setup.acceleration.point_mass_gravity(),
                    propagation_setup.acceleration.radiation_pressure()],
            'Moon': [propagation_setup.acceleration.point_mass_gravity()],
        }
    elif model_choice == 'sh':
        acceleration_settings_on_vehicle = {
            'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(sh_degree, sh_order),
                    propagation_setup.acceleration.aerodynamic()],
            'Sun':[propagation_setup.acceleration.point_mass_gravity()],
            'Moon': [propagation_setup.acceleration.point_mass_gravity()],
        }
    elif model_choice == 'sh_no':
        acceleration_settings_on_vehicle = {
            'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(sh_degree, sh_order),
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
    initial_state = get_initial_state(simulation_start_epoch, bodies) + initial_state_perturbation

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


def get_environment_settings(body_settings,
                             global_frame_orientation,
                             simulation_start_epoch,
                             atmosphere,
                             shape,
                             rotation_model,
                             tides = False):
    

    if rotation_model == 'IAU':
        body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.spice(
            global_frame_orientation, 'IAU_Earth')
    elif rotation_model == 'simple':
        body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
            global_frame_orientation, 'IAU_Earth', 'IAU_Earth', simulation_start_epoch)
    elif rotation_model == 'ITRS':
        body_settings.get( "Earth" ).rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
            environment_setup.rotation_model.IAUConventions.iau_2006, global_frame_orientation)
        body_settings.get( "Earth" ).gravity_field_settings.associated_reference_frame = "ITRS"
    else:
        raise ValueError(f"Unsupported rotation model: {rotation_model}")

    
    if atmosphere == 'exponential':
        body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.exponential_predefined('Earth')
    elif atmosphere == 'US76':
        body_settings.get( "Earth" ).atmosphere_settings = environment_setup.atmosphere.us76()
    else:
        raise ValueError(f"Unsupported atmosphere model: {atmosphere}")

    if shape == 'oblate_spheroid':
        # Shape model Constants
        body_radius = 6378.0E3
        body_flattening = 1.0 / 300.0
        body_settings.get( "Earth" ).shape_settings = environment_setup.shape.oblate_spherical(body_radius, body_flattening )
    elif shape == 'sphere':
        body_settings.get( "Earth" ).shape_settings = environment_setup.shape.spherical_spice()
    else:
        raise ValueError(f"Unsupported shape model: {shape}")

    if tides:
        tide_raising_body = "Moon"
        degree = 2
        love_number = 0.301
        gravity_field_variation_list = list()
        gravity_field_variation_list.append(environment_setup.gravity_field_variation.solid_body_tide(
            tide_raising_body, love_number, degree ))
        body_settings.get( "Earth" ).gravity_field_variation_settings = gravity_field_variation_list



def get_acceleration_dict(sh_degree, 
                          sh_order,
                          aero= False,
                          Sun= False,
                          Moon= False,
                          srp= False,
                          schwarzschild= False):

    acceleration_settings_on_vehicle = dict()

    acceleration_settings_on_vehicle['Earth'] = propagation_setup.acceleration.spherical_harmonic_gravity(sh_degree, sh_order)

    if aero:
        acceleration_settings_on_vehicle['Earth'] = propagation_setup.acceleration.aerodynamic()
    elif Sun:
        acceleration_settings_on_vehicle['Sun'] = propagation_setup.acceleration.point_mass_gravity()
    elif Moon:
        acceleration_settings_on_vehicle['Moon'] = propagation_setup.acceleration.point_mass_gravity()
    elif srp:
        acceleration_settings_on_vehicle['Sun'] = propagation_setup.acceleration.radiation_pressure()
    elif schwarzschild: 
        acceleration_settings_on_vehicle['Earth'] = propagation_setup.acceleration.relativistic_correction(use_schwarzschild= True)

    return acceleration_settings_on_vehicle




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
    bodies.get_body('Capsule').mass = new_capsule_mass
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
    constant_angles = np.zeros([3,1])
    constant_angles[ 0 ] = shape_parameters[ 5 ]
    angle_function = lambda time : constant_angles
    environment_setup.add_rotation_model( bodies, 'Capsule',
                                          environment_setup.rotation_model.aerodynamic_angle_based(
                                              'Earth', 'J2000', 'CapsuleFixed', angle_function ))
    
    # Create radiation pressure settings
    reference_area_radiation = 5  # m^2
    radiation_pressure_coefficient = 1.2
    environment_setup.add_radiation_pressure_target_model(bodies, 'Capsule',
                                                          environment_setup.radiation_pressure.cannonball_radiation_target(
                                                                reference_area_radiation, radiation_pressure_coefficient))
                                                          
    # Update the capsule shape parameters
    set_capsule_shape_parameters(shape_parameters,
                                 bodies,
                                 capsule_density)



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
    benchmark_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        first_benchmark_step_size,
        propagation_setup.integrator.CoefficientSets.rkdp_87)
    benchmark_propagator_settings.print_settings.print_dependent_variable_indices = True

    print('Running first benchmark...')
    first_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        benchmark_propagator_settings )

    # Create integrator settings for the second benchmark in the same way
    benchmark_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        second_benchmark_step_size,
        propagation_setup.integrator.CoefficientSets.rkdp_87)
    benchmark_propagator_settings.print_settings.print_dependent_variable_indices = False

    print('Running second benchmark...')
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



########################################################################################################

def compute_aerodynamic_metrics(dependent_variables_history: dict):

    altitude = np.array([np.linalg.norm(dependent_variables_history[epoch][1]) for epoch in dependent_variables_history.keys()])
    density = np.array([np.linalg.norm(dependent_variables_history[epoch][2]) for epoch in dependent_variables_history.keys()])
    airspeed = np.array([np.linalg.norm(dependent_variables_history[epoch][2]) for epoch in dependent_variables_history.keys()])
    aero_acceleration = np.array([np.linalg.norm(dependent_variables_history[epoch][4]) for epoch in dependent_variables_history.keys()])

    heat_flux = np.sqrt(density) * airspeed ** 3
    heat_load = density * airspeed ** 3
    load_factor = aero_acceleration / 9.81

    max_heat_flux = np.max(heat_flux)
    max_heat_flux_index = np.argmax(heat_flux)
    altitude_at_max_heat_flux = altitude[max_heat_flux_index]

    max_heat_load = np.max(heat_load)
    max_heat_load_index = np.argmax(heat_load)
    altitude_at_max_heat_load = altitude[max_heat_load_index]

    max_load_factor = np.max(load_factor)
    max_load_factor_index = np.argmax(load_factor)
    altitude_at_max_load_factor = altitude[max_load_factor_index]

    return {
        "max_heat_flux": max_heat_flux,
        "altitude_at_max_heat_flux": altitude_at_max_heat_flux,
        "max_heat_load": max_heat_load,
        "altitude_at_max_heat_load": altitude_at_max_heat_load,
        "max_load_factor": max_load_factor,
        "altitude_at_max_load_factor": altitude_at_max_load_factor
    }
