'''
Copyright (c) 2010-2021, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Lunar Ascent
First name: ***COMPLETE HERE***
Last name: ***COMPLETE HERE***
Student number: ***COMPLETE HERE***

This module contains the problem-specific classes and functions, which will be called by the main script where the
optimization is executed.
'''

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np

# Tudatpy imports
import tudatpy
from tudatpy.kernel import constants
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.astro import frame_conversion
from tudatpy.kernel.math import interpolators

###########################################################################
# CREATE GUIDANCE CLASS ###################################################
###########################################################################

class LunarAscentThrustGuidance:
    """
    Class that defines and updates the thrust guidance of the Lunar Ascent problem at each time step.

    Attributes
    ----------
    vehicle_body
    initial_time
    parameter_vector

    Methods
    -------
    get_current_thrust_direction(time)
    """

    def __init__(self,
                 vehicle_body: str,
                 initial_time: float,
                 parameter_vector: list):
        """
        Constructor of the LunarAscentThrustGuidance class.

        Parameters
        ----------
        vehicle_body : str
            Name of the vehicle to apply the thrust guidance to.
        initial_time: float
            Initial time of the simulation [s].
        parameter_vector : list
            List of thrust parameters to retrieve the thrust guidance from.

        Returns
        -------
        none
        """
        print('Initializing guidance...')
        # Set arguments as attributes
        self.vehicle_body = vehicle_body
        self.initial_time = initial_time
        self.parameter_vector = parameter_vector
        self.time_interval = parameter_vector[1]
        # Prepare dictionary for thrust angles
        self.thrust_angle_dict = {}
        # Initialize time
        current_time = initial_time
        # Loop over nodes
        for i in range(len(parameter_vector) - 2):
            # Store time as key, thrust angle as value
            self.thrust_angle_dict[current_time] = parameter_vector[i + 2]
            # Increase time
            current_time += self.time_interval
        # Create interpolator settings
        interpolator_settings = interpolators.linear_interpolation(
            boundary_interpolation=interpolators.use_boundary_value)
        # Create the interpolator between nodes and set it as attribute
        self.thrust_angle_interpolator = interpolators.create_one_dimensional_scalar_interpolator(
            self.thrust_angle_dict, interpolator_settings )

    def get_current_thrust_direction(self,
                                     time: float) -> np.array:
        """
        Retrieves the direction of the thrust in the inertial frame.

        This function is needed to get the thrust direction at each time step of the propagation, based on the thrust
        parameters provided.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        thrust_inertial_frame : np.array
            Thrust direction expressed in the inertial frame.
        """
        # Interpolate with time
        angle = self.thrust_angle_interpolator.interpolate(time)
        # Set thrust in vertical frame and transpose it
        thrust_direction_vertical_frame = np.array([[0, np.sin(angle), - np.cos(angle)]]).T
        # Update flight conditions (this is needed to let tudat know to update all variables)
        self.vehicle_body.flight_conditions.update_conditions(time)
        # Get aerodynamic angle calculator
        aerodynamic_angle_calculator = self.vehicle_body.flight_conditions.aerodynamic_angle_calculator
        # Retrieve rotation matrix from vertical to inertial frame from the aerodynamic angle calculator
        vertical_to_inertial_frame = aerodynamic_angle_calculator.get_rotation_matrix_between_frames(
            environment.vertical_frame,
            environment.inertial_frame)
        # Compute the thrust in the inertial frame
        thrust_inertial_frame = np.dot(vertical_to_inertial_frame,
                                       thrust_direction_vertical_frame)
        return thrust_inertial_frame


###########################################################################
# CREATE OTHER PROBLEM-SPECIFIC FUNCTIONS #################################
###########################################################################

def get_thrust_acceleration_model_from_parameters(thrust_parameters: list,
                                                  bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                                                  initial_time: float,
                                                  specific_impulse: float) -> \
        tudatpy.kernel.numerical_simulation.propagation_setup.acceleration.ThrustAccelerationSettings:
    """
    Creates the thrust acceleration models from the LunarAscentThrustGuidance class and sets it in the propagator.

    Parameters
    ----------
    thrust_parameters : list of floats
        List of thrust parameters.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    initial_time : float
        The start time of the simulation in seconds.
    specific_impulse : float
        Specific impulse of the vehicle.

    Returns
    -------
    tudatpy.kernel.numerical_simulation.propagation_setup.acceleration.ThrustAccelerationSettings
        Thrust acceleration settings object.
    """
    # Create Thrust Guidance object
    thrust_guidance = LunarAscentThrustGuidance(bodies.get_body('Vehicle'),
                                                initial_time,
                                                thrust_parameters)
    # Retrieves thrust functions
    thrust_direction_function = thrust_guidance.get_current_thrust_direction
    thrust_magnitude = thrust_parameters[0]
    # Set thrust functions in the acceleration model

    thrust_direction_settings = propagation_setup.thrust.custom_thrust_direction(thrust_direction_function)

    #thrust_magnitude_settings = propagation_setup.thrust.custom_thrust_magnitude(thrust_magnitude_function,specific_impulse)
    thrust_magnitude_settings = propagation_setup.thrust.constant_thrust_magnitude(thrust_magnitude, specific_impulse)

    acceleration_settings = propagation_setup.acceleration.thrust_from_direction_and_magnitude(
        thrust_direction_settings, thrust_magnitude_settings)

    # Create and return thrust acceleration settings
    return acceleration_settings
