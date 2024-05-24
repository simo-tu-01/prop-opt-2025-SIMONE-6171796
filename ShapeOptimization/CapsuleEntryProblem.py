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
This module contains the problem-specific classes and functions, which will be called by the main script where the
optimization is executed.
'''


###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# Problem-specific imports
import CapsuleEntryUtilities as Util

# Tudatpy imports
import tudatpy
from tudatpy.data import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import numerical_simulation

###########################################################################
# CREATE PROBLEM CLASS ####################################################
###########################################################################

class ShapeOptimizationProblem:
    """
    Class to initialize, simulate and optimize the Shape Optimization problem.
    The class is created specifically for this problem. This is done to provide better integration with Pagmo/Pygmo,
    where the optimization process (assignment 3) will be done. For the first two assignments, the presence of this
    class is not strictly needed, but it was chosen to create the code so that the same version could be used for all
    three assignments.
    Attributes
    ----------
    bodies
    integrator_settings
    propagator_settings
    capsule_density
    Methods
    -------
    get_last_run_propagated_state_history()
    get_last_run_dependent_variable_history()
    get_last_run_dynamics_simulator()
    fitness(shape_parameters)
    """

    def __init__(self,
                 bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                 termination_settings,
                 capsule_density: float,
                 simulation_start_epoch: float,
                 decision_variable_range):
        """
        Constructor for the ShapeOptimizationProblem class.
        Parameters
        ----------
        bodies : tudatpy.kernel.simulation.environment_setup.SystemOfBodies
            System of bodies present in the simulation.
        integrator_settings : tudatpy.kernel.simulation.propagation_setup.integrator.IntegratorSettings
            Integrator settings to be provided to the dynamics simulator.
        propagator_settings : tudatpy.kernel.simulation.propagation_setup.propagator.MultiTypePropagatorSettings
            Propagator settings object.
        capsule_density : float
            Constant density of the vehicle.
        Returns
        -------
        none
        """
        # Set arguments as attributes
        self.bodies_function = lambda : bodies
        self.termination_settings_function = lambda : termination_settings
        self.capsule_density = capsule_density
        self.simulation_start_epoch = simulation_start_epoch
        self.decision_variable_range = decision_variable_range


    def get_last_run_propagated_cartesian_state_history(self) -> dict:
        """
        Returns the full history of the propagated state, converted to Cartesian states
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function( ).get_equations_of_motion_numerical_solution()

    def get_last_run_propagated_state_history(self) -> dict:
        """
        Returns the full history of the propagated state, not converted to Cartesian state
        (i.e. in the actual formulation that was used during the numerical integration).
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function( ).get_equations_of_motion_numerical_solution_raw()

    def get_bounds(self):

        return self.decision_variable_range

    def get_last_run_dependent_variable_history(self) -> dict:
        """
        Returns the full history of the dependent variables.
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function( ).get_dependent_variable_history()

    def get_last_run_dynamics_simulator(self):
        """
        Returns the dynamics simulator object.
        Parameters
        ----------
        none
        Returns
        -------
        tudatpy.kernel.simulation.propagation_setup.SingleArcDynamicsSimulator
        """
        return self.dynamics_simulator_function( )

    def fitness(self,
                shape_parameters):
        """
        Propagates the trajectory with the shape parameters given as argument.
        This function uses the shape parameters to set a new aerodynamic coefficient interface, subsequently propagating
        the trajectory. The fitness, currently set to zero, can be computed here: it will be used during the
        optimization process.
        Parameters
        ----------
        shape_parameters : list of floats
            List of shape parameters to be optimized.
        Returns
        -------
        fitness : float
            Fitness value (for optimization, see assignment 3).
        """
        bodies = self.bodies_function()

        # Delete existing capsule
        bodies.remove_body('Capsule')
        # Create new capsule with a new coefficient interface based on the current parameters, add it to the body system
        Util.add_capsule_to_body_system(bodies,
                                        shape_parameters,
                                        self.capsule_density)


        # Create propagator settings for benchmark (Cowell)
        dependent_variables_to_save = Util.get_dependent_variable_save_settings()
        termination_settings = self.termination_settings_function( )
        propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                           bodies,
                                                           self.simulation_start_epoch,
                                                           termination_settings,
                                                           dependent_variables_to_save)

        propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
            2.0, propagation_setup.integrator.CoefficientSets.rkdp_87)

        # Create simulation object and propagate dynamics
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies,
            propagator_settings )

        self.dynamics_simulator_function = lambda: dynamics_simulator

        # Add the objective and constraint values into the fitness vector
        fitness = 0.0
        return [fitness]
