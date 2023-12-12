#!/usr/bin/env python

# import drake
from pydrake.all import (StartMeshcat, DiagramBuilder,
        AddMultibodyPlantSceneGraph, AddDefaultVisualization, Parser)

# import pyidto tools
from pyidto.trajectory_optimizer import TrajectoryOptimizer
from pyidto.problem_definition import ProblemDefinition
from pyidto.solver_parameters import SolverParameters
from pyidto.trajectory_optimizer_solution import TrajectoryOptimizerSolution
from pyidto.trajectory_optimizer_stats import TrajectoryOptimizerStats

import numpy as np
from copy import deepcopy
import time

class CIMPC():
    "Simple CI-MPC class for the Harpy robot."
    def __init__(self, model_file):

        # robot file
        self.model_file = model_file

        # MPC tiem horizon settings
        self.T = 2.                 # total time horizon
        self.dt = 0.05               # time step
        N = int(self.T/self.dt)      # number of steps
        
        # instantitate problem definition
        self.problem = ProblemDefinition()
        self.problem.num_steps = N

        # stage cost weights
        self.problem.Qq = np.diag([2, 2, 2, 0.1, 0.1, 0.1, 0.1])
        self.problem.Qv = np.diag([0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
        self.problem.R = np.diag([1e5, 1e5, 1e5,
                                  0.01, 0.01, 0.01, 0.01])
        # terminal cost weights
        self.problem.Qf_q = 2 * np.eye(7)
        self.problem.Qf_v = 0.2 * np.eye(7)

        # solver parameters
        self.params = SolverParameters()
        
        # Trust region solver parameters
        self.params.max_iterations = 200
        self.params.scaling = True
        self.params.equality_constraints = True
        self.params.Delta0 = 1e3
        self.params.Delta_max = 1e5
        self.params.num_threads = 4

        # Contact modeling parameters
        self.params.contact_stiffness = 5e3
        self.params.dissipation_velocity = 0.1
        self.params.smoothing_factor = 0.01
        self.params.friction_coefficient = 0.5
        self.params.stiction_velocity = 0.1

        self.params.verbose = True

        # solver intial guess and refernce trajectories
        self.q_guess = None
        self.q0 = None
        self.qf = None

        # Allocate some structs that will hold the solution
        self.sol = TrajectoryOptimizerSolution()
        self.stats = TrajectoryOptimizerStats()

    # update intial conditions
    def update_init_final_condition(self,q0,qf):
        self.q0 = q0
        self.qf = qf
        self.problem.q_init = q0
        self.problem.v_init = np.zeros(7)
    
    # make a reference trajectory
    def update_ref_traj(self):
        # noimnal trajectory containers
        q_nom = []
        v_nom = []

        # interpolate to get refernece trajectory
        for k in range(self.problem.num_steps + 1):
            sigma = k / self.problem.num_steps
            q_nom.append((1 - sigma) * self.q0 + sigma * self.qf)
            v_nom.append(np.array([0.17, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        self.problem.q_nom = q_nom
        self.problem.v_nom = v_nom

    # solve the MPC problem
    def solve(self):
        # instantiate trajectory optimizer
        opt = TrajectoryOptimizer(self.model_file,self.problem, self.params, self.dt)

        # solve the problem
        self.q_guess = deepcopy(self.problem.q_nom)
        opt.Solve(self.q_guess, self.sol, self.stats)

    # visualization
    def visualize(self,q):
        # Start meshcat for visualization
        meshcat = StartMeshcat()

        # create simple diagram
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
        Parser(plant).AddModels(self.model_file)
        plant.Finalize()

        # Connect to the meshcat visualizer
        AddDefaultVisualization(builder, meshcat)

        # Build the system diagram
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()
        plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
        plant.get_actuation_input_port().FixValue(plant_context,
                np.zeros(plant.num_actuators()))

        # Step through q, setting the plant positions at each step
        meshcat.StartRecording()
        for k in range(len(q)):
            diagram_context.SetTime(k * self.dt)
            plant.SetPositions(plant_context, q[k])
            diagram.ForcedPublish(diagram_context)
            time.sleep(self.dt)
        meshcat.StopRecording()
        meshcat.PublishRecording()

if __name__=="__main__":

    # Relative path to the model file that we'll use
    model_file = "../models/urdf/harpy_planar_CIMPC.urdf"

    # intial and final configuartion
    q0 = np.array([0.0,   # base horizontal position
                        0.515, # base vertical position
                        0.0,   # base orientation
                       -0.45,  # right hip
                        1.15,  # right knee
                       -0.45,  # left hip
                        1.15]) # left knee
    qf = deepcopy(q0)
    qf[0] = q0[0] + 0.17 * 0.5

    # insatntiate the CIMPC class
    mpc = CIMPC(model_file)

    # update the initial condition
    mpc.update_init_final_condition(q0,qf)
    mpc.update_ref_traj()

    # solve the MPC problem
    mpc.solve()

    q_sol = mpc.sol.q
    solve_time = np.sum(mpc.stats.iteration_times)

    print(q_sol)
    print(solve_time)

    mpc.visualize(q_sol)
