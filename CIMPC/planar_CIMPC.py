#!/usr/bin/env python

# import drake
from pydrake.all import (StartMeshcat, DiagramBuilder,
        AddMultibodyPlantSceneGraph, AddDefaultVisualization, Parser, BezierCurve)

# import pyidto tools
from pyidto.trajectory_optimizer import TrajectoryOptimizer
from pyidto.problem_definition import ProblemDefinition
from pyidto.solver_parameters import SolverParameters
from pyidto.trajectory_optimizer_solution import TrajectoryOptimizerSolution
from pyidto.trajectory_optimizer_stats import TrajectoryOptimizerStats

import numpy as np
from copy import deepcopy
import time
import yaml

class CIMPC():
    """Simple CI-MPC class for the Harpy robot.
    Inputs: mode file    
    """

    # TODO: Think about how to track arbitrary forces and torques on
    #       the robot. This is not the case for the real robot. 
    #       The map:  [f_r, f_l] -> [f, tau] is injective. Need to project to the set of 
    #       feasible torque wrenches or somehting.

    def __init__(self, model_file):

        # robot file
        self.model_file = model_file

        # MPC time horizon settings
        self.T = 4.0                 # total time horizon
        self.dt = 0.05               # time step
        N = int(self.T/self.dt)      # number of steps
        
        # instantitate problem definition
        self.problem = ProblemDefinition()
        self.problem.num_steps = N

        # stage cost weights
        self.problem.Qq = np.diag([2, 2, 2, 0.1, 0.1, 0.1, 0.1])
        self.problem.Qv = np.diag([2, 2, 2, 0.1, 0.1, 0.1, 0.1])
        self.problem.R = np.diag([1e5, 1e5, 1e5,
                                  0.02, 0.02, 0.02, 0.02])
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

        # control points for bezier curve
        self.ctrl_pts = None

        # Allocate some structs that will hold the solution
        self.sol = TrajectoryOptimizerSolution()
        self.stats = TrajectoryOptimizerStats()

    # create bezier curve trajectory
    def update_ref_traj_(self, ctrl_pts):
        # create bezier curve parameterized by control points and t in [0, T]
        b = BezierCurve(0, self.T, ctrl_pts)
        
        # noimnal trajectory containers
        q_nom = []
        v_nom = []

        # evaluate bezier curve at each time step
        for k in range(self.problem.num_steps + 1):
            # time at time step k
            t_k = k * self.dt

            # evaluate bezier curve at time t
            q_k = b.value(t_k)             # eval bezier curve at time t
            v_k = b.EvalDerivative(t_k, 1) # 1st derivative at time t

            # append to nominal trajectory
            q_nom.append(q_k)
            v_nom.append(v_k)

        # assign reference trajectory to problem
        self.problem.q_nom = q_nom
        self.problem.v_nom = v_nom

        # update intial conditions
        self.problem.q_init = q_nom[0]
        self.problem.v_init = v_nom[0]

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

    # import simulaiton config yaml file
    with open('sim_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # import all sim config settings
    sim_type = config["sim_type"]
    model_file = config[sim_type]["model_file"]

    # intial configuartion
    q0_b = np.array([0,    # base horizontal position
                     0.52,  # base vertical position
                     0.0])   # base orientation
    q0_j = np.array([-0.45,  # right hip
                      1.15,  # right knee
                     -0.45,  # left hip
                      1.15]) # left knee
    q0 = np.vstack((q0_b.reshape(-1, 1), 
                    q0_j.reshape(-1, 1)))

    # final configuration
    qf_b = np.array([0.5,    # base horizontal position
                     0.52,  # base vertical position
                     0.0])   # base orientation
    qf_j = np.array([-0.45,  # right hip
                      1.15,  # right knee
                     -0.45,  # left hip
                      1.15]) # left knee
    qf = np.vstack((qf_b.reshape(-1, 1), 
                    qf_j.reshape(-1, 1)))
    
    # define base x-z position and orientation, and joint bezier control points for state ref traj
    n_pts = 5
    # linear interpolation
    if n_pts == 3:
        ctrl_pts_x = np.array([q0_b[0], (q0_b[0]+qf_b[0])/2, qf_b[0]])
        ctrl_pts_z = np.array([q0_b[1], (q0_b[1]+qf_b[1])/2, qf_b[1]])
        ctrl_pts_t = np.array([q0_b[2], (q0_b[2]+qf_b[2])/2, qf_b[2]])
        ctrl_pts_j = np.array([q0_j, (q0_j+qf_j)/2, qf_j]).T
    # enforce 0 intial and final velocity
    elif n_pts == 5:
        ctrl_pts_x = np.array([q0_b[0], q0_b[0], (q0_b[0]+qf_b[0])/2, qf_b[0], qf_b[0]])
        ctrl_pts_z = np.array([q0_b[1], q0_b[1], (q0_b[1]+qf_b[1])/2, qf_b[1], qf_b[1]])
        ctrl_pts_t = np.array([q0_b[2], q0_b[2], (q0_b[2]+qf_b[2])/2, qf_b[2], qf_b[2]])
        ctrl_pts_j = np.array([q0_j, q0_j, (q0_j+qf_j)/2, qf_j, qf_j]).T
    # custom control points, should be expressive enough to do most things?
    elif n_pts == 7:
        # TODO: implement a seven point bezier curve. How should I do this?
        pass

    # define full configuration bezier control points, control points are columns
    ctrl_pts = np.vstack((ctrl_pts_x, ctrl_pts_z, ctrl_pts_t, ctrl_pts_j))

    # insatntiate the CIMPC class
    mpc = CIMPC(model_file)

    # update the reference trajecotry
    mpc.update_ref_traj_(ctrl_pts)

    # solve the MPC problem
    mpc.solve()

    q_sol = mpc.sol.q
    solve_time = np.sum(mpc.stats.iteration_times)

    print("\nSolve time:", solve_time)

    mpc.visualize(q_sol)
