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

    def __init__(self, sim_type, config):

        sim_type = "jump"

        # robot file
        self.model_file = config[sim_type]["model_file"]

        # MPC time horizon settings
        self.T = config[sim_type]["T"]    # total time horizon
        self.dt = config[sim_type]["dt"]  # time step
        N = int(self.T/self.dt)           # number of steps
        
        # instantitate problem definition
        self.problem = ProblemDefinition()
        self.problem.num_steps = N

        # stage cost weights
        Qq = config[sim_type]["Qq"]
        Qv = config[sim_type]["Qv"]
        R = config[sim_type]["R"]
        self.problem.Qq = np.diag(Qq)
        self.problem.Qv = np.diag(Qv)
        self.problem.R = np.diag(R)

        # terminal cost weights
        Qf_q = config[sim_type]["Qf_q"]
        Qf_v = config[sim_type]["Qf_v"]
        self.problem.Qf_q = np.diag(Qf_q)
        self.problem.Qf_v = np.diag(Qf_v)

        # solver parameters
        self.params = SolverParameters()
        
        # Trust region solver parameters
        self.params.max_iterations = 300
        self.params.scaling = True
        self.params.equality_constraints = True
        self.params.Delta0 = 1e3
        self.params.Delta_max = 1e52
        self.params.num_threads = 4

        # Contact modeling parameters
        self.params.contact_stiffness = 5e3
        self.params.dissipation_velocity = 0.5
        self.params.smoothing_factor = 0.01
        self.params.friction_coefficient = 0.7
        self.params.stiction_velocity = 0.05

        self.params.verbose = True

        # solver intial guess and refernce trajectories
        self.q_guess = None

        # control points for bezier curve
        self.ctrl_pts = np.array(config[sim_type]["ctrl_pts"])

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
        # make the refernce trajectory
        self.update_ref_traj_(self.ctrl_pts)

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

        print("Qq:\n",self.problem.Qq)
        print("Qv:\n",self.problem.Qv)
        print("R: \n",self.problem.R)
        print("Qf_q:\n",self.problem.Qf_q)
        print("Qf_v:\n",self.problem.Qf_v)

        print("ctrl_pts:\n",self.ctrl_pts)

if __name__=="__main__":

    # import simulaiton config yaml file
    with open('sim_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # insatntiate the CIMPC class
    sim_type = "walk"
    mpc = CIMPC(sim_type, config)

    # solve the MPC problem
    mpc.solve()

    q_sol = mpc.sol.q
    solve_time = np.sum(mpc.stats.iteration_times)

    print("\nSolve time:", solve_time)

    mpc.visualize(q_sol)
