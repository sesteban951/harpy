#!/usr/bin/env python

# import drake
from pydrake.all import (StartMeshcat, DiagramBuilder,
                         AddMultibodyPlantSceneGraph, AddDefaultVisualization, 
                         Parser, BezierCurve,
                         JacobianWrtVariable)

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
import datetime

class CIMPC():
    """Simple CI-MPC class for the Harpy robot.
    Inputs: mode file    
    """

    # TODO: Think about how to track arbitrary forces and torques on
    #       the robot. This is not the case for the real robot. 
    #       The map:  [f_r, f_l] -> [f, tau] is injective. Need to project to the set of 
    #       feasible torque wrenches or somehting.

    def __init__(self, sim_type, config):

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

        # containers for saving data
        self.t_array = None
        self.r_foot_pos = None
        self.l_foot_pos = None

        # objects for drake system diagram
        self.builder = None
        self.plant = None
        self.diagram = None
        self.diagram_context = None
        self.plant_context = None

    # create bezier curve trajectory
    def update_ref_traj_(self, ctrl_pts):
        # create bezier curve parameterized by control points and t in [0, T]
        b = BezierCurve(0, self.T, ctrl_pts)
        
        # noimnal trajectory containers
        q_nom = []
        v_nom = []
        self.t_array = []

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
            self.t_array.append(t_k)

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
        self.stats = TrajectoryOptimizerStats()

    # create model of the robot
    def create_model(self):

        # create simple diagram
        self.builder = DiagramBuilder()
        self.plant, scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=1e-3)
        Parser(self.plant).AddModels(self.model_file)
        self.plant.Finalize()

    # create diagram context
    def create_diagram_context(self):

        # Build the system diagram
        self.diagram = self.builder.Build()
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = self.diagram.GetMutableSubsystemContext(self.plant, self.diagram_context)
        self.plant.get_actuation_input_port().FixValue(self.plant_context,
                np.zeros(self.plant.num_actuators()))

    # save the reference and solution trajectories.
    def save_solution(self):

        # get current date and time
        now = datetime.datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H:%M:%S_")

        # save ROM trajectories
        q_nom_str = "data/" + time_str + sim_type + "_q_nom" + ".txt"
        q_sol_str = "data/" + time_str + sim_type + "_q_sol" + ".txt"
        v_nom_str = "data/" + time_str + sim_type + "_v_nom" + ".txt"
        v_sol_str = "data/" + time_str + sim_type + "_v_sol" + ".txt"      
        np.savetxt(q_nom_str, np.array(self.problem.q_nom))
        np.savetxt(q_sol_str, np.array(self.sol.q))
        np.savetxt(v_nom_str, np.array(self.problem.v_nom))
        np.savetxt(v_sol_str, np.array(self.sol.v))

        # save the time stamps
        t_str = "data/" + time_str + sim_type + "_time" + ".txt"
        np.savetxt(t_str, np.array(self.t_array))

        # create model and diagram context
        self.create_model()
        self.create_diagram_context()

        # create foot frame postions
        r_foot_frame = self.plant.GetFrameByName("FootRight")
        l_foot_frame = self.plant.GetFrameByName("FootLeft")

        r_pos_list = []
        l_pos_list = []
        r_vel_list = []
        l_vel_list = []
        r_acc_list = []
        l_acc_list = []

        a_sol = []

        # TODO: get the CoM velocities and accelerations in world frame
        # look into CalcBiasCenterOfMassTranslationalAcceleration

        # iterate through time steps
        for k in range(self.problem.num_steps + 1):
            # set solution conifguration pos and vel
            q_v = np.vstack((self.sol.q[k], self.sol.v[k])).reshape(-1, 1)
            self.plant.SetPositionsAndVelocities(self.plant_context, q_v)
            print(q_v)
            # get foot positions in world frame
            r_pos = self.plant.CalcPointsPositions(self.plant_context, r_foot_frame, 
                                                   [0, 0, 0], self.plant.world_frame())
            l_pos = self.plant.CalcPointsPositions(self.plant_context, l_foot_frame, 
                                                   [0, 0, 0], self.plant.world_frame())
            r_pos_list.append(np.array(r_pos.T)[0])
            l_pos_list.append(np.array(l_pos.T)[0])

            # get foot end velocities in world frame, v_trans = J(q) * v_joint
            J_r = self.plant.CalcJacobianTranslationalVelocity(self.plant_context, JacobianWrtVariable.kV,
                                                               r_foot_frame, [0, 0, 0], self.plant.world_frame(),
                                                               self.plant.world_frame())
            J_l = self.plant.CalcJacobianTranslationalVelocity(self.plant_context, JacobianWrtVariable.kV,
                                                               l_foot_frame, [0, 0, 0], self.plant.world_frame(),
                                                               self.plant.world_frame())
            r_vel = J_r @ np.array(self.sol.v[k])
            l_vel = J_l @ np.array(self.sol.v[k])
            r_vel_list.append(r_vel)
            l_vel_list.append(l_vel)

            # get foot end accelerations in world frame, v_dot = J_dot(q) * v_joint + J(q) * v_joint_dot
            Jdqd_r = self.plant.CalcBiasTranslationalAcceleration(self.plant_context, JacobianWrtVariable.kV,
                                                                     r_foot_frame, [0, 0, 0], self.plant.world_frame(),
                                                                     self.plant.world_frame()).flatten()
            Jdqd_l = self.plant.CalcBiasTranslationalAcceleration(self.plant_context, JacobianWrtVariable.kV,
                                                                     l_foot_frame, [0, 0, 0], self.plant.world_frame(),
                                                                     self.plant.world_frame()).flatten()
            if k != self.problem.num_steps:
                vdot_k = (self.sol.v[k+1] - self.sol.v[k]) / self.dt
            else:
                vdot_k = (self.sol.v[k] - self.sol.v[k-1]) / self.dt
            a_sol.append(vdot_k)

            r_acc = Jdqd_r + J_r @ np.array(vdot_k)
            l_acc = Jdqd_l + J_l @ np.array(vdot_k)
            r_acc_list.append(r_acc)
            l_acc_list.append(l_acc)

        # save foot positions, velocities, and accelerations
        r_pos_str = "data/" + time_str + sim_type + "_pos_r_foot" + ".txt"
        l_pos_str = "data/" + time_str + sim_type + "_pos_l_foot" + ".txt"
        r_vel_str = "data/" + time_str + sim_type + "_vel_r_foot" + ".txt"
        l_vel_str = "data/" + time_str + sim_type + "_vel_l_foot" + ".txt"
        r_acc_str = "data/" + time_str + sim_type + "_acc_r_foot" + ".txt"
        l_acc_str = "data/" + time_str + sim_type + "_acc_l_foot" + ".txt"
        np.savetxt(r_pos_str, np.array(r_pos_list)) 
        np.savetxt(l_pos_str, np.array(l_pos_list))
        np.savetxt(r_vel_str, np.array(r_vel_list))
        np.savetxt(l_vel_str, np.array(l_vel_list))
        np.savetxt(r_acc_str, np.array(r_acc_list))
        np.savetxt(l_acc_str, np.array(l_acc_list))

        # save solution accelerations
        a_sol_str = "data/" + time_str + sim_type + "_a_sol" + ".txt"
        np.savetxt(a_sol_str, np.array(a_sol))
        
        print("\nSaved the reference trajectory data.")

    # visualization
    def visualize(self,q):
        # Start meshcat for visualization
        meshcat = StartMeshcat()

        self.create_model()

        # Connect to the meshcat visualizer
        AddDefaultVisualization(self.builder, meshcat)

        self.create_diagram_context()

        # Step through q, setting the plant positions at each step
        meshcat.StartRecording()
        for k in range(len(q)):
            self.diagram_context.SetTime(k * self.dt)
            self.plant.SetPositions(self.plant_context, q[k])
            self.diagram.ForcedPublish(self.diagram_context)
            time.sleep(self.dt)
        meshcat.StopRecording()
        meshcat.PublishRecording()

if __name__=="__main__":

    # import simulaiton config yaml file
    with open('sim_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # insatntiate the CIMPC class
    sim_type = "jump"
    mpc = CIMPC(sim_type, config)
    mpc.update_ref_traj_(mpc.ctrl_pts)

    # see the reference trajectory or solve the MPC problem
    # 1 = see ref, 0 = solve MPC
    see_ref_traj = 0
    save_sol_traj = 1
    
    # just see the refernce trajecotry
    if see_ref_traj == 1:
        q_ref = mpc.problem.q_nom
        mpc.visualize(q_ref)    
    
    # solve the MPC problem
    elif see_ref_traj == 0:    
        mpc.solve()
        q_sol = mpc.sol.q
        solve_time = np.sum(mpc.stats.iteration_times)
        print("\nSolve time:", solve_time)
        
        if save_sol_traj == 1:
            mpc.save_solution()

        mpc.visualize(q_sol)
