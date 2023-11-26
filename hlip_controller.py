from pydrake.all import *

import numpy as np
import scipy as sp

class HybridLIPController(LeafSystem):
    """
    Controller Based on Hybrid LIP Model.

    It takes state estimates `x_hat` as input, and outputs target joint angles
    `q_nom`, target joint velocities `v_nom`, feed-forward joint torques `tau_ff`, 
    and thruster forces `thrust`.

                       ----------------
                       |              |
                       |              | ---> x_nom = [q_nom, v_nom]
                       |              |
            x_hat ---> |  Controller  | ---> tau_ff
                       |              |
                       |              | ---> thrust
                       |              |
                       ----------------

    A joint-level PD controller on the robot will compute torques on each joint as

        tau = tau_ff + Kp * (q - q_nom) + Kd * (v - v_nom)
    """
    def __init__(self):
        LeafSystem.__init__(self)

        # Create an internal system model for IK calculations, etc.
        self.plant = MultibodyPlant(0)
        Parser(self.plant).AddModels("./models/urdf/harpy_planar.urdf")
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        self.input_port = self.DeclareVectorInputPort(
                "x_hat",
                BasicVector(13 + 13))  # 13 positions and velocities

        # We'll do some fancy caching stuff so that both outputs can be
        # computed with the same method.
        self._cache = self.DeclareCacheEntry(
                description="controller output cache",
                value_producer=ValueProducer(
                    allocate=lambda: AbstractValue.Make(dict()),
                    calc=lambda context, output: output.set_value(
                        self.CalcOutput(context))))
        
        # define output ports for control
        self.DeclareVectorOutputPort(
                "tau_ff",
                BasicVector(6),  # 2 DoF per leg, plus thruster angle
                lambda context, output: output.set_value(
                    self._cache.Eval(context)["tau_ff"]),
                prerequisites_of_calc={self._cache.ticket()})

        self.DeclareVectorOutputPort(
                "x_nom",
                BasicVector(12),  # actuated positions + velocities
                lambda context, output: output.set_value(
                    self._cache.Eval(context)["x_nom"]),
                prerequisites_of_calc={self._cache.ticket()})

        self.DeclareVectorOutputPort(
                "thrust",
                BasicVector(2),
                lambda context, output: output.set_value(
                    self._cache.Eval(context)["thrust"]),
                prerequisites_of_calc={self._cache.ticket()})

        # Store some variables useful for HLIP
        self.t_phase = 0

        # initla HLIP state
        # p = (p_CoM - p_stance) in world frame
        # v = (v_CoM)            in world frame
        # x = [p ; v]
        p0 = 0.2
        v0 = -0.1
        self.x_hlip = np.array([[p0],[v0]])    
        self.u_hlip = 0
    
        self.x_plus = self.x_hlip # HLIP post_imapct state

        # Linear Inverted Pendulum tuning parameters
        self.g = 9.81                         # gravity [m/s^2]
        self.z0 = 0.5                         # const CoM height [m]
        lam = np.sqrt(self.g/self.z0)         # natural frequency [1/s]
        self.A = np.array([[0,1],[lam**2,0]]) # drift matrix of LIP
        
        self.ctrl_pts = 7                     # number of bezier control points
        self.z_apex = 0.2                     # apex height [m]
        self.T = 0.25                         # step period [s]
        self.steps = 0                        # number of steps taken

        # Frames used to track LIP trajectories
        self.torso_frame = self.plant.GetFrameByName("Torso")
        self.left_foot_frame = self.plant.GetFrameByName("FootLeft")
        self.right_foot_frame = self.plant.GetFrameByName("FootRight")

        # swing and stance foot frames and variables
        self.stance_foot_frame = None
        self.swing_foot_frame = None

        # robto CoM state variables
        self.x_robot = np.array([[0],[0]]) 

        # instantiate inverse kinematics solver
        self.ik = InverseKinematics(self.plant)

        # inverse kinematics solver settings
        self.epsilon_feet = 0.001   # foot position tolerance     [m]
        self.epsilon_base = 0.01    # torso position tolerance    [m]
        self.epsilon_orient = 0.1   # torso orientation tolerance [rad]
        self.tol_feet = np.array([[self.epsilon_feet], [np.inf], [self.epsilon_feet]]) # x-z only
        self.tol_base = np.array([[np.inf], [np.inf], [self.epsilon_base]])            # z only

        # Add distance constraints to IK for the 4-bar linkages (fixed)
        self.left_4link =  self.ik.AddPointToPointDistanceConstraint(self.plant.GetFrameByName("BallTarsusLeft"), [0, 0, 0],
                                                                     self.plant.GetFrameByName("BallFemurLeft"), [0, 0, 0], 
                                                                     0.32, 0.32)
        self.right_4link = self.ik.AddPointToPointDistanceConstraint(self.plant.GetFrameByName("BallTarsusRight"), [0, 0, 0],
                                                                     self.plant.GetFrameByName("BallFemurRight"), [0, 0, 0], 
                                                                     0.32, 0.32)
        
        # Add torso position and orientation constraints
        self.p_torso_cons = self.ik.AddPositionConstraint(self.torso_frame, [0, 0, 0],
                                                          self.plant.world_frame(), 
                                                          np.array([0,0,0]), np.array([0,0,0])) 
        self.r_torso_cons = self.ik.AddOrientationConstraint(self.torso_frame, RotationMatrix(),
                                                             self.plant.world_frame(), RotationMatrix(),
                                                             self.epsilon_orient)        
        
        # Add foot constraints
        self.p_left_cons =  self.ik.AddPositionConstraint(self.left_foot_frame, [0, 0, 0],
                                                          self.plant.world_frame(), 
                                                          np.array([0,0,0]), np.array([0,0,0]))
        self.p_right_cons = self.ik.AddPositionConstraint(self.right_foot_frame, [0, 0, 0],
                                                          self.plant.world_frame(), 
                                                          np.array([0,0,0]), np.array([0,0,0]))

    ############################### HLIP Functions ############################

    # compute the LIP solution at time t given intial condition x0
    def hlip_solution(self,x0):
        x_t = sp.linalg.expm(self.A * self.t_phase) @ x0
        self.x_hlip = x_t
        self.u = self.hlip_foot_placement(x_t)
    
    # reset map for LIP after completing swing phase
    def hlip_reset(self, x_minus, u):

        # reset map data
        p_minus = x_minus[0]
        v_minus = x_minus[1]

        # reassign state (i.e., apply reset map), Eq. 5
        p_plus = p_minus - u
        v_plus = v_minus
        
        # update HLIP state
        x_plus = np.array([p_plus,v_plus])

        return x_plus
    
    # compute foot placement target based on HLIP model
    def hlip_foot_placement(self,x):
        
        # HLIP parameters
        u_des = 0.0                        # P1 orbit step size desired
        K = np.array([1, 0.25])           # feedback gain
        x_des = np.array([[0.0], [0.0]])   # HLIP state desired

        # compute foot placement target
        u = u_des + K @ (x - x_des)
        return u

    # check if reset map need soto be applied
    def hlip_phase_check(self,t_sim):

        # update swing phase time and step count
        self.t_phase = t_sim % self.T
        self.steps = np.floor(t_sim / self.T)

        # apply reset map
        if self.t_phase == 0 and self.steps > 0:
            self.x_plus = self.hlip_reset(self.x_hlip, self.hlip_foot_placement(self.x_hlip))
            self.x_hlip = self.x_plus
     
    ############################### Robot Functions ############################

    # compute foot position based on bezier curve
    def foot_bezier(self,t, uf):
        
        # foot placement parameters
        u0 = 0.0

        # foot clearance tuning parameters
        z_offset = 0.027  # z-offset on the foot
        z0 = 0.0          # intial swing foot height
        zf = -0.01        # final swing foot height (neg to ensure foot strike)

        # compute bezier curve control points, 7-pt or 5-pt bezier
        if self.ctrl_pts == 7:
            ctrl_pts_x = np.array([[u0],[u0],[u0],[(u0+uf)/2],[uf],[uf],[uf]])
            ctrl_pts_z = np.array([[z0],[z0],[z0],[(16/5)*self.z_apex],[zf],[zf],[zf]]) + z_offset
        elif self.ctrl_pts == 5:
            ctrl_pts_x = np.array([[u0],[u0],[(u0+uf)/2],[uf],[uf]])
            ctrl_pts_z = np.array([[z0],[z0],[(8/3)*self.z_apex],[zf],[zf]]) + z_offset

        # create control points matrix
        ctrl_pts = np.hstack((ctrl_pts_x,ctrl_pts_z))

        # evaluate bezier at time t
        bezier = BezierCurve(0,self.T,ctrl_pts)

    def update_foot_pos(self):
        # compute robot foot positions
        self.right_foot_pos = self.plant.CalcPointsPositions(self.plant_context, self.right_foot_frame, 
                                                             [0,0,0], self.plant.world_frame())    
        self.left_foot_pos = self.plant.CalcPointsPositions(self.plant_context, self.left_foot_frame, 
                                                          [0,0,0], self.plant.world_frame())
    
    # Estimate CoM position and velocity wrt to world frame
    def update_CoM_state(self):
        """
        Updates the robot's center of mass position and velocity wrt world frame.
        """
        # compute p_com
        p_com = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context)

        # compute v_com, via Jacobian (3x13) and generalized velocity (13x1)
        J = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(self.plant_context, 
                                                                     JacobianWrtVariable.kV,
                                                                     self.plant.world_frame(), 
                                                                     self.plant.world_frame())
        v = self.plant.GetVelocities(self.plant_context)
        v_all = J @ v
        v_com = v_all[0:3]

        # update CoM info
        self.v_com = v_com.T
        self.p_com = p_com.T

    # inverse kinematics solver
    def DoInverseKinematics(self, p_torso, p_right, p_left):

        # Update constraint on torso position
        p_torso_lb = p_torso - self.tol_base
        p_torso_ub = p_torso + self.tol_base
        self.p_torso_cons.evaluator().UpdateLowerBound(p_torso_lb)
        self.p_torso_cons.evaluator().UpdateUpperBound(p_torso_ub)
        
        # Update constraints on the positions of the feet
        p_left_lb = p_left - self.tol_feet
        p_left_ub = p_left + self.tol_feet
        p_right_lb = p_right - self.tol_feet
        p_right_ub = p_right + self.tol_feet
        self.p_left_cons.evaluator().UpdateLowerBound(p_left_lb)
        self.p_left_cons.evaluator().UpdateUpperBound(p_left_ub)
        self.p_right_cons.evaluator().UpdateLowerBound(p_right_lb)
        self.p_right_cons.evaluator().UpdateUpperBound(p_right_ub)

        # solve the IK problem        
        res = SnoptSolver().Solve(self.ik.prog())
        assert res.is_success(), "Inverse Kinematics Failed!"
        
        return res.GetSolution(self.ik.q())

    ############################### Sys Output Function ############################

    def CalcOutput(self,context):

        # Set our internal model to match the state estimte
        x_hat = self.EvalVectorInput(context, 0).get_value()
        self.plant.SetPositionsAndVelocities(self.plant_context, x_hat)
        
        # update hlip state
        self.hlip_phase_check(context.get_time())
        self.hlip_solution(self.x_plus)

        print(50*"*")
        print("time: ",self.t_phase)
        print("x_hlip = ", self.x_hlip)
        print("u = ", self.u)

        # update robot info
        self.update_CoM_state()
        self.update_foot_pos()
        print("right foot pos = ", self.right_foot_pos)
        print("left foot pos = ", self.left_foot_pos)

        # get foot placement targets
        p_torso_target = np.array([[0],[0],[self.z0]])
        p_right_target = np.array([[0.],[0],[0.0]])
        p_left_target = np.array([[0],[0],[0.2]])
        q_ik = self.DoInverseKinematics(p_torso_target, p_right_target, p_left_target)

        # set desired joint angles and velocities
        q_nom = np.array([
            q_ik[3], q_ik[4],    # thrusters
            q_ik[5], q_ik[9],    # hip
            q_ik[6], q_ik[10]])  # knee
        v_nom = np.zeros(6)
        x_nom = np.block([q_nom, v_nom])

        return {"tau_ff": np.zeros(6), "x_nom": x_nom, "thrust": np.array([0, 0])}

