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
        self.x = np.array([[p0],[v0]])    
    
        self.x_plus = self.x # HLIP post_imapct state

        # Linear Inverted Pendulum tuning parameters
        self.g = 9.81                         # gravity [m/s^2]
        self.z0 = 0.5                         # const CoM height [m]
        lam = np.sqrt(self.g/self.z0)         # natural frequency [1/s]
        self.A = np.array([[0,1],[lam**2,0]]) # drift matrix of LIP
        
        self.n = 7                            # number of bezier control points
        self.z_apex = 0.2                     # apex height [m]
        self.T = 0.25                         # step period [s]
        self.steps = 0                        # number of steps taken

    ############################### HLIP Functions ############################

    # compute the LIP solution at time t given intial condition x0
    def hlip_solution(self,x0):
        x_t = sp.linalg.expm(self.A * self.t_phase) @ x0
        self.x = x_t
    
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
    def hlip_foot_placement(self, x):
        
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
            self.x_plus = self.hlip_reset(self.x, self.hlip_foot_placement(self.x))
            self.x = self.x_plus
     
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
        if self.n == 7:
            ctrl_pts_x = np.array([[u0],[u0],[u0],[(u0+uf)/2],[uf],[uf],[uf]])
            ctrl_pts_z = np.array([[z0],[z0],[z0],[(16/5)*self.z_apex],[zf],[zf],[zf]]) + z_offset
        elif self.n == 5:
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

    ############################### Main Functions ############################

    def CalcOutput(self,context):

        # Set our internal model to match the state estimte
        x_hat = self.EvalVectorInput(context, 0).get_value()
        self.plant.SetPositionsAndVelocities(self.plant_context, x_hat)
        
        print("*"*50)

        # update time and LIP states
        print("Sim_time:")
        print(context.get_time())

        # check if reset map should be applied
        self.hlip_phase_check(context.get_time())
        print("t_phase:",self.t_phase)

        print("steps:", self.steps)

        self.hlip_solution(self.x_plus)
        print("HLIP state:")
        print(self.x)

        # set desired joint angles and velocities
        q_nom = np.zeros(6)
        v_nom = np.zeros(6)
        x_nom = np.block([q_nom, v_nom])

        return {"tau_ff": np.zeros(6), "x_nom": x_nom, "thrust": np.array([0, 0])}

