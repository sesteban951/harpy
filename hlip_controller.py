from pydrake.all import *

import numpy as np
import scipy as sp

class HybridLIPController(LeafStystem):
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

        # Store some frames that we'll use in the future
        self.torso_frame = self.plant.GetFrameByName("Torso")
        self.left_foot_frame = self.plant.GetFrameByName("FootLeft")
        self.right_foot_frame = self.plant.GetFrameByName("FootRight")

        self.ball_tarsus_left_frame = self.plant.GetFrameByName("BallTarsusLeft")
        self.ball_tarsus_right_frame = self.plant.GetFrameByName("BallTarsusRight")
        self.ball_femur_left_frame = self.plant.GetFrameByName("BallFemurLeft")
        self.ball_femur_right_frame = self.plant.GetFrameByName("BallFemurRight")

        # Store some variables useful for HLIP
        self.t_current = 0
        self.hlip_st_foot_pos = None
        self.hlip_sw_foot_pos = None

        self.p = None
        self.v = None

        # Linear Inverted Pendulum tuning parameters
        self.g = 9.81                         # gravity [m/s^2]
        self.z0 = 0.5                         # const CoM height [m]
        lam = np.sqrt(self.g/self.z0)         # natural frequency [1/s]
        self.A = np.array([[0,1],[lam**2,0]]) # drift matrix of LIP
        
        self.n = 7                            # number of bezier control points
        self.z_apex = 0.2                     # apex height [m]
        self.T = 0.25                         # step period [s]

    # compute the LIP solution at time t given intial condition x0
    def hlip_solution(self, t, x0):
        x_t = sp.linalg.expm(self.A*t) @ x0
        return x_t
    
    # reset map for LIP after completing swing phase
    def hlip_reset(self, x_minus, u):
        # reset map data
        p_stance = u
        p_CoM = x_minus[0]
        v_CoM = x_minus[1]

        # reassign state (i.e., apply reset map)
        p = p_CoM - p_stance
        v = v_CoM
        x_plus = np.array([p,v])
        
        return x_plus
    
    # compute foot placement target based on HLIP model
    def foot_placement(self, p, v):
        # Raibert parameters
        a = 0.2
        b = 1.0
        
        # compute foot placement target
        u = a*v + b*p
        return u
    
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
        
        # compute desired 
    


