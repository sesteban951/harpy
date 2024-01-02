
from pydrake.all import *
import numpy as np

class Controller(LeafSystem):
    """
    Simple controller class for the Harpy robot that tracks a modified SRb model trajectory.
    
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

        # initialize leaf system
        LeafSystem.__init__(self)

        # load trajectory data
        data_path = "./data/2023-12-29_14:19:51_jump_"
        labels = ["q_sol.txt","pos_r_foot.txt","pos_l_foot.txt","time.txt"]
        
        # floating base trajectory (x-pos, y-pos, theta)
        data_str = data_path + labels[0]       
        self.q_base = np.loadtxt(data_str)[:, 0:3]

        # foot end trajectories
        data_str = data_path + labels[1]       
        self.pos_r_foot = np.loadtxt(data_str)
        data_str = data_path + labels[2]       
        self.pos_l_foot = np.loadtxt(data_str)
        
        # time stamps
        data_str = data_path + labels[3]       
        self.time = np.loadtxt(data_str)

        # Create an internal system model for IK calculations, etc.
        self.plant = MultibodyPlant(0)
        Parser(self.plant).AddModels("../models/urdf/harpy_planar.urdf")
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        # set the input port
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

        # store swing and stance foot variables
        self.t_current = 0.0

        # Store center of mass variables for CoM position and velocity wrt world (for LIP)
        self.p_com = None
        self.v_com = None

        # instantiate inverse kinematics solver
        self.ik = InverseKinematics(self.plant)

        # inverse kinematics solver settings
        self.epsilon_feet = 0.001   # foot position tolerance     [m]
        self.epsilon_base = 0.01    # torso position tolerance    [m]
        self.epsilon_orient = 0.1   # torso orientation tolerance [rad]
        self.tol_feet = np.array([[self.epsilon_feet], [np.inf], [self.epsilon_feet]])
        self.tol_base = np.array([[np.inf], [np.inf], [self.epsilon_base]])

        # Add distance constraints to IK for the 4-bar linkages (fixed)
        self.left_4link =  self.ik.AddPointToPointDistanceConstraint(self.plant.GetFrameByName("BallTarsusLeft"), [0, 0, 0],
                                                                     self.plant.GetFrameByName("BallFemurLeft"), [0, 0, 0], 
                                                                     0.32, 0.32)
        self.right_4link = self.ik.AddPointToPointDistanceConstraint(self.plant.GetFrameByName("BallTarsusRight"), [0, 0, 0],
                                                                     self.plant.GetFrameByName("BallFemurRight"), [0, 0, 0], 
                                                                     0.32, 0.32)

c = Controller()
