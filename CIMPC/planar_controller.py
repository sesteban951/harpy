
from pydrake.all import *
import numpy as np

class Controller(LeafSystem):
    """
    Simple controller class for the Harpy robot that tracks a modified SRB model trajectory.
    
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

        # current simulation time
        self.t_current = 0.0

        # Store center of mass variables for CoM position and velocity wrt world (for LIP)
        self.p_com = None
        self.v_com = None

        # Store the desired trajectory
        self.GetNominalTrajectory()

    # function to parse the desired SRB trajectory data
    def GetNominalTrajectory(self):
        # load trajectory data
        data_path = "./data/2024-01-02_14:57:31_jump_"
        labels = ["time.txt",
                  "q_sol.txt",
                  "a_sol.txt",
                  "pos_r_foot.txt",
                  "acc_r_foot.txt",
                  "pos_l_foot.txt",
                  "acc_l_foot.txt"]
        
        # time stamps
        data_str = data_path + labels[0]       
        self.time = np.loadtxt(data_str)
        
        # floating base trajectory (x-pos, y-pos, theta)
        data_str = data_path + labels[1]       
        self.pos_base = np.loadtxt(data_str)[:, 0:3]
        data_str = data_path + labels[2]  
        self.acc_base = np.loadtxt(data_str)[:, 0:3]

        # foot end trajectories
        data_str = data_path + labels[3]       
        self.pos_r_foot = np.loadtxt(data_str)
        data_str = data_path + labels[4]
        self.acc_r_foot = np.loadtxt(data_str)

        data_str = data_path + labels[5]       
        self.pos_l_foot = np.loadtxt(data_str)
        data_str = data_path + labels[6]
        self.acc_l_foot = np.loadtxt(data_str)

    # get torques from inverse kinematics
    def InverseDynamicsQP(self, pdd_right_nom, pdd_left_nom, pdd_com_nom, orient_com_nom):
        """
        Solve the whole-body QP

                min_{q'', τ, λ} ||p''_left - p''_left_nom||^2 + 
                                ||p''_right - p''_right_nom||^2 + 
                                ||p''_com - p''_com_nom||^2 +
                                ||τ||^2
                s.t. D(q)q'' + h(q,q') = Bτ + J^Tλ
                     J_h q'' + J'_h q' = 0
                     J_f q'' + J'_f q' = 0
                     λ >= 0
                     λ in friction cones

        for control torques τ that we'll apply on the robot. 

        Args:
                pdd_left_nom: desired acceleration of the left foot
                pdd_right_nom: desired acceleration of the right foot
                pdd_com_nom: desired acceleration of the CoM
                orient_com_nom: desired orientation of the torso body

        Returns:
                tau: joint torques that will achieve the desired accelerations

        Note: assumes that self.plant_context holds the current state
        """
        # Set up a mathematical program
        # TODO(vincekurtz): allocate this in the constructor and just update
        # constraints as needed
        mp = MathematicalProgram()

        # Add decision variables for the joint accelerations, joint torques, and
        # contact forces
        qdd = mp.NewContinuousVariables(self.plant.num_velocities(), "qdd")
        tau = mp.NewContinuousVariables(self.plant.num_actuators(), "tau")
        lambda_left = mp.NewContinuousVariables(3, "lambda_left")
        lambda_right = mp.NewContinuousVariables(3, "lambda_right")

        # Add the left foot tracking cost
        # ||p''_left - p''_left_nom||^2
        J_left = self.plant.CalcJacobianTranslationalVelocity(
                self.plant_context, JacobianWrtVariable.kV,
                self.left_foot_frame, [0, 0, 0], self.plant.world_frame(),
                self.plant.world_frame())
        Jdqd_left = self.plant.CalcBiasTranslationalAcceleration(
                self.plant_context, JacobianWrtVariable.kV,
                self.left_foot_frame, [0, 0, 0], self.plant.world_frame(),
                self.plant.world_frame()).flatten()
        pdd_left = J_left @ qdd + Jdqd_left
        mp.AddQuadraticCost((pdd_left - pdd_left_nom).dot(pdd_left - pdd_left_nom), is_convex=True)

        # Add the right foot tracking cost
        # ||p''_right - p''_right_nom||^2
        J_right = self.plant.CalcJacobianTranslationalVelocity(
                self.plant_context, JacobianWrtVariable.kV,
                self.right_foot_frame, [0, 0, 0], self.plant.world_frame(),
                self.plant.world_frame())
        Jdqd_right = self.plant.CalcBiasTranslationalAcceleration(
                self.plant_context, JacobianWrtVariable.kV,
                self.right_foot_frame, [0, 0, 0], self.plant.world_frame(),
                self.plant.world_frame()).flatten()
        pdd_right = J_right @ qdd + Jdqd_right
        mp.AddQuadraticCost((pdd_right - pdd_right_nom).dot(pdd_right - pdd_right_nom), is_convex=True)

        # Add the CoM tracking cost
        # ||p''_com - p''_com_nom||^2
        J_com = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                self.plant_context, JacobianWrtVariable.kV,
                self.plant.world_frame(), self.plant.world_frame())
        Jdqd_com = self.plant.CalcBiasCenterOfMassTranslationalAcceleration(
                self.plant_context, JacobianWrtVariable.kV,
                self.plant.world_frame(), self.plant.world_frame()).flatten()
        pdd_com = J_com @ qdd + Jdqd_com
        mp.AddQuadraticCost((pdd_com - pdd_com_nom).dot(pdd_com - pdd_com_nom), is_convex=True)

        # Add the torque penalty
        # ||τ||^2
        mp.AddQuadraticCost(tau.dot(tau), is_convex=True)

        # Add the inverse dynamics constraint
        # D(q)q'' + h(q,q') = Bτ + J^Tλ
        D = self.plant.CalcMassMatrix(self.plant_context)
        H = self.plant.CalcBiasTerm(self.plant_context)
        B = self.plant.MakeActuationMatrix()
        mp.AddConstraint(eq(D @ qdd + H, B @ tau))

        # Add the holonomic constraint that enforces the four-bar linkage distances
        # J_h q'' + J'_h q' = 0

        res = OsqpSolver().Solve(mp)
        assert res.is_success(), "Inverse Dynamics QP Failed!"

    def CalcOutput(self, context):
        # Set our internal model to match the state estimte
        x_hat = self.EvalVectorInput(context, 0).get_value()
        self.plant.SetPositionsAndVelocities(self.plant_context, x_hat)
        
        # set current simulation time
        self.t_current = context.get_time()

        # get the current time interval index
        idx = np.where(self.t_current <= self.time)
        if idx[0].size == 0:
            idx = -1
        else:
            idx = idx[0][0]

        # get the desired positions and accelerations
        pos_base_des = self.pos_base[idx]
        acc_base_des = self.acc_base[idx]
        right_foot_pos_des = self.pos_r_foot[idx]
        right_foot_acc_des = self.acc_r_foot[idx]
        left_foot_pos_des = self.pos_l_foot[idx]
        left_foot_acc_des = self.acc_l_foot[idx]

        # print("-"*50)
        # print("pos_base_des: ", pos_base_des)
        # print("acc_base_des: ", acc_base_des)
        # print("right_foot_pos_des: ", right_foot_pos_des)
        # print("right_foot_acc_des: ", right_foot_acc_des)
        # print("left_foot_pos_des: ", left_foot_pos_des)
        # print("left_foot_acc_des: ", left_foot_acc_des)
        
        # get feedwoard torques from inverse dynamics
        tau_ff = self.InverseDynamicsQP(right_foot_acc_des, 
                                        left_foot_acc_des, 
                                        acc_base_des, 
                                        pos_base_des[2])

        # Map generalized positions from IK to actuated joint angles
        q_nom = np.array([0, 0, 
                          0, 0, 
                          0, 0]) 
        v_nom = np.array([0, 0,  
                          0, 0,  
                          0, 0]) 
        x_nom = np.block([q_nom,v_nom])

        return {"tau_ff": np.zeros(6), "x_nom": x_nom, "thrust": np.array([0, 0])}
        