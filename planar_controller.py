from pydrake.all import *

import numpy as np
import sys 

sys.path.append('./ROM/')
from LinearInvertedPendulum import LinearInvertedPendulum

class PlanarRaibertController(LeafSystem):
    """
    A simple controller for the planar harpy robot based on the Raibert
    heuristic.

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

        # Store some frames that we'll use in the future
        self.torso_frame = self.plant.GetFrameByName("Torso")
        self.left_foot_frame = self.plant.GetFrameByName("FootLeft")
        self.right_foot_frame = self.plant.GetFrameByName("FootRight")

        self.ball_tarsus_left_frame = self.plant.GetFrameByName("BallTarsusLeft")
        self.ball_tarsus_right_frame = self.plant.GetFrameByName("BallTarsusRight")
        self.ball_femur_left_frame = self.plant.GetFrameByName("BallFemurLeft")
        self.ball_femur_right_frame = self.plant.GetFrameByName("BallFemurRight")

        # Store center of mass variables for CoM position and velocity wrt world
        self.p_com = None
        self.v_com = None

        # We'll do some fancy caching stuff so that both outputs can be
        # computed with the same method.
        self._cache = self.DeclareCacheEntry(
                description="controller output cache",
                value_producer=ValueProducer(
                    allocate=lambda: AbstractValue.Make(dict()),
                    calc=lambda context, output: output.set_value(
                        self.CalcOutput(context))))
        
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

        # Linear Inverted Pendulum trajectory generation
        g  = 9.81                        # gravity [m/s^2]
        z0 = 0.3                         # const CoM height [m]
        self.z_apex = 0.25               # apex height [m]
        self.T = 1.0                     # step period [s]
        alpha = 0.28                     # raibert controller parameters
        beta = 1                         # raibert controller parameters
        N = 1                            # number of steps
        x0 = np.array([0.01,0.01])       # initial state [m, m/s]
        dt = 0.01                        # time step [s]

        # Linear Inverted Pendulum trajectory generation
        LIP = LinearInvertedPendulum(N,x0)
        LIP.set_physical_params(g,z0)
        LIP.set_walking_params(self.z_apex,self.T)
        LIP.set_raibert_params(alpha,beta)
        LIP.set_traj_settings(N,dt,x0)
        self.t, self.x, self.b = LIP.make_trajectories()

    def DoInverseKinematics(self, p_left, p_right, p_torso, epsilon=1e-2):
        """
        Solve an inverse kinematics problem, reporting joint angles that will
        correspond to the desired positions of the feet and torso in the world.

        Args:
            p_left: desired position of the left foot in the world frame
            p_right: desired position of the right foot in the world frame
            p_base: desired position of the torso in the world frame
            epsilon: tolerance for positions

        Returns:
            q: Joint angles that set the feet and torso where we want them
        """
        # TODO(vincekurtz): consider allocating this in the constructor and just
        # updating the constraints when this method is called
        ik = InverseKinematics(self.plant)

        # set tolerance vector for postions
        tol = np.array([[epsilon], [np.inf], [epsilon]])

        # Fix the torso frame in the world
        # TODO(vincekurtz): consider using the joint locking API for the
        # floating base instead
        p_torso_lb = p_torso - tol
        p_torso_ub = p_torso + tol
        ik.AddPositionConstraint(self.torso_frame, [0, 0, 0],
                self.plant.world_frame(), p_torso_lb, p_torso_ub)

        # Constrain the positions of the feet with bounding boxes
        p_left_lb = p_left - tol
        p_left_ub = p_left + tol
        p_right_lb = p_right - tol
        p_right_ub = p_right + tol
        ik.AddPositionConstraint(self.left_foot_frame, [0, 0, 0],
                self.plant.world_frame(), p_left_lb, p_left_ub)
        ik.AddPositionConstraint(self.right_foot_frame, [0, 0, 0],
                self.plant.world_frame(), p_right_lb, p_right_ub)

        # Add distance constraints for the 4-bar linkages
        ik.AddPointToPointDistanceConstraint(
                self.ball_tarsus_left_frame, [0, 0, 0],
                self.ball_femur_left_frame, [0, 0, 0], 0.32, 0.32)
        ik.AddPointToPointDistanceConstraint(
                self.ball_tarsus_right_frame, [0, 0, 0],
                self.ball_femur_right_frame, [0, 0, 0], 0.32, 0.32)

        # attempt to solve the IK problem        
        res = SnoptSolver().Solve(ik.prog())
        assert res.is_success(), "Inverse Kinematics Failed!"
        
        return res.GetSolution(ik.q())

    # Query desired foot position from B-slpine trajectory
    def get_foot_pos(self, t):
        
        # compute step count and swing foot time
        step_count = np.floor(t/self.T)
        time = t % self.T
        
        # bezier curve offsets
        z_offset = -0.0  # z-offset for foot 
        u = 0            # x-direction foot placement

        # compute bezier curve control points, 5-pt bezier
        ctrl_pts_z = np.array([[0],[0],[(8/3)*self.z_apex],[0],[0]]) + z_offset
        ctrl_pts_x = np.array([[0],[0],[u/2],[u],[u]])
        ctrl_pts = np.vstack((ctrl_pts_x.T,ctrl_pts_z.T))

        # evaluate bezier at time t
        bezier = BezierCurve(0,self.T,ctrl_pts)
        b = np.array(bezier.value(time))       
        
        # choose which foot to move, one in stance and one in swing
        b_sw = np.array([b.T[0][0], 0.0, b.T[0][1]])[None].T
        b_st = np.array([0.0, 0.0, 0.0])[None].T
        if step_count % 2 == 0:
            p_right = b_sw
            p_left = b_st
        else:
            p_right = b_st
            p_left = b_sw

        return p_right, p_left
    
    # Estimate CoM position and velocity wrt to world frame
    def update_com(self):
        
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
        self.p_com = p_com
        self.v_com = v_com

    def CalcOutput(self, context):
        """
        This is where the magic happens. Compute tau_ff, q_nom, and q_nom based
        on the latest state estimate x_hat.

        All the input port data is contained in 'context'. This method must
        return a dictionary with the keys "tau_ff" and "x_nom".
        """
        # Set our internal model to match the state estimte
        x_hat = self.EvalVectorInput(context, 0).get_value()
        self.plant.SetPositionsAndVelocities(self.plant_context, x_hat)

        # update the CoM position and velocity values
        self.update_com()

        # Query the position of the torso and the feet in the world frame
        p_torso = self.plant.CalcPointsPositions(
                self.plant_context, self.torso_frame,
                [0, 0, 0], self.plant.world_frame())
        p_left = self.plant.CalcPointsPositions(
                self.plant_context, self.left_foot_frame,
                [0, 0, 0], self.plant.world_frame())
        p_right = self.plant.CalcPointsPositions(
                self.plant_context, self.right_foot_frame,
                [0, 0, 0], self.plant.world_frame())

        # Do some inverse kinematics to find joint angles that set a new foot position
        p_left_target = p_left
        p_right_target = p_right
        p_torso_target = p_torso
        q_ik = self.DoInverseKinematics(p_left_target, p_right_target, p_torso_target, epsilon=1e-4)

        # Map generalized positions from IK to actuated joint angles
        q_nom = np.array([
            q_ik[3], q_ik[4],   # thrusters
            q_ik[5], q_ik[6],   # hip
            q_ik[9], q_ik[10]])  # knee
        # q_nom = np.zeros(6)
        v_nom = np.zeros(6)
        x_nom = np.block([q_nom, v_nom])

        return {"tau_ff": np.zeros(6), "x_nom": x_nom, "thrust": np.zeros(2)}