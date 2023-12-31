from pydrake.all import *
import numpy as np

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
        self.plant = MultibodyPlant(0.0)           # 0.0 indicates continuous time
        Parser(self.plant).AddModels("./models/urdf/harpy_planar_half.urdf")
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        self.input_port = self.DeclareVectorInputPort(
                "x_hat",
                BasicVector(8 + 8))  #  4 DoF legs + 1 Dof thruster + 3 DoF base

        # Store some frames that we'll use in the future
        self.torso_frame = self.plant.GetFrameByName("Torso")
        self.right_foot_frame = self.plant.GetFrameByName("FootRight")
        self.ball_tarsus_right_frame = self.plant.GetFrameByName("BallTarsusRight")
        self.ball_femur_right_frame = self.plant.GetFrameByName("BallFemurRight")

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
                BasicVector(3),  # 2 DoF per leg + 1 DoF thruster angle
                lambda context, output: output.set_value(
                    self._cache.Eval(context)["tau_ff"]),
                prerequisites_of_calc={self._cache.ticket()})

        self.DeclareVectorOutputPort(
                "x_nom",
                BasicVector(6),  # q = (knee, hip, thruster_angle), v = "
                lambda context, output: output.set_value(
                    self._cache.Eval(context)["x_nom"]),
                prerequisites_of_calc={self._cache.ticket()})

        self.DeclareVectorOutputPort(
                "thrust",
                BasicVector(1),
                lambda context, output: output.set_value(
                    self._cache.Eval(context)["thrust"]),
                prerequisites_of_calc={self._cache.ticket()})

    def DoInverseKinematics(self, p_left, p_right, p_torso):
        """
        Solve an inverse kinematics problem, reporting joint angles that will
        correspond to the desired positions of the feet and torso in the world.

        Args:
            p_left: desired position of the left foot in the world frame
            p_right: desired position of the right foot in the world frame
            p_base: desired position of the torso in the world frame

        Returns:
            q: Joint angles that set the feet and torso where we want them
        """
        # TODO(vincekurtz): consider allocating this in the constructor and just
        # updating the constraints when this method is called
        ik = InverseKinematics(self.plant)

        # Fix the torso frame in the world
        # TODO(vincekurtz): consider using the joint locking API for the
        # floating base instead
        ik.AddPositionConstraint(self.torso_frame, [0, 0, 0],
                self.plant.world_frame(), p_torso, p_torso)

        # Add distance constraints for the 4-bar linkages
        ik.AddPointToPointDistanceConstraint(
                self.ball_tarsus_left_frame, [0, 0, 0],
                self.ball_femur_left_frame, [0, 0, 0], 0.32, 0.32)
        ik.AddPointToPointDistanceConstraint(
                self.ball_tarsus_right_frame, [0, 0, 0],
                self.ball_femur_right_frame, [0, 0, 0], 0.32, 0.32)

        # Constrain the positions of the feet
        ik.AddPositionConstraint(self.left_foot_frame, [0, 0, 0],
                self.plant.world_frame(), p_left, p_left)
        ik.AddPositionConstraint(self.right_foot_frame, [0, 0, 0],
                self.plant.world_frame(), p_right, p_right)

        res = SnoptSolver().Solve(ik.prog())
        assert res.is_success(), "Inverse Kinematics Failed!"
        
        return res.GetSolution(ik.q())

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

        # Query the position of the CoM in the world frame
        p_com = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context)

        # Query the position of the torso and the feet in the world frame
        p_torso = self.plant.CalcPointsPositions(
                self.plant_context, self.torso_frame,
                [0, 0, 0], self.plant.world_frame())
        p_right = self.plant.CalcPointsPositions(
                self.plant_context, self.right_foot_frame,
                [0, 0, 0], self.plant.world_frame())

        # some random sine function as inputs to track
        A = 3.14/12
        ang_freq = 1.0
        q_ = A * np.sin(ang_freq*context.get_time())
        v_ = ang_freq*A*np.cos(ang_freq*context.get_time())

        q_nom = np.array([q_, q_, q_]) 
        v_nom = np.array([v_, v_, v_]) 
        x_nom = np.block([q_nom, v_nom])

        return {"tau_ff": np.zeros(3), "x_nom": x_nom, "thrust": np.zeros(1)}
