from pydrake.all import *
import numpy as np

class Controller(LeafSystem):
    """
    A simple example of a Drake control system for the Harpy robot. 

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
        
        # init parent class
        LeafSystem.__init__(self)

        # create an internal system model for the controller
        self.plant = MultibodyPlant(0)
        Parser(self.plant).AddModels("./models/urdf/harpy.urdf")
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        # set input ports
        self.input_port = self.DeclareVectorInputPort(
                "x_hat",
                BasicVector(19 + 18))  # 19 positions, 18 velocities

        # Fancy caching stuff so that both outputs can be computed with the same method.
        self._cache = self.DeclareCacheEntry(
                description="controller output cache",
                value_producer=ValueProducer(
                    allocate=lambda: AbstractValue.Make(dict()),
                    calc=lambda context, output: output.set_value(
                        self.CalcOutput(context))))
        
        # set output ports
        self.DeclareVectorOutputPort(
                "tau_ff",
                BasicVector(8),  # 3 DoF per leg, 2 thruster DoFs
                lambda context, output: output.set_value(
                    self._cache.Eval(context)["tau_ff"]),
                prerequisites_of_calc={self._cache.ticket()})

        self.DeclareVectorOutputPort(
                "x_nom",
                BasicVector(16),  # actuated positions + velocities
                lambda context, output: output.set_value(
                    self._cache.Eval(context)["x_nom"]),
                prerequisites_of_calc={self._cache.ticket()})

        self.DeclareVectorOutputPort(
                "thrust",
                BasicVector(2),
                lambda context, output: output.set_value(
                    self._cache.Eval(context)["thrust"]),
                prerequisites_of_calc={self._cache.ticket()})
        
        # Store some useful frames
        self.torso_frame = self.plant.GetFrameByName("Torso")
        self.left_foot_frame = self.plant.GetFrameByName("FootLeft")
        self.right_foot_frame = self.plant.GetFrameByName("FootRight")

        # instantiate IK solver object
        self.ik = InverseKinematics(self.plant)
        
        # Add distance constraints to IK for the 4-bar linkages
        self.ik.AddPointToPointDistanceConstraint(
                self.plant.GetFrameByName("BallTarsusLeft"), [0, 0, 0],
                self.plant.GetFrameByName("BallFemurLeft"), [0, 0, 0], 0.32, 0.32)
        self.ik.AddPointToPointDistanceConstraint(
                self.plant.GetFrameByName("BallTarsusRight"), [0, 0, 0],
                self.plant.GetFrameByName("BallFemurRight"), [0, 0, 0], 0.32, 0.32)

        # inverse kinematics solver settings
        self.epsilon_feet = 0.001   # foot position tolerance     [m]
        self.epsilon_base = 0.01    # torso position tolerance    [m]
        self.epsilon_orient = 0.1   # torso orientation tolerance [rad]
        self.tol_feet = np.array([[self.epsilon_feet], [np.inf], [self.epsilon_feet]])
        self.tol_base = np.array([[np.inf], [np.inf], [self.epsilon_base]])

    def DoInverseKinematics(self, p_right,p_left,p_torso,rpy_torso):
        """
        Solve an inverse kinematics problem, reporting joint angles that will
        correspond to the desired positions of the feet and torso in the world.

        Args:
            p_left: desired position of the left foot in the world frame
            p_right: desired position of the right foot in the world frame
            p_base: desired position of the torso in the world frame
            r_torso: desired orientation of the torso in the world frame
        Returns:
            q: Joint angles that set the feet and torso where we want them
        """
        pass

    def CalcOutput(self, context):
        """
        This is where the magic happens. Compute tau_ff, q_nom, and q_nom based
        on the latest state estimate x_hat.

        All the input port data is contained in 'context'. This method must
        return a dictionary with the keys "tau_ff" and "x_nom".
        """
        # set latest sate to current internal robot state
        x_hat = self.EvalVectorInput(context, 0).get_value()
        self.plant.SetPositionsAndVelocities(self.plant_context, x_hat)

        # Feed-forward joint torques
        tau_ff = np.array([0, 0,   # Thruster: right, left
                           0, 0,   # Hip: right roll, left roll
                           0, 0,   # Hip: right pitch, left pitch 
                           0, 0])  # Knee: right pitch, left pitch 

        # Target joint angles and velocities
        q_nom = np.array([0, 0,   # Thruster: right, left [rad]
                          0, 0,   # Hip: right roll, left roll [rad]
                          0, 0,   # Hip: right pitch, left pitch [rad]
                          0, 0])  # Knee: right pitch, left pitch [rad]
        v_nom = np.array([0, 0,   # Thruster: right, left
                          0, 0,   # Hip: right roll, left roll
                          0, 0,   # Hip: right pitch, left pitch 
                          0, 0])  # Knee: right pitch, left pitch 
        x_nom = np.block([q_nom, v_nom])

        # Forces applied by the thrusters
        thrust = np.array([0, 0])   # left, right

        return {"tau_ff": tau_ff, "x_nom": x_nom, "thrust": thrust}

