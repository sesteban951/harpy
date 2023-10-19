from pydrake.all import *

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
        self.plant = MultibodyPlant(0.0)
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

        # Constrain the positions of the feet to within a small box
        eps = 1e-5
        ik.AddPositionConstraint(self.left_foot_frame, [0, 0, 0],
                self.plant.world_frame(), p_left - eps, p_left + eps)
        ik.AddPositionConstraint(self.right_foot_frame, [0, 0, 0],
                self.plant.world_frame(), p_right - eps, p_right + eps)

        res = SnoptSolver().Solve(ik.prog())
        assert res.is_success(), "Inverse Kinematics Failed!"
        
        return res.GetSolution(ik.q())
    
    def SolveInverseDynamicsQp(self, pdd_left_nom, pdd_right_nom, pdd_com_nom):
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
        """
        This is where the magic happens. Compute tau_ff, q_nom, and q_nom based
        on the latest state estimate x_hat.

        All the input port data is contained in 'context'. This method must
        return a dictionary with the keys "tau_ff" and "x_nom".
        """
        # Set our internal model to match the state estimte
        x_hat = self.EvalVectorInput(context, 0).get_value()
        self.plant.SetPositionsAndVelocities(self.plant_context, x_hat)
        q = x_hat[:self.plant.num_positions()]
        qd = x_hat[self.plant.num_positions():]

        # Query the position of the CoM in the world frame
        p_com = self.plant.CalcCenterOfMassPositionInWorld(
            self.plant_context).flatten()

        # Query the position of the feet in the world frame
        p_left = self.plant.CalcPointsPositions(
                self.plant_context, self.left_foot_frame,
                [0, 0, 0], self.plant.world_frame()).flatten()
        p_right = self.plant.CalcPointsPositions(
                self.plant_context, self.right_foot_frame,
                [0, 0, 0], self.plant.world_frame()).flatten()
        
        # Compute jacobians of the foot and CoM positions
        J_com = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                self.plant_context, JacobianWrtVariable.kV,
                self.plant.world_frame(), self.plant.world_frame())

        J_left = self.plant.CalcJacobianTranslationalVelocity(
                self.plant_context, JacobianWrtVariable.kV,
                self.left_foot_frame, [0, 0, 0], self.plant.world_frame(),
                self.plant.world_frame())
        J_right = self.plant.CalcJacobianTranslationalVelocity(
                self.plant_context, JacobianWrtVariable.kV,
                self.right_foot_frame, [0, 0, 0], self.plant.world_frame(),
                self.plant.world_frame())
        
        # Define target positions for the CoM and feet
        # TODO: compute these using a reduced-order model + swing foot splines
        p_left_target = np.array([0.0, 0.065, 0.2])
        p_right_target = np.array([0.0, -0.065, 0.04])
        p_com_target = np.array([0.0, 0.0, 0.525])

        # Use a PD controller to compute desired accelerations
        pd_left = J_left @ qd
        pd_right = J_right @ qd
        pd_com = J_com @ qd
        pdd_left_nom = 100 * (p_left_target - p_left) - 10 * pd_left
        pdd_right_nom = 100 * (p_right_target - p_right) - 10 * pd_right
        pdd_com_nom = 100 * (p_com_target - p_com) - 10 * pd_com

        # Solve the whole-body QP to compute joint torques
        tau = self.SolveInverseDynamicsQp(pdd_left_nom, pdd_right_nom, pdd_com_nom)

        x_nom = np.zeros(12)

        return {"tau_ff": np.zeros(6), "x_nom": x_nom, "thrust": np.zeros(2)}

