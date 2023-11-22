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

        # store swing and stance foot variables
        self.t_current = 0.0
        self.stance_foot_frame = None
        self.swing_foot_frame = None
        self.swing_foot_last_gnd_pos = None
        self.stance_foot_last_gnd_pos = None

        # Store center of mass variables for CoM position and velocity wrt world (for LIP)
        self.p_com = None
        self.v_com = None
        self.p = None
        self.v = None

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

        # Linear Inverted Pendulum parameters
        self.z0 = 0.5                   # const CoM height [m]
        self.z_apex = 0.2               # apex height [m]
        self.T = 0.25                   # step period [s]

    def DoInverseKinematics(self, p_right, p_left, p_torso, r_torso):
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
        # instantiate inverse kinematics solver
        ik = InverseKinematics(self.plant)

        # set tolerance vectors for postions and orientations
        epsilon_feet = 0.001
        epsilon_base = 0.01
        epsilon_orient = 0.1
        tol_feet = np.array([[epsilon_feet], [np.inf], [epsilon_feet]])
        tol_base = np.array([[np.inf], [np.inf], [epsilon_base]])

        # Torso position constraint
        p_torso_lb = p_torso - tol_base
        p_torso_ub = p_torso + tol_base
        ik.AddPositionConstraint(self.torso_frame, [0, 0, 0],
                self.plant.world_frame(), p_torso_lb, p_torso_ub)
        
        # Torso orientation constraint
        ik.AddOrientationConstraint(self.torso_frame, RotationMatrix(),
                                    self.plant.world_frame(), RotationMatrix(),
                                    epsilon_orient)

        # Constrain the positions of the feet with bounding boxes
        p_left_lb = p_left - tol_feet
        p_left_ub = p_left + tol_feet
        p_right_lb = p_right - tol_feet
        p_right_ub = p_right + tol_feet
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

    # Compute foot placement location, with respect to stance foot
    def foot_placement(self):
        """
        Updates foot placement location
        """     
        # HLIP state parameters
        p = self.p[0][0]
        v = self.v[0]

        # HLIP parameters
        u_des = 0.0                        # P1 orbit step size desired
        x = np.array([[p], [v]])           # HLIP state
        x_des = np.array([[0.0], [0.0]])   # HLIP state desired
        K = np.array([.5, 0.5])           # feedback gain
    
        u = 0.6*v 
        return u

    # Query desired foot position from B-slpine trajectory
    def foot_bezier(self):
        """
        Updates right adn left foot desired positions.
        Uses a 7-pt or 5-pt bezier curve to generate a trajectory for the swing foot.
        """
        # compute step count and swing foot time
        step_count = np.floor(self.t_current/self.T)
        time = self.t_current % self.T
        
        # bezier curve offsets
        z_offset = 0.027                          # z-offset for foot 
        z0 = 0.0                                 # inital swing foot height
        zf = -0.01                               # final swing foot height (neg to ensure foot strike)
        u0 = self.stance_foot_last_gnd_pos[0][0]  # initial swing foot position
        uf = u0 + self.foot_placement()               # final swing foot position
    
        # compute bezier curve control points, 7-pt or 5-pt bezier
        n = 7
        if n == 7:
            ctrl_pts_z = np.array([[z0],[z0],[z0],[(16/5)*self.z_apex],[zf],[zf],[zf]]) + z_offset
            ctrl_pts_x = np.array([[u0],[u0],[u0],[(u0+uf)/2],[uf],[uf],[uf]])
        elif n == 5:
            ctrl_pts_z = np.array([[z0],[z0],[(8/3)*self.z_apex],[zf],[zf]]) + z_offset
            ctrl_pts_x = np.array([[u0],[u0],[(u0+uf)/2],[uf],[uf]])
        else:
            print("Invalid number of control points")
        ctrl_pts = np.vstack((ctrl_pts_x.T,ctrl_pts_z.T))

        # evaluate bezier at time t
        bezier = BezierCurve(0,self.T,ctrl_pts)
        b = np.array(bezier.value(time))       
        
        # choose which foot to move, one in stance and one in swing
        swing_target = np.array([b.T[0][0], 0.0, b.T[0][1]])[None].T
        stance_target = np.array([self.swing_foot_last_gnd_pos[0][0], 0.0, z_offset])[None].T
        
        # left foot in swing
        if step_count % 2 == 0:
            p_right = stance_target
            p_left = swing_target
        # right foot in swing
        else:
            p_right = swing_target
            p_left = stance_target

        return p_right, p_left
    
    # update stance and swing foot variables
    def update_LIP_state(self):
        """
        Updates the LIP model state. 
        x = [p_com - p_stance, v_com]
        """
        # compute step count and swing foot time
        step_count = np.floor(self.t_current/self.T)

        # choose which foot to move, one in stance and one in swing
        # left foot in swing
        if step_count % 2 == 0: 
            self.swing_foot_last_gnd_pos = self.plant.CalcPointsPositions(self.plant_context, self.right_foot_frame,
                                                                      [0, 0, 0], self.plant.world_frame())
            self.stance_foot_last_gnd_pos = self.plant.CalcPointsPositions(self.plant_context, self.left_foot_frame,
                                                                      [0, 0, 0], self.plant.world_frame())
            self.stance_foot_frame = self.right_foot_frame
            self.swing_foot_frame = self.left_foot_frame
        # right foot in swing
        else:               
            self.swing_foot_last_gnd_pos = self.plant.CalcPointsPositions(self.plant_context, self.left_foot_frame,
                                                                      [0, 0, 0], self.plant.world_frame())
            self.stance_foot_last_gnd_pos = self.plant.CalcPointsPositions(self.plant_context, self.right_foot_frame,
                                                                      [0, 0, 0], self.plant.world_frame())
            self.stance_foot_frame = self.left_foot_frame
            self.swing_foot_frame = self.right_foot_frame

        # update LIP state: p = [p_com - p_stance] and v = [v_com]
        self.p = (self.p_com - self.plant.CalcPointsPositions(self.plant_context, self.stance_foot_frame,
                                                            [0, 0, 0], self.plant.world_frame()).T).T
        self.v = self.v_com

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
        self.t_current = context.get_time()
        
        # update the CoM position and velocity values, stance and swing foot, LIP states
        self.update_CoM_state()
        self.update_LIP_state()

        # Compute target base position. (fix orientation and height)
        p_torso = self.plant.CalcPointsPositions(
                self.plant_context, self.torso_frame,
                [0, 0, 0], self.plant.world_frame())
        torso_pos_target = np.array([[p_torso[0][0]], [p_torso[1][0]], [self.z0]])
        torso_rpy_target = RotationMatrix()

        # Compute target foot positions. (fixed stance and swinging foot)
        right_pos_target, left_pos_target = self.foot_bezier()

        # find desired configuration coordinates to track LIP
        q_ik = self.DoInverseKinematics(right_pos_target, left_pos_target, 
                                        torso_pos_target, torso_rpy_target)
        
        # Map generalized positions from IK to actuated joint angles
        # TODO: We should probably give the controller velocity info to prevent jerkiness
        q_nom = np.array([
            q_ik[3], q_ik[4],    # thrusters
            q_ik[5], q_ik[9],    # hip
            q_ik[6], q_ik[10]])  # knee
        v_nom = np.zeros(6)
        x_nom = np.block([q_nom, v_nom])

        return {"tau_ff": np.zeros(6), "x_nom": x_nom, "thrust": np.array([0, 0])}