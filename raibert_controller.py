from pydrake.all import *
import numpy as np

class RaibertController(LeafSystem):
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
        Parser(self.plant).AddModels("./models/urdf/harpy.urdf")
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        # set the input port
        self.input_port = self.DeclareVectorInputPort(
                "x_hat",
                BasicVector(12 + 12))  # 19 positions, 18 velcoities

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
                BasicVector(8),  # 3 DoF per leg, 2 thruster angle
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

        # Store some frames that we'll use in the future
        self.torso_frame = self.plant.GetFrameByName("Torso")
        self.left_foot_frame = self.plant.GetFrameByName("FootLeft")
        self.right_foot_frame = self.plant.GetFrameByName("FootRight")

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

        # instantiate inverse kinematics solver
        self.ik = InverseKinematics(self.plant)

        # set intial condition for quaternion to be on S^3
        # self.ik.prog().SetInitialGuess(self.ik.prog().decision_variables()[0],1)
        # self.ik.prog().SetInitialGuess(self.ik.prog().decision_variables()[1],0)
        # self.ik.prog().SetInitialGuess(self.ik.prog().decision_variables()[2],0)
        # self.ik.prog().SetInitialGuess(self.ik.prog().decision_variables()[3],0)
        
        # inverse kinematics solver settings
        self.epsilon_feet = 0.0001    # foot position tolerance     [m]
        self.epsilon_base = 0.01    # torso position tolerance    [m]
        self.epsilon_orient = 0.05   # torso orientation tolerance [rad]
        self.tol_feet = np.array([[self.epsilon_feet], [self.epsilon_feet], [self.epsilon_feet]])
        self.tol_base = np.array([[np.inf], [np.inf], [self.epsilon_base]])

        # Add distance constraints to IK for the 4-bar linkages (fixed)
        self.left_4link =  self.ik.AddPointToPointDistanceConstraint(self.plant.GetFrameByName("BallTarsusLeft"), [0, 0, 0],
                                                                     self.plant.GetFrameByName("BallFemurLeft"), [0, 0, 0], 
                                                                     0.32, 0.32)
        self.right_4link = self.ik.AddPointToPointDistanceConstraint(self.plant.GetFrameByName("BallTarsusRight"), [0, 0, 0],
                                                                     self.plant.GetFrameByName("BallFemurRight"), [0, 0, 0], 
                                                                     0.32, 0.32)
        # Add torso constraints
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

        # Linear Inverted Pendulum parameters
        self.z0 = 0.5                   # const CoM height [m]
        self.z_apex = 0.2               # apex height [m]
        self.T = 0.25                   # step period [s]

    def DoInverseKinematics(self, p_right, p_left, p_torso):
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

        # attempt to solve the IK problem        
        res = SnoptSolver().Solve(self.ik.prog())
        assert res.is_success(), "Inverse Kinematics Failed!"
        
        return res.GetSolution(self.ik.q())

    # Compute foot placement location, with respect to stance foot
    def foot_placement(self):
        """
        Updates foot placement location
        """     
        # LIP state parameters
        p_x = self.p[0][0]
        p_y = self.p[1][0]
        v_x = self.v[0]
        v_y = self.v[1]

        # x-direction tuning parameters
        Kv_x = 0.0
        Kp_x = 0.0
        u_x = Kv_x*(v_x) - Kp_x*(p_x)

        # y-direction tuning parameters
        Kv_y = 2.0
        Kp_y = 0.0
        u_y = Kv_y*(v_y) - Kp_y*(p_y) + 0.
        return u_x, u_y

    # Query desired foot position from B-slpine trajectory
    def foot_bezier(self):
        """
        Updates right adn left foot desired positions.
        Uses a 7-pt or 5-pt bezier curve to generate a trajectory for the swing foot.
        """
        # compute step count and swing foot time
        step_count = np.floor(self.t_current/self.T)
        time = self.t_current % self.T
    
        # z-bezier curve offsets
        z_offset = 0.0                         # z-offset for foot 
        z0 = -0.01                               # inital swing foot height
        zf = -0.01                               # final swing foot height (neg to ensure foot strike)
        
        uf_x, uf_y = self.foot_placement()       # final swing foot positions

        # x-direction foot placement
        u0_x = self.stance_foot_last_gnd_pos[0][0]  # initial swing foot position
        uf_x = u0_x + self.foot_placement()         # final swing foot position
        uf_x = uf_x[0]
        # y-direction foot placement
        u0_y = self.stance_foot_last_gnd_pos[1][0]  # initial swing foot position
        uf_y = u0_x + self.foot_placement()         # final swing foot position
        uf_y = uf_y[0]

        # compute bezier curve control points, 7-pt or 5-pt bezier
        n = 7
        if n == 7:
            ctrl_pts_x = np.array([[u0_x],[u0_x],[u0_x],[(u0_x+uf_x)/2],[uf_x],[uf_x],[uf_x]])
            ctrl_pts_y = np.array([[u0_y],[u0_y],[u0_y],[(u0_y+uf_y)/2],[uf_y],[uf_y],[uf_y]])
            ctrl_pts_z = np.array([[z0],[z0],[z0],[(16/5)*self.z_apex],[zf],[zf],[zf]]) + z_offset
        elif n == 5:
            ctrl_pts_x = np.array([[u0_x],[u0_x],[(u0_x+uf_x)/2],[uf_x],[uf_x]])
            ctrl_pts_y = np.array([[u0_y],[u0_y],[(u0_y+uf_y)/2],[uf_y],[uf_y]])
            ctrl_pts_z = np.array([[z0],[z0],[(8/3)*self.z_apex],[zf],[zf]]) + z_offset
        else:
            print("Invalid number of control points")
        
        ctrl_pts = np.vstack((ctrl_pts_x.T,
                              ctrl_pts_y.T,
                              ctrl_pts_z.T))

        # evaluate bezier at time t
        bezier = BezierCurve(0,self.T,ctrl_pts)
        b = np.array(bezier.value(time))       
        
        # choose which foot to move, one in stance and one in swing
        stance_target = np.array([self.swing_foot_last_gnd_pos[0][0],
                                  self.swing_foot_last_gnd_pos[1][0],
                                  z_offset])[None].T
    
        # left foot in swing
        if step_count % 2 == 0:
            swing_target = np.array([b.T[0][0], b.T[0][1] + 0.0, b.T[0][2]])[None].T
            p_right = stance_target
            p_left = swing_target
        # right foot in swing
        else:
            swing_target = np.array([b.T[0][0], b.T[0][1] - 0.0, b.T[0][2]])[None].T
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

        print("p_com: ")
        print(self.p_com)
        print("v_com: ")
        print(self.v_com)

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
        
        # # update the CoM position and velocity values, stance and swing foot, LIP states
        # self.update_CoM_state()
        # self.update_LIP_state()

        # # Compute target base position. (fix orientation and height)
        p_torso = self.plant.CalcPointsPositions(
                self.plant_context, self.torso_frame,
                [0, 0, 0], self.plant.world_frame())
        # torso_pos_target = np.array([[p_torso[0][0]], [p_torso[1][0]], [self.z0]])

        p_right = self.plant.CalcPointsPositions(
                self.plant_context, self.right_foot_frame,
                [0, 0, 0], self.plant.world_frame())
        p_left = self.plant.CalcPointsPositions(
                self.plant_context, self.left_foot_frame,
                [0, 0, 0], self.plant.world_frame())

        # # Compute target foot positions. (fixed stance and swinging foot)
        torso_pos_target = p_torso
        right_pos_target =  np.array([[0.3], [-0.065], [0.0 + 0.3]])
        left_pos_target =  np.array([[0.1], [0.2], [0.0 + 0.1]])

        print(50*"*")
        # print("time: ", self.t_current)
        # print("steps: ", np.floor(self.t_current/self.T))
        # print("swing foot: ", self.swing_foot_frame.name())
        print("rightfoot: ", p_right)
        print("right foot target: ", right_pos_target)
        print("right error:",np.linalg.norm(p_right - right_pos_target))
        print("leftfoot: ", p_left)
        print("left foot target: ", left_pos_target)
        print("left error:",np.linalg.norm(p_left - left_pos_target))

        # find desired configuration coordinates to track LIP
        q_ik = self.DoInverseKinematics(right_pos_target, left_pos_target, torso_pos_target)
        print(q_ik)
        # Target joint angles and velocities
        # TODO: We should probably give the controller velocity info to prevent jerkiness
        n = 7
        q_nom = np.array([q_ik[7-n],  q_ik[8-n],    # Thruster: right, left [rad]
                          q_ik[9-n],  q_ik[14-n],   # Hip: right roll, left roll [rad]
                          q_ik[10-n], q_ik[15-n],   # Hip: right pitch, left pitch [rad]
                          q_ik[11-n], q_ik[16-n]])  # Knee: right pitch, left pitch [rad]
        v_nom = np.array([0, 0,   # Thruster: right, left
                          0, 0,   # Hip: right roll, left roll
                          0, 0,   # Hip: right pitch, left pitch 
                          0, 0])  # Knee: right pitch, left pitch 
        x_nom = np.block([q_nom, v_nom])

        return {"tau_ff": np.zeros(8), "x_nom": x_nom, "thrust": np.array([0, 0])}

