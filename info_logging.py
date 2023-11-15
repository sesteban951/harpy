#!/usr/bin/env python3
from pydrake.all import *

import numpy as np
import matplotlib.pyplot as plt

class InfoLogger(LeafSystem):
    """
    Info logging block to output useful information about the simulation
    """
    def __init__(self):
        # initalize parent constructor
        LeafSystem.__init__(self)
        
        # Create and internal system model to store the data
        self.plant = MultibodyPlant(0) # continuous time plant
        Parser(self.plant).AddModelFromFile("./models/urdf/harpy_planar.urdf")
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        # create input port that receives state data
        self.input_port = self.DeclareVectorInputPort("state", 
                                                      BasicVector(13 + 13))
        
        # store some important frames
        self.torso_frame = self.plant.GetFrameByName("Torso")
        self.right_foot_frame = self.plant.GetFrameByName("FootRight")
        self.left_foot_frame = self.plant.GetFrameByName("FootLeft")

        # containers for some important data
        self.p_com = np.zeros((3,1))
        self.v_com = np.zeros((3,1))
        self.lin_mom = np.zeros((3,1))
        self.ang_mom = np.zeros((3,1))
        self.p_torso = np.zeros((3,1))
        self.p_right_foot = np.zeros((3,1))
        self.p_left_foot = np.zeros((3,1))

        # create abstract output port that outputs the data
        self._cache = self.DeclareCacheEntry(description="info output cache",
                                             value_producer=ValueProducer(
                                             allocate=lambda: AbstractValue.Make(dict()),
                                             calc=lambda context, output: output.set_value(
                                                self.CalcOutput(context))))

        # create output port for the center of mass state
        self.DeclareVectorOutputPort("com_state",
                                    BasicVector(6),
                                    lambda context, output: output.set_value(
                                        self._cache.Eval(context)["com_state"]),
                                    prerequisites_of_calc={self._cache.ticket()})
        # create output port for the momentum
        self.DeclareVectorOutputPort("momentum",
                                    BasicVector(6),
                                    lambda context, output: output.set_value(
                                        self._cache.Eval(context)["momentum"]),
                                    prerequisites_of_calc={self._cache.ticket()})
         # create output port for the base position
        self.DeclareVectorOutputPort("base_pos",
                                    BasicVector(3),
                                    lambda context, output: output.set_value(
                                        self._cache.Eval(context)["base_pos"]),
                                    prerequisites_of_calc={self._cache.ticket()})
        # create output port for the right foot position
        self.DeclareVectorOutputPort("right_foot_pos",
                                    BasicVector(3),
                                    lambda context, output: output.set_value(
                                        self._cache.Eval(context)["right_foot_pos"]),
                                    prerequisites_of_calc={self._cache.ticket()})
        # create output port for the left foot position
        self.DeclareVectorOutputPort("left_foot_pos",
                                    BasicVector(3),
                                    lambda context, output: output.set_value(
                                        self._cache.Eval(context)["left_foot_pos"]),
                                    prerequisites_of_calc={self._cache.ticket()})

    # compute CoM state, x = [p_com, v_com]
    def update_CoM_state(self):
        # compute p_com
        p_com = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context)

        # compute v_com
        J = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(self.plant_context, 
                                                                    JacobianWrtVariable.kV,
                                                                    self.plant.world_frame(), 
                                                                    self.plant.world_frame())
        v = self.plant.GetVelocities(self.plant_context)
        v_all = J @ v
        v_com = v_all[0:3]

        # update CoM info
        self.p_com = p_com.T
        self.v_com = v_com.T

    # update linear and angular momentum
    def update_momentum(self):
        pass

    # update points of intersest
    def update_robot_positions(self):
        
        # query the foot position in space
        self.p_torso = self.plant.CalcPointsPositions(self.plant_context, self.torso_frame,
                                                    [0, 0, 0], self.plant.world_frame())
        self.p_right_foot = self.plant.CalcPointsPositions(self.plant_context, self.right_foot_frame,
                                                        [0, 0, 0], self.plant.world_frame())
        self.p_left_foot = self.plant.CalcPointsPositions(self.plant_context, self.left_foot_frame,
                                                        [0, 0, 0], self.plant.world_frame())
        
    # calculate the outputs for logging in the main sim file
    def CalcOutput(self, context):

        # set current state from outer robot to this copy of the robot
        x = self.EvalVectorInput(context, 0).get_value()
        self.plant.SetPositionsAndVelocities(self.plant_context, x)
        
        # update everything
        self.update_CoM_state()
        self.update_momentum()
        self.update_robot_positions()

        # output the data
        com_info = np.concatenate((self.p_com, self.v_com), axis=0).reshape((6,1))  
        momentum_info = np.concatenate((self.lin_mom, self.ang_mom), axis=0)

        dictionary = {"com_state": com_info,
                     "momentum":  momentum_info,
                     "base_pos":  self.p_torso,
                     "right_foot_pos": self.p_right_foot,
                     "left_foot_pos": self.p_left_foot}
    
        return dictionary
